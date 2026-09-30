#!/usr/bin/env python3
# Copyright 2026 The AI Edge LiteRT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs the benchmark on a Developer Device Platform (DDP) iPhone and lays the result out as a session.

DDP runs iOS as XCTest. This script submits the archive build_xctest.sh made through
`gcloud beta device-run sessions submit xctest`, with the model and a benchmark_args.json pushed
into the app container and Documents/out/ pulled back, waits for the session, and writes what
Tests/LiteRTBenchmarkTests.mm produced in the layout run_ios.sh produces:

    <session dir>/<DDP session id>/<accelerator>-<device id>/{results.pb, runtime_info.pb, stdout.txt, benchmark.done}
    <session dir>/<DDP session id>/session.json    (platform ios, runner ddp, the device's catalog name and OS)

which ../driver/collect.py turns into a board row. A session is one device, one model and one
accelerator, so every run is a process of its own, as on the other platforms: run the script once
per accelerator. `--collect SESSION_ID` submits nothing and lays out a session submitted earlier
with the same device, accelerator and model.

Needs gcloud with the device-run beta commands, logged in to a project with the Device Run API on
(LITERT_GCP_PROJECT or --project), which DDP bills the session to. `gcloud beta device-run devices
list` shows the device ids; `gcloud beta device-run software-versions list` the Xcode versions.

Exit status: 0 the job passed and results.pb is laid out; 2 bad inputs (a missing or unreadable
file, a device that is not an iPhone, a session that does not match --collect's flags, an
existing session directory without --force); 3 gcloud missing, or a gcloud command that failed or
printed something unexpected before the session had ended; 4 the job did not pass; 5 the job
passed but the pull brought back no results.pb (also when the download itself failed); 130 when
interrupted, with the session named so it can be cancelled.
"""

import argparse
import datetime
import json
import os
import pathlib
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_SESSION_DIR = pathlib.Path.home() / ".cache" / "litert-samples-benchmark" / "ddp-ios"
DEFAULT_ARCHIVE = HERE / "out" / "LiteRTBenchmark-xctest.zip"
JOB_FILES = ("results.pb", "runtime_info.pb", "stdout.txt", "benchmark.done")
POLL_SECS = 30
DESCRIBE_TRIES = 3
EXIT_INPUT, EXIT_GCLOUD, EXIT_JOB, EXIT_RESULTS, EXIT_INTERRUPTED = 2, 3, 4, 5, 130


class GcloudError(Exception):
    """A gcloud command that could not run, failed, or printed something unexpected."""


def gcloud(*args: str, project: str | None = None) -> str:
    """Runs a gcloud command and returns its stdout; GcloudError when it is missing or fails."""
    cmd = ["gcloud", *args] + ([f"--project={project}"] if project else [])
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise GcloudError("gcloud is not on PATH; install the Google Cloud CLI and log in") from None
    if res.returncode != 0:
        raise GcloudError(f"`{' '.join(cmd[:5])} ...` failed:\n{tail(res.stderr)}")
    return res.stdout


def gcloud_json(*args: str, project: str | None = None):
    out = gcloud(*args, "--format=json", project=project)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        raise GcloudError(f"`gcloud {' '.join(args[:5])} ...` printed no JSON:\n{tail(out)}") from None


def tail(text: str, lines: int = 8) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def read_archive(archive: pathlib.Path) -> tuple[str | None, str | None]:
    """(the app's bundle id from the manifest's TestHostBundleIdentifier, the BUILD_INFO text)."""
    bundle_id = build_info = None
    try:
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if name == "BUILD_INFO":
                    build_info = zf.read(name).decode(errors="replace").strip()
                elif name.endswith(".xctestrun") and "/" not in name and bundle_id is None:
                    manifest = plistlib.loads(zf.read(name))
                    for target in manifest.values() if isinstance(manifest, dict) else []:
                        if isinstance(target, dict) and target.get("TestHostBundleIdentifier"):
                            bundle_id = target["TestHostBundleIdentifier"]
    except (zipfile.BadZipFile, plistlib.InvalidFileException, ValueError, OSError):
        return None, None
    return bundle_id, build_info


def benchmark_args(model_name: str, accelerator: str, extra: list[str]) -> list[str]:
    """The flags the XCTest runs the tool with: the model, --use_gpu=true for gpu, then the extras."""
    return [f"--graph={model_name}"] + (["--use_gpu=true"] if accelerator == "gpu" else []) + list(extra)


def submit_args(args, bundle_id: str, args_path: pathlib.Path, run: str) -> list[str]:
    """The `sessions submit xctest` arguments; `run` labels the session so it can be found again."""
    model = pathlib.Path(args.model)
    return [
        "beta", "device-run", "sessions", "submit", "xctest",
        f"--device={args.device}",
        f"--test={args.test_zip}",
        f"--other-files-to-push={model}={bundle_id}:/Documents/{model.name},"
        f"{args_path}={bundle_id}:/Documents/benchmark_args.json",
        f"--paths-to-pull={bundle_id}:/Documents/out",
        f"--xctest-timeout={args.timeout}",
        f"--xcode-version={args.xcode_version}",
        f"--labels=tool=litert-samples-ios,model={model.name},accelerator={args.accelerator},run={run}",
        "--async",
    ]


def catalog_device(device_id: str, project: str | None) -> dict | None:
    """The catalog entry of a device id (`devices list`, which carries platform, name and OS)."""
    for entry in gcloud_json("beta", "device-run", "devices", "list", project=project):
        if isinstance(entry, dict) and entry.get("name", "").rsplit("/", 1)[-1] == device_id:
            return entry
    return None


def session_id(submitted) -> str | None:
    """The session id from what `sessions submit --async` printed, when it printed one (gcloud 587
    prints an empty list; see find_session)."""
    if isinstance(submitted, list):
        submitted = next((s for s in submitted if isinstance(s, dict)), {})
    if not isinstance(submitted, dict):
        return None
    for value in (submitted.get("name"), (submitted.get("metadata") or {}).get("session")):
        if isinstance(value, str) and "/sessions/" in value:
            return value.rsplit("/", 1)[-1]
    return None


def session_labels(session: dict) -> dict | None:
    jobs = (session.get("sessionConfig") or {}).get("jobConfigs") or []
    return jobs[0].get("labels") if jobs and isinstance(jobs[0], dict) else None


def session_device(session: dict) -> str | None:
    jobs = (session.get("sessionConfig") or {}).get("jobConfigs") or []
    configs = ((jobs[0].get("allocationConfig") or {}).get("deviceConfigs") or []) if jobs else []
    return ((configs[0].get("requirement") or {}).get("deviceId")) if configs else None


def find_session(run: str, project: str | None, tries: int = 6) -> str | None:
    """The id of the session labelled run=<run>, from `sessions list`; retried while the list lags."""
    for attempt in range(tries):
        for session in gcloud_json("beta", "device-run", "sessions", "list", project=project):
            if isinstance(session, dict) and (session_labels(session) or {}).get("run") == run:
                return session.get("name", "").rsplit("/", 1)[-1] or None
        if attempt + 1 < tries:
            time.sleep(10)
    return None


def describe(sid: str, project: str | None) -> dict:
    """`sessions describe --full`, retried a few times so one dropped call does not end the wait."""
    for attempt in range(DESCRIBE_TRIES):
        try:
            session = gcloud_json("beta", "device-run", "sessions", "describe", sid, "--full", project=project)
            return session if isinstance(session, dict) else {}
        except GcloudError:
            if attempt + 1 == DESCRIBE_TRIES:
                raise
            time.sleep(POLL_SECS)
    raise AssertionError("unreachable")


def wait_for_session(sid: str, project: str | None, max_secs: float) -> dict:
    """Polls the session, checking before each wait, until it is done; GcloudError past max_secs."""
    deadline = time.monotonic() + max_secs
    while True:
        session = describe(sid, project)
        report = session.get("sessionReport") or {}
        status = (report.get("status") or {}).get("statusType", "pending")
        if status == "DONE" or report.get("endTime"):
            return session
        if time.monotonic() >= deadline:
            raise GcloudError(f"session {sid} is still {status} after {int(max_secs)}s; "
                              f"`gcloud beta device-run sessions cancel {sid}` stops it")
        print(f"# {status}; next check in {POLL_SECS}s")
        time.sleep(POLL_SECS)


def execution_prefix(job: dict) -> str | None:
    """The GCS directory of the job's execution, from any of its output files."""
    for execution in job.get("executionReports") or []:
        for f in execution.get("outputFiles") or []:
            m = re.match(r"(.*/execution-[^/]+/)", (f.get("gcsOutputFile") or {}).get("path", ""))
            if m:
                return m.group(1)
    return None


def lay_out(downloaded: pathlib.Path, job_dir: pathlib.Path, session_dir: pathlib.Path) -> None:
    """Copies the pulled out/ files into the job directory and the lab's logs beside it.

    The lab puts the pulled files under the execution's artifacts/ without their device path
    (artifacts/results.pb, not artifacts/Documents/out/results.pb), so they are found by name.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    for name in JOB_FILES:
        found = sorted(p for p in downloaded.rglob(name) if p.is_file())
        if found:
            shutil.copy2(found[0], job_dir / name)
    for name in ("system.log", "junit.xml"):
        found = sorted(p for p in downloaded.rglob(name) if p.is_file())
        if found:
            shutil.copy2(found[0], session_dir / name)


def done_status(job_dir: pathlib.Path) -> int | None:
    """The tool's exit status the test wrote, or None without a readable benchmark.done."""
    try:
        return int((job_dir / "benchmark.done").read_text().strip())
    except (OSError, ValueError):
        return None


def print_result_lines(job_dir: pathlib.Path) -> None:
    log = job_dir / "stdout.txt"
    for line in log.read_text(errors="replace").splitlines() if log.exists() else []:
        if re.search(r"Inference \(avg\)|Model initialization|Overall footprint", line):
            print("  " + line.strip())


def job_tail(job_dir: pathlib.Path, session_dir: pathlib.Path) -> str:
    """What to show for a job that did not pass: the tool's own log when it came back, else the lab's."""
    log = job_dir / "stdout.txt"
    if log.exists():
        return f"benchmark.done says {done_status(job_dir)}; stdout.txt ends:\n{tail(log.read_text(errors='replace'), 12)}"
    for name in ("junit.xml", "system.log"):
        if (session_dir / name).exists():
            return f"no stdout.txt came back; {name} ends:\n{tail((session_dir / name).read_text(errors='replace'), 12)}"
    return "no stdout.txt, junit.xml or system.log came back"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="the .tflite model to benchmark")
    p.add_argument("--device", required=True, help="a DDP iPhone id, e.g. iphone16pro-18-3")
    p.add_argument("--accelerator", required=True, choices=["cpu", "gpu"], help="what this session runs on")
    p.add_argument("--test-zip", default=str(DEFAULT_ARCHIVE), help="the archive build_xctest.sh wrote (default: %(default)s)")
    p.add_argument("--session-dir", default=str(DEFAULT_SESSION_DIR), help="where sessions are laid out (default: %(default)s)")
    p.add_argument("--timeout", default="15m", help="the lab's --xctest-timeout, 1m to 1h (default: %(default)s)")
    p.add_argument("--xcode-version", default="xcode-26-2",
                   help="the lab's Xcode, an id from software-versions list (default: %(default)s)")
    p.add_argument("--project", default=os.environ.get("LITERT_GCP_PROJECT"),
                   help="Google Cloud project the session is billed to (default: LITERT_GCP_PROJECT)")
    p.add_argument("--dry-run", action="store_true", help="print the flags and the gcloud command; call nothing")
    p.add_argument("--collect", metavar="SESSION_ID",
                   help="submit nothing: wait for this session, submitted earlier with the same flags, and lay it out")
    p.add_argument("--force", action="store_true", help="with --collect, lay out over an existing session directory")
    p.add_argument("flags", nargs="*", help="more benchmark_model flags for the run, after --")
    args = p.parse_args()

    model = pathlib.Path(args.model)
    archive = pathlib.Path(args.test_zip)
    root = pathlib.Path(args.session_dir).expanduser()
    timeout_ok = re.fullmatch(r"([1-9]\d*)([mh])", args.timeout)
    minutes = int(timeout_ok.group(1)) * (60 if timeout_ok.group(2) == "h" else 1) if timeout_ok else 0
    flags = benchmark_args(model.name, args.accelerator, args.flags)
    for ok, message in (
        (model.is_file() and model.suffix == ".tflite", f"{model} is not a .tflite file"),
        ("," not in str(model) and "=" not in str(model), f"{model}: gcloud takes no ',' or '=' in a pushed path"),
        (archive.is_file(), f"no test archive at {archive}; run ./build_xctest.sh first"),
        (1 <= minutes <= 60, f"--timeout {args.timeout!r} is not between 1m and 1h"),
        (args.accelerator == "gpu" or not any(f.startswith("--use_gpu=") for f in args.flags),
         "--use_gpu belongs to --accelerator gpu, not to the extra flags"),
    ):
        if not ok:
            print(f"run_ddp_ios.py: {message}", file=sys.stderr)
            return EXIT_INPUT
    bundle_id, build_info = read_archive(archive)
    if not bundle_id or not build_info:
        print(f"run_ddp_ios.py: {archive} is not an archive from build_xctest.sh (no manifest with a "
              "TestHostBundleIdentifier, or no BUILD_INFO)", file=sys.stderr)
        return EXIT_INPUT
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=root):
            pass
    except OSError as e:
        print(f"run_ddp_ios.py: cannot write under --session-dir {root}: {e}", file=sys.stderr)
        return EXIT_INPUT

    run = uuid.uuid4().hex[:12]
    with tempfile.TemporaryDirectory(prefix="run_ddp_ios.") as tmp:
        args_path = pathlib.Path(tmp) / "benchmark_args.json"
        args_path.write_text(json.dumps(flags) + "\n")
        cmd = submit_args(args, bundle_id, args_path, run)
        print(f"# {args.accelerator} on {args.device}: {model.name}, app {bundle_id}, flags {json.dumps(flags)}")
        if args.collect:
            print(f"# collecting session {args.collect}, submitting nothing")
        else:
            print("# gcloud " + " ".join(cmd) + (f" --project={args.project}" if args.project else ""))
        if args.dry_run:
            return 0
        sid = args.collect
        try:
            device = catalog_device(args.device, args.project)
            if device is None or str(device.get("platform")).lower() != "ios":
                print(f"run_ddp_ios.py: {args.device} is "
                      f"{'not in the catalog' if device is None else device.get('platform')}, not an iPhone; "
                      "`gcloud beta device-run devices list` shows the ids", file=sys.stderr)
                return EXIT_INPUT
            if sid is None:
                submitted = gcloud_json(*cmd, project=args.project)
                sid = session_id(submitted) or find_session(run, args.project)
                if sid is None:
                    raise GcloudError(f"the submission printed no session id and no session carries the label "
                                      f"run={run}; `gcloud beta device-run sessions list` shows what was submitted")
                print(f"# session {sid}; `gcloud beta device-run sessions cancel {sid}` stops it")
            session_dir = root / sid
            if args.collect and session_dir.exists() and not args.force:
                print(f"run_ddp_ios.py: {session_dir} already exists; --force lays the session out again",
                      file=sys.stderr)
                return EXIT_INPUT
            session = wait_for_session(sid, args.project, max_secs=(minutes + 20) * 60)
        except GcloudError as e:
            print(f"run_ddp_ios.py: {e}", file=sys.stderr)
            return EXIT_GCLOUD
        except KeyboardInterrupt:
            what = f"session {sid}" if sid else f"the session labelled run={run}, if the submission went through"
            print(f"\nrun_ddp_ios.py: interrupted; {what} keeps running until it ends or "
                  f"`gcloud beta device-run sessions cancel <id>` stops it", file=sys.stderr)
            return EXIT_INTERRUPTED

        if args.collect:
            expected = {"accelerator": args.accelerator, "model": model.name}
            labels = session_labels(session) or {}
            mismatch = [f"{k} {labels.get(k)!r}" for k, v in expected.items() if labels.get(k) != v]
            if session_device(session) != args.device:
                mismatch.append(f"device {session_device(session)!r}")
            if mismatch:
                print(f"run_ddp_ios.py: session {sid} ran {', '.join(mismatch)}, not what the flags say; "
                      "pass the flags it was submitted with", file=sys.stderr)
                return EXIT_INPUT
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "ddp_session.json").write_text(json.dumps(session, indent=1) + "\n")
        report = session.get("sessionReport") or {}
        jobs = report.get("jobReports") or []
        if len(jobs) != 1:
            print(f"run_ddp_ios.py: session {sid} reports {len(jobs)} jobs for one device; see {session_dir}/ddp_session.json",
                  file=sys.stderr)
            return EXIT_GCLOUD
        job = jobs[0]
        result = (job.get("result") or {}).get("resultType", "UNKNOWN")
        job_name = job.get("displayName") or job.get("id") or "job"
        print(f"\njob {job_name}: {result}")
        downloaded = pathlib.Path(tmp) / "download"
        downloaded.mkdir()
        prefix = execution_prefix(job)
        try:
            if prefix is not None:
                gcloud("storage", "cp", "-r", f"{prefix}artifacts", f"{prefix}system.log", f"{prefix}junit.xml",
                       f"{downloaded}/")
        except GcloudError as e:
            print(f"run_ddp_ios.py: {e}", file=sys.stderr)  # the verdict below says what is missing
        except KeyboardInterrupt:
            print(f"\nrun_ddp_ios.py: interrupted while downloading; `--collect {sid} --force` lays it out later",
                  file=sys.stderr)
            return EXIT_INTERRUPTED
        job_dir = session_dir / f"{args.accelerator}-{args.device}"
        lay_out(downloaded, job_dir, session_dir)
        m = re.search(r"v(\d+(?:\.\d+)+)", build_info)
        (session_dir / "session.json").write_text(json.dumps({
            "runner": "ddp", "platform": "ios", "device_id": args.device,
            "device": device.get("displayName", args.device), "os": f"iOS {device.get('osVersion', '')}".strip(),
            "runtime_version": m.group(1) if m else "unknown",
            "binary": f"{build_info}, XCTest (Release) in benchmark/ios",
            "ddp_session": sid, "ddp_job": job_name,
            "created": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
            .isoformat().replace("+00:00", "Z"),
        }, indent=1) + "\n")
        print_result_lines(job_dir)
        print(f"session: {session_dir}")
        if result != "PASSED":
            print(f"run_ddp_ios.py: the job ended {result}; {job_tail(job_dir, session_dir)}", file=sys.stderr)
            return EXIT_JOB
        if not (job_dir / "results.pb").exists():
            print(f"run_ddp_ios.py: the job passed but no results.pb came back; {job_tail(job_dir, session_dir)}"
                  f" (a run past --timeout {args.timeout} leaves this too)", file=sys.stderr)
            return EXIT_RESULTS
        return 0


if __name__ == "__main__":
    sys.exit(main())
