#!/usr/bin/env python3
"""Run the matrix on this Mac with LiteRT's macOS benchmark_model and collect the rows.

For every model in matrix.yaml and every accelerator it lists, this script runs
the public macOS arm64 benchmark_model of the release matrix.yaml names (fetched
once into a cache) with the flags a `litert benchmark --ddp` job passes: the
model, `--use_gpu=true` for GPU, and the two result-file flags. Each model gets one
session: its runs write results.pb, runtime_info.pb and stdout.txt under
~/.cache/litert-samples-benchmark/local/<session>/<accelerator>-<device id>/,
and session.json beside them names this machine (system_profiler), its OS
(sw_vers) and the binary. Then collect.py turns each session into rows and
build_board.py rebuilds the board. `--dry-run` prints every command and runs
nothing.

Usage:
  run_local.py --dry-run [--only litert-community/MobileNet-v2] [--accelerator cpu|gpu]
  run_local.py [--only REPO] [--accelerator cpu|gpu]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform as platform_mod
import re
import shlex
import subprocess
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_MATRIX = HERE / "matrix.yaml"
DEFAULT_DATA_DIR = HERE.parent / "leaderboard" / "data"
CACHE = pathlib.Path.home() / ".cache" / "litert-samples-benchmark"
DEFAULT_MODELS_DIR = CACHE / "models"
DEFAULT_BINARIES_DIR = CACHE / "binaries"
DEFAULT_SESSIONS_DIR = CACHE / "local"
BINARIES_URL = "https://storage.googleapis.com/litert/binaries"


def load_matrix(path: pathlib.Path) -> dict:
    try:
        import yaml  # PyYAML
    except ImportError:
        sys.exit("run_local.py: PyYAML is needed to read matrix.yaml (pip install pyyaml)")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def this_platform() -> str:
    if sys.platform == "darwin" and platform_mod.machine().lower() in ("arm64", "aarch64"):
        return "macos"
    sys.exit(f"run_local.py: this machine is {sys.platform}/{platform_mod.machine()}; only macOS on Apple silicon has a local binary here")


def this_device() -> dict:
    """Name, id and OS of this Mac from system_profiler and sw_vers."""
    hw = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True).stdout
    fields = {}
    for line in hw.splitlines():
        key, _, value = line.strip().partition(":")
        if value:
            fields[key.strip()] = value.strip()
    model = fields.get("Model Name", "Mac")
    chip = fields.get("Chip", "").removeprefix("Apple ").strip()
    name = f"{model} ({chip})" if chip else model
    device_id = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    os_version = subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True).stdout.strip()
    return {
        "device": name, "device_id": device_id, "os": f"macOS {os_version}",
        "host": {k: fields[k] for k in ("Model Name", "Model Identifier", "Chip", "Total Number of Cores", "Memory") if k in fields},
    }


def download(url: str, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url)
    if "huggingface.co" in url and os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)


def show(cmd: list[str]) -> str:
    """Prints a command with the interpreter as python3 and driver paths relative to the cwd."""
    cwd = pathlib.Path.cwd()
    parts = []
    for c in cmd:
        if c == sys.executable:
            parts.append("python3")
        elif c.startswith(str(HERE) + os.sep):
            parts.append(os.path.relpath(c, cwd))
        else:
            parts.append(c)
    return shlex.join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=pathlib.Path, default=DEFAULT_MATRIX)
    p.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--models-dir", type=pathlib.Path, default=DEFAULT_MODELS_DIR, help="where model files are downloaded")
    p.add_argument("--binaries-dir", type=pathlib.Path, default=DEFAULT_BINARIES_DIR, help="where benchmark_model is cached")
    p.add_argument("--sessions-dir", type=pathlib.Path, default=DEFAULT_SESSIONS_DIR, help="where the session outputs go")
    p.add_argument("--only", action="append", default=[], metavar="REPO", help="run only this repo (repeatable)")
    p.add_argument("--accelerator", action="append", default=[], choices=["cpu", "gpu"], help="run only this accelerator (repeatable)")
    p.add_argument("--dry-run", action="store_true", help="print the commands and run nothing")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    matrix = load_matrix(args.matrix)
    plat = this_platform()
    pconf = (matrix.get("platforms") or {}).get(plat) or {}
    version = (matrix.get("runtime") or {}).get("version")
    if not version or not pconf.get("binary"):
        sys.exit(f"run_local.py: matrix.yaml needs runtime.version and platforms.{plat}.binary")
    binary = f"{version}/{pconf['binary']}"
    binary_url = f"{BINARIES_URL}/{binary}"
    binary_path = args.binaries_dir / binary
    device = this_device()
    python = sys.executable

    plan = []
    for m in matrix.get("models") or []:
        if args.only and m["repo"] not in args.only:
            continue
        for accel in m.get("accelerators") or ["cpu"]:
            if args.accelerator and accel not in args.accelerator:
                continue
            plan.append((m, accel))
    if not plan:
        sys.exit("run_local.py: nothing to run (check --only / --accelerator against matrix.yaml)")

    print(f"# {'dry run: ' if args.dry_run else ''}{len(plan)} run(s) on {device['device']} ({device['device_id']}), {device['os']}"
          f"{'; nothing runs' if args.dry_run else ''}")
    if not binary_path.exists():
        print(f"# download {binary_url} -> {binary_path}")
        if not args.dry_run:
            download(binary_url, binary_path)
            binary_path.chmod(0o755)
    extra = []
    if args.matrix != DEFAULT_MATRIX:
        extra += ["--matrix", str(args.matrix)]
    if args.data_dir != DEFAULT_DATA_DIR:
        extra += ["--data-dir", str(args.data_dir)]

    failures = 0
    announced: set[pathlib.Path] = set()
    sessions: dict[tuple, pathlib.Path] = {}  # one session per model file
    for m, accel in plan:
        key = (m["repo"], m["file"])
        if key not in sessions:
            now = dt.datetime.now(dt.timezone.utc)
            session_dir = args.sessions_dir / f"local-{now:%Y%m%d-%H%M%S}"
            n = 1
            while session_dir.exists() or session_dir in sessions.values():
                n += 1
                session_dir = args.sessions_dir / f"local-{now:%Y%m%d-%H%M%S}-{n}"
            sessions[key] = session_dir
            print(f"# {m['repo']} {m['file']}: session {session_dir.name}")
        session_dir = sessions[key]
        local = args.models_dir / m["repo"] / m["file"]
        url = f"https://huggingface.co/{m['repo']}/resolve/main/{m['file']}"
        if not local.exists() and local not in announced:
            print(f"# download {url} -> {local}")
            announced.add(local)
            if not args.dry_run:
                download(url, local)
        job_dir = session_dir / f"{accel}-{device['device_id']}"
        bench = [str(binary_path), f"--graph={local}"]
        if accel == "gpu":
            bench.append("--use_gpu=true")
        bench += [f"--result_file_path={job_dir / 'results.pb'}",
                  f"--model_runtime_info_output_file={job_dir / 'runtime_info.pb'}"]
        print(show(bench))
        if args.dry_run:
            continue
        job_dir.mkdir(parents=True, exist_ok=True)
        with open(job_dir / "stdout.txt", "w") as log, subprocess.Popen(bench, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            for line in proc.stdout:
                log.write(line)
                if "BENCHMARK RESULTS" in line or "benchmark_litert_model.h" in line:
                    sys.stdout.write(line)
        if proc.returncode != 0:
            print(f"run_local.py: {accel} run exited {proc.returncode}; see {job_dir / 'stdout.txt'}", file=sys.stderr)
            failures += 1
        if not (session_dir / "session.json").exists():
            (session_dir / "session.json").write_text(json.dumps({
                "runner": "local", "platform": plat, **device,
                "runtime_version": version, "binary": binary,
                "binary_md5": hashlib.md5(binary_path.read_bytes()).hexdigest(),
                "created": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            }, indent=1) + "\n")

    for (repo, file), session_dir in sessions.items():
        collect = [python, str(HERE / "collect.py"), str(session_dir), "--model", repo, "--file", file, *extra]
        print(show(collect))
        if not args.dry_run and subprocess.run(collect).returncode != 0:
            failures += 1
    build = [python, str(HERE / "build_board.py"), *(["--data-dir", str(args.data_dir)] if args.data_dir != DEFAULT_DATA_DIR else [])]
    print(show(build))
    if not args.dry_run and subprocess.run(build).returncode != 0:
        failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
