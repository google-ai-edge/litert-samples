#!/usr/bin/env python3
"""Run the matrix on Developer Device Platform (DDP) devices and collect the rows.

For every model in matrix.yaml and every accelerator it lists, this script runs
one `litert benchmark --ddp` session on all measured Android devices, then
collect.py on the session's outputs, and finally build_board.py. Sessions are billed to the
Google Cloud project. `--dry-run` prints every command and runs nothing.

Usage:
  run_matrix.py --dry-run [--only litert-community/MobileNet-v2]
  run_matrix.py --gcp-project PROJECT_ID [--only REPO] [--accelerator cpu|gpu]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shlex
import subprocess
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_MATRIX = HERE / "matrix.yaml"
DEFAULT_DATA_DIR = HERE.parent / "leaderboard" / "data"
DEFAULT_MODELS_DIR = pathlib.Path.home() / ".cache" / "litert-samples-benchmark" / "models"
SAVED_RE = re.compile(r"Output files saved to: (\S+)")
BINARY_RE = re.compile(r"Binary: gs://litert/binaries/([^/\s]+)/android_arm64/benchmark_model")


def load_matrix(path: pathlib.Path) -> dict:
    try:
        import yaml  # PyYAML
    except ImportError:
        sys.exit("run_matrix.py: PyYAML is needed to read matrix.yaml (pip install pyyaml)")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def model_url(repo: str, file: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{file}"


def download(url: str, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url)
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def show(cmd: list[str]) -> str:
    """Prints a command with the interpreter as python3 and paths relative to the cwd."""
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


def run_streaming(cmd: list[str]) -> str:
    """Runs a command, echoing its output, and returns the whole output."""
    out: list[str] = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
        for line in p.stdout:
            sys.stdout.write(line)
            out.append(line)
    if p.returncode != 0:
        print(f"run_matrix.py: command exited {p.returncode}: {shlex.join(cmd)}", file=sys.stderr)
    return "".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=pathlib.Path, default=DEFAULT_MATRIX)
    p.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--models-dir", type=pathlib.Path, default=DEFAULT_MODELS_DIR, help="where model files are downloaded")
    p.add_argument("--gcp-project", default=os.environ.get("LITERT_GCP_PROJECT"), help="project the sessions are billed to (or LITERT_GCP_PROJECT)")
    p.add_argument("--litert", default="litert", help="path to the litert CLI")
    p.add_argument("--only", action="append", default=[], metavar="REPO", help="run only this repo (repeatable)")
    p.add_argument("--accelerator", action="append", default=[], choices=["cpu", "gpu"], help="run only this accelerator (repeatable)")
    p.add_argument("--dry-run", action="store_true", help="print the commands and run nothing")
    args = p.parse_args()

    matrix = load_matrix(args.matrix)
    android = (matrix.get("platforms") or {}).get("android") or {}
    devices = [d["id"] for d in (android.get("devices") or {}).get("measured") or []]
    if not devices:
        sys.exit("run_matrix.py: no measured devices under platforms.android in matrix.yaml")
    if not args.dry_run and not args.gcp_project:
        sys.exit("run_matrix.py: --gcp-project (or LITERT_GCP_PROJECT) is required to submit sessions")
    project = args.gcp_project or "PROJECT_ID"
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
        sys.exit("run_matrix.py: nothing to run (check --only / --accelerator against matrix.yaml)")

    if args.dry_run:
        print(f"# dry run: {len(plan)} session(s) x {len(devices)} device job(s) would be billed to project {project}; nothing runs")
        if not args.gcp_project:
            print("# PROJECT_ID is a placeholder: pass --gcp-project or set LITERT_GCP_PROJECT to submit")
    extra = []
    if args.matrix != DEFAULT_MATRIX:
        extra += ["--matrix", str(args.matrix)]
    if args.data_dir != DEFAULT_DATA_DIR:
        extra += ["--data-dir", str(args.data_dir)]
    failures = 0
    announced: set[pathlib.Path] = set()
    for m, accel in plan:
        local = args.models_dir / m["repo"] / m["file"]
        url = model_url(m["repo"], m["file"])
        if not local.exists() and local not in announced:
            print(f"# download {url} -> {local}")
            announced.add(local)
            if not args.dry_run:
                download(url, local)
        bench = [args.litert, "benchmark", str(local), "--ddp", f"--{accel}", "--devices", ",".join(devices), "--gcp-project", project]
        collect = [python, str(HERE / "collect.py"), "<session dir from the CLI output>", "--model", m["repo"], "--file", m["file"],
                   "--runtime-version", "<pin from the CLI output>", *extra]
        print(show(bench))
        print(show(collect))
        if args.dry_run:
            continue
        out = run_streaming(bench)
        saved = SAVED_RE.findall(out)
        if not saved:
            print(f"run_matrix.py: no output directory reported for {m['repo']} {accel}; skipping collect", file=sys.stderr)
            failures += 1
            continue
        session_dir = str(pathlib.Path(saved[0]).expanduser().parent)
        pin = BINARY_RE.search(out)
        version = pin.group(1) if pin else (matrix.get("runtime") or {}).get("version")
        collect[2] = session_dir
        collect[collect.index("<pin from the CLI output>")] = str(version)
        print(show(collect))
        if subprocess.run(collect).returncode != 0:
            failures += 1

    build = [python, str(HERE / "build_board.py"), *(["--data-dir", str(args.data_dir)] if args.data_dir != DEFAULT_DATA_DIR else [])]
    print(show(build))
    if not args.dry_run and subprocess.run(build).returncode != 0:
        failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
