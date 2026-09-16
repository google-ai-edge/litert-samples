#!/usr/bin/env python3
"""Turn `litert benchmark --ddp` session outputs into leaderboard rows.

Each job directory under a session directory
(~/.cache/litert-cli/ddp/<session>/<job>/) holds results.pb, runtime_info.pb
and logcat.txt. This script decodes results.pb with protoc against LiteRT's
benchmark_result.proto (fetched once into a cache), reads the delegate line
from logcat.txt, and writes one row per job to measurements.jsonl. A row with
the same row_id replaces the earlier one. A job without results is reported on
stderr and not written.

Usage:
  collect.py SESSION_DIR... --model litert-community/MobileNet-v2
             [--task image-classification] [--runtime-version 2.2.0]
             [--matrix matrix.yaml] [--data-dir ../leaderboard/data] [--date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import shutil
import subprocess
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_MATRIX = HERE / "matrix.yaml"
DEFAULT_DATA_DIR = HERE.parent / "leaderboard" / "data"
PROTO_URL = (
    "https://raw.githubusercontent.com/google-ai-edge/litert/main/"
    "tflite/tools/benchmark/proto/benchmark_result.proto"
)
PROTO_CACHE = pathlib.Path.home() / ".cache" / "litert-samples-benchmark"
PROTO_MESSAGE = "tflite.tools.benchmark.BenchmarkResult"

# logcat line: "MM-DD HH:MM:SS.mmm  PID  TID LEVEL TAG    : message"
LOGCAT_RE = re.compile(r"^\S+ \S+\s+(\d+)\s+(\d+)\s+([VDIWEF])\s+(\S+?)\s*:\s?(.*)$")
DELEGATE_RE = re.compile(
    r"Replacing (\d+) out of (\d+) node\(s\) with delegate \(([^)]+)\) node,"
    r" yielding (\d+) partitions"
)
LOADING_RE = re.compile(r"Loading model from: \S*/([^/\s]+)$")
TIMING_RE = re.compile(
    r"count=(\d+) first=(\d+) curr=\d+ min=(\d+) max=(\d+) avg=([\d.]+) std=([\d.]+)"
    r" p5=(\d+) median=(\d+) p95=(\d+)"
)
RESULT_LINE_RE = re.compile(r"\[benchmark_litert_model\.h:\d+\] (.+?):\s+([\d.]+)(?: (?:ms|MB/s|MB))?(?: \((\d+) runs\))?$")
MODEL_SIZE_RE = re.compile(r"The input model file size \(MB\): ([\d.]+)")


def load_matrix(path: pathlib.Path) -> dict:
    """Reads matrix.yaml; returns {} when the file is missing."""
    if not path.exists():
        return {}
    try:
        import yaml  # PyYAML
    except ImportError:
        sys.exit("collect.py: PyYAML is needed to read matrix.yaml (pip install pyyaml), or pass --task and --runtime-version")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def device_names(matrix: dict) -> dict[str, str]:
    names = {}
    for group in ("measured", "planned"):
        for d in (matrix.get("devices") or {}).get(group) or []:
            names[d["id"]] = d.get("name", d["id"])
    return names


def matrix_task(matrix: dict, repo: str, file: str) -> str | None:
    for m in matrix.get("models") or []:
        if m.get("repo") == repo and m.get("file") == file:
            return m.get("task")
    return None


def ensure_proto() -> pathlib.Path | None:
    """Fetches benchmark_result.proto into the cache once; None when offline."""
    PROTO_CACHE.mkdir(parents=True, exist_ok=True)
    proto = PROTO_CACHE / "benchmark_result.proto"
    if proto.exists():
        return proto
    try:
        with urllib.request.urlopen(PROTO_URL, timeout=30) as r:
            proto.write_bytes(r.read())
    except OSError as e:
        print(f"collect.py: could not fetch benchmark_result.proto: {e}", file=sys.stderr)
        return None
    return proto


def parse_text_proto(text: str) -> dict:
    """Parses protoc --decode output (nested `name { ... }` blocks, `name: value` leaves)."""
    root: dict = {}
    stack = [root]
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.endswith("{"):
            child: dict = {}
            stack[-1][line[:-1].strip()] = child
            stack.append(child)
        elif line == "}":
            stack.pop()
        else:
            key, _, value = line.partition(":")
            value = value.strip()
            if value.startswith('"'):
                stack[-1][key.strip()] = value.strip('"')
            else:
                try:
                    stack[-1][key.strip()] = int(value)
                except ValueError:
                    stack[-1][key.strip()] = float(value)
    return root


def decode_results_pb(pb: pathlib.Path) -> dict | None:
    """Decodes results.pb with protoc; None when protoc or the proto is unavailable."""
    if not pb.exists():
        return None
    if not shutil.which("protoc"):
        print("collect.py: protoc not found; reading logcat instead", file=sys.stderr)
        return None
    proto = ensure_proto()
    if proto is None:
        return None
    with open(pb, "rb") as f:
        run = subprocess.run(
            ["protoc", f"--decode={PROTO_MESSAGE}", proto.name],
            cwd=proto.parent, stdin=f, capture_output=True, text=True,
        )
    if run.returncode != 0:
        print(f"collect.py: protoc failed on {pb}: {run.stderr.strip()}", file=sys.stderr)
        return None
    return parse_text_proto(run.stdout)


def parse_logcat(path: pathlib.Path) -> dict:
    """Reads the benchmark process's lines: model file, delegate line, results block."""
    info: dict = {"delegate": None, "results": {}, "timing": None, "file": None, "model_size_mb": None}
    if not path.exists():
        return info
    pid = None
    for raw in path.read_text(errors="replace").splitlines():
        m = LOGCAT_RE.match(raw)
        if not m:
            continue
        line_pid, _, _, tag, msg = m.groups()
        if pid is None:
            if tag == "tflite" and msg == "STARTING!":
                pid = line_pid
            continue
        if line_pid != pid:
            continue
        if (d := DELEGATE_RE.search(msg)) and info["delegate"] is None:
            info["delegate"] = {
                "name": d.group(3), "nodes_delegated": int(d.group(1)),
                "nodes_total": int(d.group(2)), "partitions": int(d.group(4)),
            }
        elif (f := LOADING_RE.search(msg)):
            info["file"] = f.group(1)
        elif (s := MODEL_SIZE_RE.search(msg)):
            info["model_size_mb"] = float(s.group(1))
        elif (t := TIMING_RE.search(msg)):
            info["timing"] = t.groups()  # the last one is the measurement phase
        elif (r := RESULT_LINE_RE.search(msg)):
            label, value, runs = r.groups()
            info["results"][label] = (float(value), int(runs) if runs else None)
    return info


def row_from_pb(pb: dict) -> dict:
    lat = pb.get("latency_metrics", {})
    mem = pb.get("memory_metrics", {})
    misc = pb.get("misc_metrics", {})
    kb = lambda k: round(mem[k] / 1024, 2) if k in mem else None
    return {
        "latency_ms": {
            "median": lat.get("median_ms"), "avg": lat.get("avg_ms"), "p5": lat.get("p5_ms"),
            "p95": lat.get("p95_ms"), "min": lat.get("min_ms"), "max": lat.get("max_ms"),
            "stddev": lat.get("stddev_ms"), "init": lat.get("init_ms"),
            "first_inference": lat.get("first_inference_ms"), "warm_up_avg": lat.get("average_warm_up_ms"),
        },
        "memory_mb": {"init_footprint": kb("init_footprint_kb"), "overall_footprint": kb("overall_footprint_kb")},
        "runs": misc.get("num_runs"), "warmup_runs": misc.get("num_warmup_runs"),
        "throughput_mb_s": misc.get("model_throughput_in_mb_per_sec"),
        "model_size_mb": misc.get("model_size_mb"),
        "source": "results.pb",
    }


def row_from_logcat(info: dict) -> dict | None:
    """Builds the same fields from the BENCHMARK RESULTS block and the last timing line."""
    res = info["results"]
    if "Inference (avg)" not in res:
        return None
    us = lambda v: round(int(v) / 1000, 3)
    t = info["timing"]
    return {
        "latency_ms": {
            "median": us(t[7]) if t else None, "avg": res["Inference (avg)"][0],
            "p5": us(t[6]) if t else None, "p95": us(t[8]) if t else None,
            "min": res.get("Inference (min)", (None,))[0], "max": res.get("Inference (max)", (None,))[0],
            "stddev": res.get("Inference (std)", (None,))[0], "init": res.get("Model initialization", (None,))[0],
            "first_inference": res.get("Warmup (first)", (None,))[0], "warm_up_avg": res.get("Warmup (avg)", (None,))[0],
        },
        "memory_mb": {
            "init_footprint": res.get("Init footprint", (None,))[0],
            "overall_footprint": res.get("Overall footprint", (None,))[0],
        },
        "runs": res["Inference (avg)"][1], "warmup_runs": res.get("Warmup (avg)", (None, None))[1],
        "throughput_mb_s": res.get("Throughput", (None,))[0],
        "model_size_mb": info["model_size_mb"],
        "source": "logcat",
    }


def collect_job(job_dir: pathlib.Path, args, matrix: dict, names: dict[str, str]) -> dict | None:
    session = job_dir.parent.name
    job = job_dir.name
    if "-" not in job:
        print(f"collect.py: skipping {session}/{job}: job name is not <accelerator>-<device>", file=sys.stderr)
        return None
    accelerator, device_id = job.split("-", 1)
    log = parse_logcat(job_dir / "logcat.txt")
    pb = decode_results_pb(job_dir / "results.pb")
    metrics = row_from_pb(pb) if pb else row_from_logcat(log)
    if metrics is None or metrics["latency_ms"]["avg"] is None:
        print(f"FAILED {session}/{job}: no results.pb and no BENCHMARK RESULTS block in logcat.txt", file=sys.stderr)
        return None
    file = log["file"] or args.file
    if not file:
        print(f"FAILED {session}/{job}: model file name not in logcat.txt; pass --file", file=sys.stderr)
        return None
    if metrics["model_size_mb"] is None:
        metrics["model_size_mb"] = log["model_size_mb"]
    task = args.task or matrix_task(matrix, args.model, file)
    if not task:
        sys.exit(f"collect.py: task unknown for {args.model}:{file}; add it to matrix.yaml or pass --task")
    version = args.runtime_version or (matrix.get("runtime") or {}).get("version")
    if not version:
        sys.exit("collect.py: runtime version unknown; pass --runtime-version or set runtime.version in matrix.yaml")
    if args.date:
        date = args.date
    else:
        src = job_dir / "results.pb" if (job_dir / "results.pb").exists() else job_dir / "logcat.txt"
        date = dt.datetime.fromtimestamp(src.stat().st_mtime, dt.timezone.utc).date().isoformat()
    delegate = log["delegate"] or {}
    if device_id not in names:
        print(f"collect.py: device {device_id} is not in matrix.yaml; the board will show its id", file=sys.stderr)
    return {
        "row_id": f"{args.model}:{file}@{version}/{device_id}/{accelerator}",
        "model": args.model, "file": file, "task": task,
        "model_size_mb": metrics.pop("model_size_mb"),
        "device_id": device_id, "device": names.get(device_id, device_id),
        "platform": "android", "accelerator": accelerator,
        "delegate": delegate.get("name"),
        "nodes_delegated": delegate.get("nodes_delegated"), "nodes_total": delegate.get("nodes_total"),
        "partitions": delegate.get("partitions"),
        **metrics,
        "runtime": "litert", "runtime_version": version,
        "ddp_session": session, "ddp_job": job, "status": "measured", "date": date,
    }


def write_rows(data_dir: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "measurements.jsonl"
    existing: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                existing[r["row_id"]] = r
    for r in rows:
        existing[r["row_id"]] = r
    with open(path, "w") as f:
        for r in existing.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session_dirs", nargs="+", type=pathlib.Path, help="~/.cache/litert-cli/ddp/<session> (or one job directory)")
    p.add_argument("--model", required=True, help="Hugging Face repo the model file came from, e.g. litert-community/MobileNet-v2")
    p.add_argument("--file", help="model file name; read from logcat.txt when omitted")
    p.add_argument("--task", help="pipeline tag; read from matrix.yaml when omitted")
    p.add_argument("--runtime-version", help="benchmark_model release the CLI pinned; matrix.yaml runtime.version when omitted")
    p.add_argument("--matrix", type=pathlib.Path, default=DEFAULT_MATRIX)
    p.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--date", help="YYYY-MM-DD; default = the day the outputs were pulled (file mtime, UTC)")
    args = p.parse_args()

    matrix = load_matrix(args.matrix)
    names = device_names(matrix)
    rows, failed = [], 0
    for sd in args.session_dirs:
        sd = sd.expanduser()
        if not sd.is_dir():
            sys.exit(f"collect.py: not a directory: {sd}")
        jobs = [sd] if (sd / "logcat.txt").exists() or (sd / "results.pb").exists() else sorted(d for d in sd.iterdir() if d.is_dir())
        for job_dir in jobs:
            row = collect_job(job_dir, args, matrix, names)
            if row is None:
                failed += 1
                continue
            rows.append(row)
            lat = row["latency_ms"]
            print(f"{row['ddp_session']}/{row['ddp_job']}: {row['file']} {row['device']} {row['accelerator']}"
                  f" median {lat['median']} ms, p95 {lat['p95']} ms, {row['runs']} runs,"
                  f" {row['nodes_delegated']}/{row['nodes_total']} nodes ({row['delegate']}), {row['source']}")
    if rows:
        path = write_rows(args.data_dir, rows)
        print(f"{len(rows)} row(s) written to {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
