#!/usr/bin/env python3
"""Turn benchmark job outputs into leaderboard rows.

A session directory holds one job directory per accelerator and device,
named <accelerator>-<device id>:

  ~/.cache/litert-cli/ddp/<session>/<job>/            from `litert benchmark --ddp`
      results.pb, runtime_info.pb, logcat.txt
  ~/.cache/litert-samples-benchmark/local/<session>/<job>/   from run_local.py
      results.pb, runtime_info.pb, stdout.txt, and session.json beside the jobs

This script decodes results.pb with protoc against LiteRT's benchmark_result.proto
(fetched once into a cache) and runtime_info.pb against model_runtime_info.proto,
which names the delegate and counts the nodes it replaced in the model's primary
subgraph; a job without runtime_info.pb takes those from the "Replacing N out of M"
line of its log when there is one. One row per job goes to measurements.jsonl. A
row with the same row_id replaces the earlier one. A job without results is
reported on stderr and not written.

The platform, device name and OS come from the flags when given, else from
session.json when the session has one (run_local.py and run_ios.sh write it),
else from matrix.yaml for Android devices. A session with neither session.json
nor flags is taken as an Android DDP session only when its jobs have logcat.txt.

Usage:
  collect.py SESSION_DIR... --model litert-community/MobileNet-v2
             [--task image-classification] [--runtime-version 2.2.0]
             [--platform macos --device "Mac Studio (M4 Max)" --os "macOS 27.0"]
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
RUNTIME_INFO_PROTOS = (  # model_runtime_info.proto and the file it imports, kept at their repo paths
    "tflite/profiling/proto/model_runtime_info.proto",
    "tflite/profiling/proto/profiling_info.proto",
)
RUNTIME_INFO_MESSAGE = "tflite.profiling.ModelRuntimeDetails"
LITERT_RAW = "https://raw.githubusercontent.com/google-ai-edge/litert/main/"

# logcat line: "MM-DD HH:MM:SS.mmm  PID  TID LEVEL TAG    : message"
LOGCAT_RE = re.compile(r"^\S+ \S+\s+(\d+)\s+(\d+)\s+([VDIWEF])\s+(\S+?)\s*:\s?(.*)$")
DELEGATE_RE = re.compile(
    r"Replacing (\d+) out of (\d+) node\(s\) with delegate \(([^)]+)\) node,"
    r" yielding (\d+) partitions"
)
GPU_API_RE = re.compile(r"Initializing (\w+)-based API from graph")
XNNPACK_RE = re.compile(r"Created TensorFlow Lite XNNPACK delegate for CPU")
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


def matrix_devices(matrix: dict, platform: str) -> dict[str, dict]:
    """{device id: entry} for one platform, measured and planned."""
    devices = {}
    groups = ((matrix.get("platforms") or {}).get(platform) or {}).get("devices") or {}
    for group in ("measured", "planned"):
        for d in groups.get(group) or []:
            devices[d["id"]] = d
    return devices


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
        print("collect.py: protoc not found; reading the job log instead", file=sys.stderr)
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


def ensure_runtime_info_protos() -> pathlib.Path | None:
    """Fetches model_runtime_info.proto and its import into the cache once; None when offline."""
    for rel in RUNTIME_INFO_PROTOS:
        dest = PROTO_CACHE / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(LITERT_RAW + rel, timeout=30) as r:
                dest.write_bytes(r.read())
        except OSError as e:
            print(f"collect.py: could not fetch {rel}: {e}", file=sys.stderr)
            return None
    return PROTO_CACHE


def primary_subgraph_delegation(text: str, accelerator: str) -> dict | None:
    """Delegate name and node counts of subgraph 0 from decoded ModelRuntimeDetails text.

    A delegate node in the subgraph's node list lists the tflite nodes it replaced
    (delegate_node_details.tflite_node_ids_replaced). The accelerator's delegate is
    XNNPACK for cpu and any other delegate for gpu; nodes_total counts the tflite
    nodes and nodes_delegated the ones that delegate replaced. partitions is the
    count the runtime logs as "yielding N partitions": the runs of that delegate's
    nodes and of everything else along the execution plan.
    """
    for sg in re.split(r"\nsubgraphs \{", "\n" + text)[1:]:
        if "subgraph_type: TFLITE_SUBGRAPH" not in sg:
            continue
        sid = re.search(r"subgraph_id: (\d+)", sg)
        if sid and sid.group(1) != "0":
            continue
        nodes = re.findall(r"\n  nodes \{(.*?)\n  \}", sg, re.S)
        delegate_nodes = 0
        mine: dict[int, tuple[str, int]] = {}  # node id -> (delegate name, nodes replaced)
        for node in nodes:
            m = re.search(r'delegate_name: "([^"]*)"', node)
            if not m:
                continue
            delegate_nodes += 1
            if ("xnnpack" in m.group(1).lower()) == (accelerator == "cpu"):
                node_id = re.search(r"^\s*id: (\d+)", node, re.M)
                mine[int(node_id.group(1)) if node_id else -len(mine) - 1] = (
                    m.group(1), len(re.findall(r"tflite_node_ids_replaced: \d+", node)))
        plan = [int(i) for i in re.findall(r"execution_plan: (\d+)", sg)]
        runs = sum(1 for i, n in enumerate(plan) if i == 0 or (n in mine) != (plan[i - 1] in mine))
        return {
            "name": next(iter(mine.values()))[0] if mine else None,
            "nodes_delegated": sum(c for _, c in mine.values()),
            "nodes_total": len(nodes) - delegate_nodes,
            "partitions": runs if plan else len(mine),
        }
    return None


def decode_runtime_info(pb: pathlib.Path, accelerator: str) -> dict | None:
    """Decodes runtime_info.pb with protoc; None without the file, protoc or the protos."""
    if not pb.exists() or not shutil.which("protoc"):
        return None
    root = ensure_runtime_info_protos()
    if root is None:
        return None
    with open(pb, "rb") as f:
        run = subprocess.run(
            ["protoc", f"--proto_path={root}", f"--decode={RUNTIME_INFO_MESSAGE}", RUNTIME_INFO_PROTOS[0]],
            cwd=root, stdin=f, capture_output=True, text=True,
        )
    if run.returncode != 0:
        print(f"collect.py: protoc failed on {pb}: {run.stderr.strip()}", file=sys.stderr)
        return None
    return primary_subgraph_delegation(run.stdout, accelerator)


def log_messages(path: pathlib.Path):
    """Yields the benchmark process's log messages.

    logcat.txt keeps every process on the device: only the lines of the pid that
    logged `tflite: STARTING!` are the benchmark's. stdout.txt is the binary's
    own output, one message per line.
    """
    lines = path.read_text(errors="replace").splitlines()
    if not any(LOGCAT_RE.match(l) for l in lines[:50]):
        yield from lines
        return
    pid = None
    for raw in lines:
        m = LOGCAT_RE.match(raw)
        if not m:
            continue
        line_pid, _, _, tag, msg = m.groups()
        if pid is None:
            if tag == "tflite" and msg == "STARTING!":
                pid = line_pid
            continue
        if line_pid == pid:
            yield msg


def parse_log(path: pathlib.Path, accelerator: str) -> dict:
    """Reads model file, delegate, results block and timing line from the job log."""
    info: dict = {"delegate": None, "results": {}, "timing": None, "file": None, "model_size_mb": None}
    if not path.exists():
        return info
    gpu_api = xnnpack = None
    for msg in log_messages(path):
        if (d := DELEGATE_RE.search(msg)) and info["delegate"] is None:
            info["delegate"] = {
                "name": d.group(3), "nodes_delegated": int(d.group(1)),
                "nodes_total": int(d.group(2)), "partitions": int(d.group(4)),
            }
        elif (g := GPU_API_RE.search(msg)):
            gpu_api = g.group(1)
        elif XNNPACK_RE.search(msg):
            xnnpack = "XNNPACK"
        elif (f := LOADING_RE.search(msg)):
            info["file"] = f.group(1)
        elif (s := MODEL_SIZE_RE.search(msg)):
            info["model_size_mb"] = float(s.group(1))
        elif (t := TIMING_RE.search(msg)):
            info["timing"] = t.groups()  # the last one is the measurement phase
        elif (r := RESULT_LINE_RE.search(msg)):
            label, value, runs = r.groups()
            info["results"][label] = (float(value), int(runs) if runs else None)
    if info["delegate"] is None:
        # No "Replacing N out of M" line (the macOS binary's stdout has none): keep
        # the delegate the log names for this accelerator, and no node count.
        name = gpu_api if accelerator == "gpu" else xnnpack
        if name:
            info["delegate"] = {"name": name, "nodes_delegated": None, "nodes_total": None, "partitions": None}
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


def row_from_log(info: dict) -> dict | None:
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
        "source": "log",
    }


def session_meta(session_dir: pathlib.Path, jobs: list[pathlib.Path], args, matrix: dict) -> dict:
    """Platform, runner, device and binary for one session: flags > session.json > matrix > DDP layout."""
    meta: dict = {}
    sj = session_dir / "session.json"
    if sj.exists():
        meta = json.loads(sj.read_text())
    ddp_layout = all((j / "logcat.txt").exists() for j in jobs)
    platform = args.platform or meta.get("platform")
    if not platform:
        if not ddp_layout:
            sys.exit(f"collect.py: {session_dir} has no session.json and no logcat.txt; pass --platform, --device and --os")
        platform = "android"
    known = list(matrix.get("platforms") or {})
    if known and platform not in known:
        sys.exit(f"collect.py: platform {platform!r} is not in matrix.yaml (known: {', '.join(known)})")
    runner = args.runner or meta.get("runner") or ("ddp" if ddp_layout else None)
    if not runner:
        sys.exit(f"collect.py: runner unknown for {session_dir}; pass --runner ddp|local|app")
    version = args.runtime_version or meta.get("runtime_version") or (matrix.get("runtime") or {}).get("version")
    pconf = (matrix.get("platforms") or {}).get(platform) or {}
    binary = args.binary or meta.get("binary")
    if not binary and platform == "android" and runner == "ddp" and version and pconf.get("binary"):
        binary = f"{version}/{pconf['binary']}"
    return {
        "platform": platform, "runner": runner,
        "device_id": args.device_id or meta.get("device_id"),
        "device": args.device or meta.get("device"),
        "os": args.os or meta.get("os"),
        "runtime_version": version,
        "binary": binary,
        "devices": matrix_devices(matrix, platform),
    }


def collect_job(job_dir: pathlib.Path, args, matrix: dict, meta: dict) -> dict | None:
    session = job_dir.parent.name
    job = job_dir.name
    accelerator, device_id = job.split("-", 1)
    device_id = meta["device_id"] or device_id
    log_path = job_dir / ("logcat.txt" if (job_dir / "logcat.txt").exists() else "stdout.txt")
    log = parse_log(log_path, accelerator)
    pb = decode_results_pb(job_dir / "results.pb")
    metrics = row_from_pb(pb) if pb else row_from_log(log)
    if metrics is None or metrics["latency_ms"]["avg"] is None:
        print(f"FAILED {session}/{job}: no results.pb and no BENCHMARK RESULTS block in {log_path.name}", file=sys.stderr)
        return None
    file = log["file"] or args.file
    if not file:
        print(f"FAILED {session}/{job}: model file name not in {log_path.name}; pass --file", file=sys.stderr)
        return None
    if args.file and log["file"] and args.file != log["file"]:
        print(f"FAILED {session}/{job}: {log_path.name} names {log['file']}, --file says {args.file}", file=sys.stderr)
        return None
    if metrics["model_size_mb"] is None:
        metrics["model_size_mb"] = log["model_size_mb"]
    task = args.task or matrix_task(matrix, args.model, file)
    if not task:
        print(f"FAILED {session}/{job}: task unknown for {args.model}:{file}; add it to matrix.yaml or pass --task", file=sys.stderr)
        return None
    version = meta["runtime_version"]
    if not version:
        print(f"FAILED {session}/{job}: runtime version unknown; pass --runtime-version or set runtime.version in matrix.yaml", file=sys.stderr)
        return None
    if args.date:
        date = args.date
    else:
        src = job_dir / "results.pb" if (job_dir / "results.pb").exists() else log_path
        date = dt.datetime.fromtimestamp(src.stat().st_mtime, dt.timezone.utc).date().isoformat()
    delegate = decode_runtime_info(job_dir / "runtime_info.pb", accelerator) or log["delegate"] or {}
    if delegate.get("name") is None and log["delegate"]:
        delegate["name"] = log["delegate"]["name"]
    entry = meta["devices"].get(device_id, {})
    if not meta["device"] and device_id not in meta["devices"]:
        print(f"collect.py: device {device_id} is not in matrix.yaml; the board will show its id", file=sys.stderr)
    platform = meta["platform"]
    return {
        "row_id": f"{args.model}:{file}@{version}/{platform}/{device_id}/{accelerator}",
        "model": args.model, "file": file, "task": task,
        "model_size_mb": metrics.pop("model_size_mb"),
        "platform": platform, "device_id": device_id,
        "device": meta["device"] or entry.get("name", device_id),
        "os": meta["os"] or entry.get("os"),
        "accelerator": accelerator,
        "delegate": delegate.get("name"),
        "nodes_delegated": delegate.get("nodes_delegated"), "nodes_total": delegate.get("nodes_total"),
        "partitions": delegate.get("partitions"),
        **metrics,
        "runtime": "litert", "runtime_version": version, "binary": meta["binary"],
        "runner": meta["runner"], "session": session, "job": job,
        "status": "measured", "date": date,
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
    p.add_argument("session_dirs", nargs="+", type=pathlib.Path, help="a session directory (or one job directory)")
    p.add_argument("--model", required=True, help="Hugging Face repo the model file came from, e.g. litert-community/MobileNet-v2")
    p.add_argument("--file", help="model file name; read from the job log when omitted")
    p.add_argument("--task", help="pipeline tag; read from matrix.yaml when omitted")
    p.add_argument("--runtime-version", help="benchmark_model release; session.json, then matrix.yaml runtime.version when omitted")
    p.add_argument("--platform", help="a platform id from matrix.yaml; session.json when omitted, else android for a DDP session")
    p.add_argument("--runner", choices=["ddp", "local", "app"], help="what produced the session; session.json when omitted, else ddp for a DDP session")
    p.add_argument("--device", help="device name shown on the board; session.json, then matrix.yaml when omitted")
    p.add_argument("--device-id", help="device id; the job directory name when omitted")
    p.add_argument("--os", help="OS version shown with the device; session.json, then matrix.yaml when omitted")
    p.add_argument("--binary", help="the benchmark_model build the rows ran; session.json when omitted, else the matrix's Android binary for DDP sessions")
    p.add_argument("--matrix", type=pathlib.Path, default=DEFAULT_MATRIX)
    p.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--date", help="YYYY-MM-DD; default = the day the outputs were written (file mtime, UTC)")
    args = p.parse_args()

    matrix = load_matrix(args.matrix)
    rows, failed = [], 0
    for sd in args.session_dirs:
        sd = sd.expanduser()
        if not sd.is_dir():
            sys.exit(f"collect.py: not a directory: {sd}")
        is_job = (sd / "results.pb").exists() or (sd / "logcat.txt").exists() or (sd / "stdout.txt").exists()
        jobs = [sd] if is_job else sorted(d for d in sd.iterdir() if d.is_dir())
        for d in [j for j in jobs if "-" not in j.name]:
            print(f"collect.py: skipping {d}: not a <accelerator>-<device> job directory", file=sys.stderr)
        jobs = [j for j in jobs if "-" in j.name]
        if not jobs:
            print(f"collect.py: no job directories in {sd}", file=sys.stderr)
            failed += 1
            continue
        meta = session_meta(sd.parent if is_job else sd, jobs, args, matrix)
        for job_dir in jobs:
            row = collect_job(job_dir, args, matrix, meta)
            if row is None:
                failed += 1
                continue
            rows.append(row)
            lat = row["latency_ms"]
            nodes = f"{row['nodes_delegated']}/{row['nodes_total']} nodes" if row["nodes_delegated"] is not None else "nodes n/a"
            print(f"{row['session']}/{row['job']}: {row['file']} {row['platform']} {row['device']} {row['accelerator']}"
                  f" median {lat['median']} ms, p95 {lat['p95']} ms, {row['runs']} runs,"
                  f" {nodes} ({row['delegate']}), {row['source']}")
    if rows:
        path = write_rows(args.data_dir, rows)
        print(f"{len(rows)} row(s) written to {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
