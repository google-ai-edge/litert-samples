#!/usr/bin/env python3
"""Turn a LiteRT-LM benchmark session into leaderboard rows.

A session is what `litert benchmark <bundle>.litertlm --ddp` pulls back: one job
directory per accelerator and device, named <accelerator>-<device id>, the same
layout collect.py reads for benchmark_model sessions:

  ~/.cache/litert-cli/ddp/<session>/<accelerator>-<device id>/
      metrics.pb       litert.lm.proto.LitertLmMetricsList, one entry per --num-iterations
      logcat.txt       the run's logcat (names the GPU API)
      provenance.txt   device, build, the binary's arguments, sha256 of the pushed files

This script decodes metrics.pb with protoc against LiteRT-LM's
`litert_lm_metrics.proto` (fetched once into the cache at the ref matrix.yaml
names) and writes one row per job to measurements-lm.jsonl: prefill and decode
tokens/s and time to first token as the median over the iterations after the
warm-up ones (--warmup-iterations, matrix.yaml `runtime_lm.warmup_iterations`),
init from the run's one engine creation, every iteration kept in the row. The
model file, `--max_num_tokens` and the shared libraries pushed beside the binary
(name and sha256) come from provenance.txt, the GPU API from logcat.txt; the
binary that ran must have the sha256 `runtime_lm` in matrix.yaml names, since a
moved `latest` is a new row. A row with the same row_id replaces the earlier one; a job without a
decodable proto, or with no iteration beyond the warm-up, is reported on stderr
and not written.

Usage:
  collect_lm.py SESSION_DIR... --model litert-community/Qwen3-0.6B [--model-size-mb 497.66]
                [--file qwen3_0_6b_mixed_int4.litertlm] [--task text-generation]
                [--runtime-version latest@2026-09-18] [--binary ...] [--warmup-iterations 1]
                [--device "Pixel 9 Pro" --os "Android 15 (API 35)"]
                [--matrix matrix.yaml] [--data-dir ../leaderboard/data] [--date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_MATRIX = HERE / "matrix.yaml"
DEFAULT_DATA_DIR = HERE.parent / "leaderboard" / "data"
PROTO_CACHE = pathlib.Path.home() / ".cache" / "litert-samples-benchmark" / "litert-lm"
PROTOS = ("runtime/proto/litert_lm_metrics.proto", "runtime/proto/engine.proto")
LITERT_LM_RAW = "https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/{ref}/"
MESSAGE = "litert.lm.proto.LitertLmMetricsList"
GPU_API_RE = re.compile(r"Created (OpenCL|WebGPU|Metal|Vulkan) device|Initializing (\w+)-based API")
# provenance.txt: "args: --backend=gpu --model_path=/data/local/tmp/litert-cli/<file> ... --max_num_tokens=1280 ..."
MODEL_PATH_RE = re.compile(r"--model_path=(\S+)")
MAX_TOKENS_RE = re.compile(r"--max_num_tokens=(\d+)")
BINARY_SHA_RE = re.compile(r"^([0-9a-f]{64})\s+(?:\./)?litert_lm_advanced_main$", re.M)
LIB_SHA_RE = re.compile(r"^([0-9a-f]{64})\s+(?:\./)?(\S+\.so)$", re.M)


def load_matrix(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml  # PyYAML
    except ImportError:
        sys.exit("collect_lm.py: PyYAML is needed to read matrix.yaml (pip install pyyaml)")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def ensure_protos(ref: str) -> pathlib.Path | None:
    """Fetches the two protos at `ref` into the cache once; None when offline."""
    root = PROTO_CACHE / ref
    for rel in PROTOS:
        dest = root / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(LITERT_LM_RAW.format(ref=ref) + rel, timeout=30) as r:
                dest.write_bytes(r.read())
        except OSError as e:
            print(f"collect_lm.py: could not fetch {rel} at {ref}: {e}", file=sys.stderr)
            return None
    return root


def decode(pb: pathlib.Path, ref: str) -> str | None:
    if not pb.exists():
        print(f"collect_lm.py: {pb} not found", file=sys.stderr)
        return None
    if not shutil.which("protoc"):
        print("collect_lm.py: protoc not found", file=sys.stderr)
        return None
    root = ensure_protos(ref)
    if root is None:
        return None
    with open(pb, "rb") as f:
        run = subprocess.run(["protoc", "--proto_path=.", f"--decode={MESSAGE}", PROTOS[0]],
                             cwd=root, stdin=f, capture_output=True, text=True)
    if run.returncode != 0:
        print(f"collect_lm.py: protoc failed on {pb}: {run.stderr.strip()}", file=sys.stderr)
        return None
    return run.stdout


def parse_iterations(text: str) -> list[dict]:
    """One dict per `metrics { ... }` block of a decoded LitertLmMetricsList."""
    out = []
    for block in re.split(r"^metrics \{", text, flags=re.M)[1:]:
        it = {"prefill_tokens": None, "decode_tokens": None, "prefill_tok_s": None, "decode_tok_s": None,
              "ttft_s": None, "init_total_ms": None, "init_executor_ms": None, "peak_mem_mb": None}
        m = re.search(r"benchmark_params \{(.*?)\}", block, re.S)
        if m:
            p = re.search(r"num_prefill_tokens: (\d+)", m.group(1)); d = re.search(r"num_decode_tokens: (\d+)", m.group(1))
            it["prefill_tokens"] = int(p.group(1)) if p else None
            it["decode_tokens"] = int(d.group(1)) if d else None
        for key, field in (("Init Total", "init_total_ms"), ("Init Executor", "init_executor_ms")):
            m = re.search(r'init_phase_durations_us \{\s*key: "%s"\s*value: (\d+)' % re.escape(key), block)
            if m:
                it[field] = round(int(m.group(1)) / 1000, 3)
        for turn, field in (("prefill_turns", "prefill_tok_s"), ("decode_turns", "decode_tok_s")):
            m = re.search(turn + r" \{.*?tokens_per_second: ([\d.eE+-]+)", block, re.S)
            if m:
                it[field] = round(float(m.group(1)), 3)
        m = re.search(r"time_to_first_token_seconds: ([\d.eE+-]+)", block)
        if m:
            it["ttft_s"] = round(float(m.group(1)), 4)
        m = re.search(r"peak_mem_mb: ([\d.eE+-]+)", block)
        if m and float(m.group(1)) > 0:
            it["peak_mem_mb"] = round(float(m.group(1)), 1)
        out.append(it)
    return out


def median(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.median(vals), 3) if vals else None


def gpu_api(log: pathlib.Path) -> str | None:
    if not log.exists():
        return None
    for line in log.read_text(errors="replace").splitlines():
        m = GPU_API_RE.search(line)
        if m:
            return m.group(1) or m.group(2)
    return None


def provenance(job_dir: pathlib.Path) -> dict:
    """From provenance.txt: the model file, the --max_num_tokens, the binary's sha256 and the .so files pushed beside it."""
    facts: dict = {"file": None, "max_num_tokens": None, "binary_sha256": None, "libs": []}
    path = job_dir / "provenance.txt"
    if not path.exists():
        return facts
    text = path.read_text(errors="replace")
    if m := MODEL_PATH_RE.search(text):
        facts["file"] = m.group(1).rsplit("/", 1)[-1]
    if m := MAX_TOKENS_RE.search(text):
        facts["max_num_tokens"] = int(m.group(1))
    if m := BINARY_SHA_RE.search(text):
        facts["binary_sha256"] = m.group(1)
    facts["libs"] = sorted(({"name": name, "sha256": sha} for sha, name in LIB_SHA_RE.findall(text)), key=lambda l: l["name"])
    return facts


def collect_job(job_dir: pathlib.Path, session: str, args, matrix: dict, meta: dict) -> dict | None:
    job = job_dir.name
    backend, device_id = job.split("-", 1)
    prov = provenance(job_dir)
    file = args.file or prov["file"]
    if not file:
        print(f"FAILED {session}/{job}: model file name not in provenance.txt; pass --file", file=sys.stderr)
        return None
    if args.file and prov["file"] and args.file != prov["file"]:
        print(f"FAILED {session}/{job}: provenance.txt names {prov['file']}, --file says {args.file}", file=sys.stderr)
        return None
    if meta["sha256"]:
        if not prov["binary_sha256"]:
            print(f"FAILED {session}/{job}: provenance.txt names no sha256 for the binary; pass --binary and --runtime-version", file=sys.stderr)
            return None
        if prov["binary_sha256"] != meta["sha256"]:
            print(f"FAILED {session}/{job}: the binary that ran (sha256 {prov['binary_sha256'][:8]}...) is not the one"
                  f" matrix.yaml runtime_lm names ({meta['sha256'][:8]}...): a moved `latest` is a new row, so update"
                  f" runtime_lm (version and sha256) first, or pass --binary and --runtime-version", file=sys.stderr)
            return None
    text = decode(job_dir / "metrics.pb", meta["ref"])
    if text is None:
        print(f"FAILED {session}/{job}: metrics.pb not decoded", file=sys.stderr)
        return None
    its = parse_iterations(text)
    if not its or its[0]["prefill_tok_s"] is None:
        print(f"FAILED {session}/{job}: no LitertLmMetrics with a prefill turn in metrics.pb", file=sys.stderr)
        return None
    measured = its[meta["warmup"]:]
    if not measured:
        print(f"FAILED {session}/{job}: {len(its)} iteration(s), none beyond the {meta['warmup']} warm-up", file=sys.stderr)
        return None
    task = args.task or next((m.get("task") for m in matrix.get("lm_models") or []
                              if m.get("repo") == args.model and m.get("file") == file), None)
    if not task:
        print(f"FAILED {session}/{job}: task unknown for {args.model}:{file}; add it to lm_models in matrix.yaml or pass --task", file=sys.stderr)
        return None
    max_tokens = args.max_num_tokens if args.max_num_tokens is not None else prov["max_num_tokens"]
    P, D = its[0]["prefill_tokens"], its[0]["decode_tokens"]
    if P is None or D is None or max_tokens is None:
        print(f"FAILED {session}/{job}: token counts unknown (prefill {P}, decode {D}, max_num_tokens {max_tokens});"
              f" metrics.pb needs benchmark_params and provenance.txt --max_num_tokens, or pass --max-num-tokens", file=sys.stderr)
        return None
    pb = job_dir / "metrics.pb"
    date = args.date or dt.datetime.fromtimestamp(pb.stat().st_mtime, dt.timezone.utc).date().isoformat()
    entry = meta["devices"].get(device_id, {})
    if not args.device and device_id not in meta["devices"]:
        print(f"collect_lm.py: device {device_id} is not in matrix.yaml; the board will show its id", file=sys.stderr)
    delegate = gpu_api(job_dir / "logcat.txt") if backend != "cpu" else None
    if backend != "cpu" and delegate is None:
        print(f"FAILED {session}/{job}: logcat.txt names no GPU API for a {backend} run", file=sys.stderr)
        return None
    libs = [{"name": l["name"], "source": meta["binary_dir"], "sha256": l["sha256"]} for l in prov["libs"]]
    return {
        "row_id": f"{args.model}:{file}@{meta['version']}/{args.platform}/{device_id}/{backend}/p{P}-d{D}-n{max_tokens}",
        "model": args.model, "file": file, "task": task, "model_size_mb": args.model_size_mb,
        "platform": args.platform, "device_id": device_id,
        "device": args.device or entry.get("name", device_id), "os": args.os or entry.get("os"),
        "accelerator": backend, "delegate": delegate,
        "conditions": {"prefill_tokens": P, "decode_tokens": D, "max_num_tokens": max_tokens,
                       "iterations": len(its), "warmup_iterations": meta["warmup"]},
        "metrics": {
            "prefill_tok_s": median([i["prefill_tok_s"] for i in measured]),
            "decode_tok_s": median([i["decode_tok_s"] for i in measured]),
            "ttft_s": median([i["ttft_s"] for i in measured]),
            "init_total_ms": its[0]["init_total_ms"], "init_executor_ms": its[0]["init_executor_ms"],
            "peak_mem_mb": median([i["peak_mem_mb"] for i in measured]),
        },
        "iterations": [{k: i[k] for k in ("prefill_tok_s", "decode_tok_s", "ttft_s", "init_total_ms")} for i in its],
        "runtime": "litert-lm", "runtime_version": meta["version"], "binary": meta["binary"], "libs": libs,
        "runner": args.runner, "session": session, "job": job, "repeat": 1,
        "source": "metrics.pb", "status": "measured", "date": date,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session_dirs", nargs="+", type=pathlib.Path, help="a session directory (or one job directory)")
    p.add_argument("--model", required=True, help="Hugging Face repo, e.g. litert-community/Qwen3-0.6B")
    p.add_argument("--file", help="the .litertlm file; read from provenance.txt when omitted")
    p.add_argument("--task", help="pipeline tag; matrix.yaml lm_models when omitted")
    p.add_argument("--runtime-version", help="LiteRT-LM release or 'latest@<date>'; matrix.yaml runtime_lm.version when omitted")
    p.add_argument("--binary", help="the binary the rows ran; matrix.yaml runtime_lm.binary (+ sha256) when omitted")
    p.add_argument("--proto-ref", help="LiteRT-LM git ref for the protos; matrix.yaml runtime_lm.proto_ref, else main")
    p.add_argument("--max-num-tokens", type=int, help="the --max_num_tokens the run used; provenance.txt when omitted")
    p.add_argument("--warmup-iterations", type=int, help="leading iterations left out of the medians; matrix.yaml runtime_lm.warmup_iterations, else 0")
    p.add_argument("--model-size-mb", type=float, help="the bundle's size in MB (bytes / 1e6, as benchmark_model reports); shown as Model size")
    p.add_argument("--platform", default="android")
    p.add_argument("--runner", default="ddp", help="what produced the session")
    p.add_argument("--device", help="device name; matrix.yaml devices when omitted")
    p.add_argument("--os", help="OS shown with the device; matrix.yaml devices when omitted")
    p.add_argument("--matrix", type=pathlib.Path, default=DEFAULT_MATRIX)
    p.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--date", help="YYYY-MM-DD; default = the day metrics.pb was written (mtime, UTC)")
    args = p.parse_args()

    matrix = load_matrix(args.matrix)
    lm = matrix.get("runtime_lm") or {}
    version = args.runtime_version or lm.get("version")
    if not version:
        sys.exit("collect_lm.py: runtime version is needed (--runtime-version, or runtime_lm.version in matrix.yaml)")
    devices = {}
    for group in ("measured", "planned"):
        for d in (((matrix.get("platforms") or {}).get(args.platform) or {}).get("devices") or {}).get(group) or []:
            devices[d["id"]] = d
    meta = {
        "version": version,
        "binary": args.binary or (f"{lm['binary']}, sha256 {lm['sha256']}" if lm.get("binary") and lm.get("sha256") else lm.get("binary")),
        "sha256": None if args.binary or not lm.get("sha256") else str(lm["sha256"]),
        "binary_dir": (args.binary or lm.get("binary") or "").split(",")[0].rsplit("/", 1)[0] or None,
        "ref": args.proto_ref or lm.get("proto_ref") or "main",
        "warmup": args.warmup_iterations if args.warmup_iterations is not None else int(lm.get("warmup_iterations") or 0),
        "devices": devices,
    }

    session_dirs = [sd.expanduser() for sd in args.session_dirs]
    for sd in session_dirs:
        if not sd.is_dir():
            sys.exit(f"collect_lm.py: not a directory: {sd}")
    rows, failed = [], 0
    for sd in session_dirs:
        is_job = (sd / "metrics.pb").exists()
        jobs = [sd] if is_job else sorted(d for d in sd.iterdir() if d.is_dir())
        for d in [j for j in jobs if "-" not in j.name]:
            print(f"collect_lm.py: skipping {d}: not a <accelerator>-<device> job directory", file=sys.stderr)
        jobs = [j for j in jobs if "-" in j.name]
        if not jobs:
            print(f"collect_lm.py: no job directories in {sd}", file=sys.stderr)
            failed += 1
            continue
        session = (sd.parent if is_job else sd).name
        for job_dir in jobs:
            row = collect_job(job_dir, session, args, matrix, meta)
            if row is None:
                failed += 1
                continue
            rows.append(row)
            mt = row["metrics"]
            print(f"{session}/{row['job']}: {row['file']} {row['platform']} {row['device']} {row['accelerator']}"
                  f" prefill {mt['prefill_tok_s']} tok/s, decode {mt['decode_tok_s']} tok/s, ttft {mt['ttft_s']} s,"
                  f" init {mt['init_total_ms']} ms, {row['conditions']['iterations']} iteration(s) with"
                  f" {meta['warmup']} warm-up ({row['delegate'] or 'cpu'}), metrics.pb")
    if rows:
        args.data_dir.mkdir(parents=True, exist_ok=True)
        path = args.data_dir / "measurements-lm.jsonl"
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
        print(f"{len(rows)} row(s) written to {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
