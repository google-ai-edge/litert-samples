#!/usr/bin/env python3
"""Turn a LiteRT-LM benchmark session into leaderboard rows.

The session is the output directory `litert_lm_harness.sh` writes on the device
and a Device Run session pulls back: one `metrics_<run>.pb` per run of the
LiteRT-LM benchmark binary (`litert.lm.proto.LitertLmMetricsList`, one entry
per `--num_iterations`), `<run>.log` (its stdout, which names the GPU API) and
`provenance.txt`. <run> is the backend name (cpu, gpu), with `_2`, `_3` for
repeats. This script decodes the protos with protoc against LiteRT-LM's
`litert_lm_metrics.proto` (fetched once into the cache at the ref matrix.yaml
names) and writes one row per run to measurements-lm.jsonl: prefill and decode
tokens/s and time to first token as the median over the iterations, init from
the first iteration, every iteration kept in the row. A row with the same
row_id replaces the earlier one; a run without a decodable proto is reported
on stderr and not written.

Usage:
  collect_lm.py OUTPUT_DIR --model litert-community/Qwen3-0.6B --file qwen3_0_6b_mixed_int4.litertlm
                --device-id caiman-35 --session session-1234abcd
                [--platform android] [--runner ddp-http] [--task text-generation]
                [--runtime-version latest@2026-09-18] [--binary ...] [--max-num-tokens 1280]
                [--model-size-mb 497.66] [--matrix matrix.yaml] [--data-dir ../leaderboard/data] [--date YYYY-MM-DD]
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
RUN_RE = re.compile(r"^metrics_([a-z]+)(?:_(\d+))?\.pb$")


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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_dir", type=pathlib.Path, help="the harness output directory (metrics_<run>.pb, <run>.log, provenance.txt)")
    p.add_argument("--model", required=True, help="Hugging Face repo, e.g. litert-community/Qwen3-0.6B")
    p.add_argument("--file", required=True, help="the .litertlm file")
    p.add_argument("--device-id", required=True, help="device id, e.g. caiman-35")
    p.add_argument("--session", required=True, help="the session id the outputs came from")
    p.add_argument("--platform", default="android")
    p.add_argument("--runner", default="ddp-http", help="what produced the session")
    p.add_argument("--task", help="pipeline tag; matrix.yaml lm_models when omitted")
    p.add_argument("--runtime-version", help="LiteRT-LM release or 'latest@<date>'; matrix.yaml runtime_lm.version when omitted")
    p.add_argument("--binary", help="the binary the rows ran; matrix.yaml runtime_lm.binary (+ sha256) when omitted")
    p.add_argument("--proto-ref", help="LiteRT-LM git ref for the protos; matrix.yaml runtime_lm.proto_ref, else main")
    p.add_argument("--max-num-tokens", type=int, help="the --max_num_tokens the run used; provenance.txt when omitted")
    p.add_argument("--model-size-mb", type=float, help="the bundle's size in MB (bytes / 1e6, as benchmark_model reports); shown as Model size")
    p.add_argument("--device", help="device name; matrix.yaml devices when omitted")
    p.add_argument("--os", help="OS shown with the device; matrix.yaml devices when omitted")
    p.add_argument("--matrix", type=pathlib.Path, default=DEFAULT_MATRIX)
    p.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--date", help="YYYY-MM-DD; default = the day the protos were written (mtime, UTC)")
    args = p.parse_args()

    matrix = load_matrix(args.matrix)
    lm = matrix.get("runtime_lm") or {}
    version = args.runtime_version or lm.get("version")
    binary = args.binary or (f"{lm['binary']}, sha256 {lm['sha256']}" if lm.get("binary") and lm.get("sha256") else lm.get("binary"))
    ref = args.proto_ref or lm.get("proto_ref") or "main"
    task = args.task or next((m.get("task") for m in matrix.get("lm_models") or [] if m.get("repo") == args.model and m.get("file") == args.file), None)
    if not version or not task:
        sys.exit("collect_lm.py: runtime version and task are needed (flags, or runtime_lm / lm_models in matrix.yaml)")
    devices = {}
    for group in ("measured", "planned"):
        for d in (((matrix.get("platforms") or {}).get(args.platform) or {}).get("devices") or {}).get(group) or []:
            devices[d["id"]] = d
    entry = devices.get(args.device_id, {})
    out = args.output_dir.expanduser()
    if not out.is_dir():
        sys.exit(f"collect_lm.py: not a directory: {out}")
    max_tokens = args.max_num_tokens
    prov = out / "provenance.txt"
    if max_tokens is None and prov.exists():
        m = re.search(r"max_num_tokens: (\d+)", prov.read_text(errors="replace"))
        max_tokens = int(m.group(1)) if m else None

    rows, failed = [], 0
    for pb in sorted(out.glob("metrics_*.pb")):
        m = RUN_RE.match(pb.name)
        if not m:
            continue
        backend, repeat = m.group(1), int(m.group(2) or 1)
        run = pb.stem.removeprefix("metrics_")
        text = decode(pb, ref)
        its = parse_iterations(text) if text else []
        if not its or its[0]["prefill_tok_s"] is None:
            print(f"FAILED {out.name}/{run}: no decodable LitertLmMetrics in {pb.name}", file=sys.stderr)
            failed += 1
            continue
        P, D = its[0]["prefill_tokens"], its[0]["decode_tokens"]
        date = args.date or dt.datetime.fromtimestamp(pb.stat().st_mtime, dt.timezone.utc).date().isoformat()
        row = {
            "row_id": f"{args.model}:{args.file}@{version}/{args.platform}/{args.device_id}/{backend}/p{P}-d{D}-n{max_tokens}"
                      + (f"/run{repeat}" if repeat > 1 else ""),
            "model": args.model, "file": args.file, "task": task, "model_size_mb": args.model_size_mb,
            "platform": args.platform, "device_id": args.device_id,
            "device": args.device or entry.get("name", args.device_id), "os": args.os or entry.get("os"),
            "accelerator": backend, "delegate": gpu_api(out / f"{run}.log") if backend != "cpu" else None,
            "conditions": {"prefill_tokens": P, "decode_tokens": D, "max_num_tokens": max_tokens, "iterations": len(its)},
            "metrics": {
                "prefill_tok_s": median([i["prefill_tok_s"] for i in its]),
                "decode_tok_s": median([i["decode_tok_s"] for i in its]),
                "ttft_s": median([i["ttft_s"] for i in its]),
                "init_total_ms": its[0]["init_total_ms"], "init_executor_ms": its[0]["init_executor_ms"],
                "peak_mem_mb": median([i["peak_mem_mb"] for i in its]),
            },
            "iterations": [{k: i[k] for k in ("prefill_tok_s", "decode_tok_s", "ttft_s", "init_total_ms")} for i in its],
            "runtime": "litert-lm", "runtime_version": version, "binary": binary,
            "runner": args.runner, "session": args.session, "job": run, "repeat": repeat,
            "source": "metrics.pb", "status": "measured", "date": date,
        }
        rows.append(row)
        mt = row["metrics"]
        print(f"{args.session}/{run}: {args.file} {args.platform} {row['device']} {backend} prefill {mt['prefill_tok_s']} tok/s,"
              f" decode {mt['decode_tok_s']} tok/s, ttft {mt['ttft_s']} s, init {mt['init_total_ms']} ms, {len(its)} iteration(s)"
              f" ({row['delegate'] or 'cpu'}), metrics.pb")
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
