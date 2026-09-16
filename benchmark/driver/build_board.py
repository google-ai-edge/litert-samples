#!/usr/bin/env python3
"""Build board.json from measurements.jsonl.

The board keeps one row per (model, file, device, accelerator): the row measured
at the newest runtime version. Rows are ordered by task, device, accelerator and
median latency. Only rows with status "measured" are used.

Usage:
  build_board.py [--data-dir ../leaderboard/data]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE.parent / "leaderboard" / "data"


def version_key(v: str) -> tuple:
    """Orders '2.2.0' < '2.10.0'; non-numeric parts compare as text after numbers."""
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in str(v).split("."))


def build(rows: list[dict]) -> dict:
    newest: dict[tuple, dict] = {}
    for r in rows:
        if r.get("status") != "measured":
            continue
        key = (r["model"], r["file"], r["device_id"], r["accelerator"])
        cur = newest.get(key)
        if cur is None or version_key(r["runtime_version"]) > version_key(cur["runtime_version"]):
            newest[key] = r
    board_rows = sorted(
        newest.values(),
        key=lambda r: (r["task"], r["device"], r["accelerator"], r["latency_ms"]["median"] if r["latency_ms"]["median"] is not None else float("inf")),
    )
    devices = sorted({(r["device_id"], r["device"]) for r in board_rows})
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "runtime": "litert",
        "runtime_versions": sorted({r["runtime_version"] for r in board_rows}, key=version_key),
        "row_count": len(board_rows),
        "models": sorted({r["model"] for r in board_rows}),
        "devices": [{"id": i, "name": n} for i, n in devices],
        "rows": board_rows,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    args = p.parse_args()
    src = args.data_dir / "measurements.jsonl"
    if not src.exists():
        sys.exit(f"build_board.py: {src} not found; run collect.py first")
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    board = build(rows)
    out = args.data_dir / "board.json"
    out.write_text(json.dumps(board, indent=1, ensure_ascii=False) + "\n")
    print(f"{board['row_count']} row(s) from {len(rows)} measurement(s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
