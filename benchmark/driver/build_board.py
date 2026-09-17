#!/usr/bin/env python3
"""Build board.json from measurements.jsonl.

The board keeps one row per (model, file, platform, device, accelerator): the
row measured at the newest runtime version (newest date on a tie). Rows are
grouped by platform in the order matrix.yaml lists the platforms, then ordered
by task, device, accelerator and median latency. Only rows with status
"measured" are used. Platforms the matrix marks `rows: planned` and that have no
rows are listed as planned, so the page can show them.

Usage:
  build_board.py [--data-dir ../leaderboard/data] [--matrix matrix.yaml]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_MATRIX = HERE / "matrix.yaml"
DEFAULT_DATA_DIR = HERE.parent / "leaderboard" / "data"


def version_key(v: str) -> tuple:
    """Orders '2.2.0' < '2.10.0'; non-numeric parts compare as text after numbers."""
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in str(v).split("."))


def load_platforms(path: pathlib.Path) -> dict[str, dict]:
    """{platform id: {name, rows}} from matrix.yaml, in file order; {} without the file or PyYAML."""
    if not path.exists():
        return {}
    try:
        import yaml  # PyYAML
    except ImportError:
        print("build_board.py: PyYAML not installed; platforms are listed from the rows only", file=sys.stderr)
        return {}
    with open(path) as f:
        matrix = yaml.safe_load(f) or {}
    return {k: {"name": v.get("name", k), "rows": v.get("rows")} for k, v in (matrix.get("platforms") or {}).items()}


def build(rows: list[dict], platforms: dict[str, dict]) -> dict:
    newest: dict[tuple, dict] = {}
    for r in rows:
        if r.get("status") != "measured":
            continue
        key = (r["model"], r["file"], r["platform"], r["device_id"], r["accelerator"])
        cur = newest.get(key)
        rank = (version_key(r["runtime_version"]), r.get("date") or "")
        if cur is None or rank > (version_key(cur["runtime_version"]), cur.get("date") or ""):
            newest[key] = r
    order = {p: i for i, p in enumerate(platforms)}
    for r in newest.values():
        order.setdefault(r["platform"], len(order))
    board_rows = sorted(
        newest.values(),
        key=lambda r: (order[r["platform"]], r["task"], r["device"], r["accelerator"],
                       r["latency_ms"]["median"] if r["latency_ms"]["median"] is not None else float("inf")),
    )
    counts: dict[str, int] = {}
    for r in board_rows:
        counts[r["platform"]] = counts.get(r["platform"], 0) + 1
    platform_list = []
    for p in sorted(order, key=order.get):
        entry = {"id": p, "name": platforms.get(p, {}).get("name", p), "rows": counts.get(p, 0)}
        if platforms.get(p, {}).get("rows") == "planned" and not counts.get(p):
            entry["status"] = "planned"
        platform_list.append(entry)
    devices = sorted({(order[r["platform"]], r["platform"], r["device_id"], r["device"]) for r in board_rows})
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "runtime": "litert",
        "runtime_versions": sorted({r["runtime_version"] for r in board_rows}, key=version_key),
        "row_count": len(board_rows),
        "models": sorted({r["model"] for r in board_rows}),
        "platforms": platform_list,
        "devices": [{"id": i, "name": n, "platform": p} for _, p, i, n in devices],
        "rows": board_rows,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--matrix", type=pathlib.Path, default=DEFAULT_MATRIX)
    args = p.parse_args()
    src = args.data_dir / "measurements.jsonl"
    if not src.exists():
        sys.exit(f"build_board.py: {src} not found; run collect.py first")
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    board = build(rows, load_platforms(args.matrix))
    out = args.data_dir / "board.json"
    out.write_text(json.dumps(board, indent=1, ensure_ascii=False) + "\n")
    print(f"{board['row_count']} row(s) from {len(rows)} measurement(s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
