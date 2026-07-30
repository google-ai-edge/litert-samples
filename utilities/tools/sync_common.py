# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
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
# ==============================================================================
"""Check or update vendored copies of common/kotlin utilities across modules.

Every module in this repo is an independent Gradle project, so shared Kotlin
utilities are distributed as vendored copies (one physical file per module)
instead of a shared Gradle module — a sample must stay standalone and fully
readable on its own. This tool keeps those copies byte-identical to the
canonical sources in common/kotlin/ (only the `package` line may differ).

Usage:
    python tools/sync_common.py --check   # list IDENTICAL/DIVERGED copies; exit 1 on drift
    python tools/sync_common.py --apply   # rewrite diverged copies from the canonical
"""

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CANONICAL_DIR = REPO_ROOT / "utilities" / "common" / "kotlin"
PACKAGE_PLACEHOLDER = "package __MODULE_PACKAGE__"

# Copies that intentionally diverge from the canonical (see common/README.md).
# They are reported but do not fail --check and are never rewritten by --apply.
KNOWN_DIVERGENCES: set[str] = set()


def normalized(lines: list[str]) -> list[str]:
    """Drop the `package` declaration and the provenance header for comparison.

    Both lines legitimately differ between the canonical and a vendored copy
    (older copies predate the provenance header).
    """
    return [line for line in lines
            if not line.startswith(("package ", "// Vendored from "))]


def find_copies(name: str) -> list[pathlib.Path]:
    """Find every module's vendored copy of a canonical file by basename."""
    copies = []
    # Sample apps sit at varying depths (compiled_model_api/<task>/<variant>/
    # [android/]app/src/main/..., samples/litert/<name>/...), so match on the
    # source-set suffix rather than a fixed prefix.
    for pattern in (f"**/src/main/java/**/{name}", f"**/src/main/kotlin/**/{name}"):
      for path in REPO_ROOT.glob(pattern):
        if CANONICAL_DIR not in path.parents:
          copies.append(path)
    return sorted(copies)


def package_line(path: pathlib.Path) -> str:
    """Return the copy's own `package` line (falls back to the placeholder)."""
    for line in path.read_text().splitlines():
        if line.startswith("package "):
            return line
    return PACKAGE_PLACEHOLDER


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report drift, exit 1 if any")
    mode.add_argument("--apply", action="store_true", help="rewrite diverged copies")
    args = parser.parse_args()

    canonical_files = sorted(CANONICAL_DIR.glob("*.kt"))
    if not canonical_files:
        print(f"No canonical files in {CANONICAL_DIR}", file=sys.stderr)
        return 2

    drifted = 0
    for canonical in canonical_files:
        canonical_lines = normalized(canonical.read_text().splitlines())
        copies = find_copies(canonical.name)
        print(f"{canonical.name}: {len(copies)} copies")
        for copy in copies:
            if normalized(copy.read_text().splitlines()) == canonical_lines:
                continue
            rel = copy.relative_to(REPO_ROOT)
            if str(rel) in KNOWN_DIVERGENCES:
                print(f"  KNOWN-DIVERGED (allowlisted) {rel}")
                continue
            drifted += 1
            if args.apply:
                body = canonical.read_text().replace(
                    PACKAGE_PLACEHOLDER, package_line(copy), 1)
                copy.write_text(body)
                print(f"  UPDATED  {rel}")
            else:
                print(f"  DIVERGED {rel}")

    if args.check and drifted:
        print(f"\n{drifted} diverged copies — run with --apply to update them.")
        return 1
    print(f"\nAll copies in sync." if not drifted else f"\n{drifted} copies updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
