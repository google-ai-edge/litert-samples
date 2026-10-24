#!/usr/bin/env python3
# Copyright 2026 Google LLC
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
"""Ahead-of-Time (AOT) Qualcomm NPU compiler for standalone LiteRT (.tflite) graphs.

Note on .litertlm packages:
  `ai_edge_litert.aot.aot_compile` compiles standalone `.tflite` FlatBuffer graphs
  (magic identifier `TFL3`), NOT multi-component `.litertlm` bundles (`RTLM`).
  Pre-compiled Qualcomm SM8750 `.litertlm` packages for Gemma 4 E2B and FastVLM 0.5B
  are available directly on Hugging Face:
    - https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm
    - https://huggingface.co/litert-community/FastVLM-0.5B
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence

TFLITE_MAGIC = b"TFL3"
LITERTLM_MAGIC = b"RTLM"
LITERTLM_HEADER_PREFIX = b"LITERTLM"


class CompilationError(RuntimeError):
  """Raised when pre-flight validation or AOT compilation fails."""


def validate_tflite_model(model_path: Path) -> None:
  """Validates that `model_path` exists and is a `TFL3` FlatBuffer model."""
  if not model_path.is_file():
    raise CompilationError(f"Model file not found: {model_path}")

  with model_path.open("rb") as f:
    header = f.read(8)

  if len(header) < 8:
    raise CompilationError(
        f"File '{model_path}' is too small ({len(header)} bytes) to be a valid "
        "LiteRT (.tflite) FlatBuffer model."
    )

  file_identifier = header[4:8]
  if (
      header.startswith(LITERTLM_HEADER_PREFIX)
      or file_identifier == LITERTLM_MAGIC
      or model_path.suffix.lower() == ".litertlm"
  ):
    raise CompilationError(
        f"'{model_path.name}' is a LiteRT-LM container package (identifier "
        "'RTLM'), not a standalone '.tflite' FlatBuffer graph ('TFL3').\n\n"
        "Why this happens:\n"
        "  `ai_edge_litert.aot.aot_compile` compiles individual `.tflite` "
        "graphs and cannot directly ingest multi-subgraph `.litertlm` bundles.\n\n"
        "Recommended action:\n"
        "  Download a pre-compiled Qualcomm NPU `.litertlm` package directly "
        "from Hugging Face:\n"
        "    - Gemma 4 E2B (SM8750): "
        "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/blob/main/gemma-4-E2B-it_qualcomm_sm8750.litertlm\n"
        "    - FastVLM 0.5B (SM8750): "
        "https://huggingface.co/litert-community/FastVLM-0.5B/blob/main/FastVLM-0.5B.qualcomm.sm8750.litertlm\n"
        "  Or pass a standalone `.tflite` model file via `--model_path model.tflite`."
    )

  if file_identifier != TFLITE_MAGIC:
    pretty_id = file_identifier.decode("ascii", errors="replace")
    raise CompilationError(
        f"Invalid FlatBuffer identifier '{pretty_id}' in '{model_path}'. "
        "Expected a valid '.tflite' model with identifier 'TFL3'."
    )


def configure_qairt_environment(qairt_root: str | None) -> Path | None:
  """Configures `QAIRT_ROOT`, `PATH`, and `LD_LIBRARY_PATH` when provided."""
  effective_root = qairt_root or os.environ.get("QAIRT_ROOT")
  if not effective_root:
    return None

  root_path = Path(effective_root).expanduser().resolve()
  if not root_path.is_dir():
    raise CompilationError(
        f"QAIRT SDK directory does not exist: {root_path}\n"
        "Unset QAIRT_ROOT to let `ai-edge-litert` auto-detect its bundled "
        "Qualcomm toolchain, or pass a valid `--qairt_root` directory."
    )

  os.environ["QAIRT_ROOT"] = str(root_path)
  bin_dirs = [
      str(p)
      for p in (
          root_path / "bin" / "x86_64-linux-clang",
          root_path / "bin" / "aarch64-ubuntu-gcc9.4",
      )
      if p.is_dir()
  ]
  lib_dirs = [
      str(p)
      for p in (
          root_path / "lib" / "x86_64-linux-clang",
          root_path / "lib" / "aarch64-ubuntu-gcc9.4",
      )
      if p.is_dir()
  ]

  if bin_dirs:
    current_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(
        bin_dirs + ([current_path] if current_path else [])
    )
  if lib_dirs:
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        lib_dirs + ([current_ld] if current_ld else [])
    )

  return root_path


def load_aot_modules():
  """Imports LiteRT AOT compiler modules lazily with clear install guidance."""
  try:
    from ai_edge_litert.aot import aot_compile as aot_lib  # pylint: disable=g-import-not-at-top
    from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target  # pylint: disable=g-import-not-at-top
    return aot_lib, qnn_target
  except ImportError as exc:
    raise CompilationError(
        "The 'ai-edge-litert' compilation package is not installed in this "
        "Python environment.\n"
        "Install it on Linux (x86_64 / WSL2) with:\n"
        "  pip install ai-edge-litert"
    ) from exc


def compile_for_npu(
    model_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] = "compiled",
    soc_model: str = "SM8750",
    qairt_root: str | None = None,
) -> Path:
  """Compiles a standalone `.tflite` model for the target Qualcomm Hexagon NPU."""
  resolved_model = Path(model_path).expanduser().resolve()
  validate_tflite_model(resolved_model)

  configured_qairt = configure_qairt_environment(qairt_root)
  aot_lib, qnn_target = load_aot_modules()

  soc_key = soc_model.strip().upper()
  try:
    soc_enum = getattr(qnn_target.SocModel, soc_key)
  except AttributeError as exc:
    available = ", ".join(
        name for name in dir(qnn_target.SocModel) if not name.startswith("_")
    )
    raise CompilationError(
        f"Unsupported Qualcomm SoC model '{soc_model}'. Available targets: {available}"
    ) from exc

  resolved_out = Path(output_dir).expanduser().resolve()
  resolved_out.mkdir(parents=True, exist_ok=True)

  print(f"Input model : {resolved_model}")
  print(f"Target SoC  : Qualcomm {soc_key}")
  if configured_qairt:
    print(f"QAIRT SDK   : {configured_qairt}")
  else:
    print("QAIRT SDK   : auto-detected from ai-edge-litert")
  print(f"Output dir  : {resolved_out}")

  target_cfg = qnn_target.Target(soc_enum)
  compiled_models = aot_lib.aot_compile(
      str(resolved_model),
      output_dir=str(resolved_out),
      target=[target_cfg],
  )
  if hasattr(compiled_models, "export"):
    compiled_models.export(str(resolved_out))

  print(f"Compilation succeeded. Output saved in '{resolved_out}'.")
  return resolved_out


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description=(
          "Ahead-of-Time (AOT) compile a standalone LiteRT (.tflite) model for "
          "Qualcomm Hexagon NPU execution."
      )
  )
  parser.add_argument(
      "-m",
      "--model_path",
      "--model-path",
      dest="model_path",
      required=True,
      help="Path to a standalone '.tflite' FlatBuffer model ('TFL3').",
  )
  parser.add_argument(
      "-o",
      "--output_dir",
      "--output-dir",
      dest="output_dir",
      default="compiled",
      help="Directory where compiled artifacts will be written (default: compiled).",
  )
  parser.add_argument(
      "-s",
      "--soc_model",
      "--soc-model",
      dest="soc_model",
      default="SM8750",
      help="Target Qualcomm SoC identifier (default: SM8750 for Snapdragon 8 Elite).",
  )
  parser.add_argument(
      "--qairt_root",
      "--qairt-root",
      dest="qairt_root",
      default=None,
      help=(
          "Optional path to a local Qualcomm AI Runtime (QAIRT) SDK root. "
          "Defaults to $QAIRT_ROOT if set, or ai-edge-litert auto-detection."
      ),
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  try:
    compile_for_npu(
        model_path=args.model_path,
        output_dir=args.output_dir,
        soc_model=args.soc_model,
        qairt_root=args.qairt_root,
    )
    return 0
  except CompilationError as err:
    print(f"Error: {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  sys.exit(main())
