# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""\
Qwen/Qwen2.5-Coder-1.5B-Instruct -> Qwen2.5-Coder-1.5B-Instruct_int4.litertlm.

One call of litert-torch's stock HF exporter with qwen2_5_coder_extension.py
(an out-of-tree `ModelExtension`) in `extension_modules`. The extension holds
the export arguments and the metadata; this script fetches the checkpoint at
the pinned revision, exports it, and names the file as the published one at
litert-community/Qwen2.5-Coder-1.5B-Instruct. No change to litert-torch, no
edit of the bundle after the export.

Usage:
  python build_qwen2_5_coder_1_5b.py --out out_int4            # 1.12 GB file
  python build_qwen2_5_coder_1_5b.py --dry-run                 # the export plan
  python build_qwen2_5_coder_1_5b.py --dry-run --no-extension  # the stock plan
"""

import argparse
import os
import shutil
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
SOURCE_FILES = [
    "config.json", "generation_config.json", "model.safetensors",
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
]
FILE_NAME = "Qwen2.5-Coder-1.5B-Instruct_int4.litertlm"
EXTENSION = Path(__file__).resolve().parent / "qwen2_5_coder_extension.py"


def check_toolchain():
  """Stops early on a litert-torch without the out-of-tree extension path."""
  from importlib import metadata

  version = "(not installed)"
  for dist in ("litert-torch", "litert-torch-nightly"):
    try:
      version = f"{dist} {metadata.version(dist)}"
      break
    except metadata.PackageNotFoundError:
      pass
  try:
    from litert_torch.generative.export_hf.model_ext import extension
  except ImportError as e:
    extension = None
    version += f" ({e})"
  if not (hasattr(extension, "load_extension_file")
          and hasattr(getattr(extension, "ModelExtension", None),
                      "get_export_defaults")):
    raise SystemExit(
        f"{version} cannot load qwen2_5_coder_extension.py: the recipe needs "
        "a .py file in extension_modules (litert-torch 864a966, 2026-10-28) "
        "and ModelExtension.get_export_defaults (764beb0, 2026-11-03), which "
        "no release carries yet. Install the version in the README.")


def main():
  """Parses the arguments, then prints the plan or writes the file."""
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--model", default=MODEL_ID,
                  help="HF model id (fetched at the pinned revision) or a "
                       "local copy of that revision")
  ap.add_argument("--out", default="out_int4", help="output dir")
  ap.add_argument("--dry-run", action="store_true",
                  help="print the export plan: each argument and its source")
  ap.add_argument("--no-extension", action="store_true",
                  help="with --dry-run: the plan of the stock exporter alone")
  args = ap.parse_args()
  if args.no_extension and not args.dry_run:
    ap.error("--no-extension is for --dry-run")

  check_toolchain()
  src = Path(args.model)
  if not src.exists():
    from huggingface_hub import snapshot_download

    src = Path(snapshot_download(args.model, revision=REVISION,
                                 allow_patterns=SOURCE_FILES))

  from litert_torch.generative.export_hf.export import export

  if args.dry_run:
    print(export(model=str(src), dry_run=True,
                 extension_modules=None if args.no_extension
                 else str(EXTENSION)))
    return
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  work = out / "work"
  shutil.rmtree(work, ignore_errors=True)
  t0 = time.time()
  try:
    export(model=str(src), output_dir=str(work),
           extension_modules=str(EXTENSION))
  except BaseException:
    shutil.rmtree(work, ignore_errors=True)  # a failed export leaves nothing
    raise
  bundle = work / "model.litertlm"
  if not bundle.exists():
    raise SystemExit("the export finished without model.litertlm")
  final = out / FILE_NAME
  bundle.replace(final)
  shutil.rmtree(work)
  print(f"EXPORT_DONE {final} ({os.path.getsize(final):,} B, "
        f"{time.time() - t0:.0f} s)")


if __name__ == "__main__":
  main()
