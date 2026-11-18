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

"""Ministral-3-3B-Instruct-2512 (BF16) -> its q4_block32_ekv4096 .litertlm.

Converts the text decoder of mistralai/Ministral-3-3B-Instruct-2512-BF16 into
the file published at litert-community/Ministral-3-3B-Instruct-2512 through
litert-torch's stock HF exporter: one `export()` call with
`ministral3_extension.py`, an out-of-tree `ModelExtension`, in
`extension_modules`. The extension reads the decoder straight from the
multimodal checkpoint (no extracted copy), and gives the export arguments
(int4 weights by recipe_int4.json, one 128-token prefill, a 4,096-token
cache, the embedding table in its own section), the tokenizer section and the
LlmMetadata; this script holds the checkpoint's revision and the output name.
The script stops before the export on a litert-torch that cannot load an
extension file.

Usage:
  python build_ministral_3_3b.py --out out_int4
  python build_ministral_3_3b.py --dry-run                 # resolved arguments
  python build_ministral_3_3b.py --dry-run --no-extension  # the stock ones
"""

import argparse
import os
import shutil
import time
from pathlib import Path

MODEL_ID = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
REVISION = "b6d637bef2393152b3da2b2fde72eecdee30557e"
# model-*.safetensors: the HF shards; consolidated.safetensors, a second copy
# of the same weights in Mistral's own layout, is not needed.
SOURCE_FILES = ["*.json", "model-*.safetensors", "chat_template.jinja"]
HERE = Path(__file__).resolve().parent
EXTENSION = HERE / "ministral3_extension.py"
OUTPUT_NAME = "Ministral-3-3B-Instruct-2512_q4_block32_ekv4096.litertlm"


def check_toolchain():
  """Stops early on a litert-torch that cannot load an extension file."""
  try:
    from litert_torch.generative.export_hf.model_ext import extension
  except ImportError:
    extension = None
  if not (hasattr(extension, "load_extension_file")
          and hasattr(extension, "ExportDefaults")):
    from importlib import metadata

    installed = "not installed"
    for dist in ("litert-torch-nightly", "litert-torch"):
      try:
        installed = f"{dist} {metadata.version(dist)}"
        break
      except metadata.PackageNotFoundError:
        pass
    raise SystemExit(
        f"{installed}: "
        "this litert-torch cannot load ministral3_extension.py: it has no "
        "out-of-tree extension files (`extension_modules` with a .py path) or "
        "no extension export defaults. No release carries them yet (0.10.0 "
        "does not); install the nightly in the README.")
  import transformers

  print(f"toolchain: transformers {transformers.__version__}, litert-torch "
        "with out-of-tree extension files")


def main():
  """Exports the file, or prints the export arguments with --dry-run."""
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--model", default=MODEL_ID,
                  help="HF model id (fetched at the pinned revision) or a "
                       "local copy of that revision")
  ap.add_argument("--out", default="out_int4", help="output dir")
  ap.add_argument("--dry-run", action="store_true",
                  help="print the resolved export arguments and stop")
  ap.add_argument("--no-extension", action="store_true",
                  help="without ministral3_extension.py (study only)")
  args = ap.parse_args()

  check_toolchain()
  from litert_torch.generative.export_hf.export import export

  extension_modules = None if args.no_extension else str(EXTENSION)
  src = Path(args.model)
  if not src.exists():
    from huggingface_hub import snapshot_download

    patterns = ["*.json"] if args.dry_run else SOURCE_FILES
    src = Path(snapshot_download(args.model, revision=REVISION,
                                 allow_patterns=patterns))
  if args.dry_run:
    plan = export(model=str(src), dry_run=True,
                  extension_modules=extension_modules)
    print(f"task {plan.task}")
    for key, value in sorted(plan.export_kwargs.items()):
      print(f"  {key} = {value!r}")
    return

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  work = out / "ministral3.work"
  shutil.rmtree(work, ignore_errors=True)
  t0 = time.time()
  try:
    export(model=str(src), output_dir=str(work),
           extension_modules=extension_modules)
  except BaseException:
    shutil.rmtree(work, ignore_errors=True)  # A failed export leaves nothing.
    raise
  bundle = work / "model.litertlm"
  if not bundle.exists():
    raise SystemExit("the export finished without model.litertlm")
  final = out / OUTPUT_NAME
  bundle.replace(final)
  shutil.rmtree(work)
  print(f"EXPORT_DONE {final} ({os.path.getsize(final):,} B, "
        f"{time.time() - t0:.0f} s)")


if __name__ == "__main__":
  main()
