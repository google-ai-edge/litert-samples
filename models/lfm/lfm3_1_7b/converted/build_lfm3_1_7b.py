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

"""LiquidAI/LFM3-1.7B (Lfm3ForCausalLM, remote code) -> four .litertlm files.

Converts the Hugging Face checkpoint into the four files of
litert-community/LFM3-1.7B-LiteRT through litert-torch's stock HF exporter,
one `export()` call per file, with `lfm3_extension.py` (an out-of-tree
`ModelExtension`) handed to it in `extension_modules`: no change to
litert-torch, no model loader of its own, no edit of a bundle after the
export.

  name       file                                    weights                      cache
  q1b_2048   LFM3-1.7B_mixed_int4_ekv2048.litertlm   recipe_q1b.json              2048
  q1b_4096   LFM3-1.7B_mixed_int4_ekv4096.litertlm   recipe_q1b.json              4096
  int8_2048  LFM3-1.7B_int8_ekv2048.litertlm         dynamic_wi8_afp32 (stock)    2048
  int8_4096  LFM3-1.7B_int8_ekv4096.litertlm         dynamic_wi8_afp32 (stock)    4096

recipe_q1b.json = int4 blockwise-32 on the MLP gate and up projections (`w1`,
`w3`), int8 per channel on every other weight. All four files: prefill
signatures 1024 / 256 / 64 / 16 / 4 / 1, the checkpoint's tokenizer.json byte
for byte, the chat template as the extension leaves it, the sampler top-k 50 /
top-p 1 / temperature 0.1 (`sampler_top_p=1.0` over the checkpoint's
generation config), and no attention, cache, short-conv or rope composite (the
only composite is litert-torch's `odml.rms_norm`). The script stops before the
export on a transformers or a litert-torch that cannot run it.

Usage:
  python build_lfm3_1_7b.py --out out_lfm3                   # the four files
  python build_lfm3_1_7b.py --out out_lfm3 --files q1b_2048  # one of them
  python build_lfm3_1_7b.py --out out_stock --files q1b_2048 --no-extension
      # the stock export without lfm3_extension.py, for study
"""

import argparse
import os
import shutil
import time
from pathlib import Path

MODEL_ID = "LiquidAI/LFM3-1.7B"
REVISION = "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"
SOURCE_FILES = [
    "config.json", "generation_config.json", "configuration_lfm3.py",
    "modeling_lfm3.py", "lfm3_conv.py", "model.safetensors", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
]
HERE = Path(__file__).resolve().parent
EXTENSION = HERE / "lfm3_extension.py"
PREFILL = [1024, 256, 64, 16, 4, 1]
FILES = {
    "q1b_2048": ("LFM3-1.7B_mixed_int4_ekv2048.litertlm", str(HERE / "recipe_q1b.json"), 2048),
    "q1b_4096": ("LFM3-1.7B_mixed_int4_ekv4096.litertlm", str(HERE / "recipe_q1b.json"), 4096),
    "int8_2048": ("LFM3-1.7B_int8_ekv2048.litertlm", "dynamic_wi8_afp32", 2048),
    "int8_4096": ("LFM3-1.7B_int8_ekv4096.litertlm", "dynamic_wi8_afp32", 4096),
}


def check_toolchain():
  """Stops early on a transformers or a litert-torch that cannot run this."""
  import transformers
  from transformers import integrations

  if not hasattr(integrations, "use_kernel_func_from_hub_with_fallback"):
    raise SystemExit(
        f"transformers {transformers.__version__} cannot load LFM3's remote "
        "code: it imports use_kernel_func_from_hub_with_fallback from "
        "transformers.integrations, which transformers has since 5.15.0. "
        "Install the version in the README.")
  try:
    from litert_torch.generative.export_hf.model_ext import extension
  except ImportError:
    extension = None
  if not hasattr(extension, "load_extension_file"):
    raise SystemExit(
        "this litert-torch has no out-of-tree extension path (a .py file in "
        "extension_modules, litert-torch commit 864a966 of 2026-10-28; no "
        "release carries it yet). Install the commit in the README.")
  print(f"toolchain: transformers {transformers.__version__}, litert-torch "
        "with out-of-tree extension files")


def export_one(export, src, work, recipe, cache, no_extension):
  """One stock export of the checkpoint into `work`."""
  export(
      model=str(src),
      output_dir=str(work),
      trust_remote_code=True,
      prefill_lengths=PREFILL,
      cache_length=cache,
      quantization_recipe=recipe,
      # Without a sampler flag the exporter writes no sampler (it turns the
      # generation config's do_sample off when it loads the model). With p = 1
      # the file declares TOP_P with the generation config's k 50 and
      # temperature 0.1, which LiteRT-LM's CPU and GPU samplers both run.
      sampler_top_p=1.0,
      # The checkpoint's Jinja is embedded and rendered on device.
      use_jinja_template=True,
      # The tokenizer section is the checkpoint's tokenizer.json as is.
      tokenizer_path_override=str(src / "tokenizer.json"),
      externalize_embedder=False,
      single_token_embedder=False,
      extension_modules=None if no_extension else str(EXTENSION),
  )


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--model", default=MODEL_ID,
                  help="HF model id (fetched at the pinned revision) or a "
                       "local copy of that revision")
  ap.add_argument("--out", default="out_lfm3", help="output dir")
  ap.add_argument("--files", default=",".join(FILES),
                  help="comma-separated names from the table above")
  ap.add_argument("--no-extension", action="store_true",
                  help="export without lfm3_extension.py (study only)")
  args = ap.parse_args()
  names = [n.strip() for n in args.files.split(",") if n.strip()]
  unknown = sorted(set(names) - set(FILES))
  if unknown:
    raise SystemExit(f"unknown --files {unknown}; choose from {list(FILES)}")

  check_toolchain()

  src = Path(args.model)
  if not src.exists():
    from huggingface_hub import snapshot_download

    src = Path(snapshot_download(args.model, revision=REVISION,
                                 allow_patterns=SOURCE_FILES))
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  from litert_torch.generative.export_hf.export import export

  if args.no_extension:
    print("extension: NONE on request (the stock export, for study only)")
  for name in names:
    final_name, recipe, cache = FILES[name]
    work = out / f"{name}.work"
    shutil.rmtree(work, ignore_errors=True)
    t0 = time.time()
    try:
      export_one(export, src, work, recipe, cache, args.no_extension)
    except BaseException:
      shutil.rmtree(work, ignore_errors=True)  # a failed export leaves nothing
      raise
    bundle = work / "model.litertlm"
    if not bundle.exists():
      raise SystemExit(f"export of {name} finished without model.litertlm")
    final = out / final_name
    bundle.replace(final)
    shutil.rmtree(work)
    print(f"EXPORT_DONE {final} ({os.path.getsize(final):,} B, "
          f"{time.time() - t0:.0f} s)")


if __name__ == "__main__":
  main()
