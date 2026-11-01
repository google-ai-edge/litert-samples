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

"""LiquidAI/LFM3-VL-2B (Lfm3VlForConditionalGeneration, remote code) -> .litertlm.

Converts the Hugging Face checkpoint into LFM3-VL-2B_int4.litertlm of
litert-community/LFM3-VL-2B-LiteRT through litert-torch's stock HF exporter,
one `export()` call with task `image_text_to_text` and two out-of-tree
`ModelExtension` files in `extension_modules`: the LFM3-1.7B recipe's
`lfm3_extension.py` (the LFM3 text decoder, unchanged) and
`lfm3_vl_extension.py` (the vision tower, the model type, the image part of
the template). No change to litert-torch, no model loader of its own, no edit
of a bundle after the export.

  text decoder    recipe_int4.json: int4 blockwise-32 on every fully connected
                  weight, int8 per channel on the tied output projection
  token embedder  int8 per channel (the same recipe; the exporter keeps it as
                  its own graph for the image path)
  vision encoder  float32 weights (recipe_vision.json selects no encoder op)
  projector       float16 weights (recipe_vision.json)

Prefill signatures 1024 / 256 / 64 / 16 / 4 / 1 over a 2,048-token cache, the
checkpoint's tokenizer.json byte for byte, the chat template as the two
extensions leave it, model type fast_vlm with a 512 x 512 image, no sampler.

Two things no extension hook reaches stay in this script, each scoped to the
export call:

* litert-torch loads the image processor with `AutoImageProcessor` and no
  `trust_remote_code` (export_lib.load_model), and this checkpoint's image
  processor is remote code: transformers then asks on stdin whether to run it.
  The tower here does not use the processor, so the script answers yes for
  that one call (`_image_processor_remote_code`).
* litert-quantizer floors a min-max scale at 1e-9 only. An int4 block of
  near-zero weights then gets a float16 block scale below 2^-14 (subnormal or
  zero), and XNNPACK refuses a fully connected weight with such a scale. The
  script floors every int4 block scale at 2^-14 before the weights are
  quantized (`_int4_block_scale_floor`), as the published file was made.

The script stops before the export on a transformers, torchvision or
litert-torch that cannot run it.

Usage:
  python build_lfm3_vl_2b.py --out out_lfm3_vl
  python build_lfm3_vl_2b.py --out out_stock --no-extension
      # the stock export without the two extension files, for study
"""

import argparse
import contextlib
import os
import shutil
import time
from pathlib import Path

MODEL_ID = "LiquidAI/LFM3-VL-2B"
REVISION = "f83d2b0a2296aaa277f394990a9b8b949ddf9691"
SOURCE_FILES = [
    "config.json", "generation_config.json", "configuration_lfm3_vl.py",
    "configuration_lfm3.py", "modeling_lfm3_vl.py", "modeling_lfm3.py",
    "lfm3_conv.py", "image_processing_lfm3_vl.py", "processing_lfm3_vl.py",
    "video_processing_lfm3_vl.py", "processor_config.json",
    "model.safetensors", "tokenizer.json", "tokenizer_config.json",
    "chat_template.jinja",
]
HERE = Path(__file__).resolve().parent
EXTENSIONS = [
    HERE.parent.parent / "lfm3_1_7b" / "converted" / "lfm3_extension.py",
    HERE / "lfm3_vl_extension.py",
]
PREFILL = [1024, 256, 64, 16, 4, 1]
FILE_NAME = "LFM3-VL-2B_int4.litertlm"
CACHE = 2048
RECIPE = HERE / "recipe_int4.json"
VISION_RECIPE = HERE / "recipe_vision.json"


def check_toolchain():
  """Stops early on a toolchain that cannot run this."""
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
  try:
    import torchvision  # noqa: F401
  except ImportError:
    raise SystemExit(
        "torchvision is missing: the checkpoint's image processor imports it "
        "and litert-torch loads the image processor for this task. Install "
        "the version in the README.") from None
  missing = [str(p) for p in EXTENSIONS if not p.exists()]
  if missing:
    raise SystemExit(f"extension file not found: {missing} (the LFM3-1.7B "
                     "recipe's lfm3_extension.py sits next to this recipe)")
  print(f"toolchain: transformers {transformers.__version__}, litert-torch "
        "with out-of-tree extension files")


INT4_SCALE_FLOOR = 2.0 ** -14  # fp16's smallest normal number


@contextlib.contextmanager
def _int4_block_scale_floor(stats):
  """Floors every int4 blockwise scale at 2^-14 while the quantizer runs.

  litert-quantizer's min-max scale has a floor of 1e-9 only; the fp16 block
  scale of a block of near-zero weights then lands below fp16's normal range
  (subnormal or zero), and XNNPACK takes only normal scales for a fully
  connected weight. The floor is applied before the weights are quantized, so
  a floored block keeps its near-zero values. `stats` counts what it raised.
  """
  import numpy as np
  from litert_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as u

  original = u.tensor_zp_scale_from_min_max

  def floored(min_value, max_value, num_bits, symmetric, granularity, *args,
              **kwargs):
    zp, scale = original(min_value, max_value, num_bits, symmetric,
                         granularity, *args, **kwargs)
    if u.is_blockwise(granularity) and num_bits == 4:
      if not symmetric:
        raise ValueError("int4 floor: asymmetric blockwise int4 is not covered")
      low = scale < np.float32(INT4_SCALE_FLOOR)
      n = int(np.count_nonzero(low))
      if n:
        scale = np.where(low, np.float32(INT4_SCALE_FLOOR), scale).astype(
            scale.dtype)
        stats["tensors"] += 1
        stats["scales"] += n
    return zp, scale

  u.tensor_zp_scale_from_min_max = floored
  try:
    yield
  finally:
    u.tensor_zp_scale_from_min_max = original


@contextlib.contextmanager
def _image_processor_remote_code():
  """`AutoImageProcessor.from_pretrained` with `trust_remote_code=True`.

  litert-torch's loader calls it without the flag; transformers then asks on
  stdin. Scoped to the export call.
  """
  import transformers

  auto = transformers.AutoImageProcessor
  original = auto.__dict__.get("from_pretrained")  # The classmethod object.
  bound = auto.from_pretrained

  def from_pretrained(*args, **kwargs):
    kwargs.setdefault("trust_remote_code", True)
    return bound(*args, **kwargs)

  auto.from_pretrained = from_pretrained
  try:
    yield
  finally:
    if original is None:
      del auto.from_pretrained
    else:
      auto.from_pretrained = original


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--model", default=MODEL_ID,
                  help="HF model id (fetched at the pinned revision) or a "
                       "local copy of that revision")
  ap.add_argument("--out", default="out_lfm3_vl", help="output dir")
  ap.add_argument("--no-extension", action="store_true",
                  help="export without the two extension files (study only)")
  args = ap.parse_args()

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
  work = out / "int4.work"
  shutil.rmtree(work, ignore_errors=True)
  t0 = time.time()
  floor_stats = {"tensors": 0, "scales": 0}
  try:
    with _image_processor_remote_code(), _int4_block_scale_floor(floor_stats):
      export(
          model=str(src),
          output_dir=str(work),
          task="image_text_to_text",
          trust_remote_code=True,
          prefill_lengths=PREFILL,
          cache_length=CACHE,
          quantization_recipe=str(RECIPE),
          vision_encoder_quantization_recipe=str(VISION_RECIPE),
          # The checkpoint's Jinja is embedded and rendered on device.
          use_jinja_template=True,
          # The tokenizer section is the checkpoint's tokenizer.json as is.
          tokenizer_path_override=str(src / "tokenizer.json"),
          extension_modules=(
              None if args.no_extension else [str(p) for p in EXTENSIONS]),
      )
  except BaseException:
    shutil.rmtree(work, ignore_errors=True)  # a failed export leaves nothing
    raise
  bundle = work / "model.litertlm"
  if not bundle.exists():
    raise SystemExit("the export finished without model.litertlm")
  final = out / FILE_NAME
  bundle.replace(final)
  shutil.rmtree(work)
  print(f"int4 block scales raised to 2^-14: {floor_stats['scales']} in "
        f"{floor_stats['tensors']} weight tensors")
  print(f"EXPORT_DONE {final} ({os.path.getsize(final):,} B, "
        f"{time.time() - t0:.0f} s)")


if __name__ == "__main__":
  main()
