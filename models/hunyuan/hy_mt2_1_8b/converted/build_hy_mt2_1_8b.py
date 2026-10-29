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

"""tencent/Hy-MT2-1.8B (dense HunYuanDenseV1ForCausalLM) -> .litertlm.

Converts the Hugging Face checkpoint into a LiteRT-LM bundle through
litert-torch's stock HF exporter with the stock int8 recipe
(`dynamic_wi8_afp32`: dynamic int8 weights on linears + embedding): 1.82 GB for
1.79 B unique parameters. The architecture needs nothing — GQA with QK-norm,
RMSNorm, SiLU MLP, a tied embedding. What the checkpoint needs are metadata
fixes, and they live in `hy_mt2_extension.py`, an out-of-tree `ModelExtension`
this script hands to the exporter (`extension_modules`): no change to
litert-torch and no edit of the bundle after the export.

  * No metadata `start_token`. The chat template renders
    `<｜hy_begin▁of▁sentence｜>` itself as a LITERAL, the tokenizer adds no BOS
    at encode time, and the LiteRT-LM engine prepends the metadata
    `start_token` to every rendered prompt — so the default export feeds the
    model two BOS tokens. The extension's `build_llm_metadata` hook drops the
    field (and keeps id 120020, the tokenizer's eos, among the stop ids).
    `verify_hy_mt2_1_8b.py --bos-discriminator` proves the prepend inside the
    runtime.

  * The chat template reads content parts. LiteRT-LM 0.18 and later hand it
    each message's content as a list of parts; the same hook reads the
    vendor template's content sites through a `format_content` macro, as the
    published bundle's template does.

  * The rope needs no change on transformers 5.20.1 and later. The config
    declares `rope_scaling: {type: dynamic, alpha: 1000}`; from that version
    transformers' HunYuan rotary embedding maps a `dynamic` type with a fixed
    `alpha` to a static `ntk_alpha` type, so `torch.export` no longer meets the
    data-dependent growth guard that failed on 5.14
    (GuardOnDataDependentSymNode: Could not guard on Eq(u0, 1)). The script
    stops early on a transformers that still has the dynamic branch.
    `verify_hy_mt2_1_8b.py --rope-ab` shows the vendor config and the baked
    config the published bundle was built from give bitwise-equal rope and
    logits.

Usage:
  python build_hy_mt2_1_8b.py --out out_int8
  python build_hy_mt2_1_8b.py --out out_int8_bos --keep-start-token   # study
"""

import argparse
import os
from pathlib import Path

MODEL_ID = "tencent/Hy-MT2-1.8B"
FINAL_NAME = "Hy-MT2-1.8B_int8.litertlm"
EXTENSION = Path(__file__).resolve().parent / "hy_mt2_extension.py"


def check_rope(src):
  """Stops early when transformers still traces the dynamic rope branch."""
  from transformers import AutoConfig
  from transformers.models.hunyuan_v1_dense import modeling_hunyuan_v1_dense

  rope = modeling_hunyuan_v1_dense.HunYuanDenseV1RotaryEmbedding(
      config=AutoConfig.from_pretrained(str(src)))
  if rope.rope_type != "ntk_alpha":
    raise SystemExit(
        f"transformers resolves this rope as {rope.rope_type!r}, not "
        "'ntk_alpha': torch.export would fail on the dynamic-rope guard. "
        "Install the transformers version in the README (5.20.1 or later).")
  print(f"rope: dynamic + alpha resolves to {rope.rope_type!r} (static); no bake")


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--model", default=MODEL_ID,
                  help="HF model id or a local checkout of it")
  ap.add_argument("--out", default="out_int8", help="output dir")
  ap.add_argument("--cache", type=int, default=4096, help="KV cache length")
  ap.add_argument("--prefill", default="128",
                  help="comma-separated prefill signature lengths (the "
                       "published bundle carries the exporter's stock single "
                       "128-token signature)")
  ap.add_argument("--keep-start-token", action="store_true",
                  help="export without hy_mt2_extension.py: reproduces the "
                       "default double-BOS export for --bos-discriminator")
  args = ap.parse_args()

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  from huggingface_hub import snapshot_download

  src = Path(args.model)
  if not src.exists():
    src = Path(snapshot_download(args.model,
                                 ignore_patterns=["train/*", "imgs/*", "*.png"]))
  check_rope(src)

  from litert_torch.generative.export_hf.export import export

  if args.keep_start_token:
    print("start_token: KEPT on request (no extension; double-BOS bundle, "
          "for study only)")
  export(
      model=str(src),
      output_dir=str(out),
      prefill_lengths=[int(x) for x in args.prefill.split(",") if x.strip()],
      cache_length=args.cache,
      # The exporter's stock int8: dynamic int8 weights on linears + embedding,
      # fp32 activations.
      quantization_recipe="dynamic_wi8_afp32",
      # The vendor Jinja is embedded verbatim and rendered on device.
      use_jinja_template=True,
      externalize_embedder=False,
      extension_modules=None if args.keep_start_token else str(EXTENSION),
  )
  bundle = out / "model.litertlm"
  if not bundle.exists():
    raise SystemExit("export finished without producing model.litertlm")
  final = out / FINAL_NAME
  bundle.replace(final)
  print(f"EXPORT_DONE {final} ({os.path.getsize(final):,} B)")


if __name__ == "__main__":
  main()
