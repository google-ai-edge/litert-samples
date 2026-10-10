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

"""Extracts the text decoder of Ministral-3-3B into a standalone checkpoint.

`mistralai/Ministral-3-3B-Instruct-2512` is published as
`Mistral3ForConditionalGeneration` — a Pixtral vision tower, a projector and
the Ministral3 text decoder under one config. litert-torch's text export path
wants an `AutoModelForCausalLM`-loadable checkpoint, so this script loads the
multimodal wrapper, copies the language model and lm_head into a plain
`Ministral3ForCausalLM`, and saves it with the tokenizer:

  Mistral3ForConditionalGeneration
    .model.vision_tower           <- dropped
    .model.multi_modal_projector  <- dropped
    .model.language_model         -> causal.model
    .lm_head                      -> causal.lm_head  (tied to embed_tokens)

Source repo: the plain repo ships FP8 weights (config.quantization_config
quant_method=fp8), which the CPU export path cannot load; use the BF16 repo,
`mistralai/Ministral-3-3B-Instruct-2512-BF16`. Its `consolidated.safetensors`
(7.7 GB) is a second, single-file copy of the same weights that the HF loader
does not read — the download here skips it.

Usage:
  python extract_text_decoder.py --out ministral3_text
"""

import argparse
import gc
import os

DEFAULT_SRC = "mistralai/Ministral-3-3B-Instruct-2512-BF16"


def download(repo, local_dir):
  """Fetches the BF16 checkpoint without the redundant consolidated file."""
  from huggingface_hub import snapshot_download

  return snapshot_download(
      repo, local_dir=local_dir,
      ignore_patterns=["consolidated.safetensors", "*.md", "*.png", "*.jpg"])


def extract(src, out):
  import torch
  import transformers

  print(f"loading multimodal wrapper from {src} (bf16, cpu) ...")
  full = transformers.AutoModelForImageTextToText.from_pretrained(
      src, dtype=torch.bfloat16, low_cpu_mem_usage=True)
  full.eval()

  text_cfg = full.config.text_config
  print(f"text_config: model_type={text_cfg.model_type} "
        f"layers={text_cfg.num_hidden_layers} hidden={text_cfg.hidden_size} "
        f"vocab={text_cfg.vocab_size} tie={text_cfg.tie_word_embeddings}")

  causal = transformers.Ministral3ForCausalLM(text_cfg)
  causal.eval()

  # language_model (Ministral3Model) -> causal.model (Ministral3Model): the
  # key sets are identical, so this must load with nothing missing or extra.
  missing, unexpected = causal.model.load_state_dict(
      full.model.language_model.state_dict(), strict=False)
  print(f"decoder load: missing={len(missing)} unexpected={len(unexpected)}")
  if missing or unexpected:
    raise SystemExit(f"ABORT: decoder remap incomplete — missing[:5]="
                     f"{missing[:5]} unexpected[:5]={unexpected[:5]}")
  causal.lm_head.load_state_dict(full.lm_head.state_dict())
  causal.tie_weights()
  causal = causal.to(torch.bfloat16)  # keep the saved checkpoint at 6.8 GB, not fp32

  w = causal.model.embed_tokens.weight
  if not torch.isfinite(w).all():
    raise SystemExit("ABORT: non-finite values in embed_tokens")
  print(f"embed_tokens {tuple(w.shape)} ok")

  del full
  gc.collect()

  print(f"saving text-only checkpoint to {out} ...")
  causal.config._name_or_path = src  # pylint: disable=protected-access
  causal.save_pretrained(out, safe_serialization=True)
  transformers.AutoTokenizer.from_pretrained(src).save_pretrained(out)
  print("EXTRACT_DONE", out)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--src", default=DEFAULT_SRC,
                  help="HF id of the BF16 checkpoint, or a local checkout")
  ap.add_argument("--download-dir", default="ministral3_bf16",
                  help="where to place the HF download when --src is an id")
  ap.add_argument("--out", default="ministral3_text",
                  help="output dir for the standalone text decoder")
  args = ap.parse_args()

  src = args.src
  if not os.path.isdir(src):
    src = download(src, args.download_dir)
  extract(src, args.out)


if __name__ == "__main__":
  main()
