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

"""mistralai/Ministral-3-3B-Instruct-2512 (text decoder) -> .litertlm.

Converts the Ministral-3 text decoder (extracted from the multimodal
checkpoint by extract_text_decoder.py) into a LiteRT-LM bundle through
litert-torch's plain HF export path: blockwise-32 OCTAV int4 weights with the
tied vocab embedding kept at int8 and split into its own section (~2.34 GB).
`Ministral3ForCausalLM` (26 layers, hidden 3072, GQA 32:8, YaRN RoPE) needs
no re-authoring.

Three model-specific facts are baked in:

  * Template. Ministral speaks Mistral's `[INST] ... [/INST]` format with
    `</s>` as its end-of-turn token; the tekken tokenizer has NO `<|im_end|>`.
    Exported under a ChatML template the int4 model never reaches a registered
    stop token and runs on past the correct answer with `<|im_start|>` spam;
    under its native template it stops cleanly. This script forces the plain
    Mistral template (upstream's full jinja cannot run in the runtime's
    minimal jinja engine, so the structured [INST] prefixes are extracted).

  * start_token. The tokenizer declares bos `<s>` and eos `</s>` — different
    tokens — so the `start_token { token_str: "<s>" }` the builder writes is
    the Mistral convention, not the granite-4.1-3b trap (which needs
    bos == eos, or add_bos_token: False). It is kept.
    `verify_ministral_3_3b.py --bos-ab` shows the bf16 model answers with the
    leading <s>.

  * Section size. Without externalize_embedder the 3B's weights are one
    ~2.55 GiB TFLite section, above the iOS ~2 GiB single-section mmap ceiling
    (engine creation fails: "Failed to map section: Cannot allocate memory").
    Externalizing the tied embedding drops the main section to ~1.8 GiB, and
    the bundle loads on iPhone.

Usage:
  python extract_text_decoder.py --out ministral3_text     # once, ~7 GB download
  python build_ministral_3_3b.py --model ministral3_text --out out_int4
"""

import argparse
import copy
import os

# Mistral's plain chat format. Byte-identical to upstream's rendering of
# system / user / assistant turns; what it drops is upstream's tool-calling
# and default-system logic, which the structured template cannot carry.
MISTRAL_TEMPLATE = r"""{%- for message in messages -%}
{%- if message['role'] == 'system' -%}
{{- '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' -}}
{%- elif message['role'] == 'user' -%}
{{- '[INST]' + message['content'] + '[/INST]' -}}
{%- elif message['role'] == 'assistant' -%}
{{- message['content'] + '</s>' -}}
{%- endif -%}
{%- endfor -%}"""


def patch_tokenizer():
  """Forces the plain Mistral template on every tokenizer the exporter loads."""
  import transformers

  orig_from_pretrained = transformers.AutoTokenizer.from_pretrained

  def patched_from_pretrained(*args, **kwargs):
    tok = orig_from_pretrained(*args, **kwargs)
    tok.chat_template = MISTRAL_TEMPLATE
    # bos <s> != eos </s>: the start_token the builder writes from bos_token is
    # legitimate for this model and is deliberately left in place.
    assert tok.bos_token == "<s>" and tok.eos_token == "</s>", (
        f"unexpected bos/eos {tok.bos_token!r}/{tok.eos_token!r} — re-check "
        "the start_token story before exporting")
    return tok

  transformers.AutoTokenizer.from_pretrained = patched_from_pretrained


def register_int4_recipe():
  """blockwise-32 OCTAV int4 weights + int8 tied embedding (131072x3072).

  Measured GSM8K 85.0% against a bf16 89.0% (n=100); a channelwise min-max
  int4 of the same model passed the sanity gate but was not at parity — the
  blockwise-32 + OCTAV grid is what preserves the accuracy.
  """
  import ai_edge_quantizer.recipe as recipe_lib

  int4_rule = copy.deepcopy(recipe_lib.dynamic_wi4_afp32()[0])
  int4_rule["algorithm_key"] = recipe_lib.AlgorithmName.OCTAV
  int4_rule["op_config"]["weight_tensor_config"]["granularity"] = "BLOCKWISE_32"

  emb_rule = copy.deepcopy(recipe_lib.dynamic_wi4_afp32()[0])
  emb_rule["operation"] = "EMBEDDING_LOOKUP"
  emb_rule["op_config"]["weight_tensor_config"]["num_bits"] = 8

  recipe_lib.MINISTRAL3_INT4 = lambda: [int4_rule, emb_rule]
  return "MINISTRAL3_INT4"


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--model", default="ministral3_text",
                  help="the standalone text decoder written by "
                       "extract_text_decoder.py (NOT the multimodal HF repo)")
  ap.add_argument("--out", default="out_int4", help="output dir")
  ap.add_argument("--cache", type=int, default=4096, help="KV cache length")
  ap.add_argument("--prefill", default="128",
                  help="comma-separated prefill signature ladder. The published "
                       "bundle carries the single 128 signature; see README.md "
                       "for the ladder trade-off.")
  args = ap.parse_args()

  if not os.path.exists(os.path.join(args.model, "config.json")):
    raise SystemExit(f"{args.model}/config.json not found — run "
                     "extract_text_decoder.py first (the multimodal repo "
                     "cannot be exported directly).")

  patch_tokenizer()
  quant = register_int4_recipe()

  from litert_torch.generative.export_hf.export import export

  export(
      model=args.model,
      output_dir=args.out,
      prefill_lengths=[int(x) for x in args.prefill.split(",") if x.strip()],
      cache_length=args.cache,
      quantization_recipe=quant,
      # False -> structured [INST] prompt templates in the bundle, not raw jinja.
      use_jinja_template=False,
      # Required for iPhone: keeps the main weights section under the iOS
      # ~2 GiB single-section mmap ceiling (see module doc).
      externalize_embedder=True,
      trust_remote_code=True,
  )
  print("EXPORT_DONE")


if __name__ == "__main__":
  main()
