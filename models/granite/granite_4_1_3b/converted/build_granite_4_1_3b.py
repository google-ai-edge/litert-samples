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

"""ibm-granite/granite-4.1-3b (dense GraniteForCausalLM) -> .litertlm.

Converts the Hugging Face checkpoint into a LiteRT-LM bundle through
litert-torch's plain HF export path. Two recipes:

  int4  blockwise-32 OCTAV int4 weights + int8 embedding  ~2.19 GB  (ship: phone GPU)
  int8  dynamic int8 weights, fp32 activations            ~3.83 GB  (quality reference)

The one model-specific step is the start_token fix. granite's tokenizer declares
`add_bos_token: False`, and its BOS is the same token as its EOS
(`<|end_of_text|>`, id 100257). The bundle builder writes `start_token` from
`tokenizer.bos_token` unconditionally — it never consults `add_bos_token` — and
the LiteRT-LM runtime prepends that token on the first turn. The model then reads
every prompt as a document that has already ended, and echoes the question back
instead of answering (quality gate 5/8 -> 8/8 with the fix; reproduced on bf16
PyTorch, so it is a prompt bug, not quantization). This script clears
`tokenizer.bos_token` before export so the bundle carries no start_token at all.
See README.md for the measurements, and strip_start_token.py to repair a bundle
that is already built.

Usage:
  python build_granite_4_1_3b.py --recipe int4 --out out_int4
  python build_granite_4_1_3b.py --recipe int8 --out out_int8
"""

import argparse
import copy

# granite-4.1's chat template, reduced to the plain-chat form. Verified
# byte-identical to the upstream chat_template.jinja rendering for single- and
# multi-turn chat; the upstream file's tools/documents/strftime branches are what
# this drops. Exporting with the simple template lets the exporter extract the
# STRUCTURED per-role prompt templates the runtime applies directly — embedding
# the raw upstream jinja instead would leave rendering to the runtime's minimal
# jinja engine, which cannot run that logic.
GRANITE_TEMPLATE = r"""{%- for message in messages -%}
{{- '<|start_of_role|>' + message['role'] + '<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|start_of_role|>assistant<|end_of_role|>' -}}
{%- endif -%}"""


def patch_tokenizer(keep_start_token):
  """Forces the simple chat template and (by default) clears bos_token.

  Clearing `tokenizer.bos_token` is the start_token fix: the metadata builder
  sets `start_token` from it unconditionally, so the only way to keep the field
  out of the bundle at export time is for the tokenizer not to have one.
  """
  import transformers

  orig_from_pretrained = transformers.AutoTokenizer.from_pretrained

  def patched_from_pretrained(*args, **kwargs):
    tok = orig_from_pretrained(*args, **kwargs)
    tok.chat_template = GRANITE_TEMPLATE
    if not keep_start_token:
      tok.bos_token = None
      print("start_token fix: bos_token cleared -> bundle will carry no start_token")
    return tok

  transformers.AutoTokenizer.from_pretrained = patched_from_pretrained


def register_int4_recipe():
  """Registers the int4 recipe under a name the exporter can look up.

  blockwise-32 OCTAV int4 for the weights (data-free optimal clipping;
  channelwise min-max int4 degrades LLMs measurably more), with the tied
  100352x2560 vocab embedding kept at int8 (EMBEDDING_LOOKUP is where naive
  int4 hurts most).
  """
  import ai_edge_quantizer.recipe as recipe_lib

  int4_rule = copy.deepcopy(recipe_lib.dynamic_wi4_afp32()[0])
  int4_rule["algorithm_key"] = recipe_lib.AlgorithmName.OCTAV
  int4_rule["op_config"]["weight_tensor_config"]["granularity"] = "BLOCKWISE_32"

  emb_rule = copy.deepcopy(recipe_lib.dynamic_wi4_afp32()[0])
  emb_rule["operation"] = "EMBEDDING_LOOKUP"
  emb_rule["op_config"]["weight_tensor_config"]["num_bits"] = 8

  recipe_lib.GRANITE_INT4 = lambda: [int4_rule, emb_rule]
  return "GRANITE_INT4"


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--recipe", choices=["int4", "int8"], default="int4")
  ap.add_argument("--model", default="ibm-granite/granite-4.1-3b",
                  help="HF model id or a local checkout of it")
  ap.add_argument("--out", default=None, help="output dir (default: out_<recipe>)")
  ap.add_argument("--cache", type=int, default=4096, help="KV cache length")
  ap.add_argument("--prefill", default="1024,256,64,16,4,1",
                  help="comma-separated prefill signature ladder. The 6-rung "
                       "default is the ship ladder: each extra signature costs "
                       "engine memory at init, and an 11-rung ladder puts the "
                       "iPhone 17 Pro over the jetsam line at the bundle's "
                       "default context (see README.md)")
  ap.add_argument("--keep-start-token", action="store_true",
                  help="skip the bos_token fix, reproducing the start_token trap "
                       "(for study only — the resulting bundle echoes prompts)")
  args = ap.parse_args()
  out_dir = args.out or f"out_{args.recipe}"

  patch_tokenizer(args.keep_start_token)

  if args.recipe == "int4":
    quant = register_int4_recipe()
  else:
    quant = "dynamic_wi8_afp32"

  from litert_torch.generative.export_hf.export import export

  export(
      model=args.model,
      output_dir=out_dir,
      prefill_lengths=[int(x) for x in args.prefill.split(",") if x.strip()],
      cache_length=args.cache,
      quantization_recipe=quant,
      # False -> the exporter extracts structured prompt templates from the
      # (simple) chat template instead of embedding raw jinja in the bundle.
      use_jinja_template=False,
      # Splits the tied vocab embedding into its own bundle section, keeping the
      # main weights section under the iOS ~2 GiB single-section mmap ceiling.
      # No effect on weights or parity.
      externalize_embedder=True,
      trust_remote_code=True,
  )
  print("EXPORT_DONE")


if __name__ == "__main__":
  main()
