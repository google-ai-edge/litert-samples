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

"""Qwen/Qwen2.5-Coder-1.5B-Instruct (dense Qwen2ForCausalLM) -> .litertlm.

Converts the Hugging Face checkpoint into a LiteRT-LM bundle through
litert-torch's plain HF export path: blockwise-32 OCTAV int4 weights with the
tied vocab embedding kept at int8 and split into its own section — 1.12 GB
for 1.54 B parameters. Standard Qwen2 architecture, nothing to re-author.

Two model-specific facts are baked in:

  * externalize_embedder=True is required, not cosmetic. Qwen2.5-Coder ties
    its embedding and lm_head, so a recipe asking for int4 linears and an int8
    embedder describes one tensor two ways; the quantizer resolves it by
    storing the 151936x1536 table once per prefill signature. With the
    default six signatures that is seven int8 copies, 1.63 GB, 65% of a
    2.53 GB file — and no error anywhere. Externalizing the table removes the
    conflict: same weights, 1.12 GB, zero duplicated tensors.

  * The vendor's default system prompt is baked into the user prefix.
    Upstream's chat template inserts "You are Qwen, created by Alibaba Cloud.
    You are a helpful assistant." whenever the caller sends no system message.
    The runtime applies structured per-role prefixes and has no place for a
    conditional default, so a plain ChatML template would put the model in a
    state it was not tuned in on every default request. Carrying the system
    turn inside the user prefix renders byte-identical to upstream for a
    single-turn request. (A caller who sends an explicit system message gets
    it as its own turn AND the default in the user prefix — the structured
    template cannot express "only if absent".)

Usage:
  python build_qwen2_5_coder_1_5b.py --out out_int4
"""

import argparse
import copy

# ChatML with Qwen's default system turn folded into the user prefix.
CODER_TEMPLATE = r"""{%- for message in messages -%}
{%- if message['role'] == 'system' -%}
{{- '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' -}}
{%- elif message['role'] == 'user' -%}
{{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n' + message['content'] + '<|im_end|>\n' -}}
{%- elif message['role'] == 'assistant' -%}
{{- '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' -}}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|im_start|>assistant\n' -}}
{%- endif -%}"""


def patch_tokenizer():
  """Forces the coder template on every tokenizer the exporter loads."""
  import transformers

  orig_from_pretrained = transformers.AutoTokenizer.from_pretrained

  def patched_from_pretrained(*args, **kwargs):
    tok = orig_from_pretrained(*args, **kwargs)
    tok.chat_template = CODER_TEMPLATE
    # No bos_token -> no start_token in the bundle. (generation_config already
    # lists both <|im_end|> and <|endoftext|> as EOS, so both become stops.)
    assert not tok.bos_token, "unexpected bos_token — re-check the start_token story"
    return tok

  transformers.AutoTokenizer.from_pretrained = patched_from_pretrained


def register_int4_recipe():
  """blockwise-32 OCTAV int4 weights + int8 tied embedding (151936x1536)."""
  import ai_edge_quantizer.recipe as recipe_lib

  int4_rule = copy.deepcopy(recipe_lib.dynamic_wi4_afp32()[0])
  int4_rule["algorithm_key"] = recipe_lib.AlgorithmName.OCTAV
  int4_rule["op_config"]["weight_tensor_config"]["granularity"] = "BLOCKWISE_32"

  emb_rule = copy.deepcopy(recipe_lib.dynamic_wi4_afp32()[0])
  emb_rule["operation"] = "EMBEDDING_LOOKUP"
  emb_rule["op_config"]["weight_tensor_config"]["num_bits"] = 8

  recipe_lib.QWEN25_CODER_INT4 = lambda: [int4_rule, emb_rule]
  return "QWEN25_CODER_INT4"


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                  help="HF model id or a local checkout of it")
  ap.add_argument("--out", default="out_int4", help="output dir")
  ap.add_argument("--cache", type=int, default=4096, help="KV cache length")
  ap.add_argument("--prefill", default="1024,256,64,16,4,1",
                  help="comma-separated prefill signature ladder (the "
                       "published bundle carries these six)")
  ap.add_argument("--inline-embedder", action="store_true",
                  help="skip externalize_embedder, reproducing the 2.53 GB "
                       "duplicated-table bundle (for study only)")
  args = ap.parse_args()

  patch_tokenizer()
  quant = register_int4_recipe()

  from litert_torch.generative.export_hf.export import export

  export(
      model=args.model,
      output_dir=args.out,
      prefill_lengths=[int(x) for x in args.prefill.split(",") if x.strip()],
      cache_length=args.cache,
      quantization_recipe=quant,
      # False -> structured per-role prompt templates in the bundle, not raw
      # jinja (upstream's tool-calling template cannot run in the runtime's
      # minimal jinja engine).
      use_jinja_template=False,
      externalize_embedder=not args.inline_embedder,
      trust_remote_code=True,
  )
  print("EXPORT_DONE")


if __name__ == "__main__":
  main()
