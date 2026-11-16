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

"""WeiboAI/VibeThinker-3B (dense Qwen2ForCausalLM, math/reasoning) -> .litertlm.

Converts the Hugging Face checkpoint into a LiteRT-LM bundle through
litert-torch's plain HF export path: blockwise-32 OCTAV int4 weights with the
tied vocab embedding kept at int8 and split into its own section (~2.06 GB).
Standard Qwen2 architecture — nothing to re-author.

Two model-specific facts are baked in:

  * Stop tokens. VibeThinker's generation_config declares only <|endoftext|>
    (151643) as EOS, but under ChatML the model ends its turn with <|im_end|>
    (151645). The bundle builder derives the stop set from generation_config,
    so without a fix the runtime has no id-level stop for <|im_end|> and can
    keep generating past the answer. This script adds 151645 to the model's
    eos_token_id before the metadata is built (the published bundle carries
    both ids).

  * Block size. This is a precision-sensitive math model: block-32 int4 holds
    GSM8K at 90.0% (bf16 97.0%), while the coarser block-128 int4 collapses
    to 64.0% (n=100, greedy, 2048-token budget). Only block-32 is offered.
    General-purpose 4B reasoning models show the opposite (block-128 is fine
    and faster); exact arithmetic needs the finer grid.

Usage:
  python build_vibethinker_3b.py --out out_int4
"""

import argparse
import copy

# Bare ChatML — the published bundle's template. Upstream's Qwen2 template
# also inserts "You are a helpful assistant." as a default system turn; the
# bundle drops that line (measured with it dropped: GSM8K 90.0%, 8Q 8/8). To
# bake it in, put the system turn into the user prefix as the Qwen2.5-Coder
# recipe in this repo does.
CHATML_TEMPLATE = r"""{%- for message in messages -%}
{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|im_start|>assistant\n' -}}
{%- endif -%}"""

IM_END = 151645        # <|im_end|>
END_OF_TEXT = 151643   # <|endoftext|>


def patch_tokenizer():
  """Forces the bare ChatML template on every tokenizer the exporter loads."""
  import transformers

  orig_from_pretrained = transformers.AutoTokenizer.from_pretrained

  def patched_from_pretrained(*args, **kwargs):
    tok = orig_from_pretrained(*args, **kwargs)
    tok.chat_template = CHATML_TEMPLATE
    # No bos_token on this tokenizer -> no start_token in the bundle.
    assert not tok.bos_token, "unexpected bos_token — re-check the start_token story"
    return tok

  transformers.AutoTokenizer.from_pretrained = patched_from_pretrained


def patch_stop_tokens():
  """Adds <|im_end|> to generation_config.eos_token_id after the model loads.

  The bundle builder reads `model.generation_config.eos_token_id` (int or
  list) into the LlmMetadata stop set. VibeThinker ships eos_token_id=151643
  only; the ChatML turn terminator is 151645.
  """
  import transformers

  orig_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

  def patched_from_pretrained(*args, **kwargs):
    model = orig_from_pretrained(*args, **kwargs)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    if IM_END not in eos:
      eos.append(IM_END)
    if END_OF_TEXT not in eos:
      eos.append(END_OF_TEXT)
    model.generation_config.eos_token_id = eos
    print(f"stop-token fix: generation_config.eos_token_id = {eos}")
    return model

  transformers.AutoModelForCausalLM.from_pretrained = patched_from_pretrained


def register_int4_recipe():
  """blockwise-32 OCTAV int4 weights + int8 tied embedding (151936x2048).

  Block-32 only, on purpose: block-128 collapsed this model's GSM8K from 90.0%
  to 64.0% (see module doc / README.md).
  """
  import ai_edge_quantizer.recipe as recipe_lib

  int4_rule = copy.deepcopy(recipe_lib.dynamic_wi4_afp32()[0])
  int4_rule["algorithm_key"] = recipe_lib.AlgorithmName.OCTAV
  int4_rule["op_config"]["weight_tensor_config"]["granularity"] = "BLOCKWISE_32"

  emb_rule = copy.deepcopy(recipe_lib.dynamic_wi4_afp32()[0])
  emb_rule["operation"] = "EMBEDDING_LOOKUP"
  emb_rule["op_config"]["weight_tensor_config"]["num_bits"] = 8

  recipe_lib.VIBETHINKER_INT4 = lambda: [int4_rule, emb_rule]
  return "VIBETHINKER_INT4"


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--model", default="WeiboAI/VibeThinker-3B",
                  help="HF model id or a local checkout of it")
  ap.add_argument("--out", default="out_int4", help="output dir")
  ap.add_argument("--cache", type=int, default=4096,
                  help="KV cache length. Keep >= 4096: the model reasons at "
                       "length before its \\boxed{} answer.")
  ap.add_argument("--prefill", default="128",
                  help="comma-separated prefill signature ladder. The published "
                       "bundle carries the single 128 signature; see README.md "
                       "for the ladder trade-off.")
  args = ap.parse_args()

  patch_tokenizer()
  patch_stop_tokens()
  quant = register_int4_recipe()

  from litert_torch.generative.export_hf.export import export

  export(
      model=args.model,
      output_dir=args.out,
      prefill_lengths=[int(x) for x in args.prefill.split(",") if x.strip()],
      cache_length=args.cache,
      quantization_recipe=quant,
      # False -> structured per-role prompt templates in the bundle, not raw
      # jinja (the runtime's minimal jinja engine cannot run upstream's
      # tool-calling template).
      use_jinja_template=False,
      # Tied embedding into its own section: main weights section 1.62 GiB,
      # under the iOS ~2 GiB single-section mmap ceiling. No effect on parity.
      externalize_embedder=True,
      trust_remote_code=True,
  )
  print("EXPORT_DONE")


if __name__ == "__main__":
  main()
