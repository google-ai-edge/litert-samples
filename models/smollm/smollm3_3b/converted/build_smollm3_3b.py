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

"""HuggingFaceTB/SmolLM3-3B (dense SmolLM3ForCausalLM) -> .litertlm.

Converts the Hugging Face checkpoint into a LiteRT-LM bundle through
litert-torch's plain HF export path: blockwise-32 OCTAV int4 weights with the
tied vocab embedding kept at int8 and split into its own bundle section
(~2.0 GB). SmolLM3's NoPE schedule (rotary disabled on every 4th layer,
`no_rope_layer_interval=4`) lowers to generic ops — no re-authoring.

The one thing to know before running this: SmolLM3's official chat template
carries its reasoning-mode system prompt (`Reasoning Mode: /think` plus a long
instruction block) inside an `if messages[0]['role'] != 'system'` branch. The
exporter extracts STRUCTURED per-role prefixes/suffixes from the template by
rendering sample conversations that always start with a system turn, so that
branch never fires and nothing conditional survives into the bundle. The
bundle is therefore bare ChatML (`<|im_start|>role\n ... <|im_end|>\n`), and
SmolLM3 answers in direct mode by default (it closes `<think></think>` at
once). A caller who wants the reasoning mode supplies SmolLM3's /think system
prompt as a system message — the system-role prefix is in the bundle. See
README.md and `verify_smollm3_3b.py --think-ab`.

Usage:
  python build_smollm3_3b.py --out out_int4
"""

import argparse
import copy

# Bare ChatML — what the exporter can carry, and what the published bundle
# uses. Byte-identical to the upstream template's per-role markers; what it
# drops is upstream's conditional system block (date, reasoning mode,
# instructions), which cannot ride the structured template (see module doc).
CHATML_TEMPLATE = r"""{%- for message in messages -%}
{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|im_start|>assistant\n' -}}
{%- endif -%}"""


def patch_tokenizer():
  """Forces the bare ChatML template on every tokenizer the exporter loads."""
  import transformers

  orig_from_pretrained = transformers.AutoTokenizer.from_pretrained

  def patched_from_pretrained(*args, **kwargs):
    tok = orig_from_pretrained(*args, **kwargs)
    tok.chat_template = CHATML_TEMPLATE
    # SmolLM3 has no bos_token, so the bundle carries no start_token and the
    # runtime prepends nothing. (Checkpoints that DO declare one while saying
    # add_bos_token: False need it cleared here — see the granite-4.1-3b recipe.)
    assert not tok.bos_token, "unexpected bos_token — re-check the start_token story"
    return tok

  transformers.AutoTokenizer.from_pretrained = patched_from_pretrained


def register_int4_recipe():
  """Registers the int4 recipe under a name the exporter can look up.

  blockwise-32 OCTAV int4 for the weights (data-free optimal clipping;
  channelwise min-max int4 degrades LLMs measurably more) with the tied
  128256x2048 vocab embedding kept at int8 (EMBEDDING_LOOKUP is where naive
  int4 hurts most). This is the recipe that measured GSM8K 81.0% against a
  bf16 81.0% (n=100) — see README.md.
  """
  import ai_edge_quantizer.recipe as recipe_lib

  int4_rule = copy.deepcopy(recipe_lib.dynamic_wi4_afp32()[0])
  int4_rule["algorithm_key"] = recipe_lib.AlgorithmName.OCTAV
  int4_rule["op_config"]["weight_tensor_config"]["granularity"] = "BLOCKWISE_32"

  emb_rule = copy.deepcopy(recipe_lib.dynamic_wi4_afp32()[0])
  emb_rule["operation"] = "EMBEDDING_LOOKUP"
  emb_rule["op_config"]["weight_tensor_config"]["num_bits"] = 8

  recipe_lib.SMOLLM3_INT4 = lambda: [int4_rule, emb_rule]
  return "SMOLLM3_INT4"


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--model", default="HuggingFaceTB/SmolLM3-3B",
                  help="HF model id or a local checkout of it")
  ap.add_argument("--out", default="out_int4", help="output dir")
  ap.add_argument("--cache", type=int, default=4096, help="KV cache length")
  ap.add_argument("--prefill", default="128",
                  help="comma-separated prefill signature ladder. The published "
                       "bundle carries the single 128 signature (smallest "
                       "engine-init memory; long prompts are prefilled in "
                       "128-token chunks). '1024,256,64,16,4,1' trades init "
                       "memory for TTFT on long prompts — see README.md.")
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
      # False -> the exporter extracts structured prompt templates from the
      # (bare ChatML) chat template instead of embedding raw jinja in the
      # bundle. The runtime's minimal jinja engine cannot run upstream's
      # template (strftime_now, tool blocks), so this is not optional.
      use_jinja_template=False,
      # Splits the tied vocab embedding into its own bundle section, keeping the
      # main weights section (1.61 GiB) under the iOS ~2 GiB single-section
      # mmap ceiling. No effect on weights or parity.
      externalize_embedder=True,
      trust_remote_code=True,
  )
  print("EXPORT_DONE")


if __name__ == "__main__":
  main()
