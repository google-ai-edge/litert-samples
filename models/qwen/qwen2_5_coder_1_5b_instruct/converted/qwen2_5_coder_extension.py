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

"""Out-of-tree litert-torch `ModelExtension` for Qwen2.5-Coder-1.5B-Instruct.

build_qwen2_5_coder_1_5b.py hands it to the stock HF exporter
(`extension_modules`), with no change to litert-torch. The same export from
the command line (SRC is a local copy of the checkpoint):

  litert-torch export_hf --model=$SRC --output_dir=out_coder \
    --extension_modules=qwen2_5_coder_extension.py

Qwen/Qwen2.5-Coder-1.5B-Instruct is a stock `Qwen2ForCausalLM`, which the
exporter traces as transformers writes it, so the extension patches no module.
It carries the recipe of the published file in two hooks, and it replaces
litert-torch's built-in `qwen2` extension (which would write LiteRT-LM's
canonical `qwen2_5` metadata and a Jinja template instead):

  * `get_export_defaults`: the export arguments. blockwise-32 OCTAV int4
    weights with the embedding at int8 (recipe_int4.json), the embedding in
    its own section, prefill signatures 1024 / 256 / 64 / 16 / 4 / 1, a
    4,096-token cache, and per-role prompt templates; and off, the exporter's
    cpu_gpu defaults that change the file (GPU composites, the rope composite,
    a 32,771-token cache with prefill 128 / 1024). Arguments a caller passes
    still win.
  * `build_llm_metadata`: the vendor's default system turn at the head of the
    user prefix (a per-role template has no "only when the caller sends no
    system message"), after checking the checkpoint's chat template still
    renders it; and the stops in a fixed order, so two builds give the same
    metadata bytes.
"""

import os

from litert_torch.generative.export_hf.model_ext import extension

_HERE = os.path.dirname(os.path.abspath(__file__))
QUANTIZATION_RECIPE = os.path.join(_HERE, 'recipe_int4.json')

# The checkpoint this recipe is for (config.json), checked before the export.
_CONFIG = {
    'hidden_size': 1536,
    'num_hidden_layers': 28,
    'num_attention_heads': 12,
    'num_key_value_heads': 2,
    'vocab_size': 151936,
    'tie_word_embeddings': True,
}

IM_END_ID = 151645  # <|im_end|>, the ChatML end of a turn
END_OF_TEXT_ID = 151643  # <|endoftext|>
DEFAULT_SYSTEM = (
    'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
)
DEFAULT_SYSTEM_TURN = f'<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n'
# What the exporter's template parser reads out of the checkpoint's template
# for each role: (prefix, suffix).
CHATML = {
    'system': ('<|im_start|>system\n', '<|im_end|>\n'),
    'user': ('<|im_start|>user\n', '<|im_end|>\n'),
    'model': ('<|im_start|>assistant\n', '<|im_end|>\n'),
}


def check_source_template(tokenizer):
  """Raises unless the checkpoint's template renders what this recipe encodes.

  The recipe bakes the vendor's default system turn into the user prefix, so
  it checks that the template still adds exactly that turn when the caller
  sends no system message, and none when the caller sends one.

  Args:
    tokenizer: The checkpoint's tokenizer, with its chat template.

  Raises:
    ValueError: The template or the turn-end token ids changed.
  """
  cases = (
      ([{'role': 'user', 'content': 'x'}],
       DEFAULT_SYSTEM_TURN + '<|im_start|>user\nx<|im_end|>\n'
       '<|im_start|>assistant\n'),
      ([{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'x'}],
       '<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nx<|im_end|>\n'
       '<|im_start|>assistant\n'),
  )
  for messages, want in cases:
    got = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    if got != want:
      raise ValueError(
          'qwen2_5_coder_extension: the chat template renders'
          f' {got!r} for {messages!r}, not {want!r}; the source template'
          ' changed, re-check the default system turn')
  for token, token_id in (('<|im_end|>', IM_END_ID),
                          ('<|endoftext|>', END_OF_TEXT_ID)):
    if tokenizer.convert_tokens_to_ids(token) != token_id:
      raise ValueError(
          f'qwen2_5_coder_extension: {token} is not token {token_id}')


def default_system_in_user_prefix(llm_metadata):
  """Puts the default system turn at the head of the user prefix (once).

  Args:
    llm_metadata: The `LlmMetadata` with the per-role templates the exporter
      read out of the checkpoint's template.

  Raises:
    ValueError: The templates are not the ChatML ones this recipe expects
      (for example a Jinja template was asked for instead).
  """
  templates = llm_metadata.prompt_templates
  for role, (prefix, suffix) in CHATML.items():
    field = getattr(templates, role)
    if role == 'user' and field.prefix == DEFAULT_SYSTEM_TURN + prefix:
      continue  # Already applied (an explicit and a registered copy).
    if (field.prefix, field.suffix) != (prefix, suffix):
      raise ValueError(
          f'qwen2_5_coder_extension: the {role} template is'
          f' {(field.prefix, field.suffix)!r}, not {(prefix, suffix)!r};'
          ' this recipe writes per-role ChatML templates'
          ' (use_jinja_template=False)')
  if templates.user.prefix == CHATML['user'][0]:
    templates.user.prefix = DEFAULT_SYSTEM_TURN + templates.user.prefix


def stops_in_fixed_order(llm_metadata):
  """Writes the stop tokens ids first, then strings, each sorted.

  The exporter collects the stops in a Python set, so their order (and the
  metadata bytes) changes from one process to the next with the string hash
  seed.

  Args:
    llm_metadata: The `LlmMetadata` whose `stop_tokens` are rewritten in place.
  """
  stops = list(llm_metadata.stop_tokens)
  ids = sorted({tuple(t.token_ids.ids) for t in stops if t.token_ids.ids})
  strings = sorted({t.token_str for t in stops if t.token_str})
  del llm_metadata.stop_tokens[:]
  for token_ids in ids:
    llm_metadata.stop_tokens.add().token_ids.ids.extend(token_ids)
  for token_str in strings:
    llm_metadata.stop_tokens.add().token_str = token_str


@extension.register_model_extension()
class Qwen25CoderExtension(extension.ModelExtension):
  """Qwen2.5-Coder-1.5B-Instruct: the published file's export arguments and
  its default system turn."""

  model_types = ('qwen2',)

  def get_export_defaults(self, config_dict, task=None, target='cpu_gpu'):
    del task
    mismatched = {k: config_dict.get(k) for k, v in _CONFIG.items()
                  if config_dict.get(k) != v}
    if mismatched:
      raise ValueError(
          'qwen2_5_coder_extension is the recipe for'
          f' Qwen2.5-Coder-1.5B-Instruct; this config.json has {mismatched}')
    if str(target or '').startswith('npu'):
      raise ValueError(
          'qwen2_5_coder_extension writes the CPU / GPU file; target'
          f' {target!r} is not this recipe')
    return extension.ExportDefaults(
        task=extension.ExportTask.TEXT_GENERATION,
        export_kwargs={
            'quantization_recipe': QUANTIZATION_RECIPE,
            'externalize_embedder': True,
            'prefill_lengths': [1024, 256, 64, 16, 4, 1],
            'cache_length': 4096,
            # max_num_tokens = the cache length (cpu_gpu would cap it at 4096
            # over a 32,771-token cache).
            'llm_metadata_max_num_tokens_override': None,
            # Per-role prompt templates, not the checkpoint's Jinja (its tool
            # branch is more than the runtime's template engine runs).
            'use_jinja_template': False,
            'apply_gpu_composites': False,
            'use_rope_composite': False,
            'externalize_weights': False,
        },
    )

  def build_llm_metadata(
      self,
      source_model_artifacts,
      export_config,
      exported_model_artifacts,
      llm_metadata,
  ):
    del export_config, exported_model_artifacts
    tokenizer = getattr(source_model_artifacts, 'tokenizer', None)
    if tokenizer is None:
      return llm_metadata  # No checkpoint to check the template against.
    check_source_template(tokenizer)
    default_system_in_user_prefix(llm_metadata)
    stops_in_fixed_order(llm_metadata)
    print('qwen2_5_coder_extension: default system turn in the user prefix;'
          f' {len(llm_metadata.stop_tokens)} stops in a fixed order')
    return llm_metadata
