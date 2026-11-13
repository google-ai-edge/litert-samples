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

"""Out-of-tree litert-torch `ModelExtension` for WeiboAI/VibeThinker-3B.

build_vibethinker_3b.py hands it to the stock HF exporter
(`extension_modules`), with no change to litert-torch. The same export from
the command line (SRC is a local copy of the checkpoint):

  litert-torch export_hf --model=$SRC --output_dir=out_vibethinker \
    --extension_modules=vibethinker_extension.py

VibeThinker-3B is a stock `Qwen2ForCausalLM` (a math and reasoning finetune),
which the exporter traces as transformers writes it, so the extension patches
no module. It carries the recipe of the published file in two hooks, and it
replaces litert-torch's built-in `qwen2` extension (which would write
LiteRT-LM's canonical `qwen2_5` metadata and the checkpoint's Jinja instead):

  * `get_export_defaults`: the export arguments. blockwise-32 OCTAV int4
    weights with the embedding at int8 (recipe_int4.json; block 128 loses
    this model's arithmetic), the embedding in its own section, one prefill
    signature of 128 tokens, a 4,096-token cache, and per-role prompt
    templates; and off, the exporter's cpu_gpu defaults that change the file
    (GPU composites, the rope composite, a 32,771-token cache with prefill
    128 / 1024). Arguments a caller passes still win.
  * `build_llm_metadata`: the stops <|im_end|> (151645) and <|endoftext|>
    (151643) — the generation config declares only 151643, while the model
    ends a ChatML turn with 151645 — and the end-of-turn text; the published
    file's Jinja template (`BUNDLE_TEMPLATE`: ChatML, the vendor's default
    system turn when the caller sends no system message, and the content-parts
    form of LiteRT-LM 0.18), after checking that it renders as the
    checkpoint's template; and the thought channel <think> ... </think>.
"""

import hashlib
import os

from litert_torch.generative.export_hf.model_ext import extension

_HERE = os.path.dirname(os.path.abspath(__file__))
QUANTIZATION_RECIPE = os.path.join(_HERE, 'recipe_int4.json')

# The checkpoint this recipe is for (config.json), checked before the export.
_CONFIG = {
    'hidden_size': 2048,
    'num_hidden_layers': 36,
    'num_attention_heads': 16,
    'num_key_value_heads': 2,
    'vocab_size': 151936,
    'tie_word_embeddings': True,
}

IM_END_ID = 151645  # <|im_end|>, the ChatML end of a turn
END_OF_TEXT_ID = 151643  # <|endoftext|>, the generation config's EOS
DEFAULT_SYSTEM_TURN = (
    '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n')
# What the exporter's template parser reads out of the checkpoint's template
# for each role: (prefix, suffix). The published file keeps them as they are.
CHATML = {
    'system': ('<|im_start|>system\n', '<|im_end|>\n'),
    'user': ('<|im_start|>user\n', '<|im_end|>\n'),
    'model': ('<|im_start|>assistant\n', '<|im_end|>\n'),
}
THOUGHT_CHANNEL = ('thought', '<think>', '</think>')

# The published file's Jinja template, byte for byte (litert-community/
# VibeThinker-3B, revision 087512d2, sha256 below): ChatML, the vendor's
# default system turn, and the content-parts form (a message's content as a
# string or as a list of {'type': 'text', 'text': ...} parts).
BUNDLE_TEMPLATE = (r'''{%- macro format_content(content) -%}
    {%- if content is string -%}
        {{- content -}}
    {%- elif content is sequence -%}
        {%- for item in content -%}
            {%- if item['type'] == 'text' -%}
                {{- item['text'] -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}
{#- litert-community/VibeThinker-3B -- the vendor's default system turn, '''
    r'''restored (2026-08-28).

    The upstream chat template emits this block whenever the caller sends no
    system message. litert_torch's parse_chat_template probes the template '''
    r'''with a
    system message already present, so the block is never rendered during '''
    r'''probing
    and never reached the bundle. It is put back verbatim below, and fires '''
    r'''once per
    conversation -- the engine renders the whole history through this '''
    r'''template on
    every turn -- and not at all when the caller supplies a system message.
-#}
{%- if messages[0].role != 'system' -%}'''
    r'''{{ "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" }}'''
    r'''{%- endif -%}{%- for message in messages -%}'''
    r'''{%- if message.role == 'user' %}<|im_start|>user
{{ format_content(message.content) }}<|im_end|>
{% endif -%}'''
    r'''{%- if message.role == 'model' or message.role == 'assistant' %}'''
    r'''<|im_start|>assistant
{{ format_content(message.content) }}<|im_end|>
{% endif -%}{%- if message.role == 'system' %}<|im_start|>system
{{ format_content(message.content) }}<|im_end|>
{% endif -%}{%- endfor -%}{%- if add_generation_prompt %}<|im_start|>assistant
{% endif -%}''')
BUNDLE_TEMPLATE_SHA256 = (
    '8a9b6a72a2f27bf693877b0d3ab81cf44d9a31e1719a81471ee3819788a5adb7')

# Conversations the bundle template must render as the checkpoint's template.
_RENDER_CASES = (
    [{'role': 'user', 'content': 'x'}],
    [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'x'}],
    [{'role': 'user', 'content': 'x'}, {'role': 'assistant', 'content': 'y'},
     {'role': 'user', 'content': 'z'}],
    [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'x'},
     {'role': 'assistant', 'content': 'y'}, {'role': 'user', 'content': 'z'}],
)


def check_source(tokenizer, generation_config):
  """Raises unless the checkpoint still renders and stops as this recipe says.

  Args:
    tokenizer: The checkpoint's tokenizer, with its chat template.
    generation_config: The checkpoint's generation config, or None.

  Raises:
    ValueError: The chat template, the turn-end token ids or the generation
      config's EOS ids changed.
  """
  if hashlib.sha256(BUNDLE_TEMPLATE.encode()).hexdigest() != (
      BUNDLE_TEMPLATE_SHA256):
    raise ValueError('vibethinker_extension: BUNDLE_TEMPLATE was edited')
  first = tokenizer.apply_chat_template(
      _RENDER_CASES[0], tokenize=False, add_generation_prompt=True)
  if not first.startswith(DEFAULT_SYSTEM_TURN):
    raise ValueError(
        'vibethinker_extension: the chat template renders'
        f' {first!r} for one user turn, without the default system turn'
        f' {DEFAULT_SYSTEM_TURN!r}; the source template changed')
  for messages in _RENDER_CASES:
    for generation in (True, False):
      source = tokenizer.apply_chat_template(
          messages, tokenize=False, add_generation_prompt=generation)
      bundle = tokenizer.apply_chat_template(
          messages, chat_template=BUNDLE_TEMPLATE, tokenize=False,
          add_generation_prompt=generation)
      if bundle != source:
        raise ValueError(
            'vibethinker_extension: the bundle template renders'
            f' {bundle!r} where the chat template renders {source!r} for'
            f' {messages!r}; the source template changed')
  for token, token_id in (('<|im_end|>', IM_END_ID),
                          ('<|endoftext|>', END_OF_TEXT_ID)):
    if tokenizer.convert_tokens_to_ids(token) != token_id:
      raise ValueError(
          f'vibethinker_extension: {token} is not token {token_id}')
  eos = getattr(generation_config, 'eos_token_id', None)
  eos = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
  if not set(eos) <= {IM_END_ID, END_OF_TEXT_ID}:
    raise ValueError(
        f'vibethinker_extension: the generation config stops on {eos}, not'
        f' only on {[END_OF_TEXT_ID, IM_END_ID]}; re-check the stops')


def set_bundle_metadata(llm_metadata):
  """Writes the published file's stops, Jinja template and thought channel.

  Args:
    llm_metadata: The `LlmMetadata` with the per-role templates the exporter
      read out of the checkpoint's template.

  Raises:
    ValueError: The per-role templates are not ChatML (for example a Jinja
      template was asked for instead).
  """
  templates = llm_metadata.prompt_templates
  for role, (prefix, suffix) in CHATML.items():
    field = getattr(templates, role)
    if (field.prefix, field.suffix) != (prefix, suffix):
      raise ValueError(
          f'vibethinker_extension: the {role} template is'
          f' {(field.prefix, field.suffix)!r}, not {(prefix, suffix)!r};'
          ' this recipe writes per-role ChatML templates'
          ' (use_jinja_template=False)')
  # The exporter derives the stops from the generation config (151643 only)
  # and adds the end-of-turn text alone and after 13 punctuation marks (for
  # SentencePiece vocabularies). The published file stops on the end-of-turn
  # text and on both ids.
  del llm_metadata.stop_tokens[:]
  llm_metadata.stop_tokens.add().token_str = CHATML['model'][1]
  for token_id in (END_OF_TEXT_ID, IM_END_ID):
    llm_metadata.stop_tokens.add().token_ids.ids.append(token_id)
  llm_metadata.jinja_prompt_template = BUNDLE_TEMPLATE
  if not any((c.channel_name, c.start, c.end) == THOUGHT_CHANNEL
             for c in llm_metadata.channels):
    channel = llm_metadata.channels.add()
    channel.channel_name, channel.start, channel.end = THOUGHT_CHANNEL


@extension.register_model_extension()
class VibeThinkerExtension(extension.ModelExtension):
  """VibeThinker-3B: the published file's export arguments, stops, template
  and thought channel."""

  model_types = ('qwen2',)

  def get_export_defaults(self, config_dict, task=None, target='cpu_gpu'):
    del task
    mismatched = {k: config_dict.get(k) for k, v in _CONFIG.items()
                  if config_dict.get(k) != v}
    if mismatched:
      raise ValueError(
          'vibethinker_extension is the recipe for VibeThinker-3B; this'
          f' config.json has {mismatched}')
    if str(target or '').startswith('npu'):
      raise ValueError(
          'vibethinker_extension writes the CPU / GPU file; target'
          f' {target!r} is not this recipe')
    return extension.ExportDefaults(
        task=extension.ExportTask.TEXT_GENERATION,
        export_kwargs={
            'quantization_recipe': QUANTIZATION_RECIPE,
            'externalize_embedder': True,
            'prefill_lengths': [128],
            'cache_length': 4096,
            # max_num_tokens = the cache length (cpu_gpu would cap it at 4096
            # over a 32,771-token cache).
            'llm_metadata_max_num_tokens_override': None,
            # Per-role prompt templates from the checkpoint's template; the
            # Jinja the file carries is BUNDLE_TEMPLATE (build_llm_metadata).
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
    model = getattr(source_model_artifacts, 'model', None)
    check_source(tokenizer, getattr(model, 'generation_config', None))
    set_bundle_metadata(llm_metadata)
    print('vibethinker_extension: stops <|im_end|> text, 151643, 151645;'
          ' the published Jinja template; thought channel')
    return llm_metadata
