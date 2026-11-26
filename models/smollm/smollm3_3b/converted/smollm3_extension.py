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

"""Out-of-tree litert-torch `ModelExtension` for HuggingFaceTB/SmolLM3-3B.

build_smollm3_3b.py passes this file to the stock HF exporter
(`extension_modules`); litert-torch is not changed. By hand:

  litert-torch export_hf --model=$SRC --output_dir=out_int4 \
    --extension_modules=smollm3_extension.py

SmolLM3 traces through the exporter's generic decoder path as is, so no model
hook is overridden. Three hooks give the form and the metadata of the
published `SmolLM3-3B_q4_block32_ekv4096.litertlm`:

  * `get_export_defaults`: its export arguments, and explicit values for the
    CPU/GPU preset arguments that would change that form.
  * `update_export_config`: the checkpoint's own tokenizer.json as the
    tokenizer section, and the template check below before the conversion.
  * `build_llm_metadata`: the chat template, prompt templates, stops and
    thought channel.

The chat template is assembled at export time from the checkpoint's own
template (`bundle_template`); the export stops when the result is not the
published file's template or renders a covered conversation differently from
the checkpoint's template (`check_template`), right after the model loads
and again when the metadata is written.
"""

import hashlib
import json
import os

import transformers

from litert_torch.generative.export_hf.model_ext import extension

_HERE = os.path.dirname(os.path.abspath(__file__))
QUANTIZATION_RECIPE = os.path.join(_HERE, 'recipe_int4.json')

# The published file: litert-community/SmolLM3-3B at revision c29e7655.
PUBLISHED_REPO = 'litert-community/SmolLM3-3B'
PUBLISHED_TEMPLATE_SHA256 = (
    '2e3e292636e245bd46fe91d95e48d1dcaac73189413f02249cc91fc1ded9bfce')

# Per-role (prefix, suffix): the structured prompt templates of the file, and
# the role markers of its chat template.
ROLES = {
    'user': ('<|im_start|>user\n', '<|im_end|>\n'),
    'model': ('<|im_start|>assistant\n', '<|im_end|>\n'),
    'system': ('<|im_start|>system\n', '<|im_end|>\n'),
}
EOS = '<|im_end|>'
END_OF_TURN = '<|im_end|>\n'  # The string stop next to the EOS id.
THOUGHT_CHANNEL = ('thought', '<think>\n', '\n</think>')

# Message content as a string or as LiteRT-LM's list of content parts.
_FORMAT_CONTENT = (
    '{%- macro format_content(content) -%}\n'
    '    {%- if content is string -%}\n'
    '        {{- content -}}\n'
    '    {%- elif content is sequence -%}\n'
    '        {%- for item in content -%}\n'
    "            {%- if item['type'] == 'text' -%}\n"
    "                {{- item['text'] -}}\n"
    '            {%- endif -%}\n'
    '        {%- endfor -%}\n'
    '    {%- endif -%}\n'
    '{%- endmacro -%}\n')


def _note(repo):
  """The template's comment on its default system turn (2026-08-28 fix).

  Args:
    repo: the checkpoint's repo id, written at the start of the comment.

  Returns:
    The Jinja comment, as template text.
  """
  return (
      '{#- ' + repo + " -- the vendor's default system turn, restored"
      ' (2026-08-28).\n'
      '\n'
      '    The upstream chat template emits this block whenever the caller'
      ' sends no\n'
      "    system message. litert_torch's parse_chat_template probes the"
      ' template with a\n'
      '    system message already present, so the block is never rendered'
      ' during probing\n'
      '    and never reached the bundle. It is put back verbatim below, and'
      ' fires once per\n'
      '    conversation -- the engine renders the whole history through this'
      ' template on\n'
      '    every turn -- and not at all when the caller supplies a system'
      ' message.\n'
      '-#}\n')


_PROBE = 'PROBE-31d7c0'
_DATE = '\x00date\x00'


def default_turn_expression(tokenizer, user_prefix):
  """The checkpoint template's default system turn, as a jinja expression.

  Renders the checkpoint's template for one user message with no system
  message and takes what comes before the user turn. Each `strftime_now`
  call stays a live call in the expression, with its format string.

  Args:
    tokenizer: The checkpoint's tokenizer; its `chat_template` is rendered.
    user_prefix: The marker that opens a user turn.

  Returns:
    A jinja expression of quoted strings and `strftime_now(...)` calls.

  Raises:
    ValueError: The template writes no default system turn.
  """
  formats = []

  def strftime_now(fmt):
    formats.append(fmt)
    return _DATE

  text = tokenizer.apply_chat_template(
      [{'role': 'user', 'content': _PROBE}], tokenize=False,
      add_generation_prompt=True, strftime_now=strftime_now)
  bos = tokenizer.bos_token or ''
  if bos and text.startswith(bos):
    text = text[len(bos):]
  end = text.find(user_prefix + _PROBE)
  if end <= 0:
    raise ValueError(
        'The checkpoint chat template writes no default system turn before'
        f' the first user turn ({user_prefix!r}); re-check the template.')
  pieces = text[:end].split(_DATE)
  if len(pieces) != len(formats) + 1:
    raise ValueError('strftime_now output is not where the template puts it.')
  expression = json.dumps(pieces[0])
  for fmt, piece in zip(formats, pieces[1:]):
    expression += f' ~ strftime_now({json.dumps(fmt)}) ~ {json.dumps(piece)}'
  return expression


def bundle_template(tokenizer, roles=None, repo=PUBLISHED_REPO):
  """The file's chat template, assembled from the checkpoint's template.

  The checkpoint's default system turn when the caller sends none, then each
  message between its role markers (`roles`), content read by
  `format_content`, and the model prefix as the generation prompt.

  Args:
    tokenizer: The checkpoint's tokenizer.
    roles: Per-role (prefix, suffix); default ROLES.
    repo: The repo named in the template's comment.

  Returns:
    The jinja template.
  """
  roles = ROLES if roles is None else roles
  parts = [
      _FORMAT_CONTENT, _note(repo),
      "{%- if messages[0].role != 'system' -%}{{ ",
      default_turn_expression(tokenizer, roles['user'][0]),
      ' }}{%- endif -%}{%- for message in messages -%}',
  ]
  for role, test in (
      ('user', "message.role == 'user'"),
      ('model', "message.role == 'model' or message.role == 'assistant'"),
      ('system', "message.role == 'system'"),
  ):
    prefix, suffix = roles[role]
    parts += ['{%- if ', test, ' %}', prefix,
              '{{ format_content(message.content) }}', suffix, '{% endif -%}']
  parts += ['{%- endfor -%}{%- if add_generation_prompt %}', roles['model'][0],
            '{% endif -%}']
  return ''.join(parts)


def _user(text):
  """A user message.

  Args:
    text: the message content.

  Returns:
    The message, as a dict.
  """
  return {'role': 'user', 'content': text}


def _assistant(text):
  """An assistant message.

  Args:
    text: the message content.

  Returns:
    The message, as a dict.
  """
  return {'role': 'assistant', 'content': text}


# Conversations the file's template renders exactly as the checkpoint's
# template does (no system message, the template's default thinking mode). A
# system message is written as bare ChatML; the checkpoint's template wraps it
# in its metadata block instead.
TEMPLATE_CHECKS = (
    [_user('What is 17 + 25?')],
    [_user('Remember the code word TANGERINE-7319.'),
     _assistant('Noted: TANGERINE-7319.'),
     _user('What was the code word?')],
)


def check_template(
    tokenizer, template, expected_sha256=PUBLISHED_TEMPLATE_SHA256
):
  """Stops the export when the assembled template is not the expected one.

  Args:
    tokenizer: The checkpoint's tokenizer.
    template: The assembled template (`bundle_template`).
    expected_sha256: sha256 of the published file's template.

  Raises:
    ValueError: The template differs from the published file's, or renders a
      TEMPLATE_CHECKS conversation differently from the checkpoint's template.
  """
  digest = hashlib.sha256(template.encode('utf-8')).hexdigest()
  if digest != expected_sha256:
    raise ValueError(
        'smollm3_extension: the chat template assembled from the checkpoint'
        f' (sha256 {digest}) is not the published file\'s ({expected_sha256}):'
        ' the checkpoint template changed. Check the renders of the new'
        ' template before updating PUBLISHED_TEMPLATE_SHA256.')
  for messages in TEMPLATE_CHECKS:
    source = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    bundle = tokenizer.apply_chat_template(
        messages, chat_template=template, tokenize=False,
        add_generation_prompt=True)
    if source != bundle:
      raise ValueError(
          'smollm3_extension: a'
          f' {len(messages)}-message conversation renders differently from'
          f' the checkpoint template.\nsource: {source!r}\nbundle: {bundle!r}')


def checkpoint_tokenizer_json(model_path):
  """The checkpoint's own tokenizer.json: a local checkout or the HF cache.

  Args:
    model_path: a local checkpoint directory or a Hugging Face repo id.

  Returns:
    The path of tokenizer.json, or None when it is not there.
  """
  if os.path.isdir(model_path):
    path = os.path.join(model_path, 'tokenizer.json')
    return path if os.path.isfile(path) else None
  try:
    import huggingface_hub  # pylint: disable=g-import-not-at-top

    path = huggingface_hub.try_to_load_from_cache(model_path, 'tokenizer.json')
  except Exception:  # pylint: disable=broad-exception-caught
    return None
  return path if isinstance(path, str) else None


def check_checkpoint_template(checkpoint_dir):
  """Stops the export when the checkpoint's chat template changed.

  The check runs right after the model loads, before the conversion;
  build_llm_metadata checks the template again when it writes it.

  Args:
    checkpoint_dir: the local directory of the checkpoint's tokenizer files.
  """
  tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_dir)
  check_template(tokenizer, bundle_template(tokenizer))
  print('smollm3_extension: the checkpoint template assembles the '
        'published one')


@extension.register_model_extension()
class SmolLM3Extension(extension.ModelExtension):
  """SmolLM3-3B: the published file's export arguments and LlmMetadata."""

  model_types = ('smollm3',)

  def get_export_defaults(self, config_dict, task=None, target='cpu_gpu'):
    del config_dict, task
    if str(target or '').startswith('npu'):
      return None  # This recipe is the CPU/GPU file; NPU keeps its preset.
    return extension.ExportDefaults(export_kwargs={
        # int4 blockwise-32 OCTAV weights, an int8 embedding table.
        'quantization_recipe': QUANTIZATION_RECIPE,
        'prefill_lengths': [128],
        'cache_length': 4096,
        'llm_metadata_max_num_tokens_override': 4096,
        'externalize_embedder': True,
        # The template is set in build_llm_metadata. With the jinja form the
        # exporter starts from the checkpoint's template instead of a
        # structured parse of it.
        'use_jinja_template': True,
        # The CPU/GPU preset values that would change the published form:
        # GPU composite ops, weights outside the TFLite, and the rope
        # composite (the published graph has none of the three).
        'apply_gpu_composites': False,
        'externalize_weights': False,
        'use_rope_composite': False,
    })

  def update_export_config(self, export_config, model_config):
    del model_config
    path = checkpoint_tokenizer_json(export_config.model)
    if path:
      check_checkpoint_template(os.path.dirname(path))
      if not export_config.tokenizer_path_override:
        # The tokenizer section is the checkpoint's tokenizer.json as is,
        # not the tokenizer re-saved by transformers.
        export_config.tokenizer_path_override = path
        print(f'smollm3_extension: tokenizer section = {path}')
    return export_config

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
      return llm_metadata  # The exporter always passes the checkpoint's.
    if tokenizer.bos_token or llm_metadata.HasField('start_token'):
      raise ValueError(
          'smollm3_extension: the tokenizer declares a BOS token'
          f' ({tokenizer.bos_token!r}); SmolLM3 has none and the file carries'
          ' no start_token. Re-check the checkpoint.')
    if tokenizer.eos_token != EOS:
      raise ValueError(
          f'smollm3_extension: EOS is {tokenizer.eos_token!r}, not {EOS!r}.')
    template = bundle_template(tokenizer)
    check_template(tokenizer, template)
    llm_metadata.jinja_prompt_template = template
    llm_metadata.ClearField('prompt_templates')
    for role, (prefix, suffix) in ROLES.items():
      field = getattr(llm_metadata.prompt_templates, role)
      field.prefix = prefix
      field.suffix = suffix
    # The exporter adds the end-of-turn string again with 14 punctuation
    # marks in front; the file stops on the EOS id and the string alone.
    del llm_metadata.stop_tokens[:]
    eos_id = tokenizer.convert_tokens_to_ids(EOS)
    llm_metadata.stop_tokens.add().token_ids.ids.append(eos_id)
    llm_metadata.stop_tokens.add().token_str = END_OF_TURN
    # The tokenizer's pad token is the end-of-turn token; the file has none.
    llm_metadata.ClearField('pad_token')
    del llm_metadata.channels[:]
    name, start, end = THOUGHT_CHANNEL
    channel = llm_metadata.channels.add()
    channel.channel_name = name
    channel.start = start
    channel.end = end
    print('smollm3_extension: chat template assembled from the checkpoint'
          f' template (sha256 {PUBLISHED_TEMPLATE_SHA256[:12]}, renders'
          f' checked); stops [{eos_id}, {END_OF_TURN!r}]; thought channel')
    return llm_metadata
