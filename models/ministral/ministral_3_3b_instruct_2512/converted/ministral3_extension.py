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

"""Out-of-tree litert-torch `ModelExtension` for Ministral-3-3B-Instruct-2512.

build_ministral_3_3b.py passes this file to the stock HF exporter
(`extension_modules`); litert-torch is not changed. By hand:

  litert-torch export_hf --model=$SRC --output_dir=out_int4 \
    --extension_modules=ministral3_extension.py

$SRC is the BF16 checkpoint, mistralai/Ministral-3-3B-Instruct-2512-BF16. It
holds a vision encoder and the text decoder under one config; the file is the
text decoder. Three hooks read the decoder from that checkpoint directly:
`load_config` (its text config), `get_hf_model_cls` (the decoder class, built
from that config) and `checkpoint_key_prefixes` (where its weights sit in the
checkpoint). Three more give the form and the metadata of the published
`Ministral-3-3B-Instruct-2512_q4_block32_ekv4096.litertlm`:
`get_export_defaults`, `update_export_config` (the checkpoint's own
tokenizer.json as the tokenizer section, and the template check below before
the conversion) and `build_llm_metadata`.

The chat template is assembled at export time from the checkpoint's own
template (`bundle_template`); the export stops when the result is not the
published file's template or renders a covered conversation differently from
the checkpoint's template (`check_template`), right after the model loads
and again when the metadata is written.
"""

import hashlib
import json
import os

import torch
import transformers

from litert_torch.generative.export_hf.model_ext import extension

_HERE = os.path.dirname(os.path.abspath(__file__))
QUANTIZATION_RECIPE = os.path.join(_HERE, 'recipe_int4.json')

# The published file: litert-community/Ministral-3-3B-Instruct-2512 at
# revision 613e437f.
PUBLISHED_REPO = 'litert-community/Ministral-3-3B-Instruct-2512'
PUBLISHED_TEMPLATE_SHA256 = (
    '78bf773541dfe55d0bdd234eaeaf9f178760f1a29fcc86631a78bc73bb7f5e9e')

# Per-role (prefix, suffix): the structured prompt templates of the file, and
# the role markers of its chat template. A model turn has no prefix.
ROLES = {
    'user': ('[INST]', '[/INST]'),
    'model': ('', '</s>'),
    'system': ('[SYSTEM_PROMPT]', '[/SYSTEM_PROMPT]'),
}
BOS = '<s>'
EOS = '</s>'

# Where the checkpoint keeps the text decoder's weights:
# `language_model.model.layers.0.self_attn.q_proj.weight` and so on.
TEXT_DECODER_PREFIX = 'language_model.model.'

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


def _message(role, text):
  """A chat message.

  Args:
    role: the message's role.
    text: the message content.

  Returns:
    The message, as a dict.
  """
  return {'role': role, 'content': text}


# Conversations the file's template renders exactly as the checkpoint's
# template does, after the BOS the checkpoint's template writes first (the
# file's start_token).
TEMPLATE_CHECKS = (
    [_message('user', 'What is 17 + 25?')],
    [_message('user', 'Remember the code word TANGERINE-7319.'),
     _message('assistant', 'Noted: TANGERINE-7319.'),
     _message('user', 'What was the code word?')],
    [_message('system', 'Answer in one word.'),
     _message('user', 'What color is a clear daytime sky?')],
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
        'ministral3_extension: the chat template assembled from the'
        f' checkpoint (sha256 {digest}) is not the published file\'s'
        f' ({expected_sha256}): the checkpoint template changed. Check the'
        ' renders of the new template before updating'
        ' PUBLISHED_TEMPLATE_SHA256.')
  for messages in TEMPLATE_CHECKS:
    source = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    if not source.startswith(BOS):
      raise ValueError(
          f'ministral3_extension: the checkpoint template no longer starts'
          f' with {BOS!r}, the file\'s start_token.')
    bundle = tokenizer.apply_chat_template(
        messages, chat_template=template, tokenize=False,
        add_generation_prompt=True)
    if source[len(BOS):] != bundle:
      raise ValueError(
          'ministral3_extension: a'
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


def text_decoder(config):
  """The text decoder alone, built from the checkpoint's text config.

  A plain constructor with no `from_pretrained`: the exporter then builds the
  model from the config and copies every weight from the checkpoint with its
  checkpoint loader, which applies `checkpoint_key_prefixes`.

  Args:
    config: The checkpoint's text config (`ministral3`).

  Returns:
    A `Ministral3ForCausalLM`; its output layer shares the embedding table.
  """
  return transformers.Ministral3ForCausalLM(config)


def check_checkpoint_template(checkpoint_dir):
  """Stops the export when the checkpoint's chat template changed.

  The check runs right after the model loads, before the conversion;
  build_llm_metadata checks the template again when it writes it.

  Args:
    checkpoint_dir: the local directory of the checkpoint's tokenizer files.
  """
  tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_dir)
  check_template(tokenizer, bundle_template(tokenizer))
  print('ministral3_extension: the checkpoint template assembles the '
        'published one')


@extension.register_model_extension()
class Ministral3Extension(extension.ModelExtension):
  """Ministral-3-3B-Instruct-2512: the text decoder of the BF16 checkpoint."""

  # The checkpoint's config is `mistral3`; `load_config` hands on its text
  # config, `ministral3`, by which the exporter resolves the extension again.
  model_types = ('mistral3', 'ministral3')
  checkpoint_key_prefixes = (TEXT_DECODER_PREFIX,)

  def get_export_defaults(self, config_dict, task=None, target='cpu_gpu'):
    del config_dict, task
    if str(target or '').startswith('npu'):
      return None  # This recipe is the CPU/GPU file; NPU keeps its preset.
    return extension.ExportDefaults(
        task='text_generation',  # The text decoder alone.
        export_kwargs={
            # int4 blockwise-32 OCTAV weights, an int8 embedding table.
            'quantization_recipe': QUANTIZATION_RECIPE,
            'prefill_lengths': [128],
            'cache_length': 4096,
            'llm_metadata_max_num_tokens_override': 4096,
            'externalize_embedder': True,
            # The template is set in build_llm_metadata. With the jinja form
            # the exporter starts from the checkpoint's template instead of a
            # structured parse of it.
            'use_jinja_template': True,
            # The CPU/GPU preset values that would change the published
            # form: GPU composite ops, weights outside the TFLite, and the
            # rope composite (the published graph has none of the three).
            'apply_gpu_composites': False,
            'externalize_weights': False,
            'use_rope_composite': False,
        },
    )

  def load_config(self, model_path, export_config, trust_remote_code=False):
    del export_config
    config = transformers.AutoConfig.from_pretrained(
        model_path, dtype=torch.float32, trust_remote_code=trust_remote_code)
    text_config = getattr(config, 'text_config', None)
    if config.model_type != 'mistral3' or text_config is None:
      return None  # A text-only checkpoint loads as usual.
    if text_config.model_type != 'ministral3':
      raise ValueError(
          'ministral3_extension: the checkpoint text config is'
          f' {text_config.model_type!r}, not ministral3.')
    text_config.dtype = torch.float32  # As the exporter sets on sub-configs.
    print('ministral3_extension: exporting the text decoder of the mistral3'
          f' checkpoint (weights under {TEXT_DECODER_PREFIX!r}; the vision'
          ' encoder and projector are not read)')
    return text_config

  def get_hf_model_cls(self, task, model_config, export_config):
    del task, export_config
    if getattr(model_config, 'model_type', None) != 'ministral3':
      return None
    return text_decoder

  def update_export_config(self, export_config, model_config):
    del model_config
    path = checkpoint_tokenizer_json(export_config.model)
    if path:
      check_checkpoint_template(os.path.dirname(path))
      if not export_config.tokenizer_path_override:
        # The tokenizer section is the checkpoint's tokenizer.json as is,
        # not the tokenizer re-saved by transformers.
        export_config.tokenizer_path_override = path
        print(f'ministral3_extension: tokenizer section = {path}')
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
    if tokenizer.bos_token != BOS or tokenizer.eos_token != EOS:
      raise ValueError(
          'ministral3_extension: the tokenizer BOS / EOS are'
          f' {tokenizer.bos_token!r} / {tokenizer.eos_token!r}, not'
          f' {BOS!r} / {EOS!r}.')
    template = bundle_template(tokenizer)
    check_template(tokenizer, template)
    llm_metadata.jinja_prompt_template = template
    llm_metadata.ClearField('prompt_templates')
    for role, (prefix, suffix) in ROLES.items():
      field = getattr(llm_metadata.prompt_templates, role)
      field.prefix = prefix
      field.suffix = suffix
    # The checkpoint's template writes the BOS first; the runtime adds it.
    llm_metadata.ClearField('start_token')
    llm_metadata.start_token.token_str = BOS
    # The exporter adds the end-of-turn string again with 14 punctuation
    # marks in front; the file stops on the EOS id alone.
    del llm_metadata.stop_tokens[:]
    eos_id = tokenizer.convert_tokens_to_ids(EOS)
    llm_metadata.stop_tokens.add().token_ids.ids.append(eos_id)
    llm_metadata.ClearField('pad_token')  # The file declares none.
    del llm_metadata.channels[:]
    print('ministral3_extension: chat template assembled from the checkpoint'
          f' template (sha256 {PUBLISHED_TEMPLATE_SHA256[:12]}, renders'
          f' checked); start_token {BOS!r}; stops [{eos_id}]')
    return llm_metadata
