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

"""Out-of-tree litert-torch `ModelExtension` for tencent/Hy-MT2-1.8B.

build_hy_mt2_1_8b.py hands it to the stock HF exporter (`extension_modules`),
with no change to litert-torch. The same export by hand:

  litert-torch export_hf --model=tencent/Hy-MT2-1.8B --output_dir=out_int8 \
    --extension_modules=hy_mt2_extension.py \
    --prefill_lengths=128 --cache_length=4096 \
    --quantization_recipe=dynamic_wi8_afp32 \
    --use_jinja_template=True --externalize_embedder=False

The architecture (`hunyuan_v1_dense`: GQA with QK-norm, RMSNorm, SiLU MLP,
tied embedding) needs no graph change; the inspector's dry run passes with no
extension. What the export gets wrong is the LiteRT-LM metadata, and that is
the one hook this file overrides:

  * No `start_token`. The chat template writes `<｜hy_begin▁of▁sentence｜>` as
    a literal, so the BOS is in the rendered prompt even when the runtime
    renders it with `bos_token` unbound. The exporter still writes the BOS as
    `start_token` (its check sees the BOS at the start of the rendered
    template), and the engine prepends `start_token` to every prompt: two BOS
    tokens. Dropping the field makes the on-device stream equal the training
    stream.
  * `<｜hy_place▁holder▁no▁2｜>` (id 120020, the tokenizer's eos and
    `generation_config.eos_token_id`) among the stop ids, as an id-level stop.
  * The embedded template reads content parts. LiteRT-LM (0.18 and later)
    hands the template each message's content as a list of parts
    (`[{"type": "text", "text": ...}]`); the vendor template prints
    `message['content']` as is, so the runtime rendered every user turn as
    that JSON text. `content_parts_template` reads the three content sites
    through a `format_content` macro (string content renders as before): the
    template the published litert-community bundle already embeds.

The rope needs no hook: from transformers 5.20.1 the HunYuan rotary embedding
maps the config's `rope_scaling: {type: dynamic, alpha: 1000}` to a static
`ntk_alpha` type and the export traces as is (5.14.1 and 5.19.0 still carry
the dynamic branch; build_hy_mt2_1_8b.py stops early on them), so this file
does not override `load_config`.
"""

import jinja2

from litert_torch.generative.export_hf.model_ext import extension


def _template_writes_bos(tokenizer) -> bool:
  """True when the template renders the BOS with `bos_token` bound to ''."""
  template = getattr(tokenizer, 'chat_template', None)
  bos = getattr(tokenizer, 'bos_token', None)
  if not template or not isinstance(bos, str) or not bos:
    return False
  env = jinja2.Environment(extensions=['jinja2.ext.loopcontrols'])
  rendered = env.from_string(template).render(
      messages=[{'role': 'user', 'content': 'x'}],
      add_generation_prompt=True,
      bos_token='',
      eos_token='',
  )
  return rendered.startswith(bos)


# Reads a message's content as a string or as a list of parts. The same macro
# (byte for byte) as in the template of the published litert-community bundle.
_FORMAT_CONTENT = (
    "{%- macro format_content(content) -%}\n"
    "    {%- if content is string -%}\n"
    "        {{- content -}}\n"
    "    {%- elif content is sequence -%}\n"
    "        {%- for item in content -%}\n"
    "            {%- if item['type'] == 'text' -%}\n"
    "                {{- item['text'] -}}\n"
    "            {%- endif -%}\n"
    "        {%- endfor -%}\n"
    "    {%- endif -%}\n"
    "{%- endmacro -%}\n"
)

# The places in the vendor template that read a message's content.
_CONTENT_SITES = (
    ("{% set system_message = messages[0]['content'] %}",
     "{% set system_message = format_content(messages[0]['content']) %}"),
    ("{{ message['content'] }}", "{{ format_content(message['content']) }}"),
)


def content_parts_template(template: str) -> str:
  """Returns the vendor template with every content site read by format_content."""
  for site, fixed in _CONTENT_SITES:
    if site not in template:
      raise ValueError(
          f'hy_mt2_extension: {site!r} not in the chat template; the vendor '
          'template changed, re-check how it reads message content')
    template = template.replace(site, fixed)
  return _FORMAT_CONTENT + template


@extension.register_model_extension()
class HyMt2Extension(extension.ModelExtension):
  """Drops the duplicate BOS, pins the eos stop, reads content parts."""

  model_types = ('hunyuan_v1_dense',)
  is_causal_text_model = True

  def build_llm_metadata(
      self,
      source_model_artifacts,
      export_config,
      exported_model_artifacts,
      llm_metadata,
  ):
    del export_config, exported_model_artifacts
    if llm_metadata.HasField('jinja_prompt_template'):
      llm_metadata.jinja_prompt_template = content_parts_template(
          llm_metadata.jinja_prompt_template)
      print('hy_mt2_extension: chat template reads content parts')

    tokenizer = getattr(source_model_artifacts, 'tokenizer', None)
    if tokenizer is None:
      return llm_metadata

    if llm_metadata.HasField('start_token') and _template_writes_bos(tokenizer):
      llm_metadata.ClearField('start_token')
      print('hy_mt2_extension: start_token dropped (the template writes the '
            'BOS itself)')

    eos_id = getattr(tokenizer, 'eos_token_id', None)
    if isinstance(eos_id, int):
      stop_ids = {i for t in llm_metadata.stop_tokens for i in t.token_ids.ids}
      if eos_id not in stop_ids:
        llm_metadata.stop_tokens.add().token_ids.ids.append(eos_id)
        print(f'hy_mt2_extension: stop id {eos_id} added')
    return llm_metadata
