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

"""Out-of-tree litert-torch `ModelExtension` for LiquidAI/LFM3-1.7B.

build_lfm3_1_7b.py hands it to the stock HF exporter (`extension_modules`),
with no change to litert-torch. The same export by hand (the mixed int4 file
with a 2,048-token cache; SRC is a local copy of the checkpoint):

  litert-torch export_hf --model=$SRC --output_dir=out_q1b_2048 \
    --extension_modules=lfm3_extension.py --trust_remote_code=True \
    --prefill_lengths=1024,256,64,16,4,1 --cache_length=2048 \
    --quantization_recipe=recipe_q1b.json --sampler_top_p=1.0 \
    --use_jinja_template=True --tokenizer_path_override=$SRC/tokenizer.json \
    --externalize_embedder=False --single_token_embedder=False

LFM3 (`lfm3`, the checkpoint's remote code) is a hybrid decoder: gated short
convolutions of width 7, sliding-window and global attention layers, sandwich
norms, q/k norms and a sigmoid output gate per head in every attention layer,
and a rotary embedding on half of each head in the global layers and on the
whole head in the sliding layers. The stock loader reads the checkpoint
with `trust_remote_code` alone, and the attention traces as the source code
writes it through litert-torch's attention interface (the sliding window
lives in the mask the exportable builds). Three hooks:

  * `patch_model`: the short convolutions run as litert-torch's LFM2 short
    convolution (`model_ext/lfm2/short_conv.py`) at width 7, and a padded
    prefill slot stays out of the convolution state. The exportable builds
    masks for the attention layers only, and the source forward indexes the
    mask dict by every layer type, so the model forward here reads it per
    layer type with `.get`, and the layer forward gives the LFM2 module a
    `valid_mask` from the positions (LiteRT-LM writes position 0 into the
    unused slots of a prefill call), for token and embeddings input alike.
  * `update_export_config`: `use_rope_composite` off, so the config says what
    the graph has (the flag is on by default, only the built-in patches act on
    it, and LFM3's rotary embeddings are not routed through `odml.rope` here).
  * `build_llm_metadata`: the stop ids are the generation config's EOS ids
    only, and the embedded template renders an earlier assistant turn the same
    way wherever it sits (`history_stable_template`).

The exporter's own template rewrites (the `generation` tags removed, `.get`
written as a subscript) and the runtime's load-time ones cover the rest of
the source template.
"""

import contextlib
import types

import torch
from transformers import modeling_outputs

from litert_torch.generative.export_hf.model_ext import extension
from litert_torch.generative.export_hf.model_ext.lfm2 import patch as lfm2_patch
from litert_torch.generative.export_hf.model_ext.lfm2 import short_conv as lfm2_short_conv

_PATCHED = '_lfm3_litert_patched'

# Export flags with no LFM3 form in this extension (not wired or tested here);
# the export stops on them.
_UNSUPPORTED_FLAGS = (
    'use_qkv_norm_rope_composite',
    'fuse_qkv',
    'split_cache',
)

# The one line of the source template that this extension changes. The source
# keeps the thought of an assistant turn after the last user message; the
# bundle drops every earlier thought, so a turn renders the same once a later
# user message arrives (LiteRT-LM keeps the earlier prompt as a prefix of the
# next one).
KEEP_THINKING_SOURCE = (
    '{%- set keep_thinking = preserve_thinking or loop.index0 >'
    ' ns.last_user_index -%}'
)
KEEP_THINKING_BUNDLE = '{%- set keep_thinking = preserve_thinking -%}'


def history_stable_template(template: str) -> str:
  """Returns `template` with earlier assistant turns rendered without thought.

  Args:
    template: The chat template as the exporter embeds it (its `generation`
      tags removed and `.get('k')` written as `['k']`).

  Raises:
    ValueError: The template is not the one this recipe was checked on.
  """
  count = template.count(KEEP_THINKING_SOURCE)
  if count == 0 and template.count(KEEP_THINKING_BUNDLE) == 1:
    return template  # Already applied (an explicit and a registered copy).
  if count != 1:
    raise ValueError(
        f'lfm3_extension: the keep_thinking line occurs {count} times in the'
        ' chat template (expected once); the source template changed,'
        ' re-check how it renders earlier assistant turns')
  for left in ('.get(', 'generation -%}'):
    if left in template:
      raise ValueError(
          f'lfm3_extension: {left!r} is still in the chat template; the'
          " runtime's template engine cannot render it")
  return template.replace(KEEP_THINKING_SOURCE, KEEP_THINKING_BUNDLE)


class Lfm3LiteRTShortConv(torch.nn.Module):
  """LFM3 gated short conv on LFM2's LiteRT short conv (same B, C, x split).

  LFM2's forward reads only the attributes set here; LFM3 differs in the width
  (7 against 3) and in `seq_idx` (packed documents, training only).
  """

  forward = lfm2_short_conv.Lfm2ShortConv.forward

  def __init__(self, original: torch.nn.Module, use_short_conv_composite: bool):
    super().__init__()
    self.layer_idx = original.layer_idx
    self.bias = original.bias
    self.conv_L_cache_size = original.conv_kernel_size
    self.use_short_conv_composite = use_short_conv_composite
    self.in_proj = original.in_proj
    self.conv = original.conv  # Only `.weight` [hidden, 1, L] and `.bias`.
    self.out_proj = original.out_proj


def valid_mask_from_positions(position_ids: torch.Tensor) -> torch.Tensor:
  """Real slots of one call: the first, then every slot whose position grows.

  Args:
    position_ids: `[1, T]` or `[T]` int positions of the call.

  Returns:
    `[1, T]` bool.
  """
  pos = position_ids.reshape(-1)
  first = torch.ones_like(pos[:1], dtype=torch.bool)
  return torch.cat([first, pos[1:] > pos[:-1]]).unsqueeze(0)


def _decoder_layer_forward(
    self,
    hidden_states,
    position_embeddings=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    valid_mask=None,
    cache_position=None,
    **kwargs,
):
  """`Lfm3DecoderLayer.forward` with the pad guard passed to the conv."""
  residual = hidden_states
  hidden_states = self.operator_norm(hidden_states)
  if self.is_attention_layer:
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        **kwargs,
    )
  else:
    hidden_states = self.conv(
        hidden_states=hidden_states,
        past_key_values=past_key_values,
        cache_position=cache_position,
        valid_mask=valid_mask,
    )
  if self.use_sandwich_norm:
    hidden_states = self.post_operator_norm(hidden_states)
  hidden_states = hidden_states + residual
  ffn_out = self.feed_forward(self.ffn_norm(hidden_states))
  if self.use_sandwich_norm:
    ffn_out = self.post_ffn_norm(ffn_out)
  return hidden_states + ffn_out


def _model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    valid_mask=None,
    cache_position=None,
    **kwargs,
):
  """`Lfm3Model.forward` for export; the exportable supplies the masks."""
  del use_cache  # The cache is always the exportable's.
  del valid_mask  # The exportable's; replaced by the position guard below.
  if inputs_embeds is None:
    inputs_embeds = self.embed_tokens(input_ids)
  if inputs_embeds.shape[1] > 1:
    valid_mask = valid_mask_from_positions(position_ids)
  else:
    valid_mask = None  # Decode: LFM2's short conv takes the token as real.
  layer_types = self.config.layer_types
  if not isinstance(attention_mask, dict):
    if 'sliding_attention' in layer_types:
      raise ValueError(
          'lfm3_extension: sliding layers need the per-layer-type mask dict'
          ' that the exportable builds (config.sliding_window set)')
    attention_mask = {'full_attention': attention_mask}
  hidden_states = inputs_embeds
  position_embeddings = {
      layer_type: self.rotary_emb(
          hidden_states, position_ids=position_ids, layer_type=layer_type
      )
      for layer_type in self.rotary_emb.layer_types
  }
  for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
    layer_type = layer_types[i]
    hidden_states = decoder_layer(
        hidden_states,
        attention_mask=attention_mask.get(layer_type),
        position_embeddings=position_embeddings.get(layer_type),
        position_ids=position_ids,
        past_key_values=past_key_values,
        valid_mask=valid_mask,
        cache_position=cache_position,
        **kwargs,
    )
  hidden_states = self.embedding_norm(hidden_states)
  return modeling_outputs.BaseModelOutputWithPast(
      last_hidden_state=hidden_states,
      past_key_values=past_key_values,
  )


def _swap_rms_norms(module, replaced):
  """Swaps every `Lfm3RMSNorm` under `module` for LFM2's composite norm.

  litert-torch's core patches already swap the class when litert_torch is
  imported before the remote code loads (the export path); this covers a
  model loaded before that.
  """
  for name, child in module.named_children():
    if type(child).__name__ == 'Lfm3RMSNorm':
      norm = lfm2_patch.Lfm2RMSNorm(child.weight.shape[0], child.variance_epsilon)
      norm.weight = child.weight
      setattr(module, name, norm)
      replaced.append((module, name, child))
    else:
      _swap_rms_norms(child, replaced)


@contextlib.contextmanager
def patch_lfm3_model(model, export_config):
  """Applies the LFM3 export patches to every LFM3 module under `model`.

  Modules are matched by class name: the remote code's classes are distinct
  objects per checkpoint folder. A second entry while patched is a no-op.

  Args:
    model: A module tree holding an `Lfm3Model`.
    export_config: The active `ExportableModuleConfig`.

  Yields:
    Control while the patches are active.
  """
  if getattr(model, _PATCHED, False):
    yield
    return
  for flag in _UNSUPPORTED_FLAGS:
    if getattr(export_config, flag, False):
      raise NotImplementedError(f'lfm3_extension: `{flag}` has no LFM3 form')
  if not any(type(m).__name__ == 'Lfm3Model' for m in model.modules()):
    raise ValueError('lfm3_extension: no Lfm3Model under the exported module')
  fuse_mlp = bool(
      getattr(export_config, 'fuse_gate_up', False)
      or getattr(export_config, 'use_swiglu_composite', False)
  )
  use_swiglu = bool(getattr(export_config, 'use_swiglu_composite', False))
  use_short_conv = bool(getattr(export_config, 'use_short_conv_composite', False))

  replaced = []  # (parent, attribute, original module)
  forwards = []  # modules given an instance `forward`
  _swap_rms_norms(model, replaced)
  norms_swapped = len(replaced)
  for parent in list(model.modules()):
    for name, child in list(parent.named_children()):
      kind = type(child).__name__
      if kind == 'Lfm3ShortConv':
        setattr(parent, name, Lfm3LiteRTShortConv(child, use_short_conv))
        replaced.append((parent, name, child))
      elif kind == 'Lfm3MLP' and fuse_mlp:
        setattr(parent, name,
                lfm2_patch.FusedLfm2MLP(child, use_swiglu_composite=use_swiglu))
        replaced.append((parent, name, child))
  for module in model.modules():
    kind = type(module).__name__
    if kind == 'Lfm3DecoderLayer':
      module.forward = types.MethodType(_decoder_layer_forward, module)
      forwards.append(module)
    elif kind == 'Lfm3Model':
      module.forward = types.MethodType(_model_forward, module)
      forwards.append(module)
  print(f'lfm3_extension: {len(replaced) - norms_swapped} modules replaced,'
        f' {len(forwards)} forwards, {norms_swapped} RMSNorm swapped here;'
        f' fuse_gate_up={fuse_mlp}, use_short_conv_composite={use_short_conv}')
  setattr(model, _PATCHED, True)
  try:
    yield
  finally:
    delattr(model, _PATCHED)
    for module in forwards:
      del module.forward
    for parent, name, original in reversed(replaced):
      setattr(parent, name, original)


def _eos_ids(source_model_artifacts):
  """The generation config's EOS ids, else the tokenizer's; [] if neither."""
  gen = getattr(source_model_artifacts.model, 'generation_config', None)
  eos = getattr(gen, 'eos_token_id', None)
  if eos is None:
    tokenizer = getattr(source_model_artifacts, 'tokenizer', None)
    eos = getattr(tokenizer, 'eos_token_id', None)
  if eos is None:
    return []
  return [int(eos)] if isinstance(eos, int) else [int(i) for i in eos]


@extension.register_model_extension()
class Lfm3Extension(extension.ModelExtension):
  """LFM3 short-conv hybrid: the conv patch, EOS-only stops, a stable template."""

  model_types = ('lfm3',)

  def update_export_config(self, export_config, model_config):
    del model_config
    # `odml.rope` is on by default for every non-split-cache export; only the
    # built-in patches act on the flag, and LFM3's rotary embeddings are not
    # routed through it here, so the bundle has no rope composite.
    if export_config.use_rope_composite:
      export_config.use_rope_composite = False
      print('lfm3_extension: use_rope_composite off (the source rope traces'
            ' as written)')
    return export_config

  def patch_model(self, model, export_config):
    return patch_lfm3_model(model, export_config)

  def build_llm_metadata(
      self,
      source_model_artifacts,
      export_config,
      exported_model_artifacts,
      llm_metadata,
  ):
    del export_config, exported_model_artifacts
    # The generic builder adds the template's end-of-turn string, alone and
    # with punctuation marks in front, as text stops (for SentencePiece
    # vocabularies). The published files stop on the EOS id alone.
    eos_ids = _eos_ids(source_model_artifacts)
    if eos_ids:
      del llm_metadata.stop_tokens[:]
      for token_id in eos_ids:
        llm_metadata.stop_tokens.add().token_ids.ids.append(token_id)
      print(f'lfm3_extension: stop ids {eos_ids} (EOS only)')
    if llm_metadata.HasField('jinja_prompt_template'):
      llm_metadata.jinja_prompt_template = history_stable_template(
          llm_metadata.jinja_prompt_template)
      print('lfm3_extension: chat template renders earlier turns without'
            ' thought')
    return llm_metadata
