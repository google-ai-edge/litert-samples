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

"""Out-of-tree litert-torch `ModelExtension` for LiquidAI/LFM3-VL-2B.

It works together with the LFM3-1.7B recipe's extension
(`../../lfm3_1_7b/converted/lfm3_extension.py`, unchanged): LFM3-VL's text
decoder is an LFM3 decoder (`text_config.model_type` `lfm3`), and the stock
exporter composes the extension registered for the checkpoint's `lfm3_vl`
with the one registered for its `text_config` (`resolve_model_extension`).
build_lfm3_vl_2b.py hands both files to the stock HF exporter
(`extension_modules`), with no change to litert-torch. The same export by
hand, without the int4 block-scale floor build_lfm3_vl_2b.py adds (SRC is a
local copy of the checkpoint):

  litert-torch export_hf --model=$SRC --output_dir=out_int4 \
    --task=image_text_to_text --trust_remote_code=True \
    --extension_modules=../../lfm3_1_7b/converted/lfm3_extension.py,lfm3_vl_extension.py \
    --prefill_lengths=1024,256,64,16,4,1 --cache_length=2048 \
    --quantization_recipe=recipe_int4.json \
    --vision_encoder_quantization_recipe=recipe_vision.json \
    --use_jinja_template=True --tokenizer_path_override=$SRC/tokenizer.json

litert-torch loads the image processor without `trust_remote_code`, so
transformers asks once whether to run the checkpoint's image-processor code
(answer y); build_lfm3_vl_2b.py answers it. Without the floor, 3,517 block
scales of 4 weights stay under 2^-14, which XNNPACK refuses on the CPU.

LFM3-VL's vision tower takes a picture at its own size (NaViT style: patches
of 16 x 16 grouped by 2 x 2 merge blocks, 2D positions per picture). LiteRT-LM
gives a vision encoder one picture stretched to a fixed size, so this file
rewrites the tower in plain PyTorch for a 512 x 512 picture, with the same
weights and the same arithmetic:

  encoder   images [1, 512, 512, 3] in [0, 1] -> features [1, 1024, 1024]
  adapter   features -> mm_embedding [1, 256, 2048] (the projector)

* Input step: LiteRT-LM hands the encoder float(u8) * (1 / 255). The graph
  takes the exact u8 back, floor(x * 255 + 0.5), and normalizes it as the
  checkpoint's processor does, (u8 - 127.5) / 127.5 (rescale 1 / 255, mean and
  std 0.5).
* Patches: the patchifier reads a temporal pair of identical frames, so it
  equals a 16 x 16 stride-16 Conv2d whose kernel sums the pair.
* Order: inside one picture the attention has no mask, every token's rotary
  position is its own (h, w), and the MLPs and norms work per token, so the
  tower runs in raster order with constant cos / sin tables; the depthwise
  3 x 3 convolution runs on the raster grid (the source converts to raster
  for it). The projector's 2 x 2 merge is four strided slices concatenated on
  channels, which gives the source's merge-block token order.
* Attention: grouped-query attention without repeat_kv (q heads 2j and 2j + 1
  are stacked on the query axis against kv head j), and the 1 / 8 scale is
  folded into the q norm weight (a power of two, exact).

Three hooks:

  * `get_vision_exportables`: the encoder and the adapter above.
  * `get_llm_model_type_proto`: `fast_vlm` with a 512 x 512 image tensor (the
    decoder positions are 1-D, as in the source).
  * `build_llm_metadata`: image parts of the chat template render
    `<image_soft_token>`, where LiteRT-LM splices the adapter's 256 embeddings
    (the source processor expands `<|image|>` to the same count); and no
    sampler unless a sampler flag is set. For text generation the stock
    loader turns the checkpoint's `do_sample` off, for this task it does not,
    and the packager would write the generation config's TOP_K sampler.

The decoder side comes from lfm3_extension.py: the short-convolution patch,
the rope-composite flag, EOS-only stops and the history-stable template.
"""

import math

import torch
from torch.nn import functional as F

from litert_torch.generative.export_hf.core import exportable_module as exportable_module_base
from litert_torch.generative.export_hf.model_ext import extension
from litert_lm_builder.runtime.proto import llm_model_type_pb2

IMAGE_SIZE = 512

# The image part of the source chat template, and what the bundle renders.
IMAGE_PART_SOURCE = '"<|vision_start|><|image|><|vision_end|>"'
IMAGE_PART_BUNDLE = '"<|vision_start|><image_soft_token><|vision_end|>"'
IMAGE_PART_COUNT = 2  # A content string and an item of a content list.


def image_soft_token_template(template: str) -> str:
  """Returns `template` with image parts rendered as `<image_soft_token>`.

  Args:
    template: The chat template as the exporter embeds it.

  Raises:
    ValueError: The template is not the one this recipe was checked on.
  """
  count = template.count(IMAGE_PART_SOURCE)
  if count == 0 and template.count(IMAGE_PART_BUNDLE) == IMAGE_PART_COUNT:
    return template  # Already applied (an explicit and a registered copy).
  if count != IMAGE_PART_COUNT:
    raise ValueError(
        f'lfm3_vl_extension: the image part occurs {count} times in the chat'
        f' template (expected {IMAGE_PART_COUNT}); the source template'
        ' changed, re-check how it renders images')
  return template.replace(IMAGE_PART_SOURCE, IMAGE_PART_BUNDLE)


# ------------------------------------------------------------- vision tower


class _RMSNorm(torch.nn.Module):

  def __init__(self, dim, eps):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.ones(dim))
    self.eps = eps

  def forward(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class _MLP(torch.nn.Module):

  def __init__(self, dim, inter):
    super().__init__()
    self.w1 = torch.nn.Linear(dim, inter, bias=False)
    self.w3 = torch.nn.Linear(dim, inter, bias=False)
    self.w2 = torch.nn.Linear(inter, dim, bias=False)

  def forward(self, x):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))


class _ConvMixer(torch.nn.Module):
  """Gated depthwise 2D convolution on the raster patch grid."""

  def __init__(self, dim, kernel, grid):
    super().__init__()
    self.dim, self.grid = dim, grid
    self.in_proj = torch.nn.Linear(dim, 3 * dim, bias=False)
    self.spatial_conv = torch.nn.Conv2d(
        dim, dim, kernel_size=kernel, padding=kernel // 2, groups=dim,
        bias=False)
    self.out_proj = torch.nn.Linear(dim, dim, bias=False)

  def forward(self, x):  # [1, n, dim], raster order
    b, c, xx = self.in_proj(x).chunk(3, dim=-1)
    g = (b * xx).reshape(1, self.grid, self.grid, self.dim).permute(0, 3, 1, 2)
    y = self.spatial_conv(g).permute(0, 2, 3, 1)
    return self.out_proj(c * y.reshape(1, self.grid * self.grid, self.dim))


class _Attention(torch.nn.Module):
  """Bidirectional GQA with q / k head norms and a fixed axial rotary table."""

  def __init__(self, dim, heads, kv_heads, eps, n):
    super().__init__()
    self.heads, self.kv_heads, self.head_dim, self.n = (
        heads, kv_heads, dim // heads, n)
    self.group = heads // kv_heads
    self.q_proj = torch.nn.Linear(dim, heads * self.head_dim, bias=False)
    self.k_proj = torch.nn.Linear(dim, kv_heads * self.head_dim, bias=False)
    self.v_proj = torch.nn.Linear(dim, kv_heads * self.head_dim, bias=False)
    self.out_proj = torch.nn.Linear(heads * self.head_dim, dim, bias=False)
    self.q_layernorm = _RMSNorm(self.head_dim, eps)
    self.k_layernorm = _RMSNorm(self.head_dim, eps)
    self.register_buffer(
        'cos', torch.zeros(1, n, 1, self.head_dim), persistent=False)
    self.register_buffer(
        'sin', torch.zeros(1, n, 1, self.head_dim), persistent=False)

  def _rope(self, x):  # [1, n, heads, head_dim]
    half = self.head_dim // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * self.cos + rotated * self.sin

  def forward(self, x):
    n, hd = self.n, self.head_dim
    q = self.q_proj(x).reshape(1, n, self.heads, hd)
    k = self.k_proj(x).reshape(1, n, self.kv_heads, hd)
    v = self.v_proj(x).reshape(1, n, self.kv_heads, hd)
    q = self._rope(self.q_layernorm(q))  # The q norm weight carries 1 / 8.
    k = self._rope(self.k_layernorm(k))
    q = q.permute(0, 2, 1, 3).reshape(1, self.kv_heads, self.group * n, hd)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    a = torch.matmul(q, k.transpose(-2, -1)).softmax(dim=-1)
    o = torch.matmul(a, v).reshape(1, self.heads, n, hd)
    return self.out_proj(o.permute(0, 2, 1, 3).reshape(1, n, self.heads * hd))


class _Layer(torch.nn.Module):

  def __init__(self, cfg, kind, grid):
    super().__init__()
    dim = cfg['hidden_size']
    self.kind = kind
    if kind == 'full_attention':
      self.self_attn = _Attention(
          dim, cfg['num_attention_heads'], cfg['num_key_value_heads'],
          cfg['norm_eps'], grid * grid)
    elif kind == 'conv':
      self.conv = _ConvMixer(dim, cfg['conv_kernel_size'], grid)
    else:
      raise ValueError(f'lfm3_vl_extension: unknown vision layer type {kind!r}')
    self.feed_forward = _MLP(dim, cfg['intermediate_size'])
    self.operator_norm = _RMSNorm(dim, cfg['norm_eps'])
    self.ffn_norm = _RMSNorm(dim, cfg['norm_eps'])

  def forward(self, x):
    h = self.operator_norm(x)
    x = x + (self.self_attn(h) if self.kind == 'full_attention' else self.conv(h))
    return x + self.feed_forward(self.ffn_norm(x))


class VisionEncoder(torch.nn.Module):
  """images [1, G, G, 3] in [0, 1] -> [1, (G / 16)^2, 1024], raster order."""

  def __init__(self, cfg, image_size):
    super().__init__()
    patch = cfg['patch_size']
    if image_size % (patch * cfg['spatial_merge_size']):
      raise ValueError(f'lfm3_vl_extension: image size {image_size} is not a'
                       f' multiple of {patch * cfg["spatial_merge_size"]}')
    self.grid = image_size // patch
    self.image_size = image_size
    dim = cfg['hidden_size']
    self.dim = dim
    self.patch_conv = torch.nn.Conv2d(
        cfg['in_channels'], dim, kernel_size=patch, stride=patch, bias=False)
    self.embedding_norm = _RMSNorm(dim, cfg['norm_eps'])
    self.layers = torch.nn.ModuleList(
        [_Layer(cfg, kind, self.grid) for kind in cfg['layer_types']])

  def forward(self, images):
    u8 = torch.floor(images * 255.0 + 0.5)  # The runtime's u8 / 255, undone.
    x = ((u8 - 127.5) / 127.5).permute(0, 3, 1, 2)
    x = self.patch_conv(x).permute(0, 2, 3, 1)
    x = self.embedding_norm(x.reshape(1, self.grid * self.grid, self.dim))
    for layer in self.layers:
      x = layer(x)
    return x


class VisionAdapter(torch.nn.Module):
  """[1, (G / 16)^2, 1024] raster -> [1, (G / 32)^2, 2048], merge-block order."""

  def __init__(self, cfg, image_size):
    super().__init__()
    dim = cfg['hidden_size']
    self.dim, self.grid = dim, image_size // cfg['patch_size']
    self.layer_norm = torch.nn.LayerNorm(
        dim, eps=cfg['projector_layer_norm_eps'], bias=cfg['projector_bias'])
    self.linear_1 = torch.nn.Linear(
        4 * dim, cfg['projector_hidden_size'], bias=cfg['projector_bias'])
    self.linear_2 = torch.nn.Linear(
        cfg['projector_hidden_size'], cfg['text_hidden_size'],
        bias=cfg['projector_bias'])

  def forward(self, x):
    x = self.layer_norm(x).reshape(1, self.grid, self.grid, self.dim)
    m = torch.cat([x[:, 0::2, 0::2], x[:, 0::2, 1::2],
                   x[:, 1::2, 0::2], x[:, 1::2, 1::2]], dim=-1)
    m = m.reshape(1, (self.grid // 2) ** 2, 4 * self.dim)
    return self.linear_2(F.gelu(self.linear_1(m)))


def vision_config(model_config) -> dict:
  """The settings the rewritten tower reads, checked against what it supports."""
  top = model_config.to_dict()
  v = dict(top['vision_config'])
  rope = v.get('rope_parameters') or {}
  checks = {
      'rope_type': (rope.get('rope_type'), 'axial'),
      'projector_hidden_act': (top.get('projector_hidden_act', 'gelu'), 'gelu'),
      'downsample_factor': (top.get('downsample_factor', 2), 2),
      'spatial_merge_size': (v.get('spatial_merge_size'), 2),
      'temporal_patch_size': (v.get('temporal_patch_size'), 2),
  }
  for name, (got, want) in checks.items():
    if got != want:
      raise ValueError(
          f'lfm3_vl_extension: {name} is {got!r}; this tower supports {want!r}')
  v['rope_theta'] = float(rope['rope_theta'])
  v['projector_hidden_size'] = top['projector_hidden_size']
  v['projector_layer_norm_eps'] = top['projector_layer_norm_eps']
  v['projector_bias'] = bool(top['projector_bias'])
  v['text_hidden_size'] = top['text_config']['hidden_size']
  return v


def _rope_tables(cfg, grid):
  """cos / sin [1, grid^2, 1, head_dim] in raster order (the source's axial
  rotary: head_dim / 4 frequencies shared by h and w)."""
  head_dim = cfg['hidden_size'] // cfg['num_attention_heads']
  spatial = head_dim // 2
  inv_freq = 1.0 / (cfg['rope_theta'] ** (
      torch.arange(0, spatial, 2, dtype=torch.float) / spatial))
  hh, ww = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing='ij')
  pos = torch.stack([hh.flatten(), ww.flatten()], dim=-1)
  freqs = pos[..., None].float() * inv_freq.float()  # [n, 2, head_dim / 4]
  out = []
  for f in (freqs.cos(), freqs.sin()):
    fhw = torch.cat([f[:, 0], f[:, 1]], dim=-1)
    out.append(torch.cat([fhw, fhw], dim=-1).reshape(1, grid * grid, 1, head_dim))
  return out


def build_vision_encoder(model, image_size=IMAGE_SIZE) -> VisionEncoder:
  """The rewritten encoder with the weights of `model.model.vision_tower`."""
  cfg = vision_config(model.config)
  encoder = VisionEncoder(cfg, image_size)
  head_dim = cfg['hidden_size'] // cfg['num_attention_heads']
  q_scale = head_dim ** -0.5
  if not math.log2(1.0 / q_scale).is_integer():
    raise ValueError('lfm3_vl_extension: the attention scale is not a power of'
                     ' two; it cannot be folded into the q norm exactly')
  state = {}
  for key, value in model.model.vision_tower.state_dict().items():
    value = value.detach().to(torch.float32)
    if key == 'patchifier.weight':
      c, t, p = cfg['in_channels'], cfg['temporal_patch_size'], cfg['patch_size']
      state['patch_conv.weight'] = value.reshape(-1, c, t, p, p).sum(2)
    elif key.endswith('self_attn.q_layernorm.weight'):
      state[key] = value * q_scale
    elif key.endswith('inv_freq'):
      continue  # The rotary buffers; the tables below replace them.
    else:
      state[key] = value
  encoder.load_state_dict(state, strict=True)
  cos, sin = _rope_tables(cfg, encoder.grid)
  for layer in encoder.layers:
    if layer.kind == 'full_attention':
      layer.self_attn.cos.copy_(cos)
      layer.self_attn.sin.copy_(sin)
  return encoder.to(torch.float32).eval()


def build_vision_adapter(model, image_size=IMAGE_SIZE) -> VisionAdapter:
  """The projector with the weights of `model.model.multi_modal_projector`."""
  adapter = VisionAdapter(vision_config(model.config), image_size)
  state = {k: v.detach().to(torch.float32)
           for k, v in model.model.multi_modal_projector.state_dict().items()}
  adapter.load_state_dict(state, strict=True)
  return adapter.to(torch.float32).eval()


class Lfm3VlVisionEncoderExportable(exportable_module_base.ExportableModuleBase):
  """The rewritten vision encoder for one 512 x 512 picture."""

  def __init__(self, model, export_config):
    super().__init__(export_config)
    self.encoder = build_vision_encoder(model)

  def forward(self, images):
    return {'features': self.encoder(images)}

  def get_sample_inputs(self, model_config, **kwargs):
    del model_config, kwargs  # The picture size is fixed; no processor.
    images = torch.rand(1, IMAGE_SIZE, IMAGE_SIZE, 3, dtype=torch.float32)
    return {'vision_encoder': ({'images': images}, {})}


class Lfm3VlVisionAdapterExportable(exportable_module_base.ExportableModuleBase):
  """The projector: encoder features -> 256 decoder embeddings."""

  def __init__(self, model, export_config, tokenizer):
    super().__init__(export_config)
    del tokenizer
    self.adapter = build_vision_adapter(model)
    self._n = (IMAGE_SIZE // 16) ** 2
    self._dim = self.adapter.dim

  def forward(self, soft_tokens):
    return {'mm_embedding': self.adapter(soft_tokens)}

  def get_sample_inputs(self, model_config, **kwargs):
    del model_config, kwargs
    features = torch.zeros(1, self._n, self._dim, dtype=torch.float32)
    return {'vision_adapter': ({'soft_tokens': features}, {})}


@extension.register_model_extension()
class Lfm3VlExtension(extension.ModelExtension):
  """LFM3-VL: the fixed-size vision tower, fast_vlm, the image template part."""

  model_types = ('lfm3_vl',)

  def get_vision_exportables(self, model_config, export_config=None):
    del model_config, export_config
    return extension.VisionExportablesSpec(
        encoder_cls=Lfm3VlVisionEncoderExportable,
        adapter_cls=Lfm3VlVisionAdapterExportable,
        eoi_cls=None,
    )

  def get_llm_model_type_proto(self, model_config, export_config):
    del model_config, export_config
    return llm_model_type_pb2.LlmModelType(
        fast_vlm=llm_model_type_pb2.FastVlm(
            image_tensor_height=IMAGE_SIZE, image_tensor_width=IMAGE_SIZE))

  def build_llm_metadata(
      self,
      source_model_artifacts,
      export_config,
      exported_model_artifacts,
      llm_metadata,
  ):
    del source_model_artifacts, exported_model_artifacts
    if llm_metadata.HasField('jinja_prompt_template'):
      llm_metadata.jinja_prompt_template = image_soft_token_template(
          llm_metadata.jinja_prompt_template)
      print('lfm3_vl_extension: image parts render <image_soft_token>')
    # The stock loader turns the checkpoint's sampling off for text generation
    # only, so for this task the packager writes the generation config's
    # sampler: TOP_K, k 50, temperature 0.1 (its top_p is unset). The file
    # keeps no sampler, as the published one, unless a sampler flag asks.
    flags = [getattr(export_config, f'sampler_{name}', None)
             for name in ('top_k', 'top_p', 'temperature')]
    if all(f is None for f in flags) and llm_metadata.HasField(
        'sampler_params'):
      llm_metadata.ClearField('sampler_params')
      print('lfm3_vl_extension: no sampler (no sampler flag)')
    return llm_metadata
