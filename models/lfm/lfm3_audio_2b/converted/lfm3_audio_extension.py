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

"""Out-of-tree litert-torch `ModelExtension` for LiquidAI/LFM3-Audio-2B.

build_lfm3_audio_2b.py hands it to the stock HF exporter (task
`multimodal_lm`) together with the LFM3-1.7B recipe's `lfm3_extension.py`,
which keeps the text decoder's hooks (the short-conv patch, the EOS-only
stops, the history-stable template). The stock loader reads the checkpoint
through its own `auto_map` (`Lfm3AudioForConditionalGeneration`), and the
stock resolver composes this extension (`lfm3_audio`, the config's model
type) with `lfm3` (its `text_config`).

This file covers the text path: speech or text in, text out, one `.litertlm`
for LiteRT-LM. Speech output (the depth decoder, the frame embedding and the
codec) runs outside LiteRT-LM; see `speech/README.md`. Four hooks:

  * `get_audio_exportables`: the audio section LiteRT-LM's generic audio path
    runs. It takes raw PCM, `audio [1, 3000, 160]` (30 s at 16 kHz, 160
    samples per row), computes the checkpoint's log-mel spectrogram in the
    graph (400-sample Hann window, hop 160, 128 slaney mel bins, log10 floored
    at -12, (x + 4) / 4), then the checkpoint's downsampler (8 frames to one
    vector), its 17-layer LFM2 encoder (every attention layer under a causal
    window of 32) and projector, and returns `features [1, 375, 2048]` and
    `mask [1, 375]` (uint8). The runtime passes no clip length, so the graph
    finds it: a frame counts when a non-zero sample lies in its span past the
    previous frame's, and `mask` marks the first ceil(frames / 8) vectors,
    which the runtime reads as the number of audio tokens. The forward is
    written for the LiteRT GPU delegate and fp16 activations: grouped query
    heads folded into the matmul rows (no rank-5 tensor, no BROADCAST_TO), the
    validity of each frame as a 0 / 1 prefix vector instead of a sample index
    (an index up to 479,999 does not survive fp16), each frame scaled by its
    own peak before the DFT (log10(m^2) added back), and an attention mask of
    -30000 (fp16 holds it).
  * `get_prefill_decode_exportables`: the stock embeddings-input prefill and
    decode, plus a second output `hidden` (the decoder's final norm output,
    the input of the output projection): `[1, T, 2048]` from prefill and
    `[1, 1, 2048]` from decode. LiteRT-LM reads `logits` and ignores it; the
    speech loop feeds it to the depth decoder, so one file serves both.
  * `get_llm_model_type_proto`: `GenericModel` with the audio fields (the only
    model type with audio in LiteRT-LM 0.18 and 0.19): `<|audio|>` as the
    delimiter, raw PCM (no runtime mel), 16 kHz mono, 160-sample frames.
  * `build_llm_metadata`: stops the export if the chat template no longer
    renders an audio part as `<|audio|>` (the runtime splits the prompt on
    it and splices the audio section's vectors there).

The loader needs no hook. Two things stay in build_lfm3_audio_2b.py: the
feature extractor's `trust_remote_code` and the int4 block-scale floor.
"""

import dataclasses
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core.external_emb import exportable_module as external_emb_module
from litert_torch.generative.export_hf.model_ext import extension

# The checkpoint's front end (preprocessor_config.json; the feature extractor
# is checked against these values before the audio section is exported).
SAMPLE_RATE = 16000
N_FFT = 400
HOP = 160
NUM_MEL_BINS = 128
LOG_FLOOR = -12.0
NORM_OFFSET = 4.0
NORM_DIVISOR = 4.0
WINDOW_ROWS = 3000  # 30 s of 160-sample rows per call: the runtime's window.
MASK_NEG = -30000.0  # Below every score; fp16 holds it (-1e9 is -inf there).
PCM_STEP = 2.0 ** -15  # One step of 16-bit PCM: the per-frame scale's floor.

# The source template renders an audio part as this literal, once.
AUDIO_PART = '"<|audio|>"'
AUDIO_DELIMITER_REGEX = r'(<\|audio\|>)'
AUDIO_TOKEN_REGEX = r'<\|audio\|>'


@dataclasses.dataclass(frozen=True)
class AudioSpec:
  """Shapes of the speech encoder, read from `Lfm3AudioConfig`."""

  conv_channels: int
  hidden_size: int
  layer_types: tuple[str, ...]
  num_attention_heads: int
  num_key_value_heads: int
  ffn_size: int
  conv_kernel: int
  norm_eps: float
  rope_theta: float
  sliding_window: int
  out_size: int


def audio_spec(config) -> AudioSpec:
  """`AudioSpec` of an `Lfm3AudioConfig` (raises on what this file does not
  implement)."""
  enc = config.audio_encoder_config
  lfm = enc.text_config
  if enc.num_mel_bins != NUM_MEL_BINS:
    raise ValueError(f'lfm3_audio_extension: {enc.num_mel_bins} mel bins'
                     f' (this file computes {NUM_MEL_BINS})')
  if lfm.hidden_size != enc.hidden_size:
    raise ValueError('lfm3_audio_extension: encoder width differs from its'
                     ' LFM2 width')
  rope = getattr(lfm, 'rope_parameters', None) or {}
  rope = rope.get('full_attention', rope)
  if rope.get('rope_type', 'default') != 'default':
    raise ValueError(f'lfm3_audio_extension: encoder rope {rope}')
  ffn = lfm.intermediate_size
  if getattr(lfm, 'block_auto_adjust_ff_dim', False):
    ffn = int(2 * ffn / 3)
    if lfm.block_ffn_dim_multiplier is not None:
      ffn = int(lfm.block_ffn_dim_multiplier * ffn)
    multiple = lfm.block_multiple_of
    ffn = multiple * ((ffn + multiple - 1) // multiple)
  return AudioSpec(
      conv_channels=enc.conv_channels, hidden_size=enc.hidden_size,
      layer_types=tuple(lfm.layer_types),
      num_attention_heads=lfm.num_attention_heads,
      num_key_value_heads=lfm.num_key_value_heads, ffn_size=ffn,
      conv_kernel=lfm.conv_L_cache, norm_eps=lfm.norm_eps,
      rope_theta=float(rope.get('rope_theta', getattr(lfm, 'rope_theta', 1e6))),
      sliding_window=enc.sliding_window,
      out_size=config.text_config.hidden_size)


def _mel_filters():
  """The feature extractor's slaney filter bank, [201, 128] float64."""
  from transformers.audio_utils import mel_filter_bank  # pylint: disable=g-import-not-at-top

  return mel_filter_bank(
      num_frequency_bins=N_FFT // 2 + 1, num_mel_filters=NUM_MEL_BINS,
      min_frequency=0.0, max_frequency=SAMPLE_RATE / 2,
      sampling_rate=SAMPLE_RATE, norm='slaney', mel_scale='slaney')


def frame_validity(audio, rows, first):
  """`audio [1, T, 160]` -> `[1, T, 1, 1]` 0 / 1: frame t counts when t == 0
  or a non-zero sample lies at index >= 160 t + 240.

  That is the feature extractor's frame count of the clip up to its last
  non-zero sample, 1 + ceil((n - 400) / 160) at least 1, as a prefix. A
  per-half-row any (80 samples), a suffix max by doubling, every tensor rank
  4 (the GPU delegate places lower-rank runtime tensors and constants
  differently).
  """
  z = audio.dtype
  nz = (audio != 0.0).to(z).reshape(1, rows, 2, HOP // 2)
  halves = torch.amax(nz, dim=3, keepdim=True)  # [1, T, 2, 1]
  block = torch.amax(halves, dim=2, keepdim=True)  # [1, T, 1, 1]
  second = halves[:, :, 1:2, :]
  suffix = block
  shift = 1
  while shift < rows:  # suffix[j] = max(block[j:])
    suffix = torch.maximum(
        suffix,
        torch.cat([suffix[:, shift:], torch.zeros(1, shift, 1, 1, dtype=z)],
                  dim=1))
    shift *= 2
  nxt = torch.cat([second[:, 1:], torch.zeros(1, 1, 1, 1, dtype=z)], dim=1)
  nxt2 = torch.cat([suffix[:, 2:], torch.zeros(1, 2, 1, 1, dtype=z)], dim=1)
  return torch.maximum(torch.maximum(nxt, nxt2), first)


class LogMel(nn.Module):
  """The feature extractor's log-mel over a zero-padded window of PCM rows.

  Frame t covers samples [160 t, 160 t + 400). The Hann window (periodic) is
  folded into a fixed DFT basis (fp32 matmuls for the 201 one-sided bins).
  """

  def __init__(self, rows):
    super().__init__()
    self.rows = rows
    n = torch.arange(N_FFT, dtype=torch.float64).reshape(-1, 1)
    k = torch.arange(N_FFT // 2 + 1, dtype=torch.float64).reshape(1, -1)
    ang = 2.0 * math.pi * torch.remainder(n * k, N_FFT) / N_FFT
    win = torch.hann_window(N_FFT, periodic=True,
                            dtype=torch.float64).reshape(-1, 1)
    self.register_buffer('dft_cos', (win * torch.cos(ang)).float(),
                         persistent=False)
    self.register_buffer('dft_sin', (win * torch.sin(ang)).float(),
                         persistent=False)
    self.register_buffer(
        'mel_filters', torch.from_numpy(_mel_filters().astype(np.float32)),
        persistent=False)
    first = torch.zeros(1, rows, 1, 1, dtype=torch.float32)
    first[0, 0] = 1.0
    self.register_buffer('first', first, persistent=False)

  def forward(self, audio):  # [1, T, 160] -> mel [1, 128, T], v1 [1, T, 1, 1]
    rows = self.rows
    blocks = torch.cat([audio.reshape(rows, HOP),
                        torch.zeros(2, HOP, dtype=audio.dtype)], dim=0)
    frames = torch.cat([blocks[0:rows], blocks[1:rows + 1],
                        blocks[2:rows + 2, 0:N_FFT - 2 * HOP]], dim=1)
    # mel(x / m) = mel(x) / m^2: in fp16 the quiet frames' power spectrum
    # stays out of the subnormal range and the loudest below 200^2.
    m = torch.clamp(torch.amax(torch.abs(frames), dim=1, keepdim=True),
                    min=PCM_STEP)
    xs = frames / m
    re = torch.matmul(xs, self.dft_cos)
    im = torch.matmul(xs, self.dft_sin)
    mel = torch.matmul(re * re + im * im, self.mel_filters)
    log_mel = torch.log10(mel) + 2.0 * torch.log10(m)
    log_mel = torch.clamp(log_mel, min=LOG_FLOOR)
    log_mel = (log_mel + NORM_OFFSET) / NORM_DIVISOR
    return (log_mel.transpose(0, 1).unsqueeze(0),
            frame_validity(audio, rows, self.first))


class Downsampler(nn.Module):
  """`Lfm3AudioDownsampler` (same parameter names) with prefix masks."""

  def __init__(self, spec, rows):
    super().__init__()
    c = spec.conv_channels
    conv = {'kernel_size': 3, 'stride': 2, 'padding': 1}
    self.conv_in = nn.Conv2d(1, c, **conv)
    self.conv_mid = nn.Conv2d(c, c, groups=c, **conv)
    self.lin_mid = nn.Conv2d(c, c, kernel_size=1)
    self.conv_end = nn.Conv2d(c, c, groups=c, **conv)
    self.lin_end = nn.Conv2d(c, c, kernel_size=1)
    self.lin_out = nn.Conv1d(NUM_MEL_BINS * c // 8, spec.hidden_size,
                             kernel_size=1)
    self.rows = rows

  def forward(self, mel, v1):  # -> h [1, T/8, hidden], v8 [1, T/8]
    t = self.rows
    # Every stride-2 length is a prefix: u < ceil(F / 2) <=> 2u < F.
    v2 = v1.reshape(1, t // 2, 2, 1)[:, :, 0:1, :].reshape(1, 1, 1, t // 2)
    v4 = v1.reshape(1, t // 4, 4, 1)[:, :, 0:1, :].reshape(1, 1, 1, t // 4)
    v8 = v1.reshape(1, t // 8, 8, 1)[:, :, 0:1, :].reshape(1, t // 8)
    h = mel.unsqueeze(1) * v1.reshape(1, 1, 1, t)
    h = F.silu(self.conv_in(h)) * v2
    h = F.silu(self.lin_mid(self.conv_mid(h))) * v4
    h = F.silu(self.lin_end(self.conv_end(h)))
    b, c, f, n = h.shape
    h = h.permute(0, 3, 1, 2).reshape(b, n, c * f)  # flatten(1, 2) order
    h = F.linear(h, self.lin_out.weight.squeeze(-1), self.lin_out.bias)
    return h, v8


class RMSNorm(nn.Module):

  def __init__(self, dim, eps):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))
    self.eps = eps

  def forward(self, x):
    return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)
                                          + self.eps))


class EncoderAttention(nn.Module):
  """`Lfm2Attention` with the query heads of one kv head folded into rows."""

  def __init__(self, spec):
    super().__init__()
    d = spec.hidden_size
    self.nh, self.nkv = spec.num_attention_heads, spec.num_key_value_heads
    self.hd = d // self.nh
    self.q_proj = nn.Linear(d, self.nh * self.hd, bias=False)
    self.k_proj = nn.Linear(d, self.nkv * self.hd, bias=False)
    self.v_proj = nn.Linear(d, self.nkv * self.hd, bias=False)
    self.out_proj = nn.Linear(self.nh * self.hd, d, bias=False)
    self.q_layernorm = RMSNorm(self.hd, spec.norm_eps)
    self.k_layernorm = RMSNorm(self.hd, spec.norm_eps)

  @staticmethod
  def rope(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., half:], x[..., :half]], dim=-1) * sin

  def forward(self, x, cos, sin, mask):  # mask [1, 1, rep * N, N]
    n = x.shape[1]
    q = self.q_layernorm(self.q_proj(x).reshape(1, n, self.nh, self.hd))
    k = self.k_layernorm(self.k_proj(x).reshape(1, n, self.nkv, self.hd))
    v = self.v_proj(x).reshape(1, n, self.nkv, self.hd).transpose(1, 2)
    q = self.rope(q.transpose(1, 2), cos, sin)
    k = self.rope(k.transpose(1, 2), cos, sin)
    rep = self.nh // self.nkv  # Query head j * rep + r reads kv head j.
    qg = q.reshape(1, self.nkv, rep * n, self.hd)
    w = torch.softmax(
        torch.matmul(qg, k.transpose(2, 3)) * (self.hd ** -0.5) + mask, dim=-1)
    o = torch.matmul(w, v).reshape(1, self.nh, n, self.hd)
    return self.out_proj(o.transpose(1, 2).reshape(1, n, self.nh * self.hd))


class EncoderShortConv(nn.Module):
  """`Lfm2ShortConv` from a zero state: in_proj, B * x, causal depthwise
  conv, C * ., out_proj."""

  def __init__(self, spec):
    super().__init__()
    d, kk = spec.hidden_size, spec.conv_kernel
    self.kernel = kk
    self.conv = nn.Conv1d(d, d, kernel_size=kk, groups=d, bias=False,
                          padding=kk - 1)
    self.in_proj = nn.Linear(d, 3 * d, bias=False)
    self.out_proj = nn.Linear(d, d, bias=False)

  def forward(self, x):
    b, c, xx = self.in_proj(x).transpose(1, 2).chunk(3, dim=1)
    y = F.conv1d(F.pad(b * xx, (self.kernel - 1, 0)), self.conv.weight, None,
                 groups=self.conv.weight.shape[0])
    return self.out_proj((c * y).transpose(1, 2))


class EncoderMLP(nn.Module):

  def __init__(self, spec):
    super().__init__()
    d, f = spec.hidden_size, spec.ffn_size
    self.w1 = nn.Linear(d, f, bias=False)
    self.w3 = nn.Linear(d, f, bias=False)
    self.w2 = nn.Linear(f, d, bias=False)

  def forward(self, x):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))


class EncoderLayer(nn.Module):

  def __init__(self, spec, kind):
    super().__init__()
    self.is_attn = kind == 'full_attention'
    if self.is_attn:
      self.self_attn = EncoderAttention(spec)
    else:
      self.conv = EncoderShortConv(spec)
    self.feed_forward = EncoderMLP(spec)
    self.operator_norm = RMSNorm(spec.hidden_size, spec.norm_eps)
    self.ffn_norm = RMSNorm(spec.hidden_size, spec.norm_eps)

  def forward(self, h, cos, sin, mask):
    x = self.operator_norm(h)
    h = h + (self.self_attn(x, cos, sin, mask) if self.is_attn
             else self.conv(x))
    return h + self.feed_forward(self.ffn_norm(h))


class Lfm2Encoder(nn.Module):
  """`Lfm2Model` over embeddings: rope at 0..N-1, every attention layer under
  the causal window (the token itself counts), final `embedding_norm`.

  The window mask enters as a runtime rank-4 tensor (a constant times frame
  0's validity, always 1): with a constant operand the converter fed a rank-3
  matmul straight into a rank-4 add, which the GPU delegate computed wrong.
  """

  def __init__(self, spec, n):
    super().__init__()
    self.layers = nn.ModuleList(
        [EncoderLayer(spec, kind) for kind in spec.layer_types])
    self.embedding_norm = RMSNorm(spec.hidden_size, spec.norm_eps)
    hd = spec.hidden_size // spec.num_attention_heads
    inv_freq = 1.0 / (spec.rope_theta ** (
        torch.arange(0, hd, 2, dtype=torch.float) / hd))
    freqs = torch.arange(n, dtype=torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    self.register_buffer('cos', emb.cos(), persistent=False)
    self.register_buffer('sin', emb.sin(), persistent=False)
    i = torch.arange(n)[:, None]
    j = torch.arange(n)[None, :]
    allowed = (j <= i) & (j > i - spec.sliding_window)
    mask = torch.where(allowed, 0.0, MASK_NEG).float()
    rep = spec.num_attention_heads // spec.num_key_value_heads
    self.register_buffer(
        'mask4', mask.repeat(rep, 1).reshape(1, 1, rep * n, n).contiguous(),
        persistent=False)

  def forward(self, h, one):  # one [1, 1, 1, 1] at run time
    mask = self.mask4 * one
    for layer in self.layers:
      h = layer(h, self.cos, self.sin, mask)
    return self.embedding_norm(h)


class Projector(nn.Module):

  def __init__(self, spec):
    super().__init__()
    self.layer_norm = nn.LayerNorm(spec.hidden_size)
    self.linear_1 = nn.Linear(spec.hidden_size, spec.out_size)
    self.linear_2 = nn.Linear(spec.out_size, spec.out_size)

  def forward(self, x):
    return self.linear_2(F.gelu(self.linear_1(self.layer_norm(x))))


class AudioSection(nn.Module):
  """PCM `[1, T, 160]` -> `features [1, T/8, out]` + uint8 `mask [1, T/8]`."""

  def __init__(self, spec, rows=WINDOW_ROWS):
    super().__init__()
    if rows % 8:
      raise ValueError(f'lfm3_audio_extension: {rows} rows (a multiple of 8)')
    self.rows = rows
    self.logmel = LogMel(rows)
    self.downsample = Downsampler(spec, rows)
    self.lfm = Lfm2Encoder(spec, rows // 8)
    self.projector = Projector(spec)

  def forward(self, audio):
    mel, v1 = self.logmel(audio)
    h, v8 = self.downsample(mel, v1)
    features = self.projector(self.lfm(h, v1[:, 0:1]))
    return {'features': features, 'mask': v8.to(torch.uint8)}


def audio_section_from_model(model, rows=WINDOW_ROWS):
  """`AudioSection` holding the weights of an `Lfm3AudioForConditionalGeneration`
  (strict: every encoder and projector tensor, nothing else), float32."""
  inner = model.model
  section = AudioSection(audio_spec(model.config), rows)
  state = {k: v for k, v in inner.audio_encoder.state_dict().items()
           if not k.startswith('lfm.rotary_emb.')}
  state.update({f'projector.{k}': v
                for k, v in inner.audio_projector.state_dict().items()})
  section.load_state_dict({k: v.float() for k, v in state.items()},
                          strict=True)
  return section.eval()


def check_feature_extractor(feature_extractor):
  """Stops on a feature extractor whose front end differs from `LogMel`."""
  want = {'sampling_rate': SAMPLE_RATE, 'n_fft': N_FFT, 'hop_length': HOP,
          'win_length': N_FFT, 'feature_size': NUM_MEL_BINS,
          'log_floor': LOG_FLOOR, 'normalization_offset': NORM_OFFSET,
          'normalization_divisor': NORM_DIVISOR}
  got = {k: getattr(feature_extractor, k, None) for k in want}
  if got != want:
    raise ValueError(f'lfm3_audio_extension: the feature extractor is {got},'
                     f' the audio section computes {want}')


class Lfm3AudioEncoderExportable(exportable_module.ExportableModuleBase):
  """The audio section as one signature, `encode`."""

  def __init__(self, model, export_config):
    super().__init__(export_config)
    self.section = audio_section_from_model(model)

  def forward(self, audio):
    return self.section(audio)

  def get_sample_inputs(self, model_config, **kwargs):
    del model_config
    feature_extractor = kwargs.get('feature_extractor')
    if feature_extractor is not None:
      check_feature_extractor(feature_extractor)
    audio = torch.zeros((1, self.section.rows, HOP), dtype=torch.float32)
    return {'encode': ({'audio': audio}, {})}


def _text_model(model):
  return model.model.language_model


def _decoder_inputs(module, embeddings, input_pos, kv_cache, mask, local_mask,
                    kwargs):
  kwargs.update(module.composite_flags())
  inputs = module.adapt_inputs(
      None, embeddings, input_pos, kv_cache, mask, local_mask=local_mask,
      use_bool_mask=module.export_config.extra_kwargs.get(
          'use_bool_mask', False),
      **kwargs)
  inputs |= module.attention_kwargs()
  # The final hidden states are an output here, so every position is live.
  inputs['prefill_logits'] = True
  return inputs


class Lfm3AudioPrefill(external_emb_module
                       .LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder):
  """Embeddings-input prefill with the `hidden` output."""

  def forward(self, embeddings, input_pos, kv_cache, mask, local_mask=None,  # pylint: disable=arguments-renamed
              **kwargs):
    inputs = _decoder_inputs(self, embeddings, input_pos, kv_cache, mask,
                             local_mask, kwargs)
    inputs.pop('logits_to_keep', None)
    output = _text_model(self.model)(**inputs)
    hidden = output.last_hidden_state
    outputs = {'kv_cache': output.past_key_values,
               'hidden': hidden.to(torch.float32)}
    if self.export_config.prefill_logits:
      outputs['logits'] = self.model.lm_head(hidden[:, -1:]).to(torch.float32)
    return outputs


class Lfm3AudioDecode(external_emb_module
                      .LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder):
  """Embeddings-input decode with the `hidden` output."""

  def forward(self, embeddings, input_pos, kv_cache, mask, local_mask=None,  # pylint: disable=arguments-renamed
              **kwargs):
    if self.export_verifier:
      raise NotImplementedError(
          'lfm3_audio_extension: MTP verifier signatures are not wired')
    inputs = _decoder_inputs(self, embeddings, input_pos, kv_cache, mask,
                             local_mask, kwargs)
    output = _text_model(self.model)(**inputs)
    hidden = output.last_hidden_state
    return {'kv_cache': output.past_key_values,
            'logits': self.model.lm_head(hidden).to(torch.float32),
            'hidden': hidden.to(torch.float32)}


def check_audio_template(template: str) -> str:
  """Returns `template` if it renders an audio part as `<|audio|>` once."""
  count = template.count(AUDIO_PART)
  if count != 1:
    raise ValueError(
        f'lfm3_audio_extension: the audio part occurs {count} times in the'
        ' chat template (expected once); the source template changed,'
        ' re-check how it renders audio')
  return template


@extension.register_model_extension()
class Lfm3AudioExtension(extension.ModelExtension):
  """LFM3-Audio text path: the audio section, `hidden`, the audio metadata."""

  model_types = ('lfm3_audio',)

  def get_prefill_decode_exportables(self, model_config, export_config):
    del model_config
    if not export_config.externalize_embedder:
      raise NotImplementedError(
          'lfm3_audio_extension: the audio file takes embeddings'
          ' (externalize_embedder), as task multimodal_lm sets it')
    if getattr(export_config, 'split_cache', False):
      raise NotImplementedError('lfm3_audio_extension: split_cache is not wired')
    return Lfm3AudioPrefill, Lfm3AudioDecode

  def get_audio_exportables(self, model_config, export_config=None):
    del model_config, export_config
    return Lfm3AudioEncoderExportable, None, None

  def get_llm_model_type_proto(self, model_config, export_config):
    del model_config
    from litert_lm_builder.runtime.proto import llm_model_type_pb2  # pylint: disable=g-import-not-at-top

    if not export_config.export_audio_encoder:
      return llm_model_type_pb2.LlmModelType(
          generic_model=llm_model_type_pb2.GenericModel())
    return llm_model_type_pb2.LlmModelType(
        generic_model=llm_model_type_pb2.GenericModel(
            audio_enabled=True, delimiter_regex=AUDIO_DELIMITER_REGEX,
            audio_token_regex=AUDIO_TOKEN_REGEX,
            skip_mel_spectrogram_extraction=True,
            audio_sample_rate_hz=SAMPLE_RATE, audio_num_channels=1,
            audio_frame_length=HOP, audio_hop_length=HOP,
            audio_input_scale=1.0, add_audio_end=False))

  def build_llm_metadata(self, source_model_artifacts, export_config,
                         exported_model_artifacts, llm_metadata):
    del source_model_artifacts, export_config, exported_model_artifacts
    if llm_metadata.HasField('jinja_prompt_template'):
      check_audio_template(llm_metadata.jinja_prompt_template)
      print('lfm3_audio_extension: the chat template renders audio as'
            ' <|audio|>')
    return llm_metadata
