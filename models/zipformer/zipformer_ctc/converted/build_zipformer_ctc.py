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

"""Zipformer-medium CR-CTC (icefall, LibriSpeech) -> LiteRT CompiledModel GPU.

Checkpoint: Zengwei/icefall-asr-librispeech-zipformer-medium-cr-ctc-20241018
  64.25M params, pure CTC (--use-ctc 1 --use-transducer 0), WER 2.12/4.62.
Pipeline: host kaldi-fbank [1,T,80] -> [GPU] Conv2dSubsampling + Zipformer2 +
  ctc Linear -> host greedy-CTC + BPE detokenization.

Setup (see README.md for details):
  git clone https://github.com/k2-fsa/icefall   # model code, no k2 build
  hf download Zengwei/icefall-asr-librispeech-zipformer-medium-cr-ctc-20241018 \
      --local-dir en_medium                              # checkpoint + tokens

Run:
  KMP_DUPLICATE_LIB_OK=TRUE JAX_PLATFORMS=cpu \
      python build_zipformer_ctc.py {baseline,convert,all} path/to/audio.wav
"""

import collections
import contextlib
import math
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ZF_DIR = os.path.join(HERE, "icefall/egs/librispeech/ASR/zipformer")
CKPT = os.path.join(HERE, "en_medium/exp/pretrained.pt")
TOKENS = os.path.join(HERE, "en_medium/data/lang_bpe_500/tokens.txt")

SR = 16000
T_IN = 1600  # 16 s of 10 ms-hop fbank frames (snip_edges=False)
torch.manual_seed(0)

BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT",
          "SELECT_V2", "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST",
          "EMBEDDING_LOOKUP", "EQUAL", "NOT_EQUAL", "GREATER", "GREATER_EQUAL",
          "LESS", "LOGICAL_AND", "PACK", "SPLIT", "SPLIT_V",
          "RFFT2D", "FFT", "STFT", "COMPLEX", "RFFT", "IRFFT", "CUMSUM"}


# ------------------------------------------------------------------ stubs
def install_stubs():
  """Installs k2 and icefall.utils stubs so the model code imports without k2.

  The model code imports k2 only for the Swoosh activation fast-paths,
  and the pure-torch formulas below are numerically identical at
  inference. icefall.utils is stubbed with the two helpers the eval path uses.
  """
  k2 = types.ModuleType("k2")

  def _sp(z):
    return torch.relu(z) + torch.log1p(torch.exp(-torch.abs(z)))

  def _swoosh_l(x):
    return _sp(x - 4.0) - 0.08 * x - 0.035

  def _swoosh_r(x):
    return _sp(x - 1.0) - 0.08 * x - 0.313261687

  k2.swoosh_l = _swoosh_l
  k2.swoosh_l_forward = _swoosh_l
  k2.swoosh_r = _swoosh_r
  k2.swoosh_r_forward = _swoosh_r

  def _no_deriv(*a, **kw):
    raise RuntimeError("k2 deriv stub hit at inference — should not happen")

  k2.swoosh_l_forward_and_deriv = _no_deriv
  k2.swoosh_r_forward_and_deriv = _no_deriv
  sys.modules["k2"] = k2

  icefall = types.ModuleType("icefall")
  utils = types.ModuleType("icefall.utils")

  @contextlib.contextmanager
  def torch_autocast(*a, **kw):
    yield

  utils.torch_autocast = torch_autocast

  def make_pad_mask(lengths, max_len=0):
    assert lengths.ndim == 1
    max_len = max(max_len, int(lengths.max()))
    n = lengths.size(0)
    rng = torch.arange(max_len, device=lengths.device)
    return rng.unsqueeze(0).expand(n, max_len) >= lengths.unsqueeze(1)

  utils.make_pad_mask = make_pad_mask
  icefall.utils = utils
  sys.modules["icefall"] = icefall
  sys.modules["icefall.utils"] = utils


install_stubs()
sys.path.insert(0, ZF_DIR)

from scaling import ScheduledFloat                      # noqa: E402
from subsampling import Conv2dSubsampling               # noqa: E402
from zipformer import Zipformer2                        # noqa: E402
from scaling_converter import convert_scaled_to_non_scaled  # noqa: E402


# ------------------------------------------------------------------ patches
def apply_gpu_patches():
  """Rewrites GPU-incompatible ops in the icefall Zipformer modules.

  All rewrites are numerically equivalent to the icefall eval path:
    * masked_fill(mask, -1000)    -> additive float bias {0,-1000} (icefall
      itself uses -1000, so additive == fill within fp32)
    * rel-shift as_strided/gather -> pad+reshape+slice (verified exact, diff 0)
    * .chunk()                    -> slices (SPLIT is GPU-banned)
    * scaling.softmax custom fn   -> torch.softmax
    * BiasNorm / SwooshL/R custom autograd -> plain eval math
  """
  import torch.nn.functional as F
  import scaling as sc
  import zipformer as zf

  def softplus_stable(z):
    # Guard-free stable softplus: relu(z) + log1p(exp(-|z|)). The jax
    # logaddexp lowering emits inf-guard SELECT_V2/EQUAL/LOGICAL_AND chains
    # (GPU-banned).
    return torch.relu(z) + torch.log1p(torch.exp(-torch.abs(z)))

  def swoosh_l(x):
    return softplus_stable(x - 4.0) - 0.08 * x - 0.035

  def swoosh_r(x):
    return softplus_stable(x - 1.0) - 0.08 * x - 0.313261687

  sc.SwooshL.forward = lambda self, x: swoosh_l(x)
  sc.SwooshR.forward = lambda self, x: swoosh_r(x)
  sc.SwooshLForward = swoosh_l
  sc.SwooshRForward = swoosh_r

  # Was x.chunk(1)[0] -> aten.split_with_sizes, which the jax bridge cannot
  # lower ('list' object has no attribute 'dtype').
  sc._no_op = lambda x: x
  sc.Identity.forward = lambda self, x: x

  def biasnorm_forward(self, x):
    channel_dim = self.channel_dim
    if channel_dim < 0:
      channel_dim += x.ndim
    bias = self.bias
    for _ in range(channel_dim + 1, x.ndim):
      bias = bias.unsqueeze(-1)
    scales = (
        torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True) ** -0.5
    ) * self.log_scale.exp()
    return x * scales

  sc.BiasNorm.forward = biasnorm_forward

  zf.softmax = lambda x, dim: torch.softmax(x, dim=dim)

  def attn_forward(self, x, pos_emb, key_padding_mask=None, attn_mask=None):
    assert attn_mask is None
    x = self.in_proj(x)
    qhd, phd, H = self.query_head_dim, self.pos_head_dim, self.num_heads
    T, B, _ = x.shape
    qd = qhd * H
    q = x[..., 0:qd].reshape(T, B, H, qhd).permute(2, 1, 0, 3)
    k = x[..., qd:2 * qd].reshape(T, B, H, qhd).permute(2, 1, 3, 0)
    p = x[..., 2 * qd:].reshape(T, B, H, phd).permute(2, 1, 0, 3)
    attn_scores = torch.matmul(q, k)                    # (H,B,T,T)
    pos_emb = self.linear_pos(pos_emb)
    L = 2 * T - 1
    pos_emb = pos_emb.reshape(-1, L, H, phd).permute(2, 0, 3, 1)
    pos_scores = torch.matmul(p, pos_emb)               # (H,B,T,2T-1)
    # rel->abs shift, pad+reshape+slice (== as_strided path, verified exact)
    pos_scores = F.pad(pos_scores, (0, 1)).reshape(H, B, T * 2 * T)
    pos_scores = pos_scores[..., :T * 2 * T - T].reshape(H, B, T, 2 * T - 1)
    attn_scores = attn_scores + pos_scores[..., T - 1:]
    if key_padding_mask is not None:  # float (B,T): 0 valid / -1000 pad
      attn_scores = attn_scores + key_padding_mask.unsqueeze(1)
    return torch.softmax(attn_scores, dim=-1)

  zf.RelPositionMultiheadAttentionWeights.forward = attn_forward

  def nonlin_forward(self, x, attn_weights):
    x = self.in_proj(x)
    T, B, _ = x.shape
    h = self.hidden_channels
    s, x, y = x[..., :h], x[..., h:2 * h], x[..., 2 * h:]
    s = self.tanh(self.balancer(s))
    x = self.whiten1(x) * s
    x = self.identity1(x)
    H = attn_weights.shape[0]
    x = x.reshape(T, B, H, -1).permute(2, 1, 0, 3)
    x = torch.matmul(attn_weights, x)
    x = x.permute(2, 1, 0, 3).reshape(T, B, -1)
    x = x * self.identity2(y)
    x = self.identity3(x)
    return self.whiten2(self.out_proj(x))

  zf.NonlinAttention.forward = nonlin_forward

  def conv_forward(self, x, src_key_padding_mask=None, chunk_size=-1):
    x = self.in_proj(x)                                 # (T,B,2C)
    C = x.shape[-1] // 2
    x, s = x[..., :C], x[..., C:]
    s = self.sigmoid(self.balancer1(s))
    x = self.activation1(x) * s
    x = self.activation2(x)
    x = x.permute(1, 2, 0)                              # (B,C,T)
    if src_key_padding_mask is not None:  # gate {1,0} == masked_fill(..., 0)
      x = x * (1.0 + src_key_padding_mask.unsqueeze(1) / 1000.0)
    x = self.depthwise_conv(x)
    x = self.balancer2(x)
    x = x.permute(2, 0, 1)
    return self.out_proj(self.whiten(x))

  zf.ConvolutionModule.forward = conv_forward

  import subsampling as sub

  def embed_forward(self, x, x_lens):
    # icefall body minus the `assert ... == x_lens.max().item()` data-dependent
    # guard (GuardOnDataDependentSymNode at torch.export).
    x = x.unsqueeze(1)
    x = self.conv(x)
    x = self.convnext(x)
    b, c, t, f = x.size()
    x = x.transpose(1, 2).reshape(b, t, c * f)
    x = self.out(x)
    x = self.out_whiten(x)
    x = self.out_norm(x)
    x = self.dropout(x)
    return x, (x_lens - 7) // 2

  sub.Conv2dSubsampling.forward = embed_forward

  def cpe_forward(self, x, left_context_len=0):
    assert left_context_len == 0
    if self.pe is None or self.pe.size(0) < 2 * x.size(0) - 1:
      # Eager pre-warm only, a no-op at export.
      self.extend_pe(x, left_context_len)
    pe = self.pe
    T = x.size(0)
    start = pe.size(0) // 2 - T + 1
    end = pe.size(0) // 2 + T
    return pe[start:end].unsqueeze(0)      # dropout is a no-op in eval

  zf.CompactRelPositionalEncoding.forward = cpe_forward

  # expand -> cat-repeat (expand lowers to BROADCAST_TO, GPU-banned)
  def upsample_forward(self, src):
    upsample = self.upsample
    T, B, C = src.shape
    src = torch.cat([src.unsqueeze(1)] * upsample, dim=1)
    return src.reshape(T * upsample, B, C)

  zf.SimpleUpsample.forward = upsample_forward

  def downsample_forward(self, src):
    T, B, C = src.shape
    ds = self.downsample
    d_seq_len = (T + ds - 1) // ds
    pad = d_seq_len * ds - T
    if pad > 0:
      src = torch.cat([src] + [src[T - 1:]] * pad, dim=0)
    src = src.reshape(d_seq_len, ds, B, C)
    if not hasattr(self, "_w"):
      # Bake softmax(bias): as a live op it stays a rank-1 SOFTMAX on a
      # constant, which ML Drift rejects (device "Failed to compile model").
      self._w = self.bias.detach().softmax(dim=0).reshape(1, ds, 1, 1)
    return (src * self._w).sum(dim=1)

  zf.SimpleDownsample.forward = downsample_forward

  # Masks arrive as a per-rate tuple (ds 1,2,4,8) instead of in-graph
  # [..., ::ds] strided slicing, which lowers to GATHER_ND.
  def zip2_forward(self, x, x_lens, src_key_padding_mask=None):
    ds2idx = {1: 0, 2: 1, 4: 2, 8: 3}
    outputs = []
    for i, module in enumerate(self.encoders):
      ds = self.downsampling_factor[i]
      x = zf.convert_num_channels(x, self.encoder_dim[i])
      m = (None if src_key_padding_mask is None
           else src_key_padding_mask[ds2idx[ds]])
      x = module(x, chunk_size=-1, feature_mask=1.0,
                 src_key_padding_mask=m, attn_mask=None)
      outputs.append(x)
    x = self._get_full_dim_output(outputs)
    x = self.downsample_output(x)
    return x, (x_lens + 1) // 2

  zf.Zipformer2.forward = zip2_forward


# ------------------------------------------------------------------ model
def build_model():
  """Builds the medium Zipformer CR-CTC model and loads the checkpoint.

  Returns:
    The eval-mode ZipformerCtc wrapper module (fbank + 4 bias inputs ->
    CTC logits).
  """
  encoder_embed = Conv2dSubsampling(
      in_channels=80, out_channels=192,
      dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)))
  encoder = Zipformer2(
      output_downsampling_factor=2,
      downsampling_factor=(1, 2, 4, 8, 4, 2),
      num_encoder_layers=(2, 2, 3, 4, 3, 2),
      encoder_dim=(192, 256, 384, 512, 384, 256),
      encoder_unmasked_dim=(192, 192, 256, 256, 256, 192),
      query_head_dim=(32,),
      pos_head_dim=(4,),
      value_head_dim=(12,),
      pos_dim=48,
      num_heads=(4, 4, 4, 8, 4, 4),
      feedforward_dim=(512, 768, 1024, 1536, 1024, 768),
      cnn_module_kernel=(31, 31, 15, 15, 15, 31),
      dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
      warmup_batches=4000.0,
      causal=False,
      chunk_size=(-1,),
      left_context_frames=(-1,),
  )
  ctc_output = nn.Sequential(
      nn.Dropout(p=0.1),
      nn.Linear(512, 500),
      nn.LogSoftmax(dim=-1),
  )

  sd = torch.load(CKPT, map_location="cpu", weights_only=False)
  sd = sd["model"] if "model" in sd else sd
  ee = {k[len("encoder_embed."):]: v for k, v in sd.items()
        if k.startswith("encoder_embed.")}
  en = {k[len("encoder."):]: v for k, v in sd.items()
        if k.startswith("encoder.")}
  ct = {k[len("ctc_output."):]: v for k, v in sd.items()
        if k.startswith("ctc_output.")}
  encoder_embed.load_state_dict(ee, strict=True)
  encoder.load_state_dict(en, strict=True)
  ctc_output.load_state_dict(ct, strict=True)
  print(f"[build] loaded: embed {len(ee)} keys, encoder {len(en)} keys, "
        f"ctc {len(ct)} keys")

  class ZipformerCtc(nn.Module):

    def __init__(self):
      super().__init__()
      self.encoder_embed = encoder_embed
      self.encoder = encoder
      self.ctc_output = ctc_output
      self.register_buffer("x_lens", torch.tensor([T_IN], dtype=torch.int64))

    def forward(self, x, b1, b2, b4, b8):
      # x: [1,T_IN,80]; b*: [1,ceil(T50/ds)] with 0=valid / -1000=pad.
      x, x_lens = self.encoder_embed(x, self.x_lens)
      x = x.permute(1, 0, 2)                  # (T', N, C)
      out, _ = self.encoder(x, x_lens, (b1, b2, b4, b8))
      out = out.permute(1, 0, 2)              # (N, T'', C)
      return self.ctc_output(out)             # log_probs [1, T'', 500]

  return ZipformerCtc().eval()


# ------------------------------------------------------------------ features
def fbank(wav_path):
  """Computes the padded 16 s kaldi-fbank window for a wav file.

  Args:
    wav_path: Path to the input audio (any rate, resampled to 16 kHz mono).

  Returns:
    Float tensor [1, T_IN, 80]. The wave stays in [-1, 1] scale and the
    features match icefall's kaldifeat options (povey window,
    snip_edges=False, high_freq=-400, dither 0, no CMN).
  """
  import torchaudio
  wave, sr = torchaudio.load(wav_path)
  if sr != SR:
    wave = torchaudio.functional.resample(wave, sr, SR)
  wave = wave.mean(0, keepdim=True)
  feats = torchaudio.compliance.kaldi.fbank(
      wave, num_mel_bins=80, sample_frequency=SR, dither=0.0,
      snip_edges=False, high_freq=-400.0)     # [T, 80]
  T = feats.shape[0]
  if T < T_IN:
    pad = feats.new_full((T_IN - T, 80), math.log(1e-10))
    feats = torch.cat([feats, pad], 0)
  else:
    feats = feats[:T_IN]
  return feats.unsqueeze(0)                   # [1, T_IN, 80]


def load_tokens():
  """Loads the BPE-500 id -> token table from tokens.txt."""
  id2tok = {}
  for line in open(TOKENS, encoding="utf-8"):
    tok, idx = line.rsplit(maxsplit=1)
    id2tok[int(idx)] = tok
  return id2tok


def greedy_ctc(log_probs, id2tok):
  """Greedy CTC decode (blank id 0, drop repeats) + BPE detokenization.

  Args:
    log_probs: [1, T, 500] logits or log-probs (argmax-invariant).
    id2tok: id -> token table from load_tokens().

  Returns:
    The decoded transcript string.
  """
  ids = log_probs[0].argmax(-1)
  out, prev = [], -1
  for i in ids.tolist():
    if i != prev and i != 0:
      out.append(id2tok.get(i, "?"))
    prev = i
  return "".join(out).replace("▁", " ").strip()


def valid_frames(wav_path):
  """Counts the real (non-padding) frames of a wav at each pipeline rate.

  Args:
    wav_path: Path to the input audio.

  Returns:
    Tuple (fbank_frames, 50Hz_frames_after_embed, 25Hz_output_frames).
  """
  import torchaudio
  wave, sr = torchaudio.load(wav_path)
  n = wave.shape[1] * SR // sr
  t_fb = n // 160 + 1                          # snip_edges=False, 10 ms hop
  t50 = (t_fb - 7) // 2
  return t_fb, t50, (t50 + 1) // 2


def make_bias(t50_valid, t50_total):
  """Builds the per-rate additive biases (0 valid / -1000 pad).

  The in-graph [..., ::ds] mask slicing lowers to GATHER_ND, so the rate
  slicing happens here on the host instead.

  Args:
    t50_valid: Number of valid 50 Hz frames for the real audio.
    t50_total: Total 50 Hz frames of the padded window.

  Returns:
    Tuple of four [1, ceil(t50_total/ds)] tensors for ds 1, 2, 4, 8.
  """
  b = torch.full((1, t50_total), -1000.0)
  b[0, :t50_valid] = 0.0
  return (b, b[:, ::2].contiguous(), b[:, ::4].contiguous(),
          b[:, ::8].contiguous())


# ------------------------------------------------------------------ stages
def stage_gold(m, wav):
  """Runs the unpadded input through the ORIGINAL icefall eval path.

  Args:
    m: The unpatched model from build_model().
    wav: Path to the test audio.

  Returns:
    Tuple (gold log-probs, gold transcript) — the reference every later
    stage is compared against.
  """
  from icefall.utils import make_pad_mask
  import torchaudio
  wave, sr = torchaudio.load(wav)
  wave = wave.mean(0, keepdim=True)
  feats = torchaudio.compliance.kaldi.fbank(
      wave, num_mel_bins=80, sample_frequency=SR, dither=0.0,
      snip_edges=False, high_freq=-400.0).unsqueeze(0)
  T = feats.shape[1]
  with torch.no_grad():
    e, elens = m.encoder_embed(feats, torch.tensor([T]))
    mask = make_pad_mask(elens)
    out, _ = m.encoder(e.permute(1, 0, 2), elens, mask)
    lp = m.ctc_output(out.permute(1, 0, 2))
  text = greedy_ctc(lp, load_tokens())
  print(f"[gold] T_fbank {T} -> log_probs {tuple(lp.shape)}")
  print(f"[gold] TEXT: {text}")
  return lp, text


def stage_patched_parity(m, gold_lp, gold_text, wav):
  """Checks the patched model reproduces the gold transcript.

  Runs the padded 16 s window + additive biases through the patched model
  and saves the inputs and logits as the .npy reference set.

  Args:
    m: The model after apply_gpu_patches().
    gold_lp: Gold log-probs from stage_gold().
    gold_text: Gold transcript from stage_gold().
    wav: Path to the test audio.

  Returns:
    Tuple (fbank input tensor, bias tensor tuple) for the converter.
  """
  x = fbank(wav)
  t_fb, t50v, t25v = valid_frames(wav)
  t50_total = (T_IN - 7) // 2
  biases = make_bias(t50v, t50_total)
  with torch.no_grad():
    lp = m(x, *biases)  # raw logits — LogSoftmax runs host-side
  text = greedy_ctc(lp, load_tokens())
  lsm = torch.log_softmax(lp, dim=-1)
  nv = min(t25v, gold_lp.shape[1], lp.shape[1])
  corr = np.corrcoef(lsm[0, :nv].numpy().ravel(),
                     gold_lp[0, :nv].numpy().ravel())[0, 1]
  md = np.abs(lsm[0, :nv].numpy() - gold_lp[0, :nv].numpy()).max()
  print(f"[patched] valid-region vs gold: corr {corr:.6f} max|diff| {md:.4f}")
  print(f"[patched] TEXT: {text}")
  print(f"[patched] text match gold: {text == gold_text}")
  np.save(os.path.join(HERE, "ref_in.npy"), x.numpy())
  for i, b in enumerate(biases):
    np.save(os.path.join(HERE, f"ref_bias{i}.npy"), b.numpy())
  np.save(os.path.join(HERE, "ref_logprobs.npy"), lp.numpy())
  return x, biases


def stage_convert(m, x, biases):
  """Converts with litert-torch, op-checks, and verifies parity.

  Args:
    m: The patched model.
    x: Fbank input tensor from stage_patched_parity().
    biases: Bias tensor tuple from stage_patched_parity().

  Returns:
    True when the converted graph is GPU-clean.
  """
  import litert_torch  # import AFTER model construction
  out = os.path.join(HERE, "zipformer_ctc.tflite")
  cm = litert_torch.convert(m, (x, *biases))
  cm.export(out)
  print(f"[convert] wrote {out}")
  clean = opcheck(out, "zip")
  lp_t = run_tflite(out, x.numpy(), [b.numpy() for b in biases])
  ref = np.load(os.path.join(HERE, "ref_logprobs.npy"))
  corr = np.corrcoef(lp_t.ravel(), ref.ravel())[0, 1]
  md = np.abs(lp_t - ref).max()
  print(f"[convert] tflite vs patched-torch: corr {corr:.6f} "
        f"max|diff| {md:.4f}")
  text = greedy_ctc(torch.from_numpy(lp_t), load_tokens())
  print(f"[convert] TFLITE TEXT: {text}")
  if clean:
    quant_fp16(out)
  return clean


def quant_fp16(path):
  """Quantizes to fp16 weights and re-verifies parity and the op set.

  Args:
    path: Path to the fp32 .tflite. The fp16 file is written beside it.
  """
  from ai_edge_quantizer import quantizer, recipe_manager
  from ai_edge_quantizer.recipe import AlgorithmName, qtyping
  rm = recipe_manager.RecipeManager()
  rm.add_quantization_config(
      regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
      op_config=qtyping.OpQuantizationConfig(
          weight_tensor_config=qtyping.TensorQuantizationConfig(
              num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
          compute_precision=qtyping.ComputePrecision.FLOAT),
      algorithm_key=AlgorithmName.FLOAT_CASTING)
  out = path.replace(".tflite", "_fp16.tflite")
  if os.path.exists(out):
    os.remove(out)
  qt = quantizer.Quantizer(float_model=path)
  qt.load_quantization_recipe(rm.get_quantization_recipe())
  qt.quantize().export_model(out)
  print(f"[fp16] wrote {out}")
  opcheck(out, "zip_fp16")
  x = np.load(os.path.join(HERE, "ref_in.npy"))
  biases = [np.load(os.path.join(HERE, f"ref_bias{i}.npy")) for i in range(4)]
  ref = np.load(os.path.join(HERE, "ref_logprobs.npy"))
  lp = run_tflite(out, x, biases)
  corr = np.corrcoef(lp.ravel(), ref.ravel())[0, 1]
  md = np.abs(lp - ref).max()
  print(f"[fp16] vs torch: corr {corr:.6f} max|diff| {md:.4f}")
  text = greedy_ctc(torch.from_numpy(lp), load_tokens())
  print(f"[fp16] TEXT: {text}")


def opcheck(path, label):
  """Static GPU-compat scan: reads the op set from the .tflite file.

  Note: this catches the standard banned set only. Two Zipformer ops passed
  this scan but were rejected by the on-device GPU compile (rank-1 SOFTMAX on
  a constant, REDUCE_MAX from LogSoftmax) — see README.md. Both are already
  rewritten away in this script.

  Args:
    path: Path to the .tflite flatbuffer.
    label: Log label prefix.

  Returns:
    True when no banned op, >4D tensor, or FFT-family op is found.
  """
  from ai_edge_litert import schema_py_generated as schema
  with open(path, "rb") as f:
    model = schema.ModelT.InitFromPackedBuf(f.read(), 0)
  names = {v: k for k, v in vars(schema.BuiltinOperator).items()
           if not k.startswith("_")}
  ops = collections.Counter()
  over = 0
  for g in model.subgraphs:
    for op in g.operators:
      c = model.operatorCodes[op.opcodeIndex]
      code = max(c.builtinCode, c.deprecatedBuiltinCode)
      if c.customCode:
        key = c.customCode.decode()
      else:
        key = names.get(code, str(code))
      ops[key] += 1
    over += sum(1 for t in g.tensors
                if t.shape is not None and len(t.shape) > 4)
  bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
  fft = {k: v for k, v in ops.items()
         if any(t in k.upper() for t in ("FFT", "STFT", "COMPLEX"))}
  print(f"[{label}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
  print(f"[{label}] banned:{bad or 'NONE'} >4D:{over} FFT:{fft or 'NONE'} "
        f"size {os.path.getsize(path)/1e6:.1f}MB")
  clean = not bad and not over and not fft
  print(f"[{label}] VERDICT:",
        "GPU-CLEAN" if clean else f"BLOCKERS {bad}")
  return clean


def run_tflite(path, x, biases):
  """Single inference through the LiteRT CompiledModel API.

  The five inputs are matched by element count (fbank 128000, biases
  796/398/199/100), the same capacity-matching the Android runner uses, so
  the signature input order does not matter.

  Args:
    path: Path to the .tflite file.
    x: Fbank input array [1, T_IN, 80].
    biases: List of the four bias arrays.

  Returns:
    The output logits as a [1, T_out, 500] fp32 array.
  """
  from ai_edge_litert.compiled_model import CompiledModel
  model = CompiledModel.from_file(path)
  inputs = model.create_input_buffers(0)
  outputs = model.create_output_buffers(0)
  by_elems = {b.size: b for b in biases}
  for i in range(len(inputs)):
    req = model.get_input_buffer_requirements(i)
    n = req["buffer_size"] // np.dtype(np.float32).itemsize
    src = x if n == x.size else by_elems[n]
    inputs[i].write(np.ascontiguousarray(src, dtype=np.float32))
  model.run_by_index(0, inputs, outputs)
  req = model.get_output_buffer_requirements(0)
  n = req["buffer_size"] // np.dtype(np.float32).itemsize
  return outputs[0].read(n, np.float32).reshape(1, -1, 500)


if __name__ == "__main__":
  stage = sys.argv[1] if len(sys.argv) > 1 else "all"
  if len(sys.argv) < 3:
    sys.exit("usage: build_zipformer_ctc.py {baseline,convert,all} audio.wav")
  wav_path = sys.argv[2]
  m = build_model()
  gold_lp, gold_text = stage_gold(m, wav_path)
  apply_gpu_patches()
  convert_scaled_to_non_scaled(m, inplace=True, is_onnx=False)
  # LogSoftmax lowers to REDUCE_MAX (rank-drop) which ML Drift rejects on
  # device; greedy CTC is argmax-invariant, so emit raw logits and
  # log-softmax host-side when log-probs are needed.
  m.ctc_output[2] = nn.Identity()
  x, biases = stage_patched_parity(m, gold_lp, gold_text, wav_path)
  if stage in ("convert", "all"):
    ok = stage_convert(m, x, biases)
    print("[done] GPU-CLEAN" if ok else "[done] blockers remain")
  # Skip the torch/jax atexit teardown (it can hang), but flush first —
  # os._exit drops buffered stdout when the output is piped.
  sys.stdout.flush()
  os._exit(0)
