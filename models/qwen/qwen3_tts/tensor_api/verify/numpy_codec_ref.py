# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent numpy reference for the codec decoder equivalence check.

Implements the Qwen3-TTS speech_tokenizer DECODER directly from
model.safetensors (no torch), mirroring qtok12/modeling_*.py semantics:
RVQ (EMA codebooks) -> causal pre_conv -> pre_transformer (rope 1e4,
sliding-window-72 causal, LayerScale) -> TransConv+ConvNeXt upsampling ->
Snake/TransConv/ResidualUnit decoder stack -> clamp. Replicates codec_main's
production-style chunking (64-frame chunks, 25 frames left context, fixed
89-frame window, right-zero-padded) so waveforms compare 1:1.

Usage:
  # 1. write deterministic codes for both sides
  python3 numpy_codec_ref.py gen --out /tmp/cc --frames 128

  # 2. codec_main --codes_file=/tmp/cc/codes.i32 --frames=128 \
  #      --dump_wav=/tmp/cc/wav_cpp.f32 ...

  # 3. compare
  python3 numpy_codec_ref.py check --weights .../speech_tokenizer/model.safetensors \
      --codes /tmp/cc/codes.i32 --dump /tmp/cc/wav_cpp.f32 --frames 128
"""

import argparse
import json
import math
import struct
import sys

import numpy as np

NQ, BINS, VQ_DIM, CB_DIM = 16, 2048, 256, 512
LATENT, DEC_DIM = 1024, 1536
TF_L, TF_H, TF_HD, TF_HID, TF_FFN = 8, 16, 64, 512, 1024
ROPE_THETA, WINDOW, EPS = 1e4, 72, 1e-5
UP_RATIOS, UP_RATES = [2, 2], [8, 5, 4, 3]
CHUNK, CTX = 64, 25
UPSAMPLE = 1920
MASK_FLOOR = -30000.0


def load_tensors(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, info in header.items():
        if name == '__metadata__' or not name.startswith('decoder.'):
            continue
        s0, s1 = info['data_offsets']
        if info['dtype'] == 'BF16':
            raw = np.frombuffer(data, '<u2', count=(s1 - s0) // 2, offset=s0)
            arr = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
        elif info['dtype'] == 'F32':
            arr = np.frombuffer(data, '<f4', count=(s1 - s0) // 4, offset=s0)
        else:
            raise ValueError(f'{name}: {info["dtype"]}')
        out[name] = arr.reshape(info['shape']).astype(np.float32)
    return out


# ---- primitives (all [C, T] channel-first like torch) ----

def causal_conv(x, w, b, dilation=1):
    """x [Cin,T], torch weight [Cout,Cin,k] -> [Cout,T]."""
    cout, cin, k = w.shape
    pad = (k - 1) * dilation
    xp = np.pad(x, ((0, 0), (pad, 0)))
    T = x.shape[1]
    out = np.zeros((cout, T), np.float32)
    for j in range(k):
        out += w[:, :, j] @ xp[:, j * dilation:j * dilation + T]
    return out + b[:, None]


def causal_dwconv(x, w, b):
    """x [C,T], torch weight [C,1,k] -> [C,T]."""
    c, _, k = w.shape
    xp = np.pad(x, ((0, 0), (k - 1, 0)))
    T = x.shape[1]
    out = np.zeros((c, T), np.float32)
    for j in range(k):
        out += w[:, 0, j:j + 1] * xp[:, j:j + T]
    return out + b[:, None]


def causal_transconv(x, w, b, stride):
    """x [Cin,T], torch weight [Cin,Cout,k] -> [Cout,T*stride] (crop k-s)."""
    cin, cout, k = w.shape
    T = x.shape[1]
    full = (T - 1) * stride + k
    out = np.zeros((cout, full), np.float32)
    for j in range(k):
        out[:, j:j + (T - 1) * stride + 1:stride] += w[:, :, j].T @ x
    out += b[:, None]
    crop = k - stride
    return out[:, :full - crop] if crop else out


def snake(x, alpha, beta):
    a = np.exp(alpha)[:, None]
    bi = 1.0 / (np.exp(beta)[:, None] + 1e-9)
    return x + bi * np.square(np.sin(x * a))


def gelu(x):
    from scipy.special import erf  # noqa: F401  (fallback below if absent)
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


try:
    from scipy.special import erf as _erf  # noqa: F401
except ImportError:  # exact erf via math, vectorized
    _verf = np.vectorize(math.erf)

    def gelu(x):  # noqa: F811
        return 0.5 * x * (1.0 + _verf(x / np.sqrt(2.0)).astype(np.float32))


def rms(x, w):
    v = np.mean(np.square(x), axis=-1, keepdims=True, dtype=np.float32)
    return x * (1.0 / np.sqrt(v + EPS)) * w


def rope_tables(T):
    half = TF_HD // 2
    freqs = ROPE_THETA ** (-2.0 * np.arange(half) / TF_HD)
    ang = np.arange(T, dtype=np.float64)[:, None] * freqs[None, :]
    cos = np.concatenate([np.cos(ang), np.cos(ang)], -1).astype(np.float32)
    sin = np.concatenate([np.sin(ang), np.sin(ang)], -1).astype(np.float32)
    return cos, sin


def pre_transformer(x, w, base='decoder.pre_transformer'):
    """x [T, LATENT] -> [T, LATENT]."""
    T = x.shape[0]
    x = x @ w[f'{base}.input_proj.weight'].T + w[f'{base}.input_proj.bias']
    cos, sin = rope_tables(T)
    mask = np.full((T, T), MASK_FLOOR, np.float32)
    for i in range(T):
        mask[i, max(0, i - WINDOW + 1):i + 1] = 0.0
    half = TF_HD // 2

    def rope(t):  # t [H, T, hd]
        rot = np.concatenate([-t[..., half:], t[..., :half]], -1)
        return t * cos + rot * sin

    for l in range(TF_L):
        p = f'{base}.layers.{l}'
        h = rms(x, w[f'{p}.input_layernorm.weight'])
        q = (h @ w[f'{p}.self_attn.q_proj.weight'].T).reshape(T, TF_H, TF_HD)
        k = (h @ w[f'{p}.self_attn.k_proj.weight'].T).reshape(T, TF_H, TF_HD)
        v = (h @ w[f'{p}.self_attn.v_proj.weight'].T).reshape(T, TF_H, TF_HD)
        q, k, v = (t.transpose(1, 0, 2) for t in (q, k, v))
        q, k = rope(q), rope(k)
        scores = q @ k.transpose(0, 2, 1) * np.float32(TF_HD ** -0.5) + mask
        m = scores.max(-1, keepdims=True)
        e = np.exp(scores - m)
        attn = (e / e.sum(-1, keepdims=True)) @ v
        attn = attn.transpose(1, 0, 2).reshape(T, TF_H * TF_HD)
        attn = attn @ w[f'{p}.self_attn.o_proj.weight'].T
        x = x + attn * w[f'{p}.self_attn_layer_scale.scale']

        f = rms(x, w[f'{p}.post_attention_layernorm.weight'])
        gate = f @ w[f'{p}.mlp.gate_proj.weight'].T
        up = f @ w[f'{p}.mlp.up_proj.weight'].T
        ffn = (gate / (1.0 + np.exp(-gate)) * up) @ w[
            f'{p}.mlp.down_proj.weight'].T
        x = x + ffn * w[f'{p}.mlp_layer_scale.scale']

    x = rms(x, w[f'{base}.norm.weight'])
    return x @ w[f'{base}.output_proj.weight'].T + w[f'{base}.output_proj.bias']


def convnext(x, w, p):
    """x [C,T]."""
    h = causal_dwconv(x, w[f'{p}.dwconv.conv.weight'],
                      w[f'{p}.dwconv.conv.bias'])
    ht = h.T  # [T,C]
    m = ht.mean(-1, keepdims=True)
    d = ht - m
    v = np.mean(d * d, axis=-1, keepdims=True)
    ht = d / np.sqrt(v + 1e-6) * w[f'{p}.norm.weight'] + w[f'{p}.norm.bias']
    ht = ht @ w[f'{p}.pwconv1.weight'].T + w[f'{p}.pwconv1.bias']
    ht = gelu(ht)
    ht = ht @ w[f'{p}.pwconv2.weight'].T + w[f'{p}.pwconv2.bias']
    ht = ht * w[f'{p}.gamma']
    return x + ht.T


def decode_window(codes_win, w):
    """codes [16, T] -> wav [T*1920]."""
    T = codes_win.shape[1]
    tables = []
    for q in range(NQ):
        base = ('decoder.quantizer.rvq_first.vq.layers.0._codebook' if q == 0
                else f'decoder.quantizer.rvq_rest.vq.layers.{q-1}._codebook')
        usage = np.maximum(w[f'{base}.cluster_usage'], EPS)
        tables.append(w[f'{base}.embedding_sum'] / usage[:, None])
    sem = tables[0][codes_win[0]]  # [T, 256]
    sem = sem @ w['decoder.quantizer.rvq_first.output_proj.weight'][:, :, 0].T
    aco = np.zeros((T, VQ_DIM), np.float32)
    for q in range(1, NQ):
        aco += tables[q][codes_win[q]]
    aco = aco @ w['decoder.quantizer.rvq_rest.output_proj.weight'][:, :, 0].T
    latent = (sem + aco).T  # [512, T]

    h = causal_conv(latent, w['decoder.pre_conv.conv.weight'],
                    w['decoder.pre_conv.conv.bias'])  # [1024, T]
    h = pre_transformer(h.T, w).T  # [1024, T]

    for i, r in enumerate(UP_RATIOS):
        p = f'decoder.upsample.{i}'
        h = causal_transconv(h, w[f'{p}.0.conv.weight'],
                             w[f'{p}.0.conv.bias'], r)
        h = convnext(h, w, f'{p}.1')

    h = causal_conv(h, w['decoder.decoder.0.conv.weight'],
                    w['decoder.decoder.0.conv.bias'])
    for i, rate in enumerate(UP_RATES):
        p = f'decoder.decoder.{i+1}.block'
        h = snake(h, w[f'{p}.0.alpha'], w[f'{p}.0.beta'])
        h = causal_transconv(h, w[f'{p}.1.conv.weight'],
                             w[f'{p}.1.conv.bias'], rate)
        for u, dil in zip((2, 3, 4), (1, 3, 9)):
            r_ = f'{p}.{u}'
            hh = snake(h, w[f'{r_}.act1.alpha'], w[f'{r_}.act1.beta'])
            hh = causal_conv(hh, w[f'{r_}.conv1.conv.weight'],
                             w[f'{r_}.conv1.conv.bias'], dil)
            hh = snake(hh, w[f'{r_}.act2.alpha'], w[f'{r_}.act2.beta'])
            hh = causal_conv(hh, w[f'{r_}.conv2.conv.weight'],
                             w[f'{r_}.conv2.conv.bias'])
            h = h + hh
    last = len(UP_RATES) + 1
    h = snake(h, w[f'decoder.decoder.{last}.alpha'],
              w[f'decoder.decoder.{last}.beta'])
    h = causal_conv(h, w[f'decoder.decoder.{last+1}.conv.weight'],
                    w[f'decoder.decoder.{last+1}.conv.bias'])
    return np.clip(h[0], -1.0, 1.0)


def chunked_decode(codes, w):
    """Mirrors codec_main: fixed window CHUNK+CTX, right-zero-pad, crop."""
    frames = codes.shape[1]
    window = CHUNK + CTX
    wav = []
    start = 0
    while start < frames:
        ctx = min(CTX, start)
        end = min(start + CHUNK, frames)
        win = np.zeros((NQ, window), np.int64)
        n = ctx + (end - start)
        win[:, :n] = codes[:, start - ctx:end]
        out = decode_window(win, w)
        wav.append(out[ctx * UPSAMPLE:(ctx + end - start) * UPSAMPLE])
        start = end
        print(f'  chunk -> {end}/{frames} frames', file=sys.stderr)
    return np.concatenate(wav)


def make_codes(frames):
    """Same LCG as codec_main's synthetic codes."""
    state = np.uint32(12345)
    out = np.empty(NQ * frames, np.int32)
    for i in range(out.size):
        state = np.uint32(state * np.uint32(1664525) + np.uint32(1013904223))
        out[i] = int(state >> np.uint32(8)) % BINS
    return out.reshape(NQ, frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['gen', 'check'])
    ap.add_argument('--out')
    ap.add_argument('--weights')
    ap.add_argument('--codes')
    ap.add_argument('--dump')
    ap.add_argument('--frames', type=int, default=128)
    args = ap.parse_args()

    if args.mode == 'gen':
        codes = make_codes(args.frames)
        codes.tofile(f'{args.out}/codes.i32')
        print(f'wrote {args.out}/codes.i32 {codes.shape}')
        return

    codes = np.fromfile(args.codes, np.int32).reshape(NQ, args.frames)
    w = load_tensors(args.weights)
    ref = chunked_decode(codes, w)
    cpp = np.fromfile(args.dump, np.float32)
    assert cpp.size == ref.size, (cpp.size, ref.size)
    d = np.abs(ref - cpp)
    corr = float(np.corrcoef(ref, cpp)[0, 1])
    print(f'samples {ref.size}: corr={corr:.6f} max|d|={d.max():.4e} '
          f'mean|d|={d.mean():.4e} ref RMS={np.sqrt(np.mean(ref**2)):.4f} '
          f'cpp RMS={np.sqrt(np.mean(cpp**2)):.4f}')
    ok = corr > 0.999
    print('EQUIVALENCE: ' + ('PASS' if ok else 'FAIL'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
