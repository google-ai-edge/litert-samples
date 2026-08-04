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

"""GPTQ quantizer for the Qwen3-TTS talker -> C++-layout-exact int4 blockwise-32.

Produces a companion safetensors holding, for every weight that
talker_weights.cc quantizes (IsQuantizableFcWeight: *_proj.weight,
*codec_head.weight, *.lm_head.*), the GPTQ-quantized codes and scales:

    <name>        I8  [out, in]      codes in [-7, 7]
    <name>.scale  F32 [out, in/32]   fp16-rounded blockwise scales

Scales are STATIC and computed from the original weights exactly as the
C++ RTN does (fp16(amax/7) per 32-block, grid clamped to [-7, 7]); GPTQ
only re-decides the rounding inside each block with Hessian error
compensation. The file therefore drops into the exact TFLite INT4 +
BlockwiseQuantization layout the graph already uses — same kernels, same
speed, better rounding.

Calibration: the fp32 numpy reference rollout (numpy_talker_ref) on real
prompts — dual-track prefill from the checkpoint tokenizer + greedy
decode — recording each linear's input rows exactly once per sequence
position (all prefill rows at step 0, the one new row per decode step;
CP rows once, on the final group forward of each frame).

Usage:
  python3 gptq_quantize.py --weights .../model.safetensors \
      --out .../talker_gptq_int4.safetensors [--steps 64] [--layers 28]
"""

import argparse
import json
import struct
import sys
import time

import numpy as np

import numpy_talker_ref as ref

INT4_BLOCK = ref.INT4_BLOCK

# Extra calibration texts beyond the reference sentence (diversity: numbers,
# punctuation, longer clauses).
CALIB_TEXTS = [
    ref.REAL_TEXT,
    'The quick brown fox jumps over the lazy dog, while seventeen quiet '
    'engineers watch the rain outside.',
    'On Tuesday morning we measured latency on the phone: ninety-nine '
    'milliseconds per frame, which is fast enough for streaming speech.',
    'Really? I could hardly believe it — the tiny model spoke clearly, '
    'paused naturally, and even laughed a little at the right moment!',
    'Numbers to read aloud: three hundred twelve, forty-seven point five, '
    'nineteen eighty-four, and one million two hundred thousand.',
]


def is_quantizable(name):
    return ('_proj.weight' in name or name.endswith('codec_head.weight')
            or '.lm_head.' in name)


def hkey(name):
    """Maps a quantizable weight name to its shared Hessian key (weights
    fed by the same input rows share one accumulator)."""
    if name.endswith('codec_head.weight'):
        return 'codec_head_in'
    if '.lm_head.' in name:
        return name + ':in'  # each group head sees a distinct row
    layer = name.rsplit('.', 3)[0]  # e.g. talker.model.layers.0
    if name.endswith(('q_proj.weight', 'k_proj.weight', 'v_proj.weight')):
        return layer + ':attn_in'
    if name.endswith('o_proj.weight'):
        return layer + ':o_in'
    if name.endswith(('gate_proj.weight', 'up_proj.weight')):
        return layer + ':mlp_in'
    if name.endswith('down_proj.weight'):
        return layer + ':down_in'
    raise ValueError(name)


class HessianAcc:
    def __init__(self):
        self.H = {}
        self.rows = {}

    def add(self, key, x):
        x = np.asarray(x, np.float32)
        if x.ndim == 1:
            x = x[None]
        if key not in self.H:
            self.H[key] = x.T @ x
            self.rows[key] = x.shape[0]
        else:
            self.H[key] += x.T @ x
            self.rows[key] += x.shape[0]


def backbone_forward_rec(x, w, base, n_layers, positions, acc, rec_from):
    """ref.backbone_forward with input recording; records rows [rec_from:]
    of each linear's input (each sequence position exactly once across the
    rollout's repeated re-forwards)."""
    S = x.shape[0]
    cos, sin = ref.rope_tables(positions)
    causal = np.full((S, S), ref.MASK_FLOOR, np.float32)
    causal[np.tril_indices(S)] = 0.0

    for i in range(n_layers):
        p = f'{base}.layers.{i}'
        h = ref.rms(x, w[f'{p}.input_layernorm.weight'])
        acc.add(f'{p}:attn_in', h[rec_from:])
        q = (h @ w[f'{p}.self_attn.q_proj.weight'].T).reshape(
            S, ref.HEADS, ref.HD)
        k = (h @ w[f'{p}.self_attn.k_proj.weight'].T).reshape(S, ref.KV,
                                                              ref.HD)
        v = (h @ w[f'{p}.self_attn.v_proj.weight'].T).reshape(S, ref.KV,
                                                              ref.HD)
        q = ref.rms(q, w[f'{p}.self_attn.q_norm.weight']).transpose(1, 0, 2)
        k = ref.rms(k, w[f'{p}.self_attn.k_norm.weight']).transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)
        q = ref.apply_rope(q, cos, sin)
        k = ref.apply_rope(k, cos, sin)
        k = np.repeat(k, ref.HEADS // ref.KV, axis=0)
        v = np.repeat(v, ref.HEADS // ref.KV, axis=0)
        scores = q @ k.transpose(0, 2, 1) * np.float32(ref.HD ** -0.5) + causal
        o = ref.softmax(scores) @ v
        o = o.transpose(1, 0, 2).reshape(S, ref.HEADS * ref.HD)
        acc.add(f'{p}:o_in', o[rec_from:])
        x = x + o @ w[f'{p}.self_attn.o_proj.weight'].T

        h2 = ref.rms(x, w[f'{p}.post_attention_layernorm.weight'])
        acc.add(f'{p}:mlp_in', h2[rec_from:])
        ff = (ref.silu(h2 @ w[f'{p}.mlp.gate_proj.weight'].T) *
              (h2 @ w[f'{p}.mlp.up_proj.weight'].T))
        acc.add(f'{p}:down_in', ff[rec_from:])
        x = x + ff @ w[f'{p}.mlp.down_proj.weight'].T
    return ref.rms(x, w[f'{base}.norm.weight'])


def predict_frame_rec(h_last, logits, w, acc):
    """ref.predict_frame with recording: lm_head inputs per group, CP layer
    inputs once (on the final group forward, whose per-position hiddens
    equal the earlier forwards' — causal prefix stability)."""
    cb0 = int(np.argmax(logits + ref.cb0_bias()))
    cb0_emb = w['talker.model.codec_embedding.weight'][cb0]
    ids = [cb0]
    tokens = [h_last, cb0_emb]
    feedback = cb0_emb.copy()
    for g in range(ref.GROUPS - 1):
        seq = np.stack(tokens)
        last = g == ref.GROUPS - 2
        if last:
            h = backbone_forward_rec(seq, w, 'talker.code_predictor.model',
                                     ref.CP_LAYERS, np.arange(len(tokens)),
                                     acc, rec_from=0)[-1]
        else:
            h = ref.backbone_forward(seq, w, 'talker.code_predictor.model',
                                     ref.CP_LAYERS,
                                     np.arange(len(tokens)))[-1]
        acc.add(f'talker.code_predictor.lm_head.{g}.weight:in', h)
        lg = h @ w[f'talker.code_predictor.lm_head.{g}.weight'].T
        idg = int(np.argmax(lg))
        ids.append(idg)
        emb = w[f'talker.code_predictor.model.codec_embedding.{g}.weight'][idg]
        feedback = feedback + emb
        if g < ref.GROUPS - 2:
            tokens.append(emb)
    return ids, feedback


def collect_hessians(weights_path, w, layers, steps, texts):
    acc = HessianAcc()
    for ti, text in enumerate(texts):
        prefill, track = ref.build_real_prompt(weights_path, steps, text=text)
        seq = [row for row in prefill]
        feedback = None
        for step in range(steps + 1):
            if step > 0:
                seq.append(feedback + track[step - 1])
            rec_from = 0 if step == 0 else len(seq) - 1
            hidden = backbone_forward_rec(np.stack(seq), w, 'talker.model',
                                          layers, np.arange(len(seq)), acc,
                                          rec_from)
            h_last = hidden[-1]
            acc.add('codec_head_in', h_last)
            logits = h_last @ w['talker.codec_head.weight'].T
            ids, feedback = predict_frame_rec(h_last, logits, w, acc)
            if step % 16 == 0:
                print(f'calib text {ti}: frame {step}/{steps} cb0={ids[0]}',
                      file=sys.stderr)
    return acc


def static_scales(W):
    """Blockwise-32 scales exactly as talker_weights.cc: fp16(amax/7)."""
    out, inner = W.shape
    blocks = W.reshape(out, inner // INT4_BLOCK, INT4_BLOCK)
    amax = np.abs(blocks).max(axis=2)
    scale = np.where(amax > 0, amax / 7.0, 1.0)
    return scale.astype(np.float16).astype(np.float32)  # [out, blocks]


def weighted_err(E, H):
    """Mean per-row quadratic error e H e^T (proxy for output MSE)."""
    return float(np.einsum('oi,oi->', E @ H, E) / E.shape[0])


def gptq_layer(W_orig, H, perc_damp=0.01):
    """Returns (q int8 [out,in], scales f32 [out,blocks], loss_gptq,
    loss_rtn). Static scales; column-serial GPTQ with lazy blocked
    trailing updates (the standard 128-column scheme)."""
    out, inn = W_orig.shape
    scales = static_scales(W_orig)
    col_scale = np.repeat(scales, INT4_BLOCK, axis=1)  # [out, in]

    H_eval = ((H + H.T) * 0.5).astype(np.float64)

    # RTN baseline (identical to the C++ in-process quant).
    q_rtn = np.clip(np.round(W_orig / col_scale), -7, 7)
    loss_rtn = weighted_err(
        (W_orig - q_rtn * col_scale).astype(np.float64), H_eval)

    H = H_eval.copy()
    diag = np.einsum('ii->i', H)
    dead = diag == 0.0
    diag[dead] = 1.0
    W = W_orig.astype(np.float32).copy()
    W[:, dead] = 0.0
    damp = perc_damp * float(np.mean(diag))
    diag += damp

    Hinv = np.linalg.inv(H)
    # Upper factor U with H^{-1} = U^T U (torch cholesky(upper=True) == L^T).
    U = np.linalg.cholesky(Hinv).T.astype(np.float32)

    Q = np.zeros((out, inn), np.float32)
    B = 128
    for i1 in range(0, inn, B):
        i2 = min(i1 + B, inn)
        Wb = W[:, i1:i2]
        Eb = np.zeros((out, i2 - i1), np.float32)
        for j in range(i1, i2):
            jj = j - i1
            s = col_scale[:, j]
            qj = np.clip(np.round(Wb[:, jj] / s), -7, 7)
            Q[:, j] = qj
            err = (Wb[:, jj] - qj * s) / U[j, j]
            Eb[:, jj] = err
            Wb[:, jj:] -= err[:, None] * U[j, j:i2][None, :]
        if i2 < inn:
            W[:, i2:] -= Eb @ U[i1:i2, i2:]
    loss_gptq = weighted_err(
        (W_orig - Q * col_scale).astype(np.float64), H_eval)
    return Q.astype(np.int8), scales, loss_gptq, loss_rtn


def load_safetensors_generic(path, names=None):
    """Reads I8/F32/F16/BF16 tensors from a safetensors file."""
    with open(path, 'rb') as f:
        hdr_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(hdr_len))
        data = f.read()
    out = {}
    for name, info in header.items():
        if name == '__metadata__' or (names and name not in names):
            continue
        s0, s1 = info['data_offsets']
        dt = info['dtype']
        if dt == 'I8':
            arr = np.frombuffer(data, np.int8, s1 - s0, s0)
        elif dt == 'F32':
            arr = np.frombuffer(data, np.float32, (s1 - s0) // 4, s0)
        elif dt == 'F16':
            arr = np.frombuffer(data, np.float16,
                                (s1 - s0) // 2, s0).astype(np.float32)
        elif dt == 'BF16':
            raw = np.frombuffer(data, '<u2', (s1 - s0) // 2, s0)
            arr = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
        else:
            raise ValueError(f'{name}: dtype {dt}')
        out[name] = arr.reshape(info['shape']).copy()
    return out


def save_safetensors(path, tensors):
    header = {}
    blobs = []
    offset = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        b = arr.tobytes()
        dt = {'int8': 'I8', 'float32': 'F32'}[arr.dtype.name]
        header[name] = {'dtype': dt, 'shape': list(arr.shape),
                        'data_offsets': [offset, offset + len(b)]}
        blobs.append(b)
        offset += len(b)
    hjson = json.dumps(header).encode()
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(hjson)))
        f.write(hjson)
        for b in blobs:
            f.write(b)


def apply_gptq_fakequant(w, companion_path):
    """Overrides quantizable weights in `w` with dequantized GPTQ values
    (q * scale, blockwise) — the numpy mirror of the C++ prequant path."""
    comp = load_safetensors_generic(companion_path)
    n = 0
    for name in list(w.keys()):
        if name not in comp:
            continue
        q = comp[name].astype(np.float32)
        scale = comp[name + '.scale']
        out, inner = q.shape
        deq = (q.reshape(out, inner // INT4_BLOCK, INT4_BLOCK) *
               scale[:, :, None]).reshape(out, inner)
        w[name] = deq.astype(np.float32)
        n += 1
    if n == 0:
        raise RuntimeError('no overlapping tensors found in companion')
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--layers', type=int, default=28)
    ap.add_argument('--steps', type=int, default=64,
                    help='decode frames per calibration text')
    ap.add_argument('--damp', type=float, default=0.01)
    ap.add_argument('--num-texts', type=int, default=len(CALIB_TEXTS),
                    help='calibration texts to use (smoke tests)')
    ap.add_argument('--report', help='JSON path for per-layer loss report')
    args = ap.parse_args()

    names = ref.weight_names(args.layers)
    print(f'loading {len(names)} tensors...', file=sys.stderr)
    w = ref.load_bf16_tensors(args.weights, names)

    t0 = time.time()
    acc = collect_hessians(args.weights, w, args.layers, args.steps,
                           CALIB_TEXTS[:args.num_texts])
    print(f'calibration done in {time.time() - t0:.0f}s; '
          f'rows: {sorted(set(acc.rows.values()))}', file=sys.stderr)

    out_tensors = {}
    report = {}
    quant_names = [n for n in names if is_quantizable(n)]
    t0 = time.time()
    for i, name in enumerate(quant_names):
        H = acc.H[hkey(name)]
        q, scales, lg, lr = gptq_layer(w[name], H, args.damp)
        out_tensors[name] = q
        out_tensors[name + '.scale'] = scales
        report[name] = {'loss_gptq': lg, 'loss_rtn': lr,
                        'ratio': lg / lr if lr > 0 else 1.0}
        if i % 20 == 0 or lg > lr:
            print(f'[{i + 1}/{len(quant_names)}] {name}: '
                  f'gptq/rtn loss ratio {lg / max(lr, 1e-20):.3f}',
                  file=sys.stderr)
    print(f'gptq done in {time.time() - t0:.0f}s', file=sys.stderr)

    save_safetensors(args.out, out_tensors)
    ratios = [r['ratio'] for r in report.values()]
    print(f'wrote {args.out}: {len(quant_names)} layers, '
          f'loss ratio gptq/rtn median {np.median(ratios):.3f} '
          f'(min {min(ratios):.3f}, max {max(ratios):.3f})')
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=1)


if __name__ == '__main__':
    main()
