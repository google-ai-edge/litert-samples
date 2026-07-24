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

"""Independent numpy reference for the talker step-graph equivalence check.

Implements the Qwen3-TTS talker + code predictor forward directly from
model.safetensors (bf16 -> fp32, no torch/qwen_tts dependency), mirroring the
production semantics verified in hf-to-litertlm/qwen3tts_work
(export_mtp_folded.py: rope position t per CP step, lm_head[t-1],
codec_embedding tables 0..13 fed back; HF repeat_interleave GQA).

Usage:
  # 1. write the fixed prefill embedding input for talker_main
  python3 numpy_talker_ref.py gen --out /tmp/talkercheck

  # 2. run talker_main with --prefill_emb_file/--dump_dir (see README)

  # 3. compute the reference and compare against the C++ dump
  python3 numpy_talker_ref.py check --weights .../model.safetensors \
      --dump /tmp/talkercheck [--layers 28] [--prefill-len 24] [--steps 8]
"""

import argparse
import json
import pathlib
import struct
import sys

import numpy as np

EMB = 1024
HEADS = 16
KV = 8
HD = 128
FFN = 3072
EPS = 1e-6
ROPE_BASE = 1e6
TALKER_VOCAB = 3072
CODEC_VOCAB = 2048
GROUPS = 16  # cb0 + 15 sub groups
CP_LAYERS = 5
MASK_FLOOR = -30000.0  # must match kMaskFloor in the C++ graph/host loop
TEXT_ROW = 0.01        # constant text-track row used by talker_main


def load_bf16_tensors(path, wanted):
    """Loads `wanted` tensor names from a safetensors file as fp32 arrays."""
    with open(path, 'rb') as f:
        hdr_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(hdr_len))
        data = f.read()  # data section; offsets below are relative to it
    out = {}
    for name in wanted:
        info = header[name]
        assert info['dtype'] == 'BF16', (name, info['dtype'])
        s0, s1 = info['data_offsets']
        raw = np.frombuffer(data, dtype='<u2', count=(s1 - s0) // 2,
                            offset=s0)
        f32 = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
        out[name] = f32.reshape(info['shape']).copy()
    return out


INT4_BLOCK = 32


def quantize_weights_like_cpp(w, mode):
    """Mirrors talker_weights.cc QuantizeFcWeight, dequantized back to fp32
    (isolates pure weight-quant noise from backend activation quantization).
    int8: per-channel symmetric. int4: blockwise-32 symmetric, scales rounded
    to fp16 as the serializer does."""
    for name, arr in w.items():
        if not ('_proj.weight' in name or name.endswith('codec_head.weight')
                or '.lm_head.' in name):
            continue
        if mode == 'int8':
            amax = np.abs(arr).max(axis=1)
            scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
            q = np.clip(np.round(arr / scale[:, None]), -127, 127)
            w[name] = (q * scale[:, None]).astype(np.float32)
        else:  # int4 blockwise
            out, inner = arr.shape
            blocks = arr.reshape(out, inner // INT4_BLOCK, INT4_BLOCK)
            amax = np.abs(blocks).max(axis=2)
            scale = np.where(amax > 0, amax / 7.0, 1.0)
            scale = scale.astype(np.float16).astype(np.float32)
            q = np.clip(np.round(blocks / scale[:, :, None]), -7, 7)
            w[name] = (q * scale[:, :, None]).reshape(out, inner).astype(
                np.float32)
    return w


def weight_names(layers):
    names = []

    def backbone(base, n):
        for i in range(n):
            p = f'{base}.layers.{i}'
            names.extend([
                f'{p}.input_layernorm.weight',
                f'{p}.self_attn.q_proj.weight',
                f'{p}.self_attn.k_proj.weight',
                f'{p}.self_attn.v_proj.weight',
                f'{p}.self_attn.o_proj.weight',
                f'{p}.self_attn.q_norm.weight',
                f'{p}.self_attn.k_norm.weight',
                f'{p}.post_attention_layernorm.weight',
                f'{p}.mlp.gate_proj.weight',
                f'{p}.mlp.up_proj.weight',
                f'{p}.mlp.down_proj.weight',
            ])
        names.append(f'{base}.norm.weight')

    backbone('talker.model', layers)
    names.append('talker.model.codec_embedding.weight')
    names.append('talker.codec_head.weight')
    backbone('talker.code_predictor.model', CP_LAYERS)
    for g in range(GROUPS - 1):
        names.append(f'talker.code_predictor.model.codec_embedding.{g}.weight')
        names.append(f'talker.code_predictor.lm_head.{g}.weight')
    return names


def rms(x, w):
    v = np.mean(np.square(x), axis=-1, keepdims=True, dtype=np.float32)
    return x * (1.0 / np.sqrt(v + EPS)) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope_tables(positions):
    half = HD // 2
    freqs = ROPE_BASE ** (-2.0 * np.arange(half) / HD)
    ang = np.asarray(positions, np.float64)[:, None] * freqs[None, :]
    cos = np.concatenate([np.cos(ang), np.cos(ang)], -1).astype(np.float32)
    sin = np.concatenate([np.sin(ang), np.sin(ang)], -1).astype(np.float32)
    return cos, sin  # [S, HD]


def apply_rope(x, cos, sin):
    half = HD // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], -1)
    return x * cos + rot * sin


def softmax(x):
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def backbone_forward(x, w, base, n_layers, positions):
    """Pre-norm GQA transformer; x [S, EMB] fp32; returns final-normed [S, EMB]."""
    S = x.shape[0]
    cos, sin = rope_tables(positions)
    causal = np.full((S, S), MASK_FLOOR, np.float32)
    causal[np.tril_indices(S)] = 0.0

    for i in range(n_layers):
        p = f'{base}.layers.{i}'
        h = rms(x, w[f'{p}.input_layernorm.weight'])
        q = (h @ w[f'{p}.self_attn.q_proj.weight'].T).reshape(S, HEADS, HD)
        k = (h @ w[f'{p}.self_attn.k_proj.weight'].T).reshape(S, KV, HD)
        v = (h @ w[f'{p}.self_attn.v_proj.weight'].T).reshape(S, KV, HD)
        q = rms(q, w[f'{p}.self_attn.q_norm.weight']).transpose(1, 0, 2)
        k = rms(k, w[f'{p}.self_attn.k_norm.weight']).transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = np.repeat(k, HEADS // KV, axis=0)  # HF repeat_interleave
        v = np.repeat(v, HEADS // KV, axis=0)
        scores = q @ k.transpose(0, 2, 1) * np.float32(HD ** -0.5) + causal
        o = softmax(scores) @ v  # [H, S, HD]
        o = o.transpose(1, 0, 2).reshape(S, HEADS * HD)
        x = x + o @ w[f'{p}.self_attn.o_proj.weight'].T

        h2 = rms(x, w[f'{p}.post_attention_layernorm.weight'])
        ff = (silu(h2 @ w[f'{p}.mlp.gate_proj.weight'].T) *
              (h2 @ w[f'{p}.mlp.up_proj.weight'].T))
        x = x + ff @ w[f'{p}.mlp.down_proj.weight'].T
    return rms(x, w[f'{base}.norm.weight'])


def cb0_bias():
    bias = np.zeros(TALKER_VOCAB, np.float32)
    bias[CODEC_VOCAB:] = MASK_FLOOR
    return bias


def predict_frame(h_last, logits, w):
    """cb0 (greedy, suppressed) + CP unroll -> (ids[16], feedback[EMB])."""
    cb0 = int(np.argmax(logits + cb0_bias()))
    cb0_emb = w['talker.model.codec_embedding.weight'][cb0]
    ids = [cb0]
    tokens = [h_last, cb0_emb]
    feedback = cb0_emb.copy()
    for g in range(GROUPS - 1):
        seq = np.stack(tokens)  # [S, EMB], rope positions 0..S-1
        h = backbone_forward(seq, w, 'talker.code_predictor.model', CP_LAYERS,
                             np.arange(len(tokens)))[-1]
        lg = h @ w[f'talker.code_predictor.lm_head.{g}.weight'].T
        idg = int(np.argmax(lg))
        ids.append(idg)
        emb = w[f'talker.code_predictor.model.codec_embedding.{g}.weight'][idg]
        feedback = feedback + emb
        if g < GROUPS - 2:
            tokens.append(emb)
    return ids, feedback


def make_prefill_emb(prefill_len):
    rng = np.random.RandomState(123)
    return (rng.randn(prefill_len, EMB) * 0.02).astype(np.float32)


# Real-prompt construction, mirroring the production host loop
# (hf-to-litertlm/qwen3tts_work/hostloop_e2e.py): dual-track prefill of
# text-projection rows + codec special rows. Speaker x-vector defaults to
# zeros (numerics experiments); pass a real 1024-d enrollment vector via
# `spk` for audio-quality runs.
REAL_TEXT = ('Hello! This is a small test of speech synthesis running on '
             'device.')
EOS_ID = 2150
THINK, THINK_BOS, THINK_EOS = 2154, 2156, 2157
PAD_ID, BOS_ID, LANG_EN = 2148, 2149, 2050
TTS_BOS, TTS_EOS, TTS_PAD = 151672, 151673, 151671


def build_real_prompt(weights_path, steps, text=REAL_TEXT, spk=None):
    """Returns (prefill [10, EMB], text_track [steps, EMB]) fp32."""
    from transformers import AutoTokenizer
    w = load_bf16_tensors(weights_path, [
        'talker.model.text_embedding.weight',
        'talker.text_projection.linear_fc1.weight',
        'talker.text_projection.linear_fc1.bias',
        'talker.text_projection.linear_fc2.weight',
        'talker.text_projection.linear_fc2.bias',
        'talker.model.codec_embedding.weight',
    ])
    codec_emb = w['talker.model.codec_embedding.weight']

    def embed_text(ids):
        rows = w['talker.model.text_embedding.weight'][np.asarray(ids)]
        h = silu(rows @ w['talker.text_projection.linear_fc1.weight'].T +
                 w['talker.text_projection.linear_fc1.bias'])
        return (h @ w['talker.text_projection.linear_fc2.weight'].T +
                w['talker.text_projection.linear_fc2.bias']).astype(np.float32)

    tok = AutoTokenizer.from_pretrained(
        str(pathlib.Path(weights_path).parent))
    ids = tok(f'<|im_start|>assistant\n{text}<|im_end|>\n'
              '<|im_start|>assistant\n', return_tensors='np')['input_ids'][0]

    tts_bos, tts_eos, tts_pad = embed_text([TTS_BOS, TTS_EOS, TTS_PAD])
    if spk is None:
        spk = np.zeros(EMB, np.float32)
    spk = np.asarray(spk, np.float32).reshape(EMB)
    codec_pre = np.concatenate([
        codec_emb[[THINK, THINK_BOS, LANG_EN, THINK_EOS]], spk[None],
        codec_emb[[PAD_ID, BOS_ID]]], 0)                       # [7, EMB]
    role = embed_text(ids[:3])                                 # [3, EMB]
    body = np.concatenate([np.repeat(tts_pad[None], 5, 0),
                           tts_bos[None]], 0) + codec_pre[:-1]  # [6, EMB]
    first_text = embed_text(ids[3:4]) + codec_pre[-1:]         # [1, EMB]
    prefill = np.concatenate([role, body, first_text], 0)      # [10, EMB]
    trailing = np.concatenate([embed_text(ids[4:-5]), tts_eos[None]], 0)

    track = np.stack([trailing[t] if t < len(trailing) else tts_pad
                      for t in range(steps)])
    return prefill.astype(np.float32), track.astype(np.float32)


def run_reference(weights_path, layers, prefill_len, steps, quant='none',
                  prefill_emb=None, text_track=None, gptq=None):
    w = load_bf16_tensors(weights_path, weight_names(layers))
    if gptq:
        import gptq_quantize
        w = gptq_quantize.apply_gptq_fakequant(w, gptq)
    elif quant != 'none':
        w = quantize_weights_like_cpp(w, quant)
    if prefill_emb is None:
        prefill_emb = make_prefill_emb(prefill_len)
    seq = [row for row in prefill_emb]

    frames, logits_all, feedback_all = [], [], []
    for step in range(steps + 1):  # step 0 = prefill frame
        if step > 0:
            row = (text_track[step - 1] if text_track is not None
                   else np.float32(TEXT_ROW))
            seq.append(feedback_all[-1] + row)
        hidden = backbone_forward(np.stack(seq), w, 'talker.model', layers,
                                  np.arange(len(seq)))
        h_last = hidden[-1]
        logits = h_last @ w['talker.codec_head.weight'].T
        ids, feedback = predict_frame(h_last, logits, w)
        frames.append(ids)
        logits_all.append(logits)
        feedback_all.append(feedback)
        print(f'ref frame {step}: cb0={ids[0]} sub={ids[1:6]}...',
              file=sys.stderr)
    return (np.array(frames, np.int32), np.stack(logits_all),
            np.stack(feedback_all))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['gen', 'check'])
    ap.add_argument('--out', help='gen: directory for prefill_emb.f32')
    ap.add_argument('--weights', help='check: model.safetensors path')
    ap.add_argument('--dump', help='check: talker_main --dump_dir directory')
    ap.add_argument('--layers', type=int, default=28)
    ap.add_argument('--prefill-len', type=int, default=24)
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--quant', choices=['none', 'int8', 'int4'],
                    default='none',
                    help='check: quantize+dequantize FC weights like the C++ '
                         'loader before running the reference')
    ap.add_argument('--gptq',
                    help='check: fake-quant FC weights from a GPTQ companion '
                         'safetensors (gptq_quantize.py output); overrides '
                         '--quant')
    ap.add_argument('--real-prompt', action='store_true',
                    help='use the production dual-track prompt (tokenizer + '
                         'text projection from the checkpoint) instead of the '
                         'random embedding; gen also writes text_track.f32. '
                         'Forces prefill_len=10.')
    args = ap.parse_args()

    if args.real_prompt:
        args.prefill_len = 10

    if args.mode == 'gen':
        if args.real_prompt:
            emb, track = build_real_prompt(args.weights, args.steps)
            track.tofile(f'{args.out}/text_track.f32')
            print(f'wrote {args.out}/text_track.f32 {track.shape}')
        else:
            emb = make_prefill_emb(args.prefill_len)
        path = f'{args.out}/prefill_emb.f32'
        emb.tofile(path)
        print(f'wrote {path} {emb.shape}')
        return

    prefill_emb, text_track = (build_real_prompt(args.weights, args.steps)
                               if args.real_prompt else (None, None))
    frames, logits, feedback = run_reference(
        args.weights, args.layers, args.prefill_len, args.steps, args.quant,
        prefill_emb, text_track, gptq=args.gptq)

    n = args.steps + 1
    c_frames = np.fromfile(f'{args.dump}/frames.i32',
                           np.int32).reshape(n, GROUPS)
    c_logits = np.fromfile(f'{args.dump}/cb0_logits.f32',
                           np.float32).reshape(n, TALKER_VOCAB)
    c_feedback = np.fromfile(f'{args.dump}/feedback.f32',
                             np.float32).reshape(n, EMB)

    ok = True
    for t in range(n):
        id_match = np.array_equal(frames[t], c_frames[t])
        dl = np.abs(logits[t] - c_logits[t])
        rel = dl.max() / (np.abs(logits[t]).max() + 1e-9)
        df = np.abs(feedback[t] - c_feedback[t]).max()
        status = 'OK ' if id_match else 'IDS DIFFER'
        print(f'frame {t}: {status} logits max|d|={dl.max():.4e} '
              f'(rel {rel:.2e}) feedback max|d|={df:.4e}')
        if not id_match:
            ok = False
            bad = np.nonzero(frames[t] != c_frames[t])[0]
            for g in bad[:4]:
                print(f'  group {g}: ref={frames[t][g]} cpp={c_frames[t][g]}')
    print('EQUIVALENCE: ' + ('PASS (all frame ids match)' if ok else 'FAIL'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
