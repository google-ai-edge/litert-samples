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

"""Teacher-forced quality comparison: fp32 vs RTN-int4 vs GPTQ-int4.

Rolls the fp32 reference on the real prompt to fix a trajectory, then
re-runs each quantized variant ON THE SAME fp32 sequence prefixes
(teacher forcing), so per-frame metrics measure weight-quant damage only,
not autoregressive divergence:

  - cb0 logits cosine vs fp32, per frame (median / min)
  - cb0 greedy id match rate
  - full 16-group frame id match rate (variant CP run on the fp32 h_last)

Usage:
  python3 gptq_eval.py --weights .../model.safetensors \
      --gptq .../talker_gptq_int4.safetensors [--steps 48] [--layers 28]
"""

import argparse
import sys

import numpy as np

import numpy_talker_ref as ref
import gptq_quantize as gq


def cos(a, b):
    return float(np.dot(a, b) /
                 ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-30))


def rollout_fp32(w, layers, prefill, track, steps):
    """Returns (seqs [list of row-lists per step], h_lasts, logits, frames)."""
    seq = [row for row in prefill]
    h_lasts, logits_all, frames, seq_lens = [], [], [], []
    feedback = None
    for step in range(steps + 1):
        if step > 0:
            seq.append(feedback + track[step - 1])
        hidden = ref.backbone_forward(np.stack(seq), w, 'talker.model',
                                      layers, np.arange(len(seq)))
        h_last = hidden[-1]
        lg = h_last @ w['talker.codec_head.weight'].T
        ids, feedback = ref.predict_frame(h_last, lg, w)
        h_lasts.append(h_last)
        logits_all.append(lg)
        frames.append(ids)
        seq_lens.append(len(seq))
    return np.stack(seq), seq_lens, h_lasts, logits_all, frames


def eval_variant(tag, wv, layers, full_seq, seq_lens, h_lasts_fp32,
                 logits_fp32, frames_fp32):
    coss, cb0_hits, frame_hits, group_hits = [], 0, 0, 0
    n = len(seq_lens)
    for step in range(n):
        S = seq_lens[step]
        hidden = ref.backbone_forward(full_seq[:S], wv, 'talker.model',
                                      layers, np.arange(S))
        h_last = hidden[-1]
        lg = h_last @ wv['talker.codec_head.weight'].T
        coss.append(cos(lg, logits_fp32[step]))
        cb0_fp32 = frames_fp32[step][0]
        cb0_v = int(np.argmax(lg + ref.cb0_bias()))
        cb0_hits += cb0_v == cb0_fp32
        # CP damage isolated on the fp32 h_last (and fp32 cb0 via logits).
        ids, _ = ref.predict_frame(h_lasts_fp32[step],
                                   logits_fp32[step], wv)
        frame_hits += ids == frames_fp32[step]
        group_hits += sum(a == b for a, b in zip(ids, frames_fp32[step]))
    print(f'{tag}: cb0-logit cos median {np.median(coss):.5f} '
          f'(min {min(coss):.5f}) | cb0 id match {cb0_hits}/{n} | '
          f'16-group frame match {frame_hits}/{n} | '
          f'per-group match {group_hits}/{n * ref.GROUPS} '
          f'({100.0 * group_hits / (n * ref.GROUPS):.1f}%)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--gptq', required=True)
    ap.add_argument('--layers', type=int, default=28)
    ap.add_argument('--steps', type=int, default=48)
    args = ap.parse_args()

    names = ref.weight_names(args.layers)
    w = ref.load_bf16_tensors(args.weights, names)
    prefill, track = ref.build_real_prompt(args.weights, args.steps)

    print(f'fp32 rollout ({args.steps} frames)...', file=sys.stderr)
    full_seq, seq_lens, h_lasts, logits, frames = rollout_fp32(
        w, args.layers, prefill, track, args.steps)

    w_rtn = ref.quantize_weights_like_cpp(
        ref.load_bf16_tensors(args.weights, names), 'int4')
    eval_variant('RTN  int4-bw32', w_rtn, args.layers, full_seq, seq_lens,
                 h_lasts, logits, frames)
    del w_rtn

    w_gptq = gq.apply_gptq_fakequant(
        ref.load_bf16_tensors(args.weights, names), args.gptq)
    eval_variant('GPTQ int4-bw32', w_gptq, args.layers, full_seq, seq_lens,
                 h_lasts, logits, frames)


if __name__ == '__main__':
    main()
