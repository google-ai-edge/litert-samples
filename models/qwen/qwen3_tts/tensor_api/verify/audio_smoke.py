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

"""Audio smoke: real-prompt talker rollout -> codec decode -> listenable wav.

Chains the two Tensor API example binaries host-side (the production stage
split): talker_main generates codec frames with in-graph greedy + EOS stop
(--eos_id), this script transposes the frame dump to the codec's [16, T]
layout, codec_main renders the waveform, and the raw fp32 is written as a
16-bit PCM wav @ 24 kHz.

Usage:
  python3 audio_smoke.py \
      --weights .../qwen3-tts-weights/model.safetensors \
      --codec-weights .../qwen3-tts-weights/speech_tokenizer/model.safetensors \
      --talker-bin .../bazel-bin/models/qwen/qwen3_tts/tensor_api/talker_step/talker_main \
      --codec-bin .../bazel-bin/models/qwen/qwen3_tts/tensor_api/codec_decode/codec_main \
      --workdir /tmp/smoke --out /tmp/smoke/smoke.wav \
      [--text "..."] [--spk enrollment.npy] [--weight-quant int8]
      [--accelerator gpu --gpu-precision fp32]

Also renders externally supplied codes with --codes-npy (e.g. a PyTorch
reference rollout) through the same codec graph for A/B listening.
"""

import argparse
import pathlib
import subprocess
import sys
import wave

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from numpy_talker_ref import EOS_ID, GROUPS, REAL_TEXT, build_real_prompt

SR = 24000


def write_wav(path, x):
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype('<i2')
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f'wrote {path}: {len(x)} samples = {len(x) / SR:.2f} s')


def run(cmd):
    print('+ ' + ' '.join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    sys.stderr.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        sys.exit(f'command failed ({proc.returncode})')
    return proc.stdout


def decode_codes(args, work, codes, tag):
    """codes [T, 16] -> wav via codec_main; returns fp32 waveform."""
    t = codes.shape[0]
    codes_path = work / f'codes_{tag}.i32'
    codes.T.astype(np.int32).tofile(codes_path)  # [16, T] codebook-major
    wav_path = work / f'wav_{tag}.f32'
    run([args.codec_bin,
         f'--weights={args.codec_weights}',
         f'--frames={t}',
         f'--codes_file={codes_path}',
         f'--dump_wav={wav_path}',
         f'--accelerator={args.codec_accelerator}',
         f'--gpu_precision={args.codec_gpu_precision}',
         '--gpu_buffer_storage=buffer',
         f'--tflite_path={work}/codec_decode.tflite',
         '--runs=1'])
    return np.fromfile(wav_path, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--codec-weights', required=True)
    ap.add_argument('--talker-bin', required=True)
    ap.add_argument('--codec-bin', required=True)
    ap.add_argument('--workdir', required=True)
    ap.add_argument('--out', required=True, help='output .wav path')
    ap.add_argument('--text', default=REAL_TEXT)
    ap.add_argument('--spk', default='',
                    help='.npy 1024-d speaker x-vector (default zeros)')
    ap.add_argument('--steps', type=int, default=240,
                    help='decode-step cap; EOS usually stops earlier')
    ap.add_argument('--max-seq', type=int, default=256)
    ap.add_argument('--accelerator', default='cpu', help='talker: cpu|gpu')
    ap.add_argument('--gpu-precision', default='fp32')
    ap.add_argument('--weight-quant', default='none')
    ap.add_argument('--prequant-weights', default='',
                    help='talker: pre-quantized int4 companion safetensors '
                         '(gptq_quantize.py output)')
    ap.add_argument('--sample-temp', type=float, default=0.0,
                    help='talker cb0 Gumbel temperature (0 = greedy)')
    ap.add_argument('--sample-seed', type=int, default=1)
    ap.add_argument('--codec-accelerator', default='cpu')
    ap.add_argument('--codec-gpu-precision', default='fp32')
    ap.add_argument('--codes-npy', default='',
                    help='skip the talker: render these [T, 16] codes instead')
    args = ap.parse_args()

    work = pathlib.Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    if args.codes_npy:
        codes = np.load(args.codes_npy).astype(np.int32)
        assert codes.ndim == 2 and codes.shape[1] == GROUPS, codes.shape
        write_wav(args.out, decode_codes(args, work, codes, 'ext'))
        return

    assert args.steps + 10 <= args.max_seq, 'steps must fit the KV cache'
    spk = np.load(args.spk) if args.spk else None
    prefill, track = build_real_prompt(args.weights, args.steps,
                                       text=args.text, spk=spk)
    prefill.tofile(work / 'prefill_emb.f32')
    track.tofile(work / 'text_track.f32')
    print(f'prompt: prefill {prefill.shape}, text_track {track.shape}, '
          f'spk={"real" if spk is not None else "zeros"}')

    run([args.talker_bin,
         '--layers=28',
         f'--max_seq={args.max_seq}',
         '--prefill_len=10',
         f'--steps={args.steps}',
         '--warmup=3',
         f'--weights={args.weights}',
         f'--weight_quant={args.weight_quant}',
         f'--prequant_weights={args.prequant_weights}',
         f'--sample_temp={args.sample_temp}',
         f'--sample_seed={args.sample_seed}',
         f'--accelerator={args.accelerator}',
         f'--gpu_precision={args.gpu_precision}',
         '--gpu_buffer_storage=buffer',
         f'--eos_id={EOS_ID}',
         f'--prefill_emb_file={work}/prefill_emb.f32',
         f'--text_track_file={work}/text_track.f32',
         f'--dump_dir={work}',
         f'--tflite_path={work}/talker_step.tflite'])

    codes = np.fromfile(work / 'frames.i32', np.int32).reshape(-1, GROUPS)
    dur = codes.shape[0] / 12.5
    print(f'talker: {codes.shape[0]} frames = {dur:.2f} s @ 12.5 Hz')
    assert (codes >= 0).all() and (codes < 2048).all(), 'non-codec id in dump'

    write_wav(args.out, decode_codes(args, work, codes, 'talker'))


if __name__ == '__main__':
    main()
