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

"""End-to-end check of a converted Zipformer CTC .tflite (CompiledModel API).

Transcribes a wav through the converted graph: host kaldi-fbank -> tflite
encoder+CTC -> host greedy-CTC decode. When build_zipformer_ctc.py has saved
a PyTorch reference (ref_logprobs.npy), also reports valid-region correlation
and max abs diff against it.

Run:
  python verify_zipformer_ctc.py zipformer_ctc_fp16.tflite path/to/audio.wav
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENS = os.path.join(HERE, "en_medium/data/lang_bpe_500/tokens.txt")

SR = 16000
T_IN = 1600  # 16 s of 10 ms-hop fbank frames (snip_edges=False)


def fbank(wav_path):
  """Computes the padded 16 s kaldi-fbank window for a wav file.

  Args:
    wav_path: Path to the input audio (any rate, resampled to 16 kHz mono).

  Returns:
    Float32 array [1, T_IN, 80]. The wave stays in [-1, 1] scale and the
    features match icefall's kaldifeat options (povey window,
    snip_edges=False, high_freq=-400, dither 0, no CMN).
  """
  import torch
  import torchaudio
  wave, sr = torchaudio.load(wav_path)
  if sr != SR:
    wave = torchaudio.functional.resample(wave, sr, SR)
  wave = wave.mean(0, keepdim=True)
  feats = torchaudio.compliance.kaldi.fbank(
      wave, num_mel_bins=80, sample_frequency=SR, dither=0.0,
      snip_edges=False, high_freq=-400.0)
  T = feats.shape[0]
  if T < T_IN:
    pad = feats.new_full((T_IN - T, 80), math.log(1e-10))
    feats = torch.cat([feats, pad], 0)
  else:
    feats = feats[:T_IN]
  return feats.unsqueeze(0).numpy().astype(np.float32), min(T, T_IN)


def make_biases(t_fbank):
  """Builds the four per-rate additive bias inputs (0 valid / -1000 pad).

  Args:
    t_fbank: Number of valid fbank frames for the real audio.

  Returns:
    List of four [1, N] float32 arrays for downsampling rates 1, 2, 4, 8.
  """
  t50_total = (T_IN - 7) // 2
  t50_valid = (t_fbank - 7) // 2
  b = np.full((1, t50_total), -1000.0, dtype=np.float32)
  b[0, :t50_valid] = 0.0
  return [np.ascontiguousarray(b[:, ::ds]) for ds in (1, 2, 4, 8)], t50_valid


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


def greedy_ctc(logits, id2tok):
  """Greedy CTC decode (blank id 0, drop repeats) + BPE detokenization.

  Args:
    logits: [1, T, 500] logits or log-probs (argmax-invariant).
    id2tok: id -> token table from load_tokens().

  Returns:
    The decoded transcript string.
  """
  ids = logits[0].argmax(-1)
  out, prev = [], -1
  for i in ids.tolist():
    if i != prev and i != 0:
      out.append(id2tok.get(i, "?"))
    prev = i
  return "".join(out).replace("▁", " ").strip()


def load_tokens():
  """Loads the BPE-500 id -> token table from tokens.txt."""
  id2tok = {}
  for line in open(TOKENS, encoding="utf-8"):
    tok, idx = line.rsplit(maxsplit=1)
    id2tok[int(idx)] = tok
  return id2tok


def main():
  """Transcribes the wav and reports parity against the saved reference."""
  if len(sys.argv) != 3:
    sys.exit("usage: verify_zipformer_ctc.py model.tflite audio.wav")
  model_path, wav_path = sys.argv[1], sys.argv[2]

  x, t_fbank = fbank(wav_path)
  biases, t50_valid = make_biases(t_fbank)
  logits = run_tflite(model_path, x, biases)
  text = greedy_ctc(logits, load_tokens())
  print(f"[verify] logits {logits.shape}")
  print(f"[verify] TEXT: {text}")

  ref_path = os.path.join(HERE, "ref_logprobs.npy")
  if os.path.exists(ref_path):
    ref = np.load(ref_path)
    t25_valid = (t50_valid + 1) // 2
    nv = min(t25_valid, ref.shape[1], logits.shape[1])
    a, b = logits[0, :nv].ravel(), ref[0, :nv].ravel()
    corr = np.corrcoef(a, b)[0, 1]
    print(f"[verify] valid-region vs PyTorch ref: corr {corr:.6f} "
          f"max|diff| {np.abs(a - b).max():.4f}")


if __name__ == "__main__":
  main()
