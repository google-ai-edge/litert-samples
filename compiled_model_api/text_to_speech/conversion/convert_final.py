# Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Produces the shippable Matcha-TTS LiteRT artifacts (fp16) + verifies parity.

Outputs (into ./artifacts/):
  matcha_textenc_fp16.tflite   in: emb(1,Tt,192), tmask(1,1,Tt)  -> mu(1,80,Tt), logw(1,1,Tt)
  matcha_decoder_fp16.tflite   in: x,mu(1,80,Tm), t_emb(1,1024), ymask(1,1,Tm) -> v(1,80,Tm)
  matcha_vocoder_fp16.tflite   in: mel(1,80,Tm) -> wav(1,1,Tm*256)
  emb.npy            (178,192) phoneme embedding table (host lookup)
  time_embed.npz     weights to compute t_emb on host (SinusoidalPosEmb + MLP)
  config.json        symbols, MAX_TEXT/MAX_MEL, mel_mean/std, hop, length_scale, sr

Verifies fp16 graphs end-to-end (host orchestration) vs torch reference on several
sentences -> waveform corr.

Run: ~/clipconv/bin/python convert_final.py [MAX_MEL]
"""

import _stub  # noqa: F401  (must be first: scipy / getsourcefile guards)

import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

import build_matcha as B
from e2e_masked import host_pipeline_masked
from e2e_matcha import g2p_ids

os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak-ng.dylib")
ART = os.path.join(B.HERE, "artifacts")
os.makedirs(ART, exist_ok=True)
MAX_TEXT = 256
MAX_MEL = int(sys.argv[1]) if len(sys.argv) > 1 else 512

SENTENCES = [
    "Hello, this is Matcha running on the GPU.",
    "The quick brown fox jumps over the lazy dog.",
    "On device text to speech, with no internet connection required.",
    "She sells seashells by the seashore.",
]


def main():
    sd = B.load_sd()
    mel_mean = float(sd["mel_mean"])
    mel_std = float(sd["mel_std"])
    emb_w = sd["encoder.emb.weight"]
    print(f"MAX_TEXT={MAX_TEXT} MAX_MEL={MAX_MEL} "
          f"mel_mean={mel_mean:.4f} mel_std={mel_std:.4f}")

    # Torch reference modules (masked) + host time-embed.
    te = B.build_text_encoder(sd)
    dec0 = B.build_decoder(sd)
    gen = B.build_hifigan()

    def t_embed(t):
        # Weight-free, host-side; the learned time_mlp stays in-graph.
        return B.sin_pos_emb(t.reshape(-1), 160)

    dec0.time_embeddings = nn.Identity()

    def te_true(emb_x, tmask):
        x = (emb_x * math.sqrt(te.n_channels)).transpose(1, -1)
        x = te.prenet(x, tmask)
        x = te.encoder(x, tmask)
        return te.proj_m(x) * tmask, te.proj_w(x, tmask)

    def dec_true(x, mu, t_emb, mask):
        with torch.no_grad():
            return dec0(x, mask, mu, t_emb)

    def gen_true(mel):
        with torch.no_grad():
            return gen(mel).numpy()

    # Build + convert masked graphs (fp32 -> fp16).
    te_r = B.reauth_text_encoder_masked(B.build_text_encoder(sd))
    dec_r = B.reauth_decoder_masked(B.build_decoder(sd), MAX_MEL)
    gen_r = B.reauth_hifigan(B.build_hifigan(), MAX_MEL)
    ex = emb_w[torch.zeros(1, MAX_TEXT, dtype=torch.long)]
    tm0 = torch.ones(1, 1, MAX_TEXT)
    x0 = torch.randn(1, 80, MAX_MEL)
    mu0 = torch.randn(1, 80, MAX_MEL)
    te0 = t_embed(torch.zeros(1))
    m0 = torch.ones(1, 1, MAX_MEL)
    mel0 = torch.randn(1, 80, MAX_MEL)

    print("\n=== convert fp32 + quantize fp16 ===")
    specs = [("textenc", te_r, (ex, tm0)), ("decoder", dec_r, (x0, mu0, te0, m0)),
             ("vocoder", gen_r, (mel0,))]
    fp16 = {}
    for name, module, inputs in specs:
        p32 = B.convert(module, inputs, os.path.join(ART, f"matcha_{name}.tflite"))
        p16 = B.to_fp16(p32, os.path.join(ART, f"matcha_{name}_fp16.tflite"))
        _, clean = B.opcheck(p16, name + "_fp16")
        size32, size16 = os.path.getsize(p32) / 1e6, os.path.getsize(p16) / 1e6
        print(f"  {name}: fp32 {size32:.1f}MB -> fp16 {size16:.1f}MB  GPU-clean={clean}")
        fp16[name] = p16

    from ai_edge_litert.interpreter import Interpreter
    it_te = Interpreter(model_path=fp16["textenc"])
    it_te.allocate_tensors()
    it_dec = Interpreter(model_path=fp16["decoder"])
    it_dec.allocate_tensors()
    it_gen = Interpreter(model_path=fp16["vocoder"])
    it_gen.allocate_tensors()

    def tfl_te(emb_x, tmask):
        outputs = B.tfl_run(it_te, emb_x.numpy(), tmask.numpy())
        a, b = outputs
        mu, logw = (a, b) if a.shape[1] == 80 else (b, a)
        return torch.from_numpy(mu.copy()), torch.from_numpy(logw.copy())

    def tfl_dec(x, mu, t_emb, mask):
        return torch.from_numpy(
            B.tfl_run(it_dec, x.numpy(), mu.numpy(), t_emb.numpy(), mask.numpy())[0].copy())

    def tfl_gen(mel):
        return B.tfl_run(it_gen, mel.numpy())[0]

    print("\n=== end-to-end fp16 parity (waveform corr vs torch) ===")
    for sentence in SENTENCES:
        ids = g2p_ids(sentence)
        if ids.shape[-1] > MAX_TEXT:
            print(f"  SKIP (>{MAX_TEXT} tokens): {sentence!r}")
            continue
        with torch.no_grad():
            wav_true, ylen, z = host_pipeline_masked(te_true, dec_true, gen_true,
                                                     t_embed, emb_w, ids, mel_mean,
                                                     mel_std, 10, MAX_TEXT, MAX_MEL)
        if ylen > MAX_MEL:
            print(f"  SKIP (>{MAX_MEL} frames, {ylen}): {sentence!r}")
            continue
        wav_tfl, *_ = host_pipeline_masked(tfl_te, tfl_dec, tfl_gen, t_embed,
                                           emb_w, ids, mel_mean, mel_std, 10,
                                           MAX_TEXT, MAX_MEL, z=z)
        n = min(len(wav_true), len(wav_tfl))
        c = float(np.corrcoef(wav_true[:n], wav_tfl[:n])[0, 1])
        print(f"  corr={c:.5f}  frames={ylen}  {len(wav_tfl) / 22050:.2f}s  {sentence!r}")
        try:
            import soundfile as sf
            sf.write(os.path.join(ART, f"sample_{SENTENCES.index(sentence)}.wav"),
                     wav_tfl, 22050)
        except Exception:
            pass

    # Export host-side tables + config (the time-MLP now lives in the decoder graph).
    np.save(os.path.join(ART, "emb.npy"), emb_w.numpy().astype(np.float32))
    emb_w.numpy().astype("<f4").tofile(os.path.join(ART, "emb.bin"))  # Raw LE f32 for Kotlin.
    from matcha.text.symbols import symbols
    cfg = dict(symbols=symbols, n_vocab=178, n_channels=192, n_feats=80,
               MAX_TEXT=MAX_TEXT, MAX_MEL=MAX_MEL, mel_mean=mel_mean, mel_std=mel_std,
               hop=256, sample_rate=22050, length_scale=0.95, sigma_min=1e-4,
               n_timesteps_default=10, time_embed_dim=1024, in_channels=160)
    with open(os.path.join(ART, "config.json"), "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"\nartifacts in {ART}:")
    for name in sorted(os.listdir(ART)):
        print("  ", name, f"{os.path.getsize(os.path.join(ART, name)) / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
