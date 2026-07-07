# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
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

"""Shippable Matcha-TTS: masked, pad-to-max GPU graphs. End-to-end parity + op-check.

Pads phonemes to max_text and mel frames to max_mel; passes a runtime float mask
to the text-encoder and decoder graphs (additive attention bias -> GPU-clean, no
SELECT/CAST). One compiled graph per stage handles any length up to the max.

Verifies: (1) all 3 masked graphs op-check GPU-clean; (2) corr(tflite-masked,
torch-original-masked) ~ 1.0 (mask removes the pad-leak that drop-mask suffered).

Run: python e2e_masked.py "Sentence." [n_timesteps] [MAX_MEL]
"""

import _stub  # noqa: F401  (must be first: scipy / getsourcefile guards)

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

import build_matcha as B
from e2e_matcha import LENGTH_SCALE, g2p_ids, generate_path, sequence_mask


def host_pipeline_masked(text_enc, decoder, vocoder, t_embed, emb_w, ids,
                         mel_mean, mel_std, n_timesteps, max_text, max_mel, z=None):
    """Pad-to-max host synthesise() around masked graphs.

    Returns (wav_valid, y_length, z) so callers can reuse identical noise.
    """
    t_x = ids.shape[-1]
    ids_pad = torch.zeros(1, max_text, dtype=torch.long)
    ids_pad[0, :t_x] = ids[0]
    tmask = torch.zeros(1, 1, max_text)
    tmask[0, 0, :t_x] = 1.0
    emb_x = emb_w[ids_pad]
    mu_x, logw = text_enc(emb_x, tmask)
    w = torch.exp(logw) * tmask
    w_ceil = torch.ceil(w) * LENGTH_SCALE
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    ymask = sequence_mask(y_lengths.float(), max_mel).unsqueeze(1)  # (1,1,max_mel).
    attn_mask = tmask.unsqueeze(-1) * ymask.unsqueeze(2)
    attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
    mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2),
                        mu_x.transpose(1, 2)).transpose(1, 2)
    if z is None:
        z = torch.randn(1, 80, max_mel)
    x = z.clone() * ymask
    t_span = torch.linspace(0, 1, n_timesteps + 1)
    t = t_span[0]
    dt = t_span[1] - t_span[0]
    for step in range(1, len(t_span)):
        t_emb = t_embed(t.reshape(1))
        v = decoder(x, mu_y, t_emb, ymask)
        x = x + dt * v
        t = t + dt
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t
    mel = (x * mel_std + mel_mean) * ymask
    wav = np.clip(vocoder(mel), -1, 1).reshape(-1)
    return wav[:int(y_lengths.item()) * 256], int(y_lengths.item()), z


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello, this is Matcha running on the GPU."
    n_timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    max_mel = int(sys.argv[3]) if len(sys.argv) > 3 else 384
    max_text = 128
    sd = B.load_sd()
    mel_mean = float(sd["mel_mean"])
    mel_std = float(sd["mel_std"])
    emb_w = sd["encoder.emb.weight"]
    ids = g2p_ids(text)
    print(f"text={text!r} tokens={ids.shape[-1]} steps={n_timesteps} "
          f"MAX_TEXT={max_text} MAX_MEL={max_mel}")

    # Ground truth: original modules, masked.
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

    with torch.no_grad():
        wav_true, ylen, z = host_pipeline_masked(te_true, dec_true, gen_true, t_embed,
                                                 emb_w, ids, mel_mean, mel_std,
                                                 n_timesteps, max_text, max_mel)
    print(f"y_lengths={ylen} audio={len(wav_true)} ({len(wav_true) / 22050:.2f}s)")

    # Masked graphs (shippable).
    te_r = B.reauth_text_encoder_masked(B.build_text_encoder(sd))
    dec_r = B.reauth_decoder_masked(B.build_decoder(sd), max_mel)
    gen_r = B.reauth_hifigan(B.build_hifigan(), max_mel)

    ex = emb_w[torch.zeros(1, max_text, dtype=torch.long)]
    tm0 = torch.ones(1, 1, max_text)
    x0 = torch.randn(1, 80, max_mel)
    mu0 = torch.randn(1, 80, max_mel)
    te0 = t_embed(torch.zeros(1))
    m0 = torch.ones(1, 1, max_mel)
    mel0 = torch.randn(1, 80, max_mel)
    p_te = B.convert(te_r, (ex, tm0), os.path.join(B.HERE, "m_textenc.tflite"))
    p_dec = B.convert(dec_r, (x0, mu0, te0, m0), os.path.join(B.HERE, "m_decoder.tflite"))
    p_gen = B.convert(gen_r, (mel0,), os.path.join(B.HERE, "m_vocoder.tflite"))
    print("\n=== op-check (masked, fp32) ===")
    clean_te = B.opcheck(p_te, "textenc")
    clean_dec = B.opcheck(p_dec, "decoder")
    clean_gen = B.opcheck(p_gen, "vocoder")

    cm_te = B.tfl_load(p_te)
    cm_dec = B.tfl_load(p_dec)
    cm_gen = B.tfl_load(p_gen)

    def tfl_te(emb_x, tmask):
        outputs = B.tfl_run(cm_te, emb_x.numpy(), tmask.numpy())
        a, b = outputs
        mu, logw = (a, b) if a.shape[1] == 80 else (b, a)
        return torch.from_numpy(mu), torch.from_numpy(logw)

    def tfl_dec(x, mu, t_emb, mask):
        return torch.from_numpy(
            B.tfl_run(cm_dec, x.numpy(), mu.numpy(), t_emb.numpy(), mask.numpy())[0])

    def tfl_gen(mel):
        return B.tfl_run(cm_gen, mel.numpy())[0]

    wav_tfl, *_ = host_pipeline_masked(tfl_te, tfl_dec, tfl_gen, t_embed, emb_w, ids,
                                       mel_mean, mel_std, n_timesteps, max_text,
                                       max_mel, z=z)

    def corr(a, b):
        n = min(len(a), len(b))
        return float(np.corrcoef(a[:n], b[:n])[0, 1])

    print(f"\ncorr(TFL-masked, TRUE) = {corr(wav_tfl, wav_true):.6f}")
    print("GPU-CLEAN ALL:", clean_te and clean_dec and clean_gen)
    try:
        import soundfile as sf
        sf.write(os.path.join(B.HERE, "m_true.wav"), wav_true, 22050)
        sf.write(os.path.join(B.HERE, "m_tfl.wav"), wav_tfl, 22050)
        print("wrote m_true.wav / m_tfl.wav")
    except Exception as e:
        print("no soundfile", e)


if __name__ == "__main__":
    main()
