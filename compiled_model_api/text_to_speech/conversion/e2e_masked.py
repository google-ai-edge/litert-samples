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

#!/usr/bin/env python3
"""Shippable Matcha-TTS: masked, pad-to-max GPU graphs. End-to-end parity + op-check.

Pads phonemes to MAX_TEXT and mel frames to MAX_MEL; passes a runtime float mask
to the text-encoder and decoder graphs (additive attention bias -> GPU-clean, no
SELECT/CAST). One compiled graph per stage handles any length up to the max.

Verifies: (1) all 3 masked graphs op-check GPU-clean; (2) corr(tflite-masked,
torch-original-masked) ~ 1.0 (mask removes the pad-leak that drop-mask suffered).

Run: ~/clipconv/bin/python e2e_masked.py "Sentence." [n_timesteps] [MAX_MEL]
"""
import _stub
import sys, os, math, numpy as np, torch, torch.nn as nn
import build_matcha as B
from e2e_matcha import g2p_ids, sequence_mask, generate_path, LENGTH_SCALE

os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak-ng.dylib")


def host_pipeline_masked(text_enc, decoder, vocoder, t_embed, emb_w, ids,
                         mel_mean, mel_std, n_timesteps, MAX_TEXT, MAX_MEL, z=None):
    Tx = ids.shape[-1]
    ids_pad = torch.zeros(1, MAX_TEXT, dtype=torch.long); ids_pad[0, :Tx] = ids[0]
    tmask = torch.zeros(1, 1, MAX_TEXT); tmask[0, 0, :Tx] = 1.0
    emb_x = emb_w[ids_pad]
    mu_x, logw = text_enc(emb_x, tmask)
    w = torch.exp(logw) * tmask
    w_ceil = torch.ceil(w) * LENGTH_SCALE
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    ymask = sequence_mask(y_lengths.float(), MAX_MEL).unsqueeze(1)         # (1,1,MAX_MEL)
    attn_mask = tmask.unsqueeze(-1) * ymask.unsqueeze(2)
    attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
    mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)
    if z is None:
        z = torch.randn(1, 80, MAX_MEL)
    x = z.clone() * ymask
    t_span = torch.linspace(0, 1, n_timesteps + 1)
    t = t_span[0]; dt = t_span[1] - t_span[0]
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
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    MAX_MEL = int(sys.argv[3]) if len(sys.argv) > 3 else 384
    MAX_TEXT = 128
    sd = B.load_sd()
    mel_mean = float(sd["mel_mean"]); mel_std = float(sd["mel_std"]); emb_w = sd["encoder.emb.weight"]
    ids = g2p_ids(text)
    print(f"text={text!r} tokens={ids.shape[-1]} steps={N} MAX_TEXT={MAX_TEXT} MAX_MEL={MAX_MEL}")

    # ground truth: original modules, masked
    te = B.build_text_encoder(sd); dec0 = B.build_decoder(sd); gen = B.build_hifigan()
    t_embed = lambda t: B.sin_pos_emb(t.reshape(-1), 160)   # weight-free, host-side
    dec0.time_embeddings = nn.Identity()                    # learned time_mlp stays in-graph

    def te_true(emb_x, tmask):
        x = (emb_x * math.sqrt(te.n_channels)).transpose(1, -1)
        x = te.prenet(x, tmask); x = te.encoder(x, tmask)
        return te.proj_m(x) * tmask, te.proj_w(x, tmask)
    def dec_true(x, mu, t_emb, mask):
        with torch.no_grad(): return dec0(x, mask, mu, t_emb)
    def gen_true(mel):
        with torch.no_grad(): return gen(mel).numpy()

    with torch.no_grad():
        wav_true, ylen, z = host_pipeline_masked(te_true, dec_true, gen_true, t_embed,
            emb_w, ids, mel_mean, mel_std, N, MAX_TEXT, MAX_MEL)
    print(f"y_lengths={ylen} audio={len(wav_true)} ({len(wav_true)/22050:.2f}s)")

    # masked graphs (shippable)
    te_r = B.reauth_text_encoder_masked(B.build_text_encoder(sd))
    dec_r = B.reauth_decoder_masked(B.build_decoder(sd), MAX_MEL)
    gen_r = B.reauth_hifigan(B.build_hifigan(), MAX_MEL)

    ex = emb_w[torch.zeros(1, MAX_TEXT, dtype=torch.long)]
    tm0 = torch.ones(1, 1, MAX_TEXT)
    x0 = torch.randn(1, 80, MAX_MEL); mu0 = torch.randn(1, 80, MAX_MEL)
    te0 = t_embed(torch.zeros(1)); m0 = torch.ones(1, 1, MAX_MEL); mel0 = torch.randn(1, 80, MAX_MEL)
    p_te = B.convert(te_r, (ex, tm0), os.path.join(B.HERE, "m_textenc.tflite"))
    p_dec = B.convert(dec_r, (x0, mu0, te0, m0), os.path.join(B.HERE, "m_decoder.tflite"))
    p_gen = B.convert(gen_r, (mel0,), os.path.join(B.HERE, "m_vocoder.tflite"))
    print("\n=== op-check (masked, fp32) ===")
    it_te, c1 = B.opcheck(p_te, "textenc")
    it_dec, c2 = B.opcheck(p_dec, "decoder")
    it_gen, c3 = B.opcheck(p_gen, "vocoder")

    def tfl_te(emb_x, tmask):
        o = B.tfl_run(it_te, emb_x.numpy(), tmask.numpy())
        a, b = o; mu, lw = (a, b) if a.shape[1] == 80 else (b, a)
        return torch.from_numpy(mu), torch.from_numpy(lw)
    def tfl_dec(x, mu, t_emb, mask):
        return torch.from_numpy(B.tfl_run(it_dec, x.numpy(), mu.numpy(), t_emb.numpy(), mask.numpy())[0])
    def tfl_gen(mel):
        return B.tfl_run(it_gen, mel.numpy())[0]

    wav_tfl, *_ = host_pipeline_masked(tfl_te, tfl_dec, tfl_gen, t_embed, emb_w, ids,
        mel_mean, mel_std, N, MAX_TEXT, MAX_MEL, z=z)

    def corr(a, b): n = min(len(a), len(b)); return float(np.corrcoef(a[:n], b[:n])[0, 1])
    print(f"\ncorr(TFL-masked, TRUE) = {corr(wav_tfl, wav_true):.6f}")
    print("GPU-CLEAN ALL:", c1 and c2 and c3)
    try:
        import soundfile as sf
        sf.write(os.path.join(B.HERE, "m_true.wav"), wav_true, 22050)
        sf.write(os.path.join(B.HERE, "m_tfl.wav"), wav_tfl, 22050)
        print("wrote m_true.wav / m_tfl.wav")
    except Exception as e:
        print("no soundfile", e)


if __name__ == "__main__":
    main()
