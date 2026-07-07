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

"""End-to-end Matcha-TTS parity: host-orchestrated tflite pipeline vs torch.

Replicates MatchaTTS.synthesise() host-side (G2P+intersperse, embedding lookup,
duration -> length-regulator, sinusoidal time-embed, Euler ODE loop, denormalize,
clamp) and runs the 3 heavy graphs as either torch submodules or tflite graphs.

Compares three pipelines on the SAME noise z / durations:
  TRUE  : original torch modules with the real y_mask  (ground truth)
  RT    : re-authored torch modules, mask dropped       (== the converted graph)
  TFL   : tflite graphs, mask dropped                   (what ships, exact length)
corr(TFL,RT) isolates conversion fidelity; corr(TFL,TRUE) is real-world fidelity
(includes the <=3 fix_len pad-frame attention leak from dropping the mask).

Run: ~/clipconv/bin/python e2e_matcha.py "Some sentence." [n_timesteps]
"""

import _stub  # noqa: F401  (must be first: scipy / getsourcefile guards)

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

import build_matcha as B

os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak-ng.dylib")
LENGTH_SCALE = 0.95


def fix_len(n: int, k: int = 4) -> int:
    """Rounds n up to the next multiple of k."""
    return int(math.ceil(n / k) * k)


def sequence_mask(length, max_length):
    """Float mask (B, max_length): 1.0 where position < length."""
    arange = torch.arange(max_length, dtype=torch.float32)
    return (arange.unsqueeze(0) < length.unsqueeze(1)).float()


def generate_path(duration, mask):
    """Monotonic alignment path from durations. [matcha utils.model]

    duration: (b, t_x); mask: (b, t_x, t_y) -> path (b, t_x, t_y).
    """
    b, t_x, t_y = mask.shape
    cum = torch.cumsum(duration, 1)
    path = sequence_mask(cum.view(b * t_x), t_y).view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, [0, 0, 1, 0])[:, :-1]
    return path * mask


def intersperse(lst, item):
    """[a, b] -> [item, a, item, b, item]."""
    out = [item] * (len(lst) * 2 + 1)
    out[1::2] = lst
    return out


def g2p_ids(text: str):
    """Text -> interspersed matcha symbol id tensor (1, T) via espeak G2P."""
    from matcha.text import text_to_sequence

    seq = text_to_sequence(text, ["english_cleaners2"])[0]
    return torch.tensor(intersperse(seq, 0), dtype=torch.long)[None]


def host_pipeline(text_enc, decoder, vocoder, t_embed_fn, emb_w, ids,
                  mel_mean, mel_std, n_timesteps, z_full=None, use_mask=True):
    """Host-side synthesise(): text_enc(emb_x)->(mu,logw); decoder->v; vocoder->wav.

    Returns (wav_valid, y_lengths, y_max_, z_full) for reuse of identical noise.
    """
    t_x = ids.shape[-1]
    emb_x = emb_w[ids]  # Host embedding lookup (1, t_x, 192).
    mu_x, logw = text_enc(emb_x)  # (1,80,t_x), (1,1,t_x).
    x_mask = torch.ones(1, 1, t_x)
    w = torch.exp(logw) * x_mask
    w_ceil = torch.ceil(w) * LENGTH_SCALE
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_max = int(y_lengths.max())
    y_max_ = fix_len(y_max)
    y_mask = sequence_mask(y_lengths.float(), y_max_).unsqueeze(1)  # (1,1,y_max_).
    attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
    attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
    mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2),
                        mu_x.transpose(1, 2)).transpose(1, 2)

    # Euler ODE.
    if z_full is None:
        z_full = torch.randn(1, 80, y_max_)
    x = z_full.clone()
    t_span = torch.linspace(0, 1, n_timesteps + 1)
    t = t_span[0]
    dt = t_span[1] - t_span[0]
    for step in range(1, len(t_span)):
        t_emb = t_embed_fn(t.reshape(1))
        if use_mask:
            v = decoder(x, mu_y, t_emb, y_mask)
        else:
            v = decoder(x, mu_y, t_emb)
        x = x + dt * v
        t = t + dt
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t
    mel = x * mel_std + mel_mean
    if use_mask:
        mel = mel * y_mask
    wav = vocoder(mel)
    wav = np.clip(wav, -1, 1)
    valid = int(y_lengths.item()) * 256
    return wav.reshape(-1)[:valid], y_lengths, y_max_, z_full


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello, this is Matcha running on the GPU."
    n_timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    sd = B.load_sd()
    mel_mean = float(sd["mel_mean"])
    mel_std = float(sd["mel_std"])
    emb_w = sd["encoder.emb.weight"]
    ids = g2p_ids(text)
    print(f"text: {text!r}  tokens: {ids.shape[-1]}  steps: {n_timesteps}")

    # Original torch modules (ground truth, with real masking).
    te = B.build_text_encoder(sd)
    dec_orig = B.build_decoder(sd)
    gen = B.build_hifigan()
    # Capture the real time-embed modules, then make the decoder accept t_emb directly.
    temb_mod, tmlp_mod = dec_orig.time_embeddings, dec_orig.time_mlp

    def t_embed(t):
        return tmlp_mod(temb_mod(t)).detach()

    dec_orig.time_embeddings = nn.Identity()
    dec_orig.time_mlp = nn.Identity()

    def te_orig(emb_x):
        # Original encoder expects ids+lengths; reuse its internals with a real mask.
        t_x = emb_x.shape[1]
        x = (emb_x * math.sqrt(te.n_channels)).transpose(1, -1)
        mask = torch.ones(1, 1, t_x)
        x = te.prenet(x, mask)
        x = te.encoder(x, mask)
        return te.proj_m(x) * mask, te.proj_w(x, mask)

    def dec_orig_call(x, mu, t_emb, mask):
        return dec_orig(x, mask, mu, t_emb)

    def gen_call(mel):
        with torch.no_grad():
            return gen(mel).numpy()

    # Figure out y_max_ first (needs one text-enc pass); fix the noise z.
    with torch.no_grad():
        wav_true, ylen, ymax_, z = host_pipeline(
            lambda e: [t.detach() for t in te_orig(e)], dec_orig_call, gen_call,
            t_embed, emb_w, ids, mel_mean, mel_std, n_timesteps, use_mask=True)
    print(f"y_lengths={int(ylen)}  y_max_(fix4)={ymax_}  audio={len(wav_true)} samples "
          f"({len(wav_true) / 22050:.2f}s)")

    # Re-authored torch + tflite, both at exact length ymax_, mask dropped.
    te_r = B.reauth_text_encoder(B.build_text_encoder(sd))
    dec_r = B.reauth_decoder(B.build_decoder(sd), ymax_)
    gen_r = B.reauth_hifigan(B.build_hifigan(), ymax_)

    def te_r_call(emb_x):
        with torch.no_grad():
            return te_r(emb_x)

    def dec_r_call(x, mu, t_emb):
        with torch.no_grad():
            return dec_r(x, mu, t_emb)

    def gen_r_call(mel):
        with torch.no_grad():
            return gen_r(mel).numpy()

    with torch.no_grad():
        wav_rt, *_ = host_pipeline(te_r_call, dec_r_call, gen_r_call, t_embed,
                                   emb_w, ids, mel_mean, mel_std, n_timesteps,
                                   z_full=z, use_mask=False)

    # Convert the three graphs at exact length and run tflite.
    ex = emb_w[ids]
    x0 = torch.randn(1, 80, ymax_)
    mu0 = torch.randn(1, 80, ymax_)
    te0 = t_embed(torch.zeros(1))
    mel0 = torch.randn(1, 80, ymax_)
    p_te = B.convert(te_r, (ex,), os.path.join(B.HERE, "e2e_textenc.tflite"))
    p_dec = B.convert(dec_r, (x0, mu0, te0), os.path.join(B.HERE, "e2e_decoder.tflite"))
    p_gen = B.convert(gen_r, (mel0,), os.path.join(B.HERE, "e2e_vocoder.tflite"))

    cm_te = B.tfl_load(p_te)
    cm_dec = B.tfl_load(p_dec)
    cm_gen = B.tfl_load(p_gen)

    def tfl_te(emb_x):
        outputs = B.tfl_run(cm_te, emb_x.numpy())
        # Order outputs as (mu, logw) by shape: mu has 80 channels, logw has 1.
        a, b = outputs
        mu, logw = (a, b) if a.shape[1] == 80 else (b, a)
        return torch.from_numpy(mu), torch.from_numpy(logw)

    def tfl_dec(x, mu, t_emb):
        return torch.from_numpy(B.tfl_run(cm_dec, x.numpy(), mu.numpy(), t_emb.numpy())[0])

    def tfl_gen(mel):
        return B.tfl_run(cm_gen, mel.numpy())[0]

    wav_tfl, *_ = host_pipeline(tfl_te, tfl_dec, tfl_gen, t_embed, emb_w, ids,
                                mel_mean, mel_std, n_timesteps, z_full=z, use_mask=False)

    def corr(a, b):
        n = min(len(a), len(b))
        return float(np.corrcoef(a[:n], b[:n])[0, 1])

    print(f"\ncorr(TFL, RT)   = {corr(wav_tfl, wav_rt):.6f}   <- conversion fidelity")
    print(f"corr(TFL, TRUE) = {corr(wav_tfl, wav_true):.6f}   <- real-world (incl. pad-leak)")
    print(f"corr(RT,  TRUE) = {corr(wav_rt, wav_true):.6f}   <- mask-drop pad-leak only")

    # Save wavs for listening.
    try:
        import soundfile as sf
        sf.write(os.path.join(B.HERE, "out_true.wav"), wav_true, 22050)
        sf.write(os.path.join(B.HERE, "out_tfl.wav"), wav_tfl, 22050)
        print("wrote out_true.wav / out_tfl.wav")
    except Exception as e:
        np.save(os.path.join(B.HERE, "out_tfl.npy"), wav_tfl)
        print("soundfile missing; saved out_tfl.npy", e)


if __name__ == "__main__":
    main()
