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
import _stub
import sys, os, math, numpy as np, torch, torch.nn as nn
import build_matcha as B

os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak-ng.dylib")
LENGTH_SCALE = 0.95
TEMперature = 1.0  # noqa (kept literal below)


def fix_len(n, k=4):
    return int(math.ceil(n / k) * k)


def sequence_mask(length, max_length):
    a = torch.arange(max_length, dtype=torch.float32)
    return (a.unsqueeze(0) < length.unsqueeze(1)).float()


def generate_path(duration, mask):
    # duration (b,t_x), mask (b,t_x,t_y) -> path (b,t_x,t_y)  [matcha utils.model]
    b, t_x, t_y = mask.shape
    cum = torch.cumsum(duration, 1)
    path = sequence_mask(cum.view(b * t_x), t_y).view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, [0, 0, 1, 0])[:, :-1]
    return path * mask


def intersperse(lst, item):
    out = [item] * (len(lst) * 2 + 1)
    out[1::2] = lst
    return out


def g2p_ids(text):
    from matcha.text import text_to_sequence
    seq = text_to_sequence(text, ["english_cleaners2"])[0]
    return torch.tensor(intersperse(seq, 0), dtype=torch.long)[None]   # (1, T)


def host_pipeline(text_enc, decoder, vocoder, t_embed_fn, emb_w, ids,
                  mel_mean, mel_std, n_timesteps, z_full=None, use_mask=True):
    """text_enc(emb_x)->(mu,logw); decoder(x,mu,t_emb[,mask])->v; vocoder(mel)->wav.
    Returns (wav_valid, y_lengths, y_max_, z_full) for reuse of identical noise."""
    Tx = ids.shape[-1]
    emb_x = emb_w[ids]                                   # host embedding lookup (1,Tx,192)
    mu_x, logw = text_enc(emb_x)                         # (1,80,Tx),(1,1,Tx)
    x_mask = torch.ones(1, 1, Tx)
    w = torch.exp(logw) * x_mask
    w_ceil = torch.ceil(w) * LENGTH_SCALE
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_max = int(y_lengths.max()); y_max_ = fix_len(y_max)
    y_mask = sequence_mask(y_lengths.float(), y_max_).unsqueeze(1)        # (1,1,y_max_)
    attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
    attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
    mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)
    # Euler ODE
    if z_full is None:
        z_full = torch.randn(1, 80, y_max_)
    x = z_full.clone()
    t_span = torch.linspace(0, 1, n_timesteps + 1)
    t = t_span[0]; dt = t_span[1] - t_span[0]
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
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    sd = B.load_sd()
    mel_mean = float(sd["mel_mean"]); mel_std = float(sd["mel_std"])
    emb_w = sd["encoder.emb.weight"]
    ids = g2p_ids(text)
    print(f"text: {text!r}  tokens: {ids.shape[-1]}  steps: {N}")

    # original torch modules (ground truth, with real masking)
    te = B.build_text_encoder(sd)
    dec_orig = B.build_decoder(sd)
    gen = B.build_hifigan()
    # capture the real time-embed modules, then make the decoder accept t_emb directly
    _temb_mod, _tmlp_mod = dec_orig.time_embeddings, dec_orig.time_mlp
    t_embed = lambda t: _tmlp_mod(_temb_mod(t)).detach()
    dec_orig.time_embeddings = nn.Identity(); dec_orig.time_mlp = nn.Identity()

    def te_orig(emb_x):
        # original encoder expects ids+lengths; reuse its internals with real mask
        Tx = emb_x.shape[1]
        x = (emb_x * math.sqrt(te.n_channels)).transpose(1, -1)
        m = torch.ones(1, 1, Tx)
        x = te.prenet(x, m); x = te.encoder(x, m)
        return te.proj_m(x) * m, te.proj_w(x, m)

    def dec_orig_call(x, mu, t_emb, mask):
        return dec_orig(x, mask, mu, t_emb)

    def gen_call(mel):
        with torch.no_grad():
            return gen(mel).numpy()

    # figure out y_max_ first (needs one text-enc pass); fix noise z
    with torch.no_grad():
        wav_true, ylen, ymax_, z = host_pipeline(
            lambda e: [t.detach() for t in te_orig(e)], dec_orig_call, gen_call,
            t_embed, emb_w, ids, mel_mean, mel_std, N, use_mask=True)
    print(f"y_lengths={int(ylen)}  y_max_(fix4)={ymax_}  audio={len(wav_true)} samples "
          f"({len(wav_true)/22050:.2f}s)")

    # re-authored torch + tflite, both at exact length ymax_, mask dropped
    te_r = B.reauth_text_encoder(B.build_text_encoder(sd))
    dec_r = B.reauth_decoder(B.build_decoder(sd), ymax_)
    gen_r = B.reauth_hifigan(B.build_hifigan(), ymax_)

    def te_r_call(emb_x):
        with torch.no_grad(): return te_r(emb_x)
    def dec_r_call(x, mu, t_emb):
        with torch.no_grad(): return dec_r(x, mu, t_emb)
    def gen_r_call(mel):
        with torch.no_grad(): return gen_r(mel).numpy()

    with torch.no_grad():
        wav_rt, *_ = host_pipeline(te_r_call, dec_r_call, gen_r_call, t_embed,
                                   emb_w, ids, mel_mean, mel_std, N, z_full=z, use_mask=False)

    # convert the three graphs at exact length and run tflite
    ex = emb_w[ids]
    x0 = torch.randn(1, 80, ymax_); mu0 = torch.randn(1, 80, ymax_); te0 = t_embed(torch.zeros(1))
    mel0 = torch.randn(1, 80, ymax_)
    p_te = B.convert(te_r, (ex,), os.path.join(B.HERE, "e2e_textenc.tflite"))
    p_dec = B.convert(dec_r, (x0, mu0, te0), os.path.join(B.HERE, "e2e_decoder.tflite"))
    p_gen = B.convert(gen_r, (mel0,), os.path.join(B.HERE, "e2e_vocoder.tflite"))
    from ai_edge_litert.interpreter import Interpreter
    it_te = Interpreter(model_path=p_te); it_te.allocate_tensors()
    it_dec = Interpreter(model_path=p_dec); it_dec.allocate_tensors()
    it_gen = Interpreter(model_path=p_gen); it_gen.allocate_tensors()

    def tfl_te(emb_x):
        o = B.tfl_run(it_te, emb_x.numpy())
        # order outputs as (mu,logw) by shape: mu has 80 ch, logw has 1 ch
        a, b = o
        mu, logw = (a, b) if a.shape[1] == 80 else (b, a)
        return torch.from_numpy(mu), torch.from_numpy(logw)
    def tfl_dec(x, mu, t_emb):
        return torch.from_numpy(B.tfl_run(it_dec, x.numpy(), mu.numpy(), t_emb.numpy())[0])
    def tfl_gen(mel):
        return B.tfl_run(it_gen, mel.numpy())[0]

    wav_tfl, *_ = host_pipeline(tfl_te, tfl_dec, tfl_gen, t_embed, emb_w, ids,
                                mel_mean, mel_std, N, z_full=z, use_mask=False)

    def corr(a, b):
        n = min(len(a), len(b)); return float(np.corrcoef(a[:n], b[:n])[0, 1])
    print(f"\ncorr(TFL, RT)   = {corr(wav_tfl, wav_rt):.6f}   <- conversion fidelity")
    print(f"corr(TFL, TRUE) = {corr(wav_tfl, wav_true):.6f}   <- real-world (incl. pad-leak)")
    print(f"corr(RT,  TRUE) = {corr(wav_rt, wav_true):.6f}   <- mask-drop pad-leak only")
    # save wavs for listening
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
