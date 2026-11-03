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

"""LFM3-Audio speech-output parts as fp32 torch modules that export to LiteRT (litert-torch), built from a state dict.

Parts (signature: inputs -> outputs), each a re-authoring of the vendor code at LiquidAI/LFM3-Audio-2B@9d49701c:
  FrameEmbed  frame_embed: codes [1,12] int32 -> embeds [1,1,2048]
              sum over k of codebook_embedding[k * 2017 + code_k]   (Lfm3AudioModel.get_audio_out_embeds)
  DepthStep   depth_step: hidden [1,2048] f32, codes [1,12] int32, masked [1,12] f32 (1 = hidden from the decoder,
              its code is ignored) -> logits [1,12,2017]           (Lfm3AudioDepthDecoder.forward = one maskgit pass)
  CodecChunk  decode: codes [1,12,F] int32, start [1] int32 = 75 Hz position of the chunk's first step (6 x the codec
              frames decoded before), k0 v0 k1 v1 k2 v2 [1,2,29,64] (rotated keys / values of the last 29 steps of
              decoder layers 2 / 4 / 6), c0..c4 [1,6,512] (the last 6 gated inputs of the short convs of layers
              0 / 1 / 3 / 5 / 7) -> frames [1,6F,1280] (irfft(1280) x synthesis window of every 75 Hz step) + the
              updated state under the same names with an `_out` suffix
              (LiquidCodecModel.decode up to the overlap-add; the host side is CodecStream below)

Weights come from a state dict through a key map: KEYMAPS["audio2b"] = the top-level names of
LiquidAI/LFM3-Audio-2B model.safetensors; KEYMAPS["hf"] = the names inside Lfm3AudioForConditionalGeneration, which
are also the names of the LFM3-Omni-2B checkpoint (model.codebook_embedding.weight, model.depth_decoder.*). The codec
reads audio_tokenizer/model.safetensors (the same file in both repos).

Graph rules applied: GQA folds the query heads that share a kv
head into the matmul rows (no repeat / BROADCAST_TO); masks and RoPE tables of fixed positions are eager constants;
every gather index is clamped in-graph; embedding tables stay fp32 in the graph; no torch.polar (cos / sin explicit);
irfft = a fixed fp32 DFT basis with the synthesis window folded in.
"""
import math
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CB = 12
CB_SIZE = 2017
EOS_CODE = 2016
LM_HIDDEN = 2048
DEPTH_HIDDEN = 1024

KEYMAPS = {
    # top-level names of the Audio-2B training export
    "audio2b": dict(codebook_embedding="codebook_embedding.weight", depth="", trunk="depthformer.model."),
    # module names of Lfm3AudioForConditionalGeneration = the LFM3-Omni-2B checkpoint
    "hf": dict(codebook_embedding="model.codebook_embedding.weight", depth="model.depth_decoder.",
               trunk="model.depth_decoder.model."),
}

# depth decoder geometry (config.json "depthformer"), codec decoder geometry (LiquidCodecConfig.lfm_config)
DEPTH_CFG = dict(layers=3, heads=32, kv_heads=8, head_dim=32, eps=1e-5, theta=1_000_000.0, positions=NUM_CB)
CODEC_CFG = dict(layers=8, attn_layers=(2, 4, 6), heads=8, kv_heads=2, head_dim=64, eps=1e-5, theta=1_000_000.0,
                 window=30, conv_kernel=7, dim=512, frame_stack=6, n_fft=1280, hop=320, fsq_levels=(8, 7, 6, 6))
CODEC_CONV_LAYERS = tuple(i for i in range(CODEC_CFG["layers"]) if i not in CODEC_CFG["attn_layers"])
CODEC_KV_LEN = CODEC_CFG["window"] - 1          # transformers keeps sliding_window - 1 past steps; the mask lets a
                                                # query read itself and the 29 before it (kv_idx > q_idx - window)
CODEC_CONV_LEN = CODEC_CFG["conv_kernel"] - 1


def read_tensors(path, names=None, prefix=None):
    """safetensors -> {name: fp32 tensor}; `names` (exact) or `prefix` (startswith) select a subset."""
    from safetensors import safe_open
    out = {}
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            if (names is None and prefix is None) or (names is not None and k in names) or \
                    (prefix is not None and k.startswith(prefix)):
                out[k] = f.get_tensor(k).to(torch.float32)
    return out


def depth_names(keymap="audio2b"):
    km = KEYMAPS[keymap] if isinstance(keymap, str) else keymap
    names = [km["depth"] + "depth_linear.weight", km["depth"] + "depth_linear.bias",
             km["depth"] + "mask_embedding", km["trunk"] + "embedding_norm.weight"]
    for k in range(NUM_CB):
        names += [f"{km['depth']}depth_embeddings.{k}.embedding.weight",
                  f"{km['depth']}depth_embeddings.{k}.embedding_norm.weight"]
    for i in range(DEPTH_CFG["layers"]):
        p = f"{km['trunk']}layers.{i}."
        names += [p + s for s in ("operator_norm.weight", "ffn_norm.weight", "self_attn.q_proj.weight",
                                  "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                                  "self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.gate_proj.weight",
                                  "feed_forward.w1.weight", "feed_forward.w2.weight", "feed_forward.w3.weight")]
    return names


def rms_norm(x, w, eps):
    # Lfm3RMSNorm / Lfm3AudioRMSNorm in fp32: weight * (x * rsqrt(mean(x^2) + eps))
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def inv_freq(head_dim, theta, partial=1.0):
    # Lfm3RotaryEmbedding.compute_default_rope_parameters (zero-tail partial rotation), fp32
    ra = int(partial * head_dim) // 2
    inv = 1.0 / (theta ** (torch.arange(0, 2 * ra, 2, dtype=torch.float) / head_dim))
    return torch.cat((inv, torch.zeros(head_dim // 2 - ra, dtype=torch.float)))


def rope_cos_sin(positions, inv):
    freqs = positions.float()[:, None] * inv[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


class Attention(nn.Module):
    """Lfm3Attention: QK RMSNorm, RoPE, GQA folded into the matmul rows, sigmoid output gate per head."""

    def __init__(self, w, heads, kv_heads, head_dim, eps):
        super().__init__()
        self.h, self.kv, self.hd, self.eps = heads, kv_heads, head_dim, eps
        for n in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"):
            self.register_buffer(n, w[f"self_attn.{n}.weight"].contiguous())
        self.register_buffer("q_norm", w["self_attn.q_norm.weight"].contiguous())
        self.register_buffer("k_norm", w["self_attn.k_norm.weight"].contiguous())

    def forward(self, h, cos, sin, bias=None, k_past=None, v_past=None):
        b, t, _ = h.shape
        q = rms_norm(F.linear(h, self.q_proj).view(b, t, self.h, self.hd), self.q_norm, self.eps).transpose(1, 2)
        k = rms_norm(F.linear(h, self.k_proj).view(b, t, self.kv, self.hd), self.k_norm, self.eps).transpose(1, 2)
        v = F.linear(h, self.v_proj).view(b, t, self.kv, self.hd).transpose(1, 2)
        q = q * cos + rot_half(q) * sin
        k = k * cos + rot_half(k) * sin
        if k_past is not None:
            k = torch.cat((k_past, k), dim=2)
            v = torch.cat((v_past, v), dim=2)
        rep = self.h // self.kv
        # query head j*rep + r reads kv head j (repeat_kv); rows of a group are r*t + position
        qg = q.reshape(b, self.kv, rep * t, self.hd)
        att = torch.matmul(qg, k.transpose(2, 3)) * (self.hd ** -0.5)
        if bias is not None:
            att = att + bias
        att = torch.softmax(att, dim=-1)
        o = torch.matmul(att, v).reshape(b, self.h, t, self.hd).transpose(1, 2)
        gate = torch.sigmoid(F.linear(h, self.gate_proj)).unsqueeze(-1)
        o = (o * gate).reshape(b, t, self.h * self.hd)
        return F.linear(o, self.o_proj), k, v


class ShortConv(nn.Module):
    """Lfm3ShortConv (no bias): in_proj -> B, C, x; depthwise causal conv (width 7) of B*x; C * conv -> out_proj.
    Channels-last: the conv is 7 shifted multiply-adds over [b, 6 + t, d] (state = the last 6 gated inputs)."""

    def __init__(self, w, dim, kernel):
        super().__init__()
        self.d, self.kernel = dim, kernel
        self.register_buffer("in_proj", w["conv.in_proj.weight"].contiguous())
        self.register_buffer("out_proj", w["conv.out_proj.weight"].contiguous())
        self.register_buffer("taps", w["conv.conv.weight"].reshape(dim, kernel).t().contiguous())   # [kernel, d]

    def forward(self, h, state):
        t = h.shape[1]
        bcx = F.linear(h, self.in_proj)
        bg, c, x = bcx[..., :self.d], bcx[..., self.d:2 * self.d], bcx[..., 2 * self.d:]
        full = torch.cat((state, bg * x), dim=1)                                  # [b, kernel-1+t, d]
        y = full[:, 0:t] * self.taps[0]
        for j in range(1, self.kernel):
            y = y + full[:, j:j + t] * self.taps[j]
        return F.linear(c * y, self.out_proj), full[:, t:]


class Layer(nn.Module):
    def __init__(self, w, attn=None, conv=None, eps=1e-5, sandwich=False):
        super().__init__()
        self.attn, self.conv, self.eps, self.sandwich = attn, conv, eps, sandwich
        self.register_buffer("op_norm", w["operator_norm.weight"].contiguous())
        self.register_buffer("ffn_norm", w["ffn_norm.weight"].contiguous())
        if sandwich:
            self.register_buffer("post_op_norm", w["post_operator_norm.weight"].contiguous())
            self.register_buffer("post_ffn_norm", w["post_ffn_norm.weight"].contiguous())
        for n in ("w1", "w2", "w3"):
            self.register_buffer(n, w[f"feed_forward.{n}.weight"].contiguous())

    def ffn(self, x):
        return F.linear(F.silu(F.linear(x, self.w1)) * F.linear(x, self.w3), self.w2)

    def forward(self, x, **kw):
        h = rms_norm(x, self.op_norm, self.eps)
        state = ()
        if self.attn is not None:
            o, k, v = self.attn(h, kw["cos"], kw["sin"], kw.get("bias"), kw.get("k_past"), kw.get("v_past"))
            state = (k, v)
        else:
            o, c = self.conv(h, kw["conv_state"])
            state = (c,)
        if self.sandwich:
            o = rms_norm(o, self.post_op_norm, self.eps)
        x = x + o
        f = self.ffn(rms_norm(x, self.ffn_norm, self.eps))
        if self.sandwich:
            f = rms_norm(f, self.post_ffn_norm, self.eps)
        return x + f, state


def _strip(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


class FrameEmbed(nn.Module):
    """codes [1,12] int32 -> [1,1,2048]: the LM input of a speech frame."""

    def __init__(self, sd, keymap="audio2b"):
        super().__init__()
        km = KEYMAPS[keymap] if isinstance(keymap, str) else keymap
        self.register_buffer("table", sd[km["codebook_embedding"]].contiguous())             # [12*2017, 2048]
        self.register_buffer("offsets", (torch.arange(NUM_CB, dtype=torch.int32) * CB_SIZE)[None])

    def forward(self, codes):
        idx = torch.clamp(codes, 0, CB_SIZE - 1) + self.offsets
        return F.embedding(idx, self.table).sum(dim=1, keepdim=True)


class DepthStep(nn.Module):
    """One maskgit pass of the depth decoder: hidden [1,2048], codes [1,12] int32, masked [1,12] f32 -> [1,12,2017]."""

    def __init__(self, sd, keymap="audio2b", split=False):
        super().__init__()
        km = KEYMAPS[keymap] if isinstance(keymap, str) else keymap
        c = DEPTH_CFG
        self.eps, self.split = c["eps"], split
        self.register_buffer("dl_w", sd[km["depth"] + "depth_linear.weight"].contiguous())
        self.register_buffer("dl_b", sd[km["depth"] + "depth_linear.bias"].contiguous())
        self.register_buffer("mask_emb", sd[km["depth"] + "mask_embedding"].contiguous())
        tables = [sd[f"{km['depth']}depth_embeddings.{k}.embedding.weight"] for k in range(NUM_CB)]
        self.register_buffer("code_table", torch.cat(tables).contiguous())                    # [12*2017, 1024]
        for k in range(NUM_CB):
            self.register_buffer(f"head{k}", tables[k].contiguous())                          # tied output head
            self.register_buffer(f"head_norm{k}",
                                 sd[f"{km['depth']}depth_embeddings.{k}.embedding_norm.weight"].contiguous())
        self.register_buffer("offsets", (torch.arange(NUM_CB, dtype=torch.int32) * CB_SIZE)[None])
        cos, sin = rope_cos_sin(torch.arange(c["positions"]), inv_freq(c["head_dim"], c["theta"]))
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.register_buffer("final_norm", sd[km["trunk"] + "embedding_norm.weight"].contiguous())
        layers = []
        for i in range(c["layers"]):
            w = _strip(sd, f"{km['trunk']}layers.{i}.")
            layers.append(Layer(w, attn=Attention(w, c["heads"], c["kv_heads"], c["head_dim"], c["eps"]),
                                eps=c["eps"]))
        self.layers = nn.ModuleList(layers)

    def project(self, hidden):
        """depth_linear: [1,2048] -> [1,12,1024] (the split depth form computes this once per frame)."""
        return F.linear(hidden, self.dl_w, self.dl_b).view(1, NUM_CB, DEPTH_HIDDEN)

    def passes(self, depth_in, codes, masked):
        emb = F.embedding(torch.clamp(codes, 0, CB_SIZE - 1) + self.offsets, self.code_table)   # [1,12,1024]
        m = masked.unsqueeze(-1)
        x = depth_in + (m * self.mask_emb + (1.0 - m) * emb)       # torch.where(masked, mask_embedding, code_embeds)
        for layer in self.layers:
            x, _ = layer(x, cos=self.cos, sin=self.sin)            # bidirectional: no mask
        x = rms_norm(x, self.final_norm, self.eps)
        logits = [F.linear(rms_norm(x[:, k], getattr(self, f"head_norm{k}"), self.eps), getattr(self, f"head{k}"))
                  for k in range(NUM_CB)]
        return torch.stack(logits, dim=1)

    def forward(self, hidden, codes, masked):
        if self.split:
            return self.passes(hidden, codes, masked)   # hidden = the projected [1,12,1024]
        return self.passes(self.project(hidden), codes, masked)


def fsq_latent_table(levels=CODEC_CFG["fsq_levels"]):
    """code -> 4 latents, LiquidCodecQuantizer.decode in its own arithmetic (int // and %, then a float division)."""
    lv = torch.tensor(levels, dtype=torch.int32)
    basis = torch.tensor([math.prod(levels[:i]) for i in range(len(levels))], dtype=torch.int32)
    idx = torch.arange(CB_SIZE, dtype=torch.long)[:, None]
    half = lv // 2
    return ((idx // basis) % lv - half) / half                                                 # [2017, 4] f32


def irfft_window_basis(window, n_fft=CODEC_CFG["n_fft"]):
    """[re; im] (2 x (n_fft/2+1) bins) -> windowed time frame: torch.fft.irfft(n=n_fft, norm='backward') * window.
    The imaginary parts of bin 0 and bin n_fft/2 do not reach the output (C2R)."""
    nb = n_fft // 2 + 1
    k = np.arange(nb)[:, None]
    n = np.arange(n_fft)[None, :]
    ang = 2.0 * np.pi * ((k * n) % n_fft) / n_fft
    a = np.full((nb, 1), 2.0)
    a[0] = a[-1] = 1.0
    w = np.asarray(window, np.float64)[None, :]
    c = a * np.cos(ang) / n_fft * w
    s = -a * np.sin(ang) / n_fft * w
    s[0] = s[-1] = 0.0
    return torch.from_numpy(np.concatenate([c, s], 0).astype(np.float32))                     # [2*nb, n_fft]


class CodecChunk(nn.Module):
    """Codec decoder of F frames with state; see the module docstring for the I/O."""

    def __init__(self, sd, frames, window=None):
        super().__init__()
        c = CODEC_CFG
        self.f, self.t, self.eps = frames, frames * c["frame_stack"], c["eps"]
        self.register_buffer("lat_table", fsq_latent_table())
        self.register_buffer("po_w", sd["quantizer.project_out.weight"].contiguous())
        self.register_buffer("po_b", sd["quantizer.project_out.bias"].contiguous())
        self.register_buffer("fq_w", sd["from_quantizer.weight"].contiguous())
        self.register_buffer("fq_b", sd["from_quantizer.bias"].contiguous())
        self.register_buffer("lin_w", sd["decoder.lin.weight"].contiguous())
        self.register_buffer("lin_b", sd["decoder.lin.bias"].contiguous())
        self.register_buffer("final_norm", sd["decoder.lfm.embedding_norm.weight"].contiguous())
        win = sd["decoder.istft.window"] if window is None else window
        self.register_buffer("basis_t", irfft_window_basis(win.double().numpy()).t().contiguous())  # [1280, 1282]
        self.register_buffer("inv_freq", inv_freq(c["head_dim"], c["theta"]))
        self.register_buffer("steps", torch.arange(self.t, dtype=torch.float32))
        L = CODEC_KV_LEN + self.t
        krel = torch.arange(-CODEC_KV_LEN, self.t, dtype=torch.float32)                         # key position - start
        d = self.steps[:, None] - krel[None, :]
        static = torch.where((d >= 0) & (d < c["window"]), 0.0, -1e9).float()                  # [t, L]
        rep = c["heads"] // c["kv_heads"]
        self.register_buffer("static_bias", static.repeat(rep, 1).contiguous())                # [rep*t, L]
        self.register_buffer("krel", krel)
        layers = []
        for i in range(c["layers"]):
            w = _strip(sd, f"decoder.lfm.layers.{i}.")
            if i in c["attn_layers"]:
                layers.append(Layer(w, attn=Attention(w, c["heads"], c["kv_heads"], c["head_dim"], c["eps"]),
                                    eps=c["eps"], sandwich=True))
            else:
                layers.append(Layer(w, conv=ShortConv(w, c["dim"], c["conv_kernel"]), eps=c["eps"], sandwich=True))
        self.layers = nn.ModuleList(layers)
        self.nbins = c["n_fft"] // 2 + 1
        assert L == self.static_bias.shape[1]

    def forward(self, codes, start, k0, v0, k1, v1, k2, v2, c0, c1, c2, c3, c4):
        c = CODEC_CFG
        lat = F.embedding(torch.clamp(codes, 0, CB_SIZE - 1), self.lat_table)                  # [1,12,F,4]
        lat = lat.permute(0, 2, 1, 3).reshape(1, self.f, NUM_CB * len(c["fsq_levels"]))
        emb = F.linear(F.linear(lat, self.po_w, self.po_b), self.fq_w, self.fq_b)              # [1,F,512]
        x = torch.cat([emb] * c["frame_stack"], dim=-1).reshape(1, self.t, c["dim"])           # nearest x6
        s = start.float().reshape(1)
        cos, sin = rope_cos_sin(s + self.steps, self.inv_freq)
        # cache slots before position 0 (the stream's first chunks) are empty: mask them
        bias = self.static_bias + torch.clamp(-(self.krel + s), 0.0, 1.0) * -1e9
        kv = [(k0, v0), (k1, v1), (k2, v2)]
        convs = [c0, c1, c2, c3, c4]
        kv_out, conv_out = [], []
        ia = ic = 0
        for i, layer in enumerate(self.layers):
            if i in c["attn_layers"]:
                x, (k, v) = layer(x, cos=cos, sin=sin, bias=bias, k_past=kv[ia][0], v_past=kv[ia][1])
                kv_out += [k[:, :, -CODEC_KV_LEN:], v[:, :, -CODEC_KV_LEN:]]
                ia += 1
            else:
                x, (cs,) = layer(x, conv_state=convs[ic])
                conv_out.append(cs)
                ic += 1
        x = rms_norm(x, self.final_norm, self.eps)
        y = F.linear(x, self.lin_w, self.lin_b)                                                 # [1,t,1282]
        mag = torch.exp(y[..., :self.nbins])
        ang = y[..., self.nbins:]
        z = torch.cat((mag * torch.cos(ang), mag * torch.sin(ang)), dim=-1)
        frames = F.linear(z, self.basis_t)                                                      # [1,t,1280]
        return (frames, *kv_out, *conv_out)


CODEC_STATE_NAMES = ["k0", "v0", "k1", "v1", "k2", "v2", "c0", "c1", "c2", "c3", "c4"]


def codec_zero_state():
    kv = np.zeros((1, CODEC_CFG["kv_heads"], CODEC_KV_LEN, CODEC_CFG["head_dim"]), np.float32)
    cv = np.zeros((1, CODEC_CONV_LEN, CODEC_CFG["dim"]), np.float32)
    return {n: (kv.copy() if n[0] in "kv" else cv.copy()) for n in CODEC_STATE_NAMES}


class CodecStream:
    """Host side of the codec: overlap-add (hop 320) of the windowed frames, the window-power envelope division, the
    960-sample tail held between chunks, and the 480-sample trims at both ends (LiquidCodecModel.decode, streaming
    branch). push(frames [t,1280]) -> finished samples; finish() -> the rest."""

    def __init__(self, window):
        c = CODEC_CFG
        self.n_fft, self.hop = c["n_fft"], c["hop"]
        self.trim = (self.n_fft - self.hop) // 2
        self.tail_len = 2 * self.trim
        self.w2 = np.asarray(window, np.float32) ** 2
        self.audio = None
        self.env = None
        self.started = False

    def _ola(self, frames):
        t = frames.shape[0]
        size = (t - 1) * self.hop + self.n_fft
        audio = np.zeros(size, np.float32)
        env = np.zeros(size, np.float32)
        for i in range(t):
            audio[i * self.hop:i * self.hop + self.n_fft] += frames[i]
            env[i * self.hop:i * self.hop + self.n_fft] += self.w2
        return audio, env

    def push(self, frames, finalize=False):
        audio, env = self._ola(np.asarray(frames, np.float32).reshape(-1, self.n_fft))
        if self.audio is not None:
            audio[:self.tail_len] += self.audio
            env[:self.tail_len] += self.env
        if not self.started:
            audio, env = audio[self.trim:], env[self.trim:]
            self.started = True
        if finalize:
            audio, env = audio[:-self.trim], env[:-self.trim]
            self.audio = self.env = None
        else:
            self.audio, self.env = audio[-self.tail_len:].copy(), env[-self.tail_len:].copy()
            audio, env = audio[:-self.tail_len], env[:-self.tail_len]
        return audio / env

    def finish(self):
        """Flush the held tail when the last chunk was pushed with finalize=False."""
        if self.audio is None:
            return np.zeros(0, np.float32)
        audio, env = self.audio[:-self.trim], self.env[:-self.trim]
        self.audio = self.env = None
        return audio / env


def load_codec_state_dict(path):
    return read_tensors(path, prefix=None)


def load_depth_state_dict(path, keymap="audio2b"):
    return read_tensors(path, names=set(depth_names(keymap)))


def load_frame_state_dict(path, keymap="audio2b"):
    km = KEYMAPS[keymap] if isinstance(keymap, str) else keymap
    return read_tensors(path, names={km["codebook_embedding"]})


# ---------------------------------------------------------------- export (litert-torch)

def _export(module, name, sample_kwargs, out_path):
    import litert_torch
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = litert_torch.signature(name, module.eval(), sample_kwargs=sample_kwargs).convert()
    model.export(str(out_path))
    return out_path


class FrameEmbedSig(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, codes):
        return {"embeds": self.m(codes)}


class DepthStepSig(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, hidden, codes, masked):
        return {"logits": self.m(hidden, codes, masked)}


class CodecChunkSig(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, codes, start, k0, v0, k1, v1, k2, v2, c0, c1, c2, c3, c4):
        out = self.m(codes, start, k0, v0, k1, v1, k2, v2, c0, c1, c2, c3, c4)
        return dict(zip(["frames"] + [n + "_out" for n in CODEC_STATE_NAMES], out))


def frame_embed_sample():
    return {"codes": torch.zeros(1, NUM_CB, dtype=torch.int32)}


def depth_step_sample():
    return {"hidden": torch.zeros(1, LM_HIDDEN), "codes": torch.zeros(1, NUM_CB, dtype=torch.int32),
            "masked": torch.ones(1, NUM_CB)}


def codec_chunk_sample(frames):
    kw = {"codes": torch.zeros(1, NUM_CB, frames, dtype=torch.int32), "start": torch.zeros(1, dtype=torch.int32)}
    kw.update({n: torch.from_numpy(v) for n, v in codec_zero_state().items()})
    return kw


def export_frame_embed(sd, out_path, keymap="audio2b"):
    """sd: a state dict holding the codebook_embedding table under the keymap's name."""
    return _export(FrameEmbedSig(FrameEmbed(sd, keymap)), "frame_embed", frame_embed_sample(), out_path)


def export_depth_step(sd, out_path, keymap="audio2b"):
    """sd: a state dict holding depth_names(keymap) (load_depth_state_dict reads only those)."""
    return _export(DepthStepSig(DepthStep(sd, keymap)), "depth_step", depth_step_sample(), out_path)


def export_codec_chunk(sd, frames, out_path):
    """sd: audio_tokenizer/model.safetensors (load_codec_state_dict); frames = codec frames per call (F)."""
    return _export(CodecChunkSig(CodecChunk(sd, frames)), "decode", codec_chunk_sample(frames), out_path)
