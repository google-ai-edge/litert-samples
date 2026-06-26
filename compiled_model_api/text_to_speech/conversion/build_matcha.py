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
"""Matcha-TTS -> CompiledModel GPU: REAL-WEIGHT conversion + parity.

Loads matcha_ljspeech.ckpt + generator_v1 (HiFi-GAN T2 v1), applies the proven
re-authoring recipe to the 3 heavy graphs (text encoder / CFM decoder / HiFi-GAN
vocoder), converts each with litert-torch, and verifies:
  (a) re-authored torch == original torch  (mask-drop is a no-op at full length)
  (b) tflite == re-authored torch          (conversion fidelity)
  (c) end-to-end waveform: full tflite-orchestrated pipeline vs torch synthesise()

Host (CPU) does: phoneme embedding lookup, duration -> length-regulator (expand mu),
sinusoidal time-embed per ODE step, the Euler ODE loop, mel denormalize. The 3 GPU
graphs are text_encoder(emb)->(mu,logw), decoder(x,mu,t_emb)->v, vocoder(mel)->wav.

Run:  ~/clipconv/bin/python build_matcha.py [stage]
      stage in {opcheck, parity, e2e, all}   (default: all)
"""
import _stub  # FIRST: scipy/_propack + getsourcefile guards
import sys, os, types, math, logging, collections
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from types import SimpleNamespace as NS

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "matcha_ljspeech.ckpt")
HIFI = os.path.join(HERE, "generator_v1")
PHON_LIB = "/opt/homebrew/lib/libespeak-ng.dylib"
os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", PHON_LIB)
torch.manual_seed(0)

BANNED = {"GATHER","GATHER_ND","TOPK_V2","GELU","ERF","WHERE","SELECT","SELECT_V2",
          "BROADCAST_TO","POW","TRANSPOSE_CONV","CAST","EMBEDDING_LOOKUP",
          "RFFT2D","FFT","STFT","COMPLEX","RFFT","IRFFT"}


# ----------------------------------------------------------------------------- stubs
def stub_matcha_utils():
    def sequence_mask(length, max_length=None):
        if max_length is None: max_length = int(length.max())
        a = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return a.unsqueeze(0) < length.unsqueeze(1)
    class _UMod(types.ModuleType):
        __file__ = "<stub:matcha.utils>"
        def __getattr__(s, n):
            if n.startswith("__"): raise AttributeError(n)
            if n == "get_pylogger": return lambda name=None: logging.getLogger(name)
            return sequence_mask if n == "sequence_mask" else (lambda *a, **k: None)
    upkg = _UMod("matcha.utils"); upkg.__path__ = []
    um = types.ModuleType("matcha.utils.model"); um.sequence_mask = sequence_mask
    sys.modules["matcha.utils"] = upkg; sys.modules["matcha.utils.model"] = um
stub_matcha_utils()


# ----------------------------------------------------------- proven re-authoring ops
class GN4D(nn.Module):
    """GroupNorm w/o GATHER_ND: reshape (B,G,C//G,T), mean/var over (2,3)."""
    def __init__(s, gn):
        super().__init__(); s.g = gn.num_groups; s.eps = gn.eps
        s.register_buffer("w", gn.weight.detach().reshape(1, -1, 1))
        s.register_buffer("b", gn.bias.detach().reshape(1, -1, 1))
    def forward(s, x):
        b, c, t = x.shape; x = x.reshape(b, s.g, c // s.g, t)
        m = x.mean((2, 3), keepdim=True); v = (x - m).pow(2).mean((2, 3), keepdim=True)
        return (((x - m) * torch.rsqrt(v + s.eps)).reshape(b, c, t)) * s.w + s.b


class MishClean(nn.Module):
    """Mish w/ SELECT-free fp16-safe softplus relu(x)+log1p(exp(-|x|)). Exact. [C6]"""
    def forward(s, x):
        return x * torch.tanh(torch.relu(x) + torch.log1p(torch.exp(-torch.abs(x))))


class ZeroStuffConvT1d(nn.Module):
    """Exact GPU-clean ConvTranspose1d (DAC): 2D nearest upsample x const mask +
    flipped conv1d + crop. [C20/C31/C32]"""
    def __init__(s, ct, L):
        super().__init__()
        s.s = ct.stride[0]; s.k = ct.kernel_size[0]; s.p = ct.padding[0]
        s.op = ct.output_padding[0]; s.L = L
        s.register_buffer("w", ct.weight.flip(2).transpose(0, 1).contiguous())
        s.register_buffer("b", ct.bias.detach().clone() if ct.bias is not None
                          else torch.zeros(ct.out_channels))
        mk = np.zeros((L * s.s,), np.float32); mk[::s.s] = 1.0
        s.register_buffer("mask", torch.from_numpy(mk)[None, None])
    def forward(s, x):
        xn = F.interpolate(x.unsqueeze(2), size=(1, s.L * s.s), mode="nearest").squeeze(2) * s.mask
        y = F.conv1d(xn, s.w, bias=s.b, padding=s.k - 1)
        ol = (s.L - 1) * s.s + s.k - 2 * s.p + s.op
        return y[:, :, s.p:s.p + ol]


def swap_norm_act(m):
    for n, c in list(m.named_children()):
        if isinstance(c, nn.GroupNorm): setattr(m, n, GN4D(c))
        elif isinstance(c, nn.Mish): setattr(m, n, MishClean())
        else: swap_norm_act(c)


def swap_convtranspose(model, lengths):
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.ConvTranspose1d):
            par = model; *pth, last = name.split(".")
            for q in pth: par = getattr(par, q)
            setattr(par, last, ZeroStuffConvT1d(mod, lengths[name]))


# ------------------------------------------------------------------- model builders
def load_sd():
    return torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]


def build_text_encoder(sd):
    from matcha.models.components.text_encoder import TextEncoder
    enc = NS(n_feats=80, n_channels=192, filter_channels=768, filter_channels_dp=256,
             n_heads=2, n_layers=6, kernel_size=3, p_dropout=0.0, spk_emb_dim=64,
             n_spks=1, prenet=True)
    dp = NS(filter_channels_dp=256, kernel_size=3, p_dropout=0.0)
    te = TextEncoder("RoPE Encoder", enc, dp, n_vocab=178, n_spks=1, spk_emb_dim=64)
    w = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    te.load_state_dict(w, strict=True)
    return te.eval()


def build_decoder(sd):
    from matcha.models.components.decoder import Decoder
    dec = Decoder(in_channels=160, out_channels=80, channels=(256, 256), dropout=0.0,
                  attention_head_dim=64, n_blocks=1, num_mid_blocks=2, num_heads=2,
                  act_fn="snakebeta")
    w = {k[len("decoder.estimator."):]: v for k, v in sd.items()
         if k.startswith("decoder.estimator.")}
    dec.load_state_dict(w, strict=True)
    return dec.eval()


def build_hifigan():
    from matcha.hifigan.models import Generator
    from matcha.hifigan.config import v1
    from matcha.hifigan.env import AttrDict
    h = AttrDict(v1)
    g = Generator(h)
    ckpt = torch.load(HIFI, map_location="cpu", weights_only=False)
    g.load_state_dict(ckpt["generator"])
    g.eval(); g.remove_weight_norm()
    return g


# --------------------------------------------------- re-authored graph wrappers
def reauth_text_encoder(te):
    """Drop the all-ones mask (kills SELECT_V2 x6 + CAST); RoPE/attn already clean.
    Graph input = gathered phoneme embeddings (B,T,192), unscaled. Output (mu,logw)."""
    import types as _t
    nc = te.n_channels
    # patch matcha MultiHeadAttention to skip masked_fill (mask is all-ones)
    for mo in te.modules():
        if type(mo).__name__ == "MultiHeadAttention":
            def mha_forward(self, x, c, attn_mask=None):
                q = self.conv_q(x); k = self.conv_k(c); v = self.conv_v(c)
                out, _ = self.attention(q, k, v, mask=None)
                return self.conv_o(out)
            mo.forward = _t.MethodType(mha_forward, mo)

    class TEWrap(nn.Module):
        def __init__(s, te): super().__init__(); s.te = te; s.nc = nc
        def forward(s, emb_x):                 # emb_x: (1, T, 192)
            t = s.te
            x = (emb_x * math.sqrt(s.nc)).transpose(1, -1)   # (1,192,T)
            m = torch.ones(1, 1, x.shape[-1])
            x = t.prenet(x, m)
            x = t.encoder(x, m)
            mu = t.proj_m(x) * m
            logw = t.proj_w(x, m)
            return mu, logw
    return TEWrap(te).eval()


def reauth_decoder(dec, T):
    """Host-side time embed; GN->4D, Mish->softplus, ConvT->ZeroStuff, drop attn mask.
    Graph input = (x(1,80,T), mu(1,80,T), t_emb(1,1024)); output v(1,80,T)."""
    import types as _t
    # host-compute the time embedding -> Identity inside graph
    dec.time_embeddings = nn.Identity(); dec.time_mlp = nn.Identity()
    # discover ConvTranspose1d input length, then swap
    L = {}; hk = []
    for n, mo in dec.named_modules():
        if isinstance(mo, nn.ConvTranspose1d):
            hk.append(mo.register_forward_hook(
                (lambda nm: (lambda m, i, o: L.__setitem__(nm, i[0].shape[-1])))(n)))
    with torch.no_grad():
        dec(torch.randn(1, 80, T), torch.ones(1, 1, T), torch.randn(1, 80, T), torch.randn(1, 1024))
    for h in hk: h.remove()
    swap_convtranspose(dec, L)
    swap_norm_act(dec)
    # drop the all-ones attention mask in diffusers Attention
    for mo in dec.modules():
        if type(mo).__name__ == "Attention":
            mo.forward = (lambda orig: (lambda hidden_states, encoder_hidden_states=None,
                          attention_mask=None, **k: orig(hidden_states,
                          encoder_hidden_states=encoder_hidden_states)))(mo.forward)

    class DecWrap(nn.Module):
        def __init__(s, d): super().__init__(); s.d = d
        def forward(s, x, mu, t_emb):
            return s.d(x, torch.ones(1, 1, x.shape[-1]), mu, t_emb)
    return DecWrap(dec).eval()


def reauth_text_encoder_masked(te):
    """Shippable: keep the mask as a runtime float input (additive attn bias, no
    SELECT/CAST). Handles padded (variable-length) phoneme sequences GPU-clean.
    Graph inputs: emb_x (1,Tmax,192), tmask (1,1,Tmax) float. Outputs (mu,logw)."""
    import types as _t
    from einops import rearrange
    nc = te.n_channels
    for mo in te.modules():
        if type(mo).__name__ == "MultiHeadAttention":
            def mha_forward(self, x, c, attn_mask=None):
                q = self.conv_q(x); k = self.conv_k(c); v = self.conv_v(c)
                h = self.n_heads
                q = rearrange(q, "b (h c) t -> b h t c", h=h)
                k = rearrange(k, "b (h c) t -> b h t c", h=h)
                v = rearrange(v, "b (h c) t -> b h t c", h=h)
                q = self.query_rotary_pe(q); k = self.key_rotary_pe(k)
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.k_channels)
                if attn_mask is not None:
                    scores = scores + (attn_mask - 1.0) * 1e4   # additive, no SELECT
                p = torch.softmax(scores, dim=-1)
                out = torch.matmul(p, v).transpose(2, 3).contiguous().view(
                    q.shape[0], h * self.k_channels, -1)
                return self.conv_o(out)
            mo.forward = _t.MethodType(mha_forward, mo)

    class TEWrapM(nn.Module):
        def __init__(s, te): super().__init__(); s.te = te; s.nc = nc
        def forward(s, emb_x, tmask):
            t = s.te
            x = (emb_x * math.sqrt(s.nc)).transpose(1, -1)
            x = t.prenet(x, tmask)
            x = t.encoder(x, tmask)
            return t.proj_m(x) * tmask, t.proj_w(x, tmask)
    return TEWrapM(te).eval()


def _decoder_forward_clean(self, x, mask, mu, t, spks=None, cond=None):
    """Copy of matcha Decoder.forward with the one GPU-hostile op fixed: the
    half-res mask `mask_down[:, :, ::2]` (step-2 slice -> GATHER_ND) is replaced
    by a reshape+index decimation (RESHAPE+SLICE, GPU-clean). Identical result."""
    from einops import pack, rearrange, repeat

    def dec2(m):                      # m (b,1,L) with L even -> (b,1,L//2) == m[:,:,::2]
        b, c, L = m.shape
        return m.reshape(b, c, L // 2, 2)[:, :, :, 0]

    t = self.time_embeddings(t); t = self.time_mlp(t)
    x = pack([x, mu], "b * t")[0]
    if spks is not None:
        spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
        x = pack([x, spks], "b * t")[0]
    hiddens = []; masks = [mask]
    for resnet, transformer_blocks, downsample in self.down_blocks:
        mask_down = masks[-1]
        x = resnet(x, mask_down, t)
        x = rearrange(x, "b c t -> b t c")
        mask_down = rearrange(mask_down, "b 1 t -> b t")
        for tb in transformer_blocks:
            x = tb(hidden_states=x, attention_mask=mask_down, timestep=t)
        x = rearrange(x, "b t c -> b c t")
        mask_down = rearrange(mask_down, "b t -> b 1 t")
        hiddens.append(x)
        x = downsample(x * mask_down)
        masks.append(dec2(mask_down))
    masks = masks[:-1]
    mask_mid = masks[-1]
    for resnet, transformer_blocks in self.mid_blocks:
        x = resnet(x, mask_mid, t)
        x = rearrange(x, "b c t -> b t c")
        mask_mid = rearrange(mask_mid, "b 1 t -> b t")
        for tb in transformer_blocks:
            x = tb(hidden_states=x, attention_mask=mask_mid, timestep=t)
        x = rearrange(x, "b t c -> b c t")
        mask_mid = rearrange(mask_mid, "b t -> b 1 t")
    for resnet, transformer_blocks, upsample in self.up_blocks:
        mask_up = masks.pop()
        x = resnet(pack([x, hiddens.pop()], "b * t")[0], mask_up, t)
        x = rearrange(x, "b c t -> b t c")
        mask_up = rearrange(mask_up, "b 1 t -> b t")
        for tb in transformer_blocks:
            x = tb(hidden_states=x, attention_mask=mask_up, timestep=t)
        x = rearrange(x, "b t c -> b c t")
        mask_up = rearrange(mask_up, "b t -> b 1 t")
        x = upsample(x * mask_up)
    x = self.final_block(x, mask_up)
    output = self.final_proj(x * mask_up)
    return output * mask


def sin_pos_emb(t, dim=160, scale=1000.0):
    """matcha SinusoidalPosEmb, host-side (no learned weights). t: (B,) -> (B,dim)."""
    half = dim // 2
    freqs = torch.exp(torch.arange(half, dtype=torch.float32) * (-math.log(10000) / (half - 1)))
    e = scale * t.reshape(-1, 1) * freqs.reshape(1, -1)
    return torch.cat([e.sin(), e.cos()], dim=-1)


def reauth_decoder_masked(dec, T):
    """Shippable: keep mask as runtime float input. Diffusers Attention -> manual
    additive-masked attention. GN->4D, Mish->softplus, ConvT->ZeroStuff. The learned
    time-MLP stays IN-graph (GPU); host feeds only the weight-free SinusoidalPosEmb.
    Graph inputs: x(1,80,T), mu(1,80,T), t_sin(1,160), mask(1,1,T). Output v(1,80,T)."""
    import types as _t
    dec.time_embeddings = nn.Identity()      # host computes sin_pos_emb; keep time_mlp on GPU
    L = {}; hk = []
    for n, mo in dec.named_modules():
        if isinstance(mo, nn.ConvTranspose1d):
            hk.append(mo.register_forward_hook(
                (lambda nm: (lambda m, i, o: L.__setitem__(nm, i[0].shape[-1])))(n)))
    with torch.no_grad():
        dec(torch.randn(1, 80, T), torch.ones(1, 1, T), torch.randn(1, 80, T), torch.randn(1, 160))
    for h in hk: h.remove()
    swap_convtranspose(dec, L)
    swap_norm_act(dec)
    for mo in dec.modules():
        if type(mo).__name__ == "Attention":
            def attn_forward(self, hidden_states, encoder_hidden_states=None,
                             attention_mask=None, **kw):
                b, seq, _ = hidden_states.shape
                h = self.heads
                q = self.to_q(hidden_states); k = self.to_k(hidden_states); v = self.to_v(hidden_states)
                d = q.shape[-1] // h
                q = q.reshape(b, seq, h, d).transpose(1, 2)
                k = k.reshape(b, seq, h, d).transpose(1, 2)
                v = v.reshape(b, seq, h, d).transpose(1, 2)
                scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
                if attention_mask is not None:                 # (b, seq) float 1/0
                    # replicate diffusers AttnProcessor2_0: SDPA adds the raw 0/1
                    # mask to scores (a soft +bias on valid cols, NOT -inf masking)
                    scores = scores + attention_mask.reshape(b, 1, 1, seq)
                p = torch.softmax(scores, dim=-1)
                out = torch.matmul(p, v).transpose(1, 2).reshape(b, seq, h * d)
                return self.to_out[0](out)
            mo.forward = _t.MethodType(attn_forward, mo)

    dec.forward = _t.MethodType(_decoder_forward_clean, dec)   # GPU-clean half-res mask

    class DecWrapM(nn.Module):
        def __init__(s, d): super().__init__(); s.d = d
        def forward(s, x, mu, t_emb, mask):
            return s.d(x, mask, mu, t_emb)
    return DecWrapM(dec).eval()


def reauth_hifigan(g, T):
    L = {}; hk = []
    for n, mo in g.named_modules():
        if isinstance(mo, nn.ConvTranspose1d):
            hk.append(mo.register_forward_hook(
                (lambda nm: (lambda m, i, o: L.__setitem__(nm, i[0].shape[-1])))(n)))
    with torch.no_grad():
        g(torch.randn(1, 80, T))
    for h in hk: h.remove()
    swap_convtranspose(g, L)
    return g.eval()


# --------------------------------------------------------------- op-check helper
def opcheck(path, label):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path); it.allocate_tensors()
    ops = collections.Counter(d.get("op_name", "?") for d in it._get_ops_details())
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
    over = sum(1 for d in it.get_tensor_details() if len(d.get("shape", [])) > 4)
    fft = {k: v for k, v in ops.items() if any(t in k.upper() for t in ("FFT", "STFT", "COMPLEX"))}
    print(f"[{label}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    print(f"[{label}] banned:{bad or 'NONE'} >4D:{over} FFT:{fft or 'NONE'} "
          f"size {os.path.getsize(path)/1e6:.1f}MB")
    verdict = "GPU-CLEAN" if not bad and not over and not fft else f"BLOCKERS {bad}"
    print(f"[{label}] VERDICT: {verdict}")
    return it, (not bad and not over and not fft)


def tfl_run(it, *inputs):
    ins = it.get_input_details()
    for d, x in zip(ins, inputs):
        it.set_tensor(d["index"], x.astype(d["dtype"]))
    it.invoke()
    outs = it.get_output_details()
    return [it.get_tensor(d["index"]) for d in outs]


# --------------------------------------------------------------- convert one graph
def convert(mod, example_inputs, out):
    import litert_torch
    litert_torch.convert(mod, example_inputs).export(out)
    return out


def to_fp16(fp32_path, fp16_path):
    """fp16 weights via AI Edge Quantizer FLOAT_CASTING (the zoo standard)."""
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT),
        algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(fp16_path):
        os.remove(fp16_path)
    qt = quantizer.Quantizer(float_model=fp32_path)
    qt.load_quantization_recipe(rm.get_quantization_recipe())
    qt.quantize().export_model(fp16_path)
    return fp16_path


# =============================================================================
def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    sd = load_sd()
    mel_mean = float(sd["mel_mean"]); mel_std = float(sd["mel_std"])
    print(f"mel_mean={mel_mean:.4f} mel_std={mel_std:.4f}")

    # representative fixed length for op-check / per-graph parity
    Ttext, Tmel = 48, 128

    # ---- build + re-author
    te = build_text_encoder(sd)
    dec = build_decoder(sd)
    gen = build_hifigan()
    te_r = reauth_text_encoder(build_text_encoder(sd))
    dec_r = reauth_decoder(build_decoder(sd), Tmel)
    gen_r = reauth_hifigan(build_hifigan(), Tmel)

    emb_w = sd["encoder.emb.weight"]           # (178,192) host lookup table

    # (a) re-authored == original (full-length, mask is no-op)
    print("\n=== (a) re-authoring is a no-op vs original torch ===")
    ids = torch.randint(0, 178, (1, Ttext))
    with torch.no_grad():
        mu0, logw0, _ = te(ids, torch.tensor([Ttext]))
        ex = emb_w[ids]                        # (1,Ttext,192)
        mu1, logw1 = te_r(ex)
    print(f"textenc mu  corr {np.corrcoef(mu0.numpy().ravel(), mu1.numpy().ravel())[0,1]:.6f} "
          f"max|d| {(mu0-mu1).abs().max():.2e}")
    print(f"textenc logw corr {np.corrcoef(logw0.numpy().ravel(), logw1.numpy().ravel())[0,1]:.6f} "
          f"max|d| {(logw0-logw1).abs().max():.2e}")
    # decoder: original vs re-authored at fixed T
    x = torch.randn(1, 80, Tmel); mu = torch.randn(1, 80, Tmel); t = torch.rand(1)
    with torch.no_grad():
        t_emb = dec.time_mlp(dec.time_embeddings(t)) if not isinstance(dec.time_mlp, nn.Identity) else None
    # dec was re-authored in-place for dec_r build via separate instance; rebuild clean orig:
    dec_orig = build_decoder(sd)
    with torch.no_grad():
        t_emb = dec_orig.time_mlp(dec_orig.time_embeddings(t))
        v0 = dec_orig(x, torch.ones(1, 1, Tmel), mu, t)
        v1 = dec_r(x, mu, t_emb)
    print(f"decoder corr {np.corrcoef(v0.numpy().ravel(), v1.numpy().ravel())[0,1]:.6f} "
          f"max|d| {(v0-v1).abs().max():.2e}")
    mel_in = torch.randn(1, 80, Tmel)
    with torch.no_grad():
        w0 = gen(mel_in); w1 = gen_r(mel_in)
    print(f"vocoder corr {np.corrcoef(w0.numpy().ravel(), w1.numpy().ravel())[0,1]:.6f} "
          f"max|d| {(w0-w1).abs().max():.2e}")

    if stage in ("opcheck", "parity", "all"):
        print("\n=== convert + op-check (fp32) ===")
        te_p = convert(te_r, (ex,), os.path.join(HERE, "matcha_textenc.tflite"))
        dec_p = convert(dec_r, (x, mu, t_emb), os.path.join(HERE, "matcha_decoder.tflite"))
        gen_p = convert(gen_r, (mel_in,), os.path.join(HERE, "matcha_vocoder.tflite"))
        it_te, c1 = opcheck(te_p, "textenc")
        it_dec, c2 = opcheck(dec_p, "decoder")
        it_gen, c3 = opcheck(gen_p, "vocoder")

        print("\n=== (b) tflite == re-authored torch ===")
        def corr(a, b): n = min(a.size, b.size); return np.corrcoef(a.ravel()[:n], b.ravel()[:n])[0, 1]
        o = tfl_run(it_te, ex.numpy())
        print(f"textenc tflite vs torch: mu corr {corr(o[0], mu1.numpy()):.6f}  "
              f"logw corr {corr(o[1], logw1.numpy()):.6f}")
        o = tfl_run(it_dec, x.numpy(), mu.numpy(), t_emb.numpy())
        print(f"decoder tflite vs torch: corr {corr(o[0], v1.numpy()):.6f}")
        o = tfl_run(it_gen, mel_in.numpy())
        print(f"vocoder tflite vs torch: corr {corr(o[0], w1.numpy()):.6f}")
        print("\nGPU-CLEAN ALL:", c1 and c2 and c3)


if __name__ == "__main__":
    main()
