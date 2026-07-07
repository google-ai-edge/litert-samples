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

"""Mimi (Kyutai 2024 codec) DECODE path -> CompiledModel GPU: re-authoring + parity.

Decode path (host RVQ -> GPU graph):
  codes[1,32,T] --HOST--> quantizer.decode -> emb[1,512,T]
  emb --GPU--> upsample(ConvT1d,gr=512,s2) -> dectrans(8L) -> decoder(SEANet 4x ConvT1d) -> audio[1,1,L]

Re-authoring recipe (all proven C-patterns; numerically equivalent except sigmoid-GELU):
  - MimiMLP gelu(erf)  -> sigmoid-GELU  x*sigmoid(1.702x)        [GELU is banned/FlexErf]
  - MimiRotaryEmbedding-> baked const cos/sin + rotate_half      [kills GATHER_ND pos-gather]
  - causal/sliding mask-> baked const additive bias (1,1,S,S)    [kills CUMSUM/EQUAL/AND/SELECT_V2]
  - >4D in attn        -> manual matmul+softmax, all tensors <=4D
  - MimiLayerScale     -> bake gamma into preceding Linear (o_proj / fc2)   [C19]
  - ConvTranspose1d    -> ZeroStuffConvT1d (grouped-aware)        [C32/C20, kills TRANSPOSE_CONV]
  - MimiConv1d         -> causal CONSTANT pad already GPU-clean (no reflect GATHER_ND)

Also exports the decoder_transformer STANDALONE (graph a) + the full fused decode (graph b)
so Pixel 8a can run the C33 test: standalone-correct-but-fused-collapse => C33 generalizes.

Run:  python build_mimi.py [stage]   stage in {opcheck,parity,c33,all}
"""
import _stub  # FIRST: scipy/_propack + getsourcefile guards
import sys
import os
import math
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MimiModel
from transformers.models.mimi.modeling_mimi import rotate_half

HERE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)

BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP",
          "RFFT2D", "FFT", "STFT", "COMPLEX", "RFFT", "IRFFT", "CUMSUM"}


# ----------------------------------------------------------- proven re-authoring ops
class ZeroStuffConvT1d(nn.Module):
    """Exact GPU-clean ConvTranspose1d (DAC C20/C32), GROUPED-aware: 2D nearest upsample
    x const zero-stuff mask + flipped grouped conv1d + crop. Replaces nn.ConvTranspose1d
    in-place so the MimiConvTranspose1d wrapper's trim still applies on the result."""
    def __init__(self, ct, L):
        super().__init__()
        self.s = ct.stride[0]
        self.k = ct.kernel_size[0]
        self.p = ct.padding[0]
        self.op = ct.output_padding[0]
        self.L = L
        self.g = ct.groups
        Cin, Cout, G = ct.in_channels, ct.out_channels, ct.groups
        # ConvT weight (Cin, Cout//G, K) -> grouped conv1d weight (Cout, Cin//G, K), flipped
        w = ct.weight.detach()
        w = w.view(G, Cin // G, Cout // G, self.k).permute(0, 2, 1, 3).reshape(Cout, Cin // G, self.k)
        self.register_buffer("w", w.flip(2).contiguous())
        self.register_buffer("b", ct.bias.detach().clone() if ct.bias is not None
                          else torch.zeros(Cout))
        mk = np.zeros((L * self.s,), np.float32)
        mk[::self.s] = 1.0
        self.register_buffer("mask", torch.from_numpy(mk)[None, None])

    def forward(self, x):
        xn = F.interpolate(x.unsqueeze(2), size=(1, self.L * self.s), mode="nearest").squeeze(2) * self.mask
        y = F.conv1d(xn, self.w, bias=self.b, padding=self.k - 1, groups=self.g)
        ol = (self.L - 1) * self.s + self.k - 2 * self.p + self.op
        return y[:, :, self.p:self.p + ol]


def swap_convtranspose(model, lengths):
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.ConvTranspose1d):
            par = model
            *pth, last = name.split(".")
            for q in pth: par = getattr(par, q)
            setattr(par, last, ZeroStuffConvT1d(mod, lengths[name]))


def bake_mimi_convs(model, conv_lengths):
    """MimiConv1d causal pad uses int64-buffer .item() (dynamic) -> jax ConcretizationError.
    Bake the (left,right) constant pad for each conv's fixed input length. pad_mode=constant
    on the whole decode path -> plain F.pad (no reflect GATHER_ND)."""
    from transformers.models.mimi.modeling_mimi import MimiConv1d
    import types
    for name, mo in list(model.named_modules()):
        if isinstance(mo, MimiConv1d) and name in conv_lengths:
            length = conv_lengths[name]
            conv = mo.conv
            k_eff = (conv.kernel_size[0] - 1) * conv.dilation[0] + 1
            stride = conv.stride[0]
            ptot = k_eff - stride
            n_frames = math.ceil((length - k_eff + ptot) / stride + 1) - 1
            extra = n_frames * stride + k_eff - ptot - length
            pl, pr = ptot, extra          # causal

            if mo.pad_mode == "replicate":
                # tflite PAD is constant-only; replicate = edge-slice replication (SLICE+CONCAT)
                def fwd(self, hidden_states, padding_cache=None, _pl=pl, _pr=pr):
                    x = hidden_states
                    parts = [x[:, :, :1]] * _pl + [x] + [x[:, :, -1:]] * _pr
                    return self.conv(torch.cat(parts, dim=2) if len(parts) > 1 else x)
            else:
                def fwd(self, hidden_states, padding_cache=None, _pl=pl, _pr=pr, _pm=mo.pad_mode):
                    return self.conv(F.pad(hidden_states, (_pl, _pr), _pm))
            mo.forward = types.MethodType(fwd, mo)


class TanhGELU(nn.Module):
    """GELU(erf) is banned (FlexErf). tanh approximation (MUL/ADD/TANH, no POW) is the
    closest GPU-clean form: 0.5x(1+tanh(sqrt(2/pi)(x+0.044715 x^3)))."""
    C = math.sqrt(2.0 / math.pi)

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(self.C * (x + 0.044715 * x * x * x)))


class SigmoidGELU(nn.Module):
    """GELU -> x*sigmoid(1.702x). No x^3 term -> no fp16 overflow risk (the hardened variant)."""
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class CleanELU(nn.Module):
    """nn.ELU(alpha=1) lowers to SELECT (banned). Exact SELECT-free identity:
    ELU(x) = relu(x) - relu(1 - exp(min(x,0))). min(x,0) keeps exp from fp16-overflowing."""
    def forward(self, x):
        xm = -torch.relu(-x)                     # = min(x, 0)
        return torch.relu(x) - torch.relu(1.0 - torch.exp(xm))


def swap_elu(m):
    for n, c in list(m.named_children()):
        if isinstance(c, nn.ELU):
            setattr(m, n, CleanELU())
        else:
            swap_elu(c)


class SafeLayerNorm(nn.Module):
    """fp16-robust LayerNorm: scale x down before the squared-diff reduction so fp16 never sees
    large (|x|~27)^2 values. LN is scale-invariant, so the result is identical (eps*scale^2 keeps
    even the eps exact). On-device ML Drift computes fp16 internally regardless of stored precision,
    so this is the only lever for the large-magnitude residual stream."""
    def __init__(self, ln, scale=0.03125):
        super().__init__()
        self.eps = ln.eps * scale * scale
        self.scale = scale
        self.register_buffer("w", ln.weight.detach().clone())
        self.register_buffer("b", ln.bias.detach().clone())

    def forward(self, x):
        xs = x * self.scale
        mu = xs.mean(-1, keepdim=True)
        d = xs - mu
        var = (d * d).mean(-1, keepdim=True)
        return d * torch.rsqrt(var + self.eps) * self.w + self.b


def rope_cos_sin(seq, head_dim, base=10000.0):
    """MimiRotaryEmbedding (default rope, attention_scaling=1.0) baked for fixed length."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)                  # (seq, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)             # (seq, head_dim)
    return emb.cos()[None, None], emb.sin()[None, None]  # (1,1,seq,head_dim)


def causal_bias(seq, sliding_window=None, neg=-1e9):
    i = torch.arange(seq)[:, None]
    j = torch.arange(seq)[None, :]
    mask = j <= i
    if sliding_window is not None:
        mask = mask & (j > i - sliding_window)
    bias = torch.where(mask, torch.zeros(()), torch.full((), neg))
    return bias[None, None].float()                     # (1,1,seq,seq)


def make_clean_attn(n_heads, head_dim, scaling):
    def fwd(self, hidden_states, attention_mask=None, position_ids=None,
            past_key_values=None, output_attentions=False, use_cache=False, **kw):
        b, t, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, t, n_heads, head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, t, n_heads, head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, t, n_heads, head_dim).transpose(1, 2)
        q = q * self.bcos + rotate_half(q) * self.bsin
        k = k * self.bcos + rotate_half(k) * self.bsin
        scores = torch.matmul(q, k.transpose(-1, -2)) * scaling + self.cbias
        p = torch.softmax(scores, dim=-1)
        o = torch.matmul(p, v).transpose(1, 2).reshape(b, t, n_heads * head_dim)
        return self.o_proj(o), None
    return fwd


def reauth_transformer(tr, cfg, seq, harden=False):
    """In-place re-author a MimiTransformerModel for fixed length `seq`.
    harden=True applies fp16-on-Mali robustness: fp16-safe causal bias (-1e4 not -1e9->-inf),
    sigmoid-GELU (no x^3 overflow), SafeLayerNorm (no large-magnitude squared-diff)."""
    bcos, bsin = rope_cos_sin(seq, cfg.head_dim)
    cbias = causal_bias(seq, sliding_window=cfg.sliding_window, neg=(-1e4 if harden else -1e9))
    scaling = 1.0 / math.sqrt(cfg.head_dim)
    import types
    for lyr in tr.layers:
        a = lyr.self_attn
        a.register_buffer("bcos", bcos)
        a.register_buffer("bsin", bsin)
        a.register_buffer("cbias", cbias)
        a.forward = types.MethodType(make_clean_attn(cfg.num_attention_heads, cfg.head_dim, scaling), a)
        lyr.mlp.activation_fn = SigmoidGELU() if harden else TanhGELU()
        if harden:
            lyr.input_layernorm = SafeLayerNorm(lyr.input_layernorm)
            lyr.post_attention_layernorm = SafeLayerNorm(lyr.post_attention_layernorm)
        # bake LayerScale gamma into the preceding Linear (o_proj for attn, fc2 for mlp)
        with torch.no_grad():
            g_a = lyr.self_attn_layer_scale.scale.detach()
            a.o_proj.weight.mul_(g_a[:, None])
            if a.o_proj.bias is not None:
                a.o_proj.bias.mul_(g_a)
            g_m = lyr.mlp_layer_scale.scale.detach()
            lyr.mlp.fc2.weight.mul_(g_m[:, None])
            if lyr.mlp.fc2.bias is not None:
                lyr.mlp.fc2.bias.mul_(g_m)
        lyr.self_attn_layer_scale = nn.Identity()
        lyr.mlp_layer_scale = nn.Identity()

    def tr_forward(x):
        for lyr in tr.layers:
            x = lyr(x)[0]
        return x
    return tr_forward


# ------------------------------------------------------------------ graph wrappers
class TransGraph(nn.Module):
    """(a) standalone re-authored decoder_transformer. in/out (1,seq,512)."""
    def __init__(self, tr_forward):
        super().__init__()
        self.fn = tr_forward

    def forward(self, x):
        return self.fn(x)


class DecodeGraph(nn.Module):
    """(b) full fused decode: emb(1,512,T) -> upsample -> dectrans -> decoder -> audio(1,1,L)."""
    def __init__(self, upsample, tr_forward, decoder):
        super().__init__()
        self.upsample = upsample
        self.fn = tr_forward
        self.decoder = decoder

    def forward(self, emb):
        x = self.upsample(emb)              # (1,512,2T)
        x = x.transpose(1, 2)           # (1,2T,512)
        x = self.fn(x)                     # transformer
        x = x.transpose(1, 2)           # (1,512,2T)
        return self.decoder(x)             # (1,1,L)


class EncodeGraph(nn.Module):
    """encode: audio(1,1,L) -> encoder(SEANet) -> encoder_transformer -> downsample -> emb(1,512,T).
    emb then feeds host RVQ encode -> codes (Euclidean argmin, int64 -> host like DAC)."""
    def __init__(self, encoder, tr_forward, downsample):
        super().__init__()
        self.encoder = encoder
        self.fn = tr_forward
        self.downsample = downsample

    def forward(self, audio):
        x = self.encoder(audio)            # (1,512,Senc)
        x = self.fn(x.transpose(1, 2))     # encoder_transformer
        return self.downsample(x.transpose(1, 2))   # (1,512,T)


class DecOnlyGraph(nn.Module):
    """SEANet decoder only (no transformer): trans_out(1,seq,512) -> audio(1,1,L). Tests whether
    the convs are fp16-exact on Mali => validates a CPU-transformer + GPU-conv hybrid (Matcha precedent)."""
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, trans_out):
        return self.decoder(trans_out.transpose(1, 2))


class DecodeGraphTapped(nn.Module):
    """(b') fused decode with an extra tap on the transformer output. The C33 witness:
    the SAME transformer weights/input as the standalone graph, but fused with upsample +
    SEANet decoder. On device, in-context tap == standalone => fusion is fine; tap collapses
    while standalone is correct => C33 (transformer-residual-collapse-when-fused) generalizes.
    Returns (trans_out(1,2T,512), audio(1,1,L))."""
    def __init__(self, upsample, tr_forward, decoder):
        super().__init__()
        self.upsample = upsample
        self.fn = tr_forward
        self.decoder = decoder

    def forward(self, emb):
        x = self.upsample(emb).transpose(1, 2)   # (1,2T,512)
        tap = self.fn(x)                         # transformer output (the C33 tap)
        audio = self.decoder(tap.transpose(1, 2))
        return tap, audio


# --------------------------------------------------------------- op-check helpers
def opcheck(path, label):
    """Static GPU-compat scan: read the op set straight from the .tflite flatbuffer."""
    from ai_edge_litert import schema_py_generated as schema
    with open(path, "rb") as f:
        model = schema.ModelT.InitFromPackedBuf(f.read(), 0)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items() if not k.startswith("_")}
    ops = collections.Counter()
    over = 0
    for g in model.subgraphs:
        for op in g.operators:
            c = model.operatorCodes[op.opcodeIndex]
            code = max(c.builtinCode, c.deprecatedBuiltinCode)
            ops[c.customCode.decode() if c.customCode else names.get(code, str(code))] += 1
        over += sum(1 for t in g.tensors if t.shape is not None and len(t.shape) > 4)
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
    fft = {k: v for k, v in ops.items() if any(t in k.upper() for t in ("FFT", "STFT", "COMPLEX"))}
    print(f"[{label}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    print(f"[{label}] banned:{bad or 'NONE'} >4D:{over} FFT:{fft or 'NONE'} "
          f"size {os.path.getsize(path)/1e6:.1f}MB")
    print(f"[{label}] VERDICT:", "GPU-CLEAN" if not bad and not over and not fft else f"BLOCKERS {bad}")
    return not bad and not over and not fft


def tfl_run(path, *inputs):
    """Inference through the LiteRT CompiledModel API; returns shaped outputs in
    signature order (buffer order == get_signature_list() name order)."""
    from ai_edge_litert.compiled_model import CompiledModel
    model = CompiledModel.from_file(path)
    sig = model.get_signature_list()
    key = list(sig)[0]
    ins = model.create_input_buffers(0)
    outs = model.create_output_buffers(0)
    for buf, x in zip(ins, inputs):
        buf.write(np.ascontiguousarray(x, dtype=np.float32))
    model.run_by_index(0, ins, outs)
    details = model.get_output_tensor_details(key)
    result = []
    for j, name in enumerate(sig[key]["outputs"]):
        d = details[name]
        n = int(np.prod(d["shape"])) if len(d["shape"]) else 1
        result.append(outs[j].read(n, np.dtype(d["dtype"])).reshape(d["shape"]))
    return result


def corr(a, b):
    n = min(a.size, b.size)
    return np.corrcoef(a.ravel()[:n], b.ravel()[:n])[0, 1]


def convert(mod, example_inputs, out):
    import litert_torch
    litert_torch.convert(mod, example_inputs).export(out)
    return out


def to_fp16(fp32_path, fp16_path):
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(
        regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
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


def real_latent(m):
    """codes + dequantized embeddings from a real forward (matches probe)."""
    sr = m.config.sampling_rate
    t = torch.linspace(0, 1, sr)
    audio = (0.3 * torch.sin(2 * np.pi * 220 * t) + 0.2 * torch.sin(2 * np.pi * 440 * t)).reshape(1, 1, -1)
    with torch.no_grad():
        codes = m.encode(audio).audio_codes        # (1,32,T)
        emb = m.quantizer.decode(codes)            # (1,512,T)  host-side RVQ
    return audio, codes, emb


# =============================================================================
def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    m = MimiModel.from_pretrained("kyutai/mimi").eval()
    cfg = m.config
    audio, codes, emb = real_latent(m)
    Tc = emb.shape[-1]
    seq = 2 * Tc                                    # upsample stride 2 -> exact 2*Tc
    print(f"codes {tuple(codes.shape)} emb {tuple(emb.shape)}  Tc={Tc} seq={seq}")

    # reference torch decode (full path) from the SAME embeddings
    with torch.no_grad():
        up = m.upsample(emb)
        ref_trans_in = up.transpose(1, 2).clone()  # (1,seq,512) standalone-transformer input
        # instrument the residual stream entering each layer (C33-relevance check)
        res_max = []
        hk0 = [lyr.register_forward_pre_hook(
            (lambda L: (lambda mod, inp: L.append(inp[0].abs().max().item())))(res_max))
            for lyr in m.decoder_transformer.layers]
        dt = m.decoder_transformer(ref_trans_in, return_dict=False)[0]
        for h in hk0: h.remove()
        ref_audio = m.decoder(dt.transpose(1, 2))  # (1,1,L)
        ref_trans_out = dt.clone()
    print(f"ref audio {tuple(ref_audio.shape)}  trans_in|x|max {ref_trans_in.abs().max():.1f}  "
          f"trans_out|x|max {ref_trans_out.abs().max():.1f}")
    print(f"  residual stream entering L0..L7: {[round(v,1) for v in res_max]}  PEAK {max(res_max):.1f}")

    # ---- build re-authored graphs (fresh model instances so refs stay intact)
    m2 = MimiModel.from_pretrained("kyutai/mimi").eval()
    tr_fwd = reauth_transformer(m2.decoder_transformer, cfg, seq)
    # capture input lengths for ConvTranspose1d (swap) AND MimiConv1d (bake pads), one pass
    from transformers.models.mimi.modeling_mimi import MimiConv1d
    L = {}
    LC = {}
    hk = []
    for n, mo in m2.named_modules():
        if isinstance(mo, nn.ConvTranspose1d):
            hk.append(mo.register_forward_pre_hook(
                (lambda nm: (lambda mod, i: L.__setitem__(nm, i[0].shape[-1])))(n)))
        elif isinstance(mo, MimiConv1d):
            hk.append(mo.register_forward_pre_hook(
                (lambda nm: (lambda mod, i: LC.__setitem__(nm, i[0].shape[-1])))(n)))
    with torch.no_grad():
        _u = m2.upsample(emb)
        _ = m2.decoder(_u)  # exercise upsample + all decoder convs
    for h in hk: h.remove()
    bake_mimi_convs(m2, LC)
    swap_convtranspose(m2, L)
    swap_elu(m2.decoder)               # SEANet ELU -> SELECT-free identity

    trans_g = TransGraph(tr_fwd).eval()
    decode_g = DecodeGraph(m2.upsample, tr_fwd, m2.decoder).eval()
    tapped_g = DecodeGraphTapped(m2.upsample, tr_fwd, m2.decoder).eval()

    # (a) re-authoring no-op vs original torch (sigmoid-GELU is the only approximation)
    print("\n=== (a) re-authored torch vs original torch ===")
    with torch.no_grad():
        ra_trans = trans_g(ref_trans_in)
        ra_audio = decode_g(emb)
    print(f"transformer corr {corr(ra_trans.numpy(), ref_trans_out.numpy()):.6f} "
          f"max|d| {(ra_trans-ref_trans_out).abs().max():.2e}")
    print(f"audio       corr {corr(ra_audio.numpy(), ref_audio.numpy()):.6f} "
          f"max|d| {(ra_audio-ref_audio).abs().max():.2e}")

    if stage in ("opcheck", "parity", "all"):
        print("\n=== convert + op-check (fp32) ===")
        dec_p = convert(decode_g, (emb,), os.path.join(HERE, "mimi_decode.tflite"))
        it_dec, c_dec = opcheck(dec_p, "decode")
        print("\n=== (b) tflite decode vs re-authored torch ===")
        o = tfl_run(it_dec, emb.numpy())
        print(f"decode tflite vs torch: corr {corr(o[0], ra_audio.numpy()):.6f}")
        print(f"decode tflite vs ORIG torch: corr {corr(o[0], ref_audio.numpy()):.6f}")

    if stage in ("c33", "all"):
        print("\n=== C33 graphs: standalone transformer (a) + tapped fused decode (b') ===")
        tr_p = convert(trans_g, (ref_trans_in,), os.path.join(HERE, "mimi_dectrans_standalone.tflite"))
        it_tr, c_tr = opcheck(tr_p, "dectrans_standalone")
        o = tfl_run(it_tr, ref_trans_in.numpy())
        print(f"standalone transformer tflite vs torch: corr {corr(o[0], ra_trans.numpy()):.6f}")
        tap_p = convert(tapped_g, (emb,), os.path.join(HERE, "mimi_decode_tapped.tflite"))
        it_tap, c_tap = opcheck(tap_p, "decode_tapped")
        ot = tfl_run(it_tap, emb.numpy())
        # outputs may be returned in any order; match by shape
        tap_t = next(a for a in ot if a.ndim == 3 and a.shape[1] == seq)
        aud_t = next(a for a in ot if a.shape[-1] == ra_audio.shape[-1])
        print(f"tapped fused: trans-tap corr {corr(tap_t, ra_trans.numpy()):.6f}  "
              f"audio corr {corr(aud_t, ra_audio.numpy()):.6f}")
        # dump device-test fixtures (same latent feeds both graphs).
        # .npy for desktop compare; raw float32 C-order .bin for the on-device probe input.
        for nm, arr in [("emb", emb), ("trans_in", ref_trans_in),
                        ("trans_out_cpu", ra_trans), ("audio_cpu", ra_audio)]:
            a = arr.numpy() if isinstance(arr, torch.Tensor) else arr
            np.save(os.path.join(HERE, f"fixture_{nm}.npy"), a)
            a.astype("<f4").tofile(os.path.join(HERE, f"fixture_{nm}.bin"))
        print("  wrote fixtures: emb / trans_in / trans_out_cpu / audio_cpu  (.npy + .bin)")

    if stage in ("deconly", "all"):
        print("\n=== SEANet decoder-only (CPU-transformer + GPU-conv hybrid validation) ===")
        deconly_g = DecOnlyGraph(m2.decoder).eval()
        with torch.no_grad():
            da = deconly_g(ref_trans_out)            # exact transformer output -> audio
        print(f"deconly vs orig audio corr {corr(da.numpy(), ref_audio.numpy()):.6f}")
        dp = convert(deconly_g, (ref_trans_out,), os.path.join(HERE, "mimi_deconly.tflite"))
        opcheck(dp, "deconly")
        to_fp16(dp, os.path.join(HERE, "mimi_deconly_fp16.tflite"))

    if stage in ("harden", "all"):
        print("\n=== hardened standalone transformer (fp16-on-Mali robustness) ===")
        mh = MimiModel.from_pretrained("kyutai/mimi").eval()
        htr_fwd = reauth_transformer(mh.decoder_transformer, cfg, seq, harden=True)
        htr_g = TransGraph(htr_fwd).eval()
        with torch.no_grad():
            hra = htr_g(ref_trans_in)
        print(f"hardened transformer vs orig torch corr {corr(hra.numpy(), ref_trans_out.numpy()):.6f} "
              f"max|d| {(hra-ref_trans_out).abs().max():.2e}")
        hp = convert(htr_g, (ref_trans_in,), os.path.join(HERE, "mimi_dectrans_hardened.tflite"))
        opcheck(hp, "dectrans_hardened")
        to_fp16(hp, os.path.join(HERE, "mimi_dectrans_hardened_fp16.tflite"))

    if stage in ("enc", "all"):
        print("\n=== encoder path: audio -> encoder -> encoder_transformer -> downsample -> emb ===")
        with torch.no_grad():
            e_emb_ref = m.encoder(audio)
            Se = e_emb_ref.shape[-1]
            e_dt = m.encoder_transformer(e_emb_ref.transpose(1, 2), return_dict=False)[0]
            enc_emb_ref = m.downsample(e_dt.transpose(1, 2))    # (1,512,T)
        print(f"encoder SEANet out {tuple(e_emb_ref.shape)}  Senc={Se}  enc_emb {tuple(enc_emb_ref.shape)}")
        me = MimiModel.from_pretrained("kyutai/mimi").eval()
        etr_fwd = reauth_transformer(me.encoder_transformer, cfg, Se)
        from transformers.models.mimi.modeling_mimi import MimiConv1d as _MC
        L2 = {}
        LC2 = {}
        hk2 = []
        for n, mo in me.named_modules():
            if isinstance(mo, nn.ConvTranspose1d):
                hk2.append(mo.register_forward_pre_hook(
                    (lambda nm: (lambda mod, i: L2.__setitem__(nm, i[0].shape[-1])))(n)))
            elif isinstance(mo, _MC):
                hk2.append(mo.register_forward_pre_hook(
                    (lambda nm: (lambda mod, i: LC2.__setitem__(nm, i[0].shape[-1])))(n)))
        with torch.no_grad():
            _x = me.encoder(audio)
            me.downsample(_x)          # exercise encoder + downsample convs
        for h in hk2: h.remove()
        bake_mimi_convs(me, LC2)
        swap_elu(me.encoder)
        enc_g = EncodeGraph(me.encoder, etr_fwd, me.downsample).eval()
        with torch.no_grad():
            enc_emb_mine = enc_g(audio)
        print(f"re-authored encode vs orig torch corr {corr(enc_emb_mine.numpy(), enc_emb_ref.numpy()):.6f} "
              f"max|d| {(enc_emb_mine-enc_emb_ref).abs().max():.2e}")
        enc_p = convert(enc_g, (audio,), os.path.join(HERE, "mimi_encode.tflite"))
        it_enc, c_enc = opcheck(enc_p, "encode")
        oe = tfl_run(it_enc, audio.numpy())
        print(f"encode tflite vs torch: corr {corr(oe[0], enc_emb_ref.numpy()):.6f}")

    if stage == "all":
        print("\n=== fp16 (device runs LITERT_CL fp16 -> C33 must reproduce here) ===")
        for base in ("mimi_decode", "mimi_dectrans_standalone", "mimi_decode_tapped", "mimi_encode"):
            to_fp16(os.path.join(HERE, base + ".tflite"), os.path.join(HERE, base + "_fp16.tflite"))
            opcheck(os.path.join(HERE, base + "_fp16.tflite"), base + "_fp16")


if __name__ == "__main__":
    main()
