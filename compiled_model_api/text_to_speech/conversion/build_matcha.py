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

"""Matcha-TTS -> CompiledModel GPU: real-weight conversion + parity checks.

Loads matcha_ljspeech.ckpt + generator_v1 (HiFi-GAN T2 v1), applies the proven
re-authoring recipe to the 3 heavy graphs (text encoder / CFM decoder / HiFi-GAN
vocoder), converts each with litert-torch, and verifies:
  (a) re-authored torch == original torch  (mask-drop is a no-op at full length)
  (b) tflite == re-authored torch          (conversion fidelity)
  (c) end-to-end waveform: full tflite-orchestrated pipeline vs torch
      synthesise()

Host (CPU) does: phoneme embedding lookup, duration -> length-regulator
(expand mu), sinusoidal time-embed per ODE step, the Euler ODE loop, mel
denormalize. The 3 GPU graphs are text_encoder(emb)->(mu,logw),
decoder(x,mu,t_emb)->v, vocoder(mel)->wav.

Run:  python build_matcha.py [stage]
      stage in {opcheck, parity, e2e, all}   (default: all)
"""

import _stub  # noqa: F401  (must be first: scipy / getsourcefile guards)

import collections
import logging
import math
import os
import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "matcha_ljspeech.ckpt")
HIFI = os.path.join(HERE, "generator_v1")


def setup_espeak_library():
    """Points phonemizer at libespeak-ng on macOS or Linux.

    Only used to build the G2P assets (espeak generates the reference IPA); the
    shipped Android app needs no espeak. Respects an existing
    PHONEMIZER_ESPEAK_LIBRARY, then falls back to ctypes' library resolver and a
    few common install locations. Leaves the env var unset if none is found, so
    phonemizer can still fall back to a system espeak binary.
    """
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return
    import ctypes.util

    found = (ctypes.util.find_library("espeak-ng")
             or ctypes.util.find_library("espeak"))
    candidates = [
        found,
        "/opt/homebrew/lib/libespeak-ng.dylib",  # macOS (Apple silicon)
        "/usr/local/lib/libespeak-ng.dylib",  # macOS (Intel)
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",  # Debian/Ubuntu
        "/usr/lib/libespeak-ng.so.1",  # other Linux
    ]
    for path in candidates:
        if path and os.path.exists(path):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = path
            return


setup_espeak_library()
torch.manual_seed(0)

BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT",
          "SELECT_V2", "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST",
          "EMBEDDING_LOOKUP", "RFFT2D", "FFT", "STFT", "COMPLEX", "RFFT",
          "IRFFT"}


def stub_matcha_utils():
    """Stubs matcha.utils so importing matcha models skips hydra/lightning."""

    def sequence_mask(length, max_length=None):
        if max_length is None:
            max_length = int(length.max())
        arange = torch.arange(
            max_length, dtype=length.dtype, device=length.device)
        return arange.unsqueeze(0) < length.unsqueeze(1)

    class _UtilsModule(types.ModuleType):
        __file__ = "<stub:matcha.utils>"

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            if name == "get_pylogger":
                return lambda name=None: logging.getLogger(name)
            if name == "sequence_mask":
                return sequence_mask
            return lambda *a, **k: None

    utils_pkg = _UtilsModule("matcha.utils")
    utils_pkg.__path__ = []
    utils_model = types.ModuleType("matcha.utils.model")
    utils_model.sequence_mask = sequence_mask
    sys.modules["matcha.utils"] = utils_pkg
    sys.modules["matcha.utils.model"] = utils_model


stub_matcha_utils()


class GN4D(nn.Module):
    """GroupNorm w/o GATHER_ND: reshape (B,G,C//G,T), mean/var over (2,3)."""

    def __init__(self, gn):
        super().__init__()
        self.groups = gn.num_groups
        self.eps = gn.eps
        self.register_buffer("w", gn.weight.detach().reshape(1, -1, 1))
        self.register_buffer("b", gn.bias.detach().reshape(1, -1, 1))

    def forward(self, x):
        b, c, t = x.shape
        x = x.reshape(b, self.groups, c // self.groups, t)
        mean = x.mean((2, 3), keepdim=True)
        var = (x - mean).pow(2).mean((2, 3), keepdim=True)
        normalized = ((x - mean) * torch.rsqrt(var + self.eps)).reshape(b, c, t)
        return normalized * self.w + self.b


class MishClean(nn.Module):
    """Mish via SELECT-free fp16-safe softplus relu(x)+log1p(exp(-|x|)).

    Exact rewrite. [C6]
    """

    def forward(self, x):
        softplus = torch.relu(x) + torch.log1p(torch.exp(-torch.abs(x)))
        return x * torch.tanh(softplus)


class ZeroStuffConvT1d(nn.Module):
    """Exact GPU-clean ConvTranspose1d (DAC): 2D nearest upsample x const mask +
    flipped conv1d + crop. [C20/C31/C32]"""

    def __init__(self, ct, length):
        super().__init__()
        self.stride = ct.stride[0]
        self.kernel = ct.kernel_size[0]
        self.pad = ct.padding[0]
        self.out_pad = ct.output_padding[0]
        self.length = length
        weight = ct.weight.flip(2).transpose(0, 1).contiguous()
        self.register_buffer("w", weight)
        if ct.bias is not None:
            bias = ct.bias.detach().clone()
        else:
            bias = torch.zeros(ct.out_channels)
        self.register_buffer("b", bias)
        mask = np.zeros((length * self.stride,), np.float32)
        mask[::self.stride] = 1.0
        self.register_buffer("mask", torch.from_numpy(mask)[None, None])

    def forward(self, x):
        target = (1, self.length * self.stride)
        stuffed = F.interpolate(
            x.unsqueeze(2), size=target, mode="nearest").squeeze(2) * self.mask
        y = F.conv1d(stuffed, self.w, bias=self.b, padding=self.kernel - 1)
        out_len = ((self.length - 1) * self.stride + self.kernel
                   - 2 * self.pad + self.out_pad)
        return y[:, :, self.pad:self.pad + out_len]


def swap_norm_act(module):
    """Recursively replaces GroupNorm -> GN4D and Mish -> MishClean.

    Args:
        module: Module whose children are rewritten in place.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, GN4D(child))
        elif isinstance(child, nn.Mish):
            setattr(module, name, MishClean())
        else:
            swap_norm_act(child)


def swap_convtranspose(model, lengths):
    """Replaces each ConvTranspose1d with ZeroStuffConvT1d at its length.

    Args:
        model: Module whose ConvTranspose1d children are replaced in place.
        lengths: Mapping of module name -> traced input length.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.ConvTranspose1d):
            parent = model
            *path, last = name.split(".")
            for part in path:
                parent = getattr(parent, part)
            setattr(parent, last, ZeroStuffConvT1d(module, lengths[name]))


def load_sd():
    """Loads the Matcha-LJSpeech checkpoint state dict."""
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    return ckpt["state_dict"]


def build_text_encoder(sd):
    """Builds the Matcha TextEncoder and loads checkpoint weights.

    Args:
        sd: Checkpoint state dict from load_sd().

    Returns:
        The TextEncoder in eval mode.
    """
    from matcha.models.components.text_encoder import TextEncoder

    enc = NS(n_feats=80, n_channels=192, filter_channels=768,
             filter_channels_dp=256, n_heads=2, n_layers=6, kernel_size=3,
             p_dropout=0.0, spk_emb_dim=64, n_spks=1, prenet=True)
    dp = NS(filter_channels_dp=256, kernel_size=3, p_dropout=0.0)
    te = TextEncoder("RoPE Encoder", enc, dp, n_vocab=178, n_spks=1,
                     spk_emb_dim=64)
    weights = {k[len("encoder."):]: v for k, v in sd.items()
               if k.startswith("encoder.")}
    te.load_state_dict(weights, strict=True)
    return te.eval()


def build_decoder(sd):
    """Builds the Matcha CFM decoder (U-Net estimator) and loads weights.

    Args:
        sd: Checkpoint state dict from load_sd().

    Returns:
        The Decoder in eval mode.
    """
    from matcha.models.components.decoder import Decoder

    dec = Decoder(in_channels=160, out_channels=80, channels=(256, 256),
                  dropout=0.0, attention_head_dim=64, n_blocks=1,
                  num_mid_blocks=2, num_heads=2, act_fn="snakebeta")
    weights = {k[len("decoder.estimator."):]: v for k, v in sd.items()
               if k.startswith("decoder.estimator.")}
    dec.load_state_dict(weights, strict=True)
    return dec.eval()


def build_hifigan():
    """Builds the HiFi-GAN v1 generator from the generator_v1 checkpoint."""
    from matcha.hifigan.models import Generator
    from matcha.hifigan.config import v1
    from matcha.hifigan.env import AttrDict

    generator = Generator(AttrDict(v1))
    ckpt = torch.load(HIFI, map_location="cpu", weights_only=False)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    generator.remove_weight_norm()
    return generator


def reauth_text_encoder(te):
    """Drops the all-ones mask (kills SELECT_V2 x6 + CAST) from the encoder.

    RoPE/attention are already GPU-clean. Graph input = gathered phoneme
    embeddings (B, T, 192), unscaled. Output (mu, logw).

    Args:
        te: TextEncoder from build_text_encoder(); patched in place.

    Returns:
        An eval-mode nn.Module with forward(emb_x) -> (mu, logw).
    """
    n_channels = te.n_channels

    # Patch matcha MultiHeadAttention to skip masked_fill (mask is all-ones).
    for module in te.modules():
        if type(module).__name__ == "MultiHeadAttention":

            def mha_forward(self, x, c, attn_mask=None):
                q = self.conv_q(x)
                k = self.conv_k(c)
                v = self.conv_v(c)
                out, _ = self.attention(q, k, v, mask=None)
                return self.conv_o(out)

            module.forward = types.MethodType(mha_forward, module)

    class TEWrap(nn.Module):
        def __init__(self, te):
            super().__init__()
            self.te = te
            self.n_channels = n_channels

        def forward(self, emb_x):
            # emb_x: (1, T, 192).
            te = self.te
            # (1, 192, T).
            x = (emb_x * math.sqrt(self.n_channels)).transpose(1, -1)
            mask = torch.ones(1, 1, x.shape[-1])
            x = te.prenet(x, mask)
            x = te.encoder(x, mask)
            mu = te.proj_m(x) * mask
            logw = te.proj_w(x, mask)
            return mu, logw

    return TEWrap(te).eval()


def _trace_convtranspose_lengths(model, run_dummy):
    """Records each ConvTranspose1d input length seen during run_dummy().

    Args:
        model: Module whose ConvTranspose1d inputs are traced via hooks.
        run_dummy: Zero-arg callable that runs one forward pass.

    Returns:
        Mapping of module name -> input length (last dim).
    """
    lengths = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ConvTranspose1d):

            def record(nm):
                return lambda m, i, o: lengths.__setitem__(
                    nm, i[0].shape[-1])

            hooks.append(module.register_forward_hook(record(name)))
    with torch.no_grad():
        run_dummy()
    for hook in hooks:
        hook.remove()
    return lengths


def reauth_decoder(dec, T):
    """Re-authors the CFM decoder for the drop-mask (fixed-length) graph.

    Host-side time embed; GroupNorm -> GN4D, Mish -> softplus form,
    ConvTranspose1d -> ZeroStuffConvT1d, and the all-ones attention mask is
    dropped. Graph input = (x(1,80,T), mu(1,80,T), t_emb(1,1024)); output
    v(1,80,T).

    Args:
        dec: Decoder from build_decoder(); patched in place.
        T: Fixed mel frame count baked into the graph.

    Returns:
        An eval-mode nn.Module with forward(x, mu, t_emb) -> v.
    """
    # Host-compute the time embedding -> Identity inside graph.
    dec.time_embeddings = nn.Identity()
    dec.time_mlp = nn.Identity()
    lengths = _trace_convtranspose_lengths(
        dec, lambda: dec(torch.randn(1, 80, T), torch.ones(1, 1, T),
                         torch.randn(1, 80, T), torch.randn(1, 1024)))
    swap_convtranspose(dec, lengths)
    swap_norm_act(dec)
    # Drop the all-ones attention mask in diffusers Attention.
    for module in dec.modules():
        if type(module).__name__ == "Attention":

            def drop_mask(orig):
                def wrapped(hidden_states, encoder_hidden_states=None,
                            attention_mask=None, **kwargs):
                    return orig(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states)
                return wrapped

            module.forward = drop_mask(module.forward)

    class DecWrap(nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.d = decoder

        def forward(self, x, mu, t_emb):
            return self.d(x, torch.ones(1, 1, x.shape[-1]), mu, t_emb)

    return DecWrap(dec).eval()


def reauth_text_encoder_masked(te):
    """Re-authors the text encoder with the mask kept as a runtime input.

    Shippable variant: the mask stays a runtime float input (additive
    attention bias, no SELECT/CAST), so the graph handles padded
    (variable-length) phoneme sequences GPU-clean. Graph inputs: emb_x
    (1, Tmax, 192), tmask (1, 1, Tmax) float. Outputs (mu, logw).

    Args:
        te: TextEncoder from build_text_encoder(); patched in place.

    Returns:
        An eval-mode nn.Module with forward(emb_x, tmask) -> (mu, logw).
    """
    from einops import rearrange

    n_channels = te.n_channels
    for module in te.modules():
        if type(module).__name__ == "MultiHeadAttention":

            def mha_forward(self, x, c, attn_mask=None):
                q = self.conv_q(x)
                k = self.conv_k(c)
                v = self.conv_v(c)
                heads = self.n_heads
                q = rearrange(q, "b (h c) t -> b h t c", h=heads)
                k = rearrange(k, "b (h c) t -> b h t c", h=heads)
                v = rearrange(v, "b (h c) t -> b h t c", h=heads)
                q = self.query_rotary_pe(q)
                k = self.key_rotary_pe(k)
                scale = math.sqrt(self.k_channels)
                scores = torch.matmul(q, k.transpose(-2, -1)) / scale
                if attn_mask is not None:
                    # Additive, no SELECT.
                    scores = scores + (attn_mask - 1.0) * 1e4
                probs = torch.softmax(scores, dim=-1)
                out = torch.matmul(probs, v).transpose(2, 3).contiguous().view(
                    q.shape[0], heads * self.k_channels, -1)
                return self.conv_o(out)

            module.forward = types.MethodType(mha_forward, module)

    class TEWrapM(nn.Module):
        def __init__(self, te):
            super().__init__()
            self.te = te
            self.n_channels = n_channels

        def forward(self, emb_x, tmask):
            te = self.te
            x = (emb_x * math.sqrt(self.n_channels)).transpose(1, -1)
            x = te.prenet(x, tmask)
            x = te.encoder(x, tmask)
            return te.proj_m(x) * tmask, te.proj_w(x, tmask)

    return TEWrapM(te).eval()


def _decoder_forward_clean(self, x, mask, mu, t, spks=None, cond=None):
    """Copy of matcha Decoder.forward with the one GPU-hostile op fixed.

    The half-res mask `mask_down[:, :, ::2]` (step-2 slice -> GATHER_ND) is
    replaced by a reshape+index decimation (RESHAPE+SLICE, GPU-clean).
    Identical result.

    Args:
        self: The matcha Decoder this function is bound to.
        x: Noisy mel (B, 80, T).
        mask: Float mel mask (B, 1, T).
        mu: Expanded encoder output (B, 80, T).
        t: Time embedding input.
        spks: Optional speaker embedding.
        cond: Unused (matcha signature compatibility).

    Returns:
        The estimated flow v (B, 80, T).
    """
    from einops import pack, rearrange, repeat

    def dec2(m):
        # m (b,1,L) with L even -> (b,1,L//2) == m[:,:,::2].
        b, c, length = m.shape
        return m.reshape(b, c, length // 2, 2)[:, :, :, 0]

    t = self.time_embeddings(t)
    t = self.time_mlp(t)
    x = pack([x, mu], "b * t")[0]
    if spks is not None:
        spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
        x = pack([x, spks], "b * t")[0]
    hiddens = []
    masks = [mask]
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
    """matcha SinusoidalPosEmb, host-side (no learned weights).

    Args:
        t: Time tensor (B,).
        dim: Embedding dimension.
        scale: matcha's fixed time scale.

    Returns:
        The sinusoidal embedding (B, dim).
    """
    half = dim // 2
    freqs = torch.exp(torch.arange(half, dtype=torch.float32)
                      * (-math.log(10000) / (half - 1)))
    angles = scale * t.reshape(-1, 1) * freqs.reshape(1, -1)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


def reauth_decoder_masked(dec, T):
    """Re-authors the CFM decoder with the mask kept as a runtime input.

    Shippable variant: diffusers Attention -> manual additive-masked
    attention; GroupNorm -> GN4D; Mish -> softplus form; ConvTranspose1d ->
    ZeroStuffConvT1d. The learned time-MLP stays IN-graph (GPU); the host
    feeds only the weight-free SinusoidalPosEmb. Graph inputs: x(1,80,T),
    mu(1,80,T), t_sin(1,160), mask(1,1,T). Output v(1,80,T).

    Args:
        dec: Decoder from build_decoder(); patched in place.
        T: Fixed mel frame count baked into the graph.

    Returns:
        An eval-mode nn.Module with forward(x, mu, t_emb, mask) -> v.
    """
    # Host computes sin_pos_emb; time_mlp stays on GPU.
    dec.time_embeddings = nn.Identity()
    lengths = _trace_convtranspose_lengths(
        dec, lambda: dec(torch.randn(1, 80, T), torch.ones(1, 1, T),
                         torch.randn(1, 80, T), torch.randn(1, 160)))
    swap_convtranspose(dec, lengths)
    swap_norm_act(dec)
    for module in dec.modules():
        if type(module).__name__ == "Attention":

            def attn_forward(self, hidden_states, encoder_hidden_states=None,
                             attention_mask=None, **kwargs):
                b, seq, _ = hidden_states.shape
                heads = self.heads
                q = self.to_q(hidden_states)
                k = self.to_k(hidden_states)
                v = self.to_v(hidden_states)
                head_dim = q.shape[-1] // heads
                q = q.reshape(b, seq, heads, head_dim).transpose(1, 2)
                k = k.reshape(b, seq, heads, head_dim).transpose(1, 2)
                v = v.reshape(b, seq, heads, head_dim).transpose(1, 2)
                scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
                if attention_mask is not None:
                    # attention_mask: (b, seq) float 1/0. Replicate diffusers
                    # AttnProcessor2_0: SDPA adds the raw 0/1 mask to scores
                    # (a soft +bias on valid cols, NOT -inf masking).
                    scores = scores + attention_mask.reshape(b, 1, 1, seq)
                probs = torch.softmax(scores, dim=-1)
                out = torch.matmul(probs, v).transpose(1, 2)
                out = out.reshape(b, seq, heads * head_dim)
                return self.to_out[0](out)

            module.forward = types.MethodType(attn_forward, module)

    # GPU-clean half-res mask.
    dec.forward = types.MethodType(_decoder_forward_clean, dec)

    class DecWrapM(nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.d = decoder

        def forward(self, x, mu, t_emb, mask):
            return self.d(x, mask, mu, t_emb)

    return DecWrapM(dec).eval()


def reauth_hifigan(generator, T):
    """Replaces the HiFi-GAN ConvTranspose1d stack with ZeroStuffConvT1d.

    Args:
        generator: HiFi-GAN generator from build_hifigan(); patched in place.
        T: Fixed mel frame count baked into the graph.

    Returns:
        The generator in eval mode.
    """
    lengths = _trace_convtranspose_lengths(
        generator, lambda: generator(torch.randn(1, 80, T)))
    swap_convtranspose(generator, lengths)
    return generator.eval()


def opcheck(path, label):
    """Statically scans a .tflite flatbuffer for GPU-hostile ops.

    Args:
        path: Path to the .tflite file.
        label: Tag prepended to the printed report lines.

    Returns:
        True when no banned op, >4-D tensor, or FFT-family op is present.
    """
    from ai_edge_litert import schema_py_generated as schema

    with open(path, "rb") as f:
        model = schema.ModelT.InitFromPackedBuf(f.read(), 0)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items()
             if not k.startswith("_")}
    ops = collections.Counter()
    over_4d = 0
    for g in model.subgraphs:
        for op in g.operators:
            c = model.operatorCodes[op.opcodeIndex]
            code = max(c.builtinCode, c.deprecatedBuiltinCode)
            if c.customCode:
                op_name = c.customCode.decode()
            else:
                op_name = names.get(code, str(code))
            ops[op_name] += 1
        over_4d += sum(1 for t in g.tensors
                       if t.shape is not None and len(t.shape) > 4)
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
    fft = {k: v for k, v in ops.items()
           if any(term in k.upper() for term in ("FFT", "STFT", "COMPLEX"))}
    print(f"[{label}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    print(f"[{label}] banned:{bad or 'NONE'} >4D:{over_4d} FFT:{fft or 'NONE'} "
          f"size {os.path.getsize(path) / 1e6:.1f}MB")
    clean = not bad and not over_4d and not fft
    print(f"[{label}] VERDICT: {'GPU-CLEAN' if clean else f'BLOCKERS {bad}'}")
    return clean


def tfl_load(path):
    """Loads a .tflite as a LiteRT CompiledModel.

    Args:
        path: Path to the .tflite file.

    Returns:
        The CompiledModel.
    """
    from ai_edge_litert.compiled_model import CompiledModel

    return CompiledModel.from_file(path)


def tfl_run(model, *inputs):
    """Runs a LiteRT CompiledModel on numpy inputs.

    Args:
        model: CompiledModel from tfl_load().
        *inputs: One numpy array per signature input, in signature order.

    Returns:
        A list of output arrays shaped per the signature details.
    """
    signatures = model.get_signature_list()
    key = list(signatures)[0]
    in_details = model.get_input_tensor_details(key)
    out_details = model.get_output_tensor_details(key)
    input_buffers = model.create_input_buffers(0)
    output_buffers = model.create_output_buffers(0)
    for name, buffer, x in zip(signatures[key]["inputs"], input_buffers,
                               inputs):
        dtype = np.dtype(in_details[name]["dtype"])
        buffer.write(np.ascontiguousarray(x, dtype=dtype))
    model.run_by_index(0, input_buffers, output_buffers)
    outputs = []
    for name, buffer in zip(signatures[key]["outputs"], output_buffers):
        detail = out_details[name]
        count = int(np.prod(detail["shape"]))
        flat = buffer.read(count, np.dtype(detail["dtype"]))
        outputs.append(flat.reshape(detail["shape"]))
    return outputs


def convert(module, example_inputs, out):
    """Converts a torch module to a .tflite flatbuffer with litert-torch.

    Args:
        module: The torch module to convert.
        example_inputs: Example input tuple for tracing.
        out: Output .tflite path.

    Returns:
        The output path.
    """
    import litert_torch

    litert_torch.convert(module, example_inputs).export(out)
    return out


def to_fp16(fp32_path, fp16_path):
    """fp16 weights via AI Edge Quantizer FLOAT_CASTING (the zoo standard).

    Args:
        fp32_path: Source fp32 .tflite path.
        fp16_path: Destination fp16 .tflite path.

    Returns:
        The fp16 path.
    """
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


def main():
    """Builds, converts, and parity-checks the 3 Matcha graphs."""
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    sd = load_sd()
    mel_mean = float(sd["mel_mean"])
    mel_std = float(sd["mel_std"])
    print(f"mel_mean={mel_mean:.4f} mel_std={mel_std:.4f}")

    # Representative fixed lengths for op-check / per-graph parity.
    t_text, t_mel = 48, 128

    # Build + re-author.
    te = build_text_encoder(sd)
    dec = build_decoder(sd)
    gen = build_hifigan()
    te_r = reauth_text_encoder(build_text_encoder(sd))
    dec_r = reauth_decoder(build_decoder(sd), t_mel)
    gen_r = reauth_hifigan(build_hifigan(), t_mel)

    emb_w = sd["encoder.emb.weight"]  # (178,192) host lookup table.

    # (a) Re-authored == original (full-length, mask is a no-op).
    print("\n=== (a) re-authoring is a no-op vs original torch ===")
    ids = torch.randint(0, 178, (1, t_text))
    with torch.no_grad():
        mu0, logw0, _ = te(ids, torch.tensor([t_text]))
        ex = emb_w[ids]  # (1,t_text,192)
        mu1, logw1 = te_r(ex)
    mu_corr = np.corrcoef(mu0.numpy().ravel(), mu1.numpy().ravel())[0, 1]
    print(f"textenc mu  corr {mu_corr:.6f} "
          f"max|d| {(mu0 - mu1).abs().max():.2e}")
    logw_corr = np.corrcoef(logw0.numpy().ravel(),
                            logw1.numpy().ravel())[0, 1]
    print(f"textenc logw corr {logw_corr:.6f} "
          f"max|d| {(logw0 - logw1).abs().max():.2e}")

    # Decoder: original vs re-authored at fixed T. dec_r was re-authored from a
    # separate instance; rebuild a clean original for the reference.
    x = torch.randn(1, 80, t_mel)
    mu = torch.randn(1, 80, t_mel)
    t = torch.rand(1)
    dec_orig = build_decoder(sd)
    with torch.no_grad():
        t_emb = dec_orig.time_mlp(dec_orig.time_embeddings(t))
        v0 = dec_orig(x, torch.ones(1, 1, t_mel), mu, t)
        v1 = dec_r(x, mu, t_emb)
    dec_corr = np.corrcoef(v0.numpy().ravel(), v1.numpy().ravel())[0, 1]
    print(f"decoder corr {dec_corr:.6f} "
          f"max|d| {(v0 - v1).abs().max():.2e}")
    mel_in = torch.randn(1, 80, t_mel)
    with torch.no_grad():
        w0 = gen(mel_in)
        w1 = gen_r(mel_in)
    voc_corr = np.corrcoef(w0.numpy().ravel(), w1.numpy().ravel())[0, 1]
    print(f"vocoder corr {voc_corr:.6f} "
          f"max|d| {(w0 - w1).abs().max():.2e}")

    if stage in ("opcheck", "parity", "all"):
        print("\n=== convert + op-check (fp32) ===")
        te_path = convert(te_r, (ex,),
                          os.path.join(HERE, "matcha_textenc.tflite"))
        dec_path = convert(dec_r, (x, mu, t_emb),
                           os.path.join(HERE, "matcha_decoder.tflite"))
        gen_path = convert(gen_r, (mel_in,),
                           os.path.join(HERE, "matcha_vocoder.tflite"))
        clean_te = opcheck(te_path, "textenc")
        clean_dec = opcheck(dec_path, "decoder")
        clean_gen = opcheck(gen_path, "vocoder")

        print("\n=== (b) tflite == re-authored torch ===")

        def corr(a, b):
            n = min(a.size, b.size)
            return np.corrcoef(a.ravel()[:n], b.ravel()[:n])[0, 1]

        out = tfl_run(tfl_load(te_path), ex.numpy())
        print(f"textenc tflite vs torch: "
              f"mu corr {corr(out[0], mu1.numpy()):.6f}  "
              f"logw corr {corr(out[1], logw1.numpy()):.6f}")
        out = tfl_run(tfl_load(dec_path), x.numpy(), mu.numpy(), t_emb.numpy())
        print(f"decoder tflite vs torch: corr {corr(out[0], v1.numpy()):.6f}")
        out = tfl_run(tfl_load(gen_path), mel_in.numpy())
        print(f"vocoder tflite vs torch: corr {corr(out[0], w1.numpy()):.6f}")
        print("\nGPU-CLEAN ALL:", clean_te and clean_dec and clean_gen)


if __name__ == "__main__":
    main()
