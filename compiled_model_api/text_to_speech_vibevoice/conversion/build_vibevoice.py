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

"""Build the four VibeVoice LiteRT GPU graphs + device assets from the source.

Runs in a Python 3.10 conversion env (torch, transformers, safetensors,
litert-torch, ai-edge-quantizer). The conv decoder + tiny head are exported
fp16; the two Qwen2 LMs are exported fp32 (fp16 collapses them on Android ARM
XNNPACK). Conversions print eager-vs-tflite parity. Assets are read straight
from the checkpoint state dict + the voice preset, so no full model
construction is needed.

    python build_vibevoice.py --stage all --src ./VibeVoice \
        --ckpt ./VibeVoice-Realtime-0.5B --out ./out
    python build_vibevoice.py --stage decoder --src ... --ckpt ... --out ...
"""
import argparse
import importlib.util
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6
HD = 64


# --- Shared helpers ---
def load_vv_modules(src):
    """Import the VibeVoice tokenizer + diffusion-head modules.

    Imports directly from the cloned repo, bypassing the package __init__
    (which pulls a text tokenizer that moved in newer transformers).

    Args:
      src: path to the cloned microsoft/VibeVoice repo.

    Returns:
      A `load(name)` helper that imports a `vibevoice.modular.<name>` module.
    """
    from transformers.models.auto import auto_factory as af
    orig = af._LazyAutoMapping.register
    af._LazyAutoMapping.register = lambda self, k, v, exist_ok=False: (
        orig(self, k, v, exist_ok=True) if True else None)
    vv = os.path.join(src, "vibevoice")
    mod = os.path.join(vv, "modular")
    for name, p in [("vibevoice", vv), ("vibevoice.modular", mod)]:
        pkg = types.ModuleType(name)
        pkg.__path__ = [p]
        sys.modules[name] = pkg

    def load(name):
        spec = importlib.util.spec_from_file_location(
            f"vibevoice.modular.{name}", os.path.join(mod, f"{name}.py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        return m
    load("configuration_vibevoice")
    return load


def safe_rms(x, weight, eps=EPS):
    """RMSNorm with an fp16 overflow guard (max-normalize before squaring).

    Args:
      x: input tensor, normalized over the last dim.
      weight: optional elementwise scale (None to skip).
      eps: numerical epsilon.

    Returns:
      The RMS-normalized tensor.
    """
    m = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    xs = x / m
    out = xs * torch.rsqrt((xs * xs).mean(dim=-1, keepdim=True) + eps / (m * m))
    return out * weight if weight is not None else out


def rotate_half(x):
    """Rotate the last dim by half for rotary position embeddings.

    Args:
      x: input tensor with an even-sized last dim.

    Returns:
      The half-rotated tensor.
    """
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


class TanhGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (
            1.0 + torch.tanh(
                0.7978845608028654 * (x + 0.044715 * x * x * x)))


class ZeroStuffConvT1d(nn.Module):
    """GPU-clean ConvTranspose1d: nearest-upsample x zero-stuff mask +
    flipped conv + crop."""
    def __init__(self, ct, length):
        super().__init__()
        self.s = ct.stride[0]
        self.k = ct.kernel_size[0]
        self.p = ct.padding[0]
        self.op = ct.output_padding[0]
        self.length = length
        self.register_buffer(
            "w", ct.weight.flip(2).transpose(0, 1).contiguous())
        self.register_buffer(
            "b",
            ct.bias.detach().clone() if ct.bias is not None
            else torch.zeros(ct.out_channels))
        mk = np.zeros((length * self.s,), np.float32)
        mk[::self.s] = 1.0
        self.register_buffer("mask", torch.from_numpy(mk)[None, None])

    def forward(self, x):
        xn = F.interpolate(
            x.unsqueeze(2), size=(1, self.length * self.s),
            mode="nearest").squeeze(2)
        y = F.conv1d(xn * self.mask, self.w, bias=self.b, padding=self.k - 1)
        out_len = (self.length - 1) * self.s + self.k - 2 * self.p + self.op
        return y[:, :, self.p:self.p + out_len]


def fp16_recipe():
    """Build an ai-edge-quantizer recipe that casts all weights to fp16."""
    from ai_edge_quantizer import recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    oc = qtyping.OpQuantizationConfig(
        weight_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
        compute_precision=qtyping.ComputePrecision.FLOAT)
    rm.add_quantization_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=oc,
        algorithm_key=AlgorithmName.FLOAT_CASTING)
    return rm.get_quantization_recipe()


def export_fp16(module, example_inputs, out_path, fp16=True):
    """Convert a torch module to a tflite graph (fp16 or fp32).

    The deep Qwen2 LM graphs need fp32 on-device (Android ARM XNNPACK computes
    fp16 graphs in fp16, which collapses the 20-layer residual stream to noise);
    the conv decoder + tiny head stay fp16 on the GPU (with
    GpuOptions.Precision.FP32 they are exact and much smaller).

    Args:
      module: the torch module to export.
      example_inputs: a tuple of example input tensors for tracing.
      out_path: destination path for the tflite graph.
      fp16: if True, fp16-quantize the weights; otherwise keep fp32.
    """
    import litert_torch
    tmp = out_path.replace(".tflite", "_tmpfp32.tflite")
    litert_torch.convert(module, example_inputs).export(tmp)
    if fp16:
        from ai_edge_quantizer import quantizer
        qt = quantizer.Quantizer(float_model=tmp)
        qt.load_quantization_recipe(fp16_recipe())
        qt.quantize().export_model(out_path)
        os.remove(tmp)
    else:
        os.replace(tmp, out_path)
    print(
        f"  {os.path.basename(out_path)}: "
        f"{os.path.getsize(out_path) / 1e6:.1f} MB")


def corr(a, b):
    """Pearson correlation of two flattened arrays.

    Args:
      a: first array-like.
      b: second array-like.

    Returns:
      The correlation coefficient as a float.
    """
    return float(
        np.corrcoef(np.asarray(a).flatten(), np.asarray(b).flatten())[0, 1])


def state_dict(ckpt):
    """Load the checkpoint's safetensors state dict.

    Args:
      ckpt: checkpoint directory containing model.safetensors.

    Returns:
      The state dict mapping parameter names to tensors.
    """
    from safetensors.torch import load_file
    return load_file(os.path.join(ckpt, "model.safetensors"))


def cfg(ckpt):
    """Load the checkpoint's config.json.

    Args:
      ckpt: checkpoint directory containing config.json.

    Returns:
      The parsed config as a dict.
    """
    return json.load(open(os.path.join(ckpt, "config.json")))


# --- Conversion stages ---
def build_decoder(src, ckpt, out, frames=128):
    """Export the σ-VAE conv decoder as a GPU-clean fp32 tflite graph.

    Re-authors the ConvNeXt blocks (GELU -> tanh-GELU, ConvRMSNorm -> safe
    RMS, ConvTranspose1d -> ZeroStuffConvT1d) and prints eager parity.

    Args:
      src: cloned microsoft/VibeVoice repo.
      ckpt: VibeVoice-Realtime-0.5B checkpoint dir.
      out: output directory for the tflite graph.
      frames: fixed number of decoded latent frames.
    """
    load = load_vv_modules(src)
    cm = load("configuration_vibevoice")
    tm = load("modular_vibevoice_tokenizer")
    acfg = cm.VibeVoiceAcousticTokenizerConfig(
        **cfg(ckpt)["acoustic_tokenizer_config"])
    model = tm.VibeVoiceAcousticTokenizerModel(acfg).eval()
    sd = state_dict(ckpt)
    pre = "model.acoustic_tokenizer."
    model.load_state_dict(
        {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)},
        strict=False)
    dec = model.decoder.eval()
    latent = torch.randn(1, acfg.vae_dim, frames)
    with torch.no_grad():
        ref = dec(latent).numpy()

    for m in dec.modules():
        if isinstance(m, tm.FFN):
            m.gelu = TanhGELU()
    for m in dec.modules():
        if isinstance(m, tm.ConvRMSNorm):
            eps, w = m.eps, m.weight
            m.forward = (lambda eps, w: lambda x: (
                safe_rms(x.transpose(1, 2).float(), w, eps)).transpose(
                    1, 2))(eps, w)
    lengths, hooks = {}, []
    for nm, m in dec.named_modules():
        if isinstance(m, nn.ConvTranspose1d):
            hooks.append(m.register_forward_hook(
                (lambda n: lambda mod, i, o: lengths.__setitem__(
                    n, i[0].shape[-1]))(nm)))
    with torch.no_grad():
        dec(latent)
    for h in hooks:
        h.remove()
    for name, m in list(dec.named_modules()):
        if isinstance(m, nn.ConvTranspose1d):
            parent = dec
            *path, last = name.split(".")
            for q in path:
                parent = getattr(parent, q)
            setattr(parent, last, ZeroStuffConvT1d(m, lengths[name]))
    with torch.no_grad():
        print(
            f"  decoder eager parity "
            f"corr={corr(ref, dec(latent).numpy()):.6f}")
    export_fp16(
        dec, (latent,), os.path.join(out, "vv_decoder_fp32.tflite"),
        fp16=False)


def build_head(src, ckpt, out):
    """Export the DDPM diffusion head as a GPU-clean fp16 tflite graph.

    Re-authors the head to take a host-computed timestep embedding and uses
    safe RMSNorm; prints eager parity.

    Args:
      src: cloned microsoft/VibeVoice repo.
      ckpt: VibeVoice-Realtime-0.5B checkpoint dir.
      out: output directory for the tflite graph.
    """
    load = load_vv_modules(src)
    cm = load("configuration_vibevoice")
    dm = load("modular_vibevoice_diffusion_head")
    dcfg = cm.VibeVoiceDiffusionHeadConfig(**cfg(ckpt)["diffusion_head_config"])
    head = dm.VibeVoiceDiffusionHead(dcfg).eval()
    sd = state_dict(ckpt)
    pre = "model.prediction_head."
    head.load_state_dict(
        {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)},
        strict=False)

    class ExportHead(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.h = h

        def forward(self, noisy, t_freq, condition):
            h = self.h
            x = h.noisy_images_proj(noisy)
            c = h.cond_proj(condition) + h.t_embedder.mlp(t_freq)
            for layer in h.layers:
                mod = layer.adaLN_modulation(c)
                d = layer.embed_dim
                shift, scale, gate = (
                    mod[..., :d], mod[..., d:2 * d], mod[..., 2 * d:])
                nx = (safe_rms(x, layer.norm.weight, layer.norm.eps)
                      * (1 + scale) + shift)
                x = x + gate * layer.ffn(nx)
            f = h.final_layer
            mod = f.adaLN_modulation(c)
            d = h.cond_dim
            shift, scale = mod[..., :d], mod[..., d:]
            return f.linear(
                safe_rms(x, None, f.norm_final.eps) * (1 + scale) + shift)

    noisy = torch.randn(1, dcfg.latent_size)
    cond = torch.randn(1, dcfg.hidden_size)
    t_freq = head.t_embedder.timestep_embedding(torch.tensor([500.0]), 256)
    net = ExportHead(head).eval()
    with torch.no_grad():
        ref = head(noisy, torch.tensor([500.0]), cond).numpy()
        print(
            f"  head eager parity "
            f"corr={corr(ref, net(noisy, t_freq, cond).numpy()):.6f}")
    export_fp16(
        net, (noisy, t_freq, cond),
        os.path.join(out, "vv_diffhead_fp16.tflite"))


def build_backbone(src, ckpt, out, which, pmax):
    """Export one Qwen2 LM as an fp32 KV-cache-step tflite graph.

    Re-authors attention as manual matmul+softmax with a host-managed packed
    KV cache and safe RMSNorm; the LMs ship fp32 (fp16 collapses on ARM).

    Args:
      src: cloned microsoft/VibeVoice repo (unused; kept for a uniform stage
        signature).
      ckpt: VibeVoice-Realtime-0.5B checkpoint dir.
      out: output directory for the tflite graph.
      which: "base" (4-layer text LM) or "tts" (20-layer TTS LM).
      pmax: KV-cache capacity (padded key/value length).
    """
    from transformers import Qwen2Config, Qwen2Model
    dc = cfg(ckpt)["decoder_config"]
    n_tts = cfg(ckpt).get("tts_backbone_num_hidden_layers", 20)
    layers = n_tts if which == "tts" else dc["num_hidden_layers"] - n_tts
    prefix = (
        "model."
        f"{'tts_language_model' if which == 'tts' else 'language_model'}.")
    conf = Qwen2Config(**{**dc, "num_hidden_layers": layers})
    model = Qwen2Model(conf).eval()
    sd = state_dict(ckpt)
    model.load_state_dict(
        {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)},
        strict=False)
    fnorm = model.norm.weight if which == "tts" else None
    nkv = conf.num_key_value_heads
    nq = conf.num_attention_heads
    hd = conf.hidden_size // nq
    rep = nq // nkv
    eps = dc.get("rms_norm_eps", 1e-6)
    scale = hd ** -0.5

    class KVStep(nn.Module):
        def __init__(self):
            super().__init__()
            # Register as a submodule so the weights are captured.
            self.layers = model.layers
            self.fnorm = fnorm

        def forward(self, x, cos, sin, mask, pk, pv):
            h = x
            nks, nvs = [], []
            for i, layer in enumerate(self.layers):
                a = layer.self_attn
                hn = safe_rms(h, layer.input_layernorm.weight, eps)
                q = F.linear(
                    hn, a.q_proj.weight, a.q_proj.bias).reshape(nq, 1, hd)
                k = F.linear(
                    hn, a.k_proj.weight, a.k_proj.bias).reshape(nkv, 1, hd)
                v = F.linear(
                    hn, a.v_proj.weight, a.v_proj.bias).reshape(nkv, 1, hd)
                c = cos.reshape(1, 1, hd)
                s = sin.reshape(1, 1, hd)
                q = q * c + rotate_half(q) * s
                k = k * c + rotate_half(k) * s
                nks.append(k.reshape(1, nkv, 1, hd))
                nvs.append(v.reshape(1, nkv, 1, hd))
                pk_i = pk[:, i * nkv:(i + 1) * nkv, :, :].reshape(nkv, pmax, hd)
                pv_i = pv[:, i * nkv:(i + 1) * nkv, :, :].reshape(nkv, pmax, hd)
                K = torch.cat(
                    [torch.cat([pk_i, k], dim=1).unsqueeze(1)] * rep,
                    dim=1).reshape(nq, pmax + 1, hd)
                V = torch.cat(
                    [torch.cat([pv_i, v], dim=1).unsqueeze(1)] * rep,
                    dim=1).reshape(nq, pmax + 1, hd)
                sc = (torch.matmul(q, K.transpose(-1, -2)) * scale
                      + mask.reshape(1, 1, pmax + 1))
                ctx = torch.matmul(
                    torch.softmax(sc, dim=-1), V).reshape(1, 1, nq * hd)
                h = h + F.linear(ctx, a.o_proj.weight, a.o_proj.bias)
                hn2 = safe_rms(h, layer.post_attention_layernorm.weight, eps)
                m = layer.mlp
                h = h + F.linear(
                    F.silu(F.linear(hn2, m.gate_proj.weight))
                    * F.linear(hn2, m.up_proj.weight),
                    m.down_proj.weight)
            hidden = (
                safe_rms(h, self.fnorm, eps)
                if self.fnorm is not None else h)
            return hidden, torch.cat(nks, dim=1), torch.cat(nvs, dim=1)

    ex = (
        torch.randn(1, 1, conf.hidden_size),
        torch.randn(1, 1, 1, hd), torch.randn(1, 1, 1, hd),
        torch.zeros(1, 1, 1, pmax + 1),
        torch.randn(1, layers * nkv, pmax, hd),
        torch.randn(1, layers * nkv, pmax, hd))
    # LMs ship fp32 (fp16 collapses to noise on Android ARM XNNPACK — see
    # export_fp16 docstring).
    name = f"vv_{'tts_lm' if which == 'tts' else 'base_lm'}_kv_fp32.tflite"
    export_fp16(KVStep().eval(), ex, os.path.join(out, name), fp16=False)


def build_assets(src, ckpt, out, voice):
    """Export the host-side assets (token embeddings, glue weights, voice).

    Writes embed_tokens.f16 (fp16 token table), glue.f32 (connector + type
    embedding + EOS classifier weights) with glue_layout.json, and the voice
    preset's precomputed prompt KV cache.

    Args:
      src: cloned microsoft/VibeVoice repo (unused; kept for a uniform stage
        signature).
      ckpt: VibeVoice-Realtime-0.5B checkpoint dir.
      out: output directory for the assets.
      voice: path to a voice preset .pt file.
    """
    sd = state_dict(ckpt)

    def npf16(path, t):
        t.detach().float().numpy().astype(np.float16).tofile(path)

    npf16(
        os.path.join(out, "embed_tokens.f16"),
        sd["model.language_model.embed_tokens.weight"])
    order = [
        ("conn_fc1_w", "model.acoustic_connector.fc1.weight"),
        ("conn_fc1_b", "model.acoustic_connector.fc1.bias"),
        ("conn_norm_w", "model.acoustic_connector.norm.weight"),
        ("conn_fc2_w", "model.acoustic_connector.fc2.weight"),
        ("conn_fc2_b", "model.acoustic_connector.fc2.bias"),
        ("type_emb", "model.tts_input_types.weight"),
        ("eos_fc1_w", "tts_eos_classifier.fc1.weight"),
        ("eos_fc1_b", "tts_eos_classifier.fc1.bias"),
        ("eos_fc2_w", "tts_eos_classifier.fc2.weight"),
        ("eos_fc2_b", "tts_eos_classifier.fc2.bias"),
    ]
    layout, off = {}, 0
    with open(os.path.join(out, "glue.f32"), "wb") as f:
        for name, key in order:
            a = sd[key].detach().float().numpy().ravel()
            a.tofile(f)
            layout[name] = {
                "offset": off,
                "count": int(a.size),
                "shape": list(sd[key].shape),
            }
            off += a.size
    layout["speech_scaling_factor"] = float(
        sd["model.speech_scaling_factor"].item())
    layout["speech_bias_factor"] = float(
        sd["model.speech_bias_factor"].item())
    json.dump(
        layout, open(os.path.join(out, "glue_layout.json"), "w"), indent=2)

    # Registers the output type so torch.load can unpickle the voice preset.
    from transformers.modeling_outputs import (
        BaseModelOutputWithPast,  # noqa: F401
    )
    pre = torch.load(voice, map_location="cpu", weights_only=False)
    base = os.path.basename(voice).replace(".pt", "")
    with open(os.path.join(out, f"voice_{base}.bin"), "wb") as f:
        for key in ["lm", "tts_lm", "neg_tts_lm"]:
            c = pre[key].past_key_values
            for cache in (c.key_cache, c.value_cache):
                (np.stack([cache[i][0].float().numpy()
                           for i in range(len(cache))])
                 .astype(np.float16).tofile(f))
        pre["tts_lm"].last_hidden_state[:, -1].float().numpy().tofile(f)
        pre["neg_tts_lm"].last_hidden_state[:, -1].float().numpy().tofile(f)
    print(
        f"  assets: embed_tokens.f16, glue.f32, voice_{base}.bin "
        f"(scale={layout['speech_scaling_factor']}, "
        f"bias={layout['speech_bias_factor']})")


def main():
    """Parse args and build the requested stage(s)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "decoder", "head", "backbone", "assets"])
    ap.add_argument(
        "--src", required=True, help="cloned microsoft/VibeVoice repo")
    ap.add_argument(
        "--ckpt", required=True,
        help="VibeVoice-Realtime-0.5B checkpoint dir")
    ap.add_argument("--out", default="./out")
    ap.add_argument(
        "--voice", default=None,
        help="voice preset .pt (default: en-Emma_woman)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    voice = args.voice or os.path.join(
        args.src, "demo/voices/streaming_model/en-Emma_woman.pt")

    if args.stage in ("all", "decoder"):
        build_decoder(args.src, args.ckpt, args.out)
    if args.stage in ("all", "head"):
        build_head(args.src, args.ckpt, args.out)
    if args.stage in ("all", "backbone"):
        build_backbone(args.src, args.ckpt, args.out, "base", 128)
        build_backbone(args.src, args.ckpt, args.out, "tts", 384)
    if args.stage in ("all", "assets"):
        build_assets(args.src, args.ckpt, args.out, voice)
    print("done.")


if __name__ == "__main__":
    main()
