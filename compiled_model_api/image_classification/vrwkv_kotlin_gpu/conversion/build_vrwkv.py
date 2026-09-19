#!/usr/bin/env python3
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

"""Vision-RWKV (VRWKV-S) -> LiteRT CompiledModel GPU conversion.

VRWKV is an RWKV-style vision backbone (ICLR 2025). Its core op is a
bidirectional WKV (a CUDA kernel), which we re-author GPU-clean. Because the
token count is fixed (T = 14*14 = 196), the bidirectional WKV is exactly a
per-channel decay-biased attention over the token sequence:

  L[c,t,i] = k[c,i] - (spatial_decay[c]/T)|t-i| + (spatial_first[c]/T)delta
  y[c,t]   = sum_i softmax_i(L[c,t,:]) * v[c,i]

i.e. C independent [T,T] attention matrices -> plain 4D softmax + matmul, no
scan. The token-distance matrix is fed as a runtime input so the per-channel
[C,T,T] decay bias is computed live instead of being const-folded into a large
constant (which makes an unshippable 1.5 GB model). Q-Shift becomes pad + slice
+ concat (<=4D). LayerScale is baked into the following norm. RWKV uses
square-relu + sigmoid gating (no GELU).

Run:  python build_vrwkv.py [parity|convert|fp16|all]  (then validate_vrwkv.py)
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "vrwkv_s_in1k_224.pth")
FP32 = os.path.join(HERE, "vrwkv_s.tflite")
FP16 = os.path.join(HERE, "vrwkv_s_fp16.tflite")

IMG_SIZE = 224
PATCH = 16
GRID = IMG_SIZE // PATCH          # 14
T = GRID * GRID                   # 196 tokens
C = 384                           # embed dims (VRWKV-S)
DEPTH = 12
IN_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
IN_STD = np.array([58.395, 57.12, 57.375], np.float32)

_IDX = torch.arange(T, dtype=torch.float32)
DIST_HOST = (_IDX.view(T, 1) - _IDX.view(1, T)).abs().view(1, 1, T, T)
# The token-distance geometry is a runtime input so the per-channel [C,T,T]
# decay bias (w * dist) is computed live instead of const-folded into a
# ~59 MB-per-block flatbuffer constant. eye = relu(1 - dist).
_RT = {}


def load_state():
    """Loads the VRWKV-S checkpoint as a float32 state dict.

    Returns:
        Mapping from parameter name to a float32 tensor.
    """
    sd = torch.load(CKPT, map_location="cpu")["state_dict"]
    return {k: v.float() for k, v in sd.items()}


def bi_wkv_matrix(k, v, decay, first):
    """GPU-clean bidirectional WKV via a fixed-size [B,C,T,T] softmax.

    Args:
        k: Key tensor, shape [B, T, C].
        v: Value tensor, shape [B, T, C].
        decay: Raw per-channel spatial_decay, shape [C].
        first: Raw per-channel spatial_first, shape [C].

    Returns:
        The mixed output, shape [B, T, C].
    """
    b = k.shape[0]
    dist = _RT["dist"]                          # [1,1,T,T] runtime input
    eye = torch.relu(1.0 - dist)
    w = (decay / T).view(1, C, 1, 1)            # per-channel decay
    u = (first / T).view(1, C, 1, 1)            # per-channel self weight
    kc = k.transpose(1, 2).reshape(b, C, 1, T)  # [B,C,1,T] over i
    vc = v.transpose(1, 2).reshape(b, C, T, 1)  # [B,C,T,1] over i
    logits = kc - w * dist + u * eye            # [B,C,T,T]
    attn = F.softmax(logits, dim=-1)            # over i
    y = torch.matmul(attn, vc)                  # [B,C,T,1]
    return y.reshape(b, C, T).transpose(1, 2)   # [B,T,C]


def bi_wkv_reference(k, v, decay, first):
    """Explicit double-loop reference of the same math (oracle).

    Args:
        k: Key tensor, shape [B, T, C].
        v: Value tensor, shape [B, T, C].
        decay: Raw per-channel spatial_decay, shape [C].
        first: Raw per-channel spatial_first, shape [C].

    Returns:
        The mixed output, shape [B, T, C].
    """
    b = k.shape[0]
    w = decay / T
    u = first / T
    out = torch.zeros(b, T, C)
    for t in range(T):
        num = torch.zeros(b, C)
        den = torch.zeros(b, C)
        for i in range(T):
            if i == t:
                weight = torch.exp(u + k[:, t])
            else:
                weight = torch.exp(k[:, i] - w * abs(t - i))
            num = num + weight * v[:, i]
            den = den + weight
        out[:, t] = num / (den + 1e-6)
    return out


def q_shift(x):
    """Spatial q-shift: shifts channel quarters by 1 px.

    Args:
        x: Token features, shape [B, T, C], over the GRID*GRID grid.

    Returns:
        The shifted features, shape [B, T, C].
    """
    b = x.shape[0]
    g = x.transpose(1, 2).reshape(b, C, GRID, GRID)     # [B,C,H,W]
    q = C // 4
    q0 = F.pad(g[:, 0:q, :, 0:GRID - 1], (1, 0, 0, 0))          # right
    q1 = F.pad(g[:, q:2 * q, :, 1:GRID], (0, 1, 0, 0))         # left
    q2 = F.pad(g[:, 2 * q:3 * q, 0:GRID - 1, :], (0, 0, 1, 0))  # down
    q3 = F.pad(g[:, 3 * q:4 * q, 1:GRID, :], (0, 0, 0, 1))     # up
    rest = g[:, 4 * q:, :, :]
    out = torch.cat([q0, q1, q2, q3, rest], dim=1)
    return out.reshape(b, C, T).transpose(1, 2)


class SpatialMix(nn.Module):
    """VRWKV spatial-mix: q-shift + Bi-WKV + gated output."""

    def __init__(self, p, pre):
        """Builds the mixer from the checkpoint parameters.

        Args:
            p: The model state dict.
            pre: Key prefix (e.g. "backbone.layers.0.att.").
        """
        super().__init__()
        self.decay = nn.Parameter(p[pre + "spatial_decay"],
                                  requires_grad=False)
        self.first = nn.Parameter(p[pre + "spatial_first"],
                                  requires_grad=False)
        self.mix_k = nn.Parameter(p[pre + "spatial_mix_k"].view(C),
                                  requires_grad=False)
        self.mix_v = nn.Parameter(p[pre + "spatial_mix_v"].view(C),
                                  requires_grad=False)
        self.mix_r = nn.Parameter(p[pre + "spatial_mix_r"].view(C),
                                  requires_grad=False)
        self.key = nn.Parameter(p[pre + "key.weight"], requires_grad=False)
        self.value = nn.Parameter(p[pre + "value.weight"],
                                  requires_grad=False)
        self.recept = nn.Parameter(p[pre + "receptance.weight"],
                                   requires_grad=False)
        self.output = nn.Parameter(p[pre + "output.weight"],
                                   requires_grad=False)

    def forward(self, x):
        """Mixes tokens for [B, T, C] and returns [B, T, C]."""
        xx = q_shift(x)
        xk = x * self.mix_k + xx * (1 - self.mix_k)
        xv = x * self.mix_v + xx * (1 - self.mix_v)
        xr = x * self.mix_r + xx * (1 - self.mix_r)
        k = xk @ self.key.t()
        v = xv @ self.value.t()
        sr = torch.sigmoid(xr @ self.recept.t())
        y = bi_wkv_matrix(k, v, self.decay, self.first)
        return (sr * y) @ self.output.t()


class ChannelMix(nn.Module):
    """VRWKV channel-mix (FFN): q-shift + square-relu + gating."""

    def __init__(self, p, pre):
        """Builds the FFN from the checkpoint parameters.

        Args:
            p: The model state dict.
            pre: Key prefix (e.g. "backbone.layers.0.ffn.").
        """
        super().__init__()
        self.mix_k = nn.Parameter(p[pre + "spatial_mix_k"].view(C),
                                  requires_grad=False)
        self.mix_r = nn.Parameter(p[pre + "spatial_mix_r"].view(C),
                                  requires_grad=False)
        self.key = nn.Parameter(p[pre + "key.weight"], requires_grad=False)
        self.recept = nn.Parameter(p[pre + "receptance.weight"],
                                   requires_grad=False)
        self.value = nn.Parameter(p[pre + "value.weight"],
                                  requires_grad=False)

    def forward(self, x):
        """Applies the gated FFN to [B, T, C] and returns [B, T, C]."""
        xx = q_shift(x)
        xk = x * self.mix_k + xx * (1 - self.mix_k)
        xr = x * self.mix_r + xx * (1 - self.mix_r)
        k = torch.square(torch.relu(xk @ self.key.t()))
        kv = k @ self.value.t()
        return torch.sigmoid(xr @ self.recept.t()) * kv


def layer_norm(x, w, b):
    """LayerNorm over the last (channel) dim of [B, T, C].

    Args:
        x: Input features, shape [B, T, C].
        w: LayerNorm weight, shape [C].
        b: LayerNorm bias, shape [C].

    Returns:
        The normalized features, shape [B, T, C].
    """
    return F.layer_norm(x, (C,), w, b)


class Block(nn.Module):
    """A VRWKV block in POST-norm mode (config post_norm=True).

    The norm follows the mixer and the LayerScale gamma is baked into
    that norm's affine params.
    """

    def __init__(self, p, i):
        """Builds block ``i`` from the checkpoint parameters.

        Args:
            p: The model state dict.
            i: Zero-based block index.
        """
        super().__init__()
        pre = f"backbone.layers.{i}."
        self.i = i
        g1 = p[pre + "gamma1"]
        g2 = p[pre + "gamma2"]
        self.ln1w = nn.Parameter(g1 * p[pre + "ln1.weight"],
                                 requires_grad=False)
        self.ln1b = nn.Parameter(g1 * p[pre + "ln1.bias"],
                                 requires_grad=False)
        self.ln2w = nn.Parameter(g2 * p[pre + "ln2.weight"],
                                 requires_grad=False)
        self.ln2b = nn.Parameter(g2 * p[pre + "ln2.bias"],
                                 requires_grad=False)
        if i == 0:
            self.ln0w = nn.Parameter(p[pre + "ln0.weight"],
                                     requires_grad=False)
            self.ln0b = nn.Parameter(p[pre + "ln0.bias"],
                                     requires_grad=False)
        self.att = SpatialMix(p, pre + "att.")
        self.ffn = ChannelMix(p, pre + "ffn.")

    def forward(self, x):
        """Runs the block on [B, T, C] and returns [B, T, C]."""
        if self.i == 0:
            x = layer_norm(x, self.ln0w, self.ln0b)
        x = x + layer_norm(self.att(x), self.ln1w, self.ln1b)
        x = x + layer_norm(self.ffn(x), self.ln2w, self.ln2b)
        return x


class VRWKV(nn.Module):
    """VRWKV-S ImageNet-1K classifier (patch embed + blocks + head)."""

    def __init__(self, p):
        """Builds the classifier from the checkpoint parameters.

        Args:
            p: The model state dict.
        """
        super().__init__()
        self.patch_w = nn.Parameter(
            p["backbone.patch_embed.projection.weight"], requires_grad=False)
        self.patch_b = nn.Parameter(
            p["backbone.patch_embed.projection.bias"], requires_grad=False)
        self.pos = nn.Parameter(p["backbone.pos_embed"], requires_grad=False)
        self.blocks = nn.ModuleList([Block(p, i) for i in range(DEPTH)])
        self.lnw = nn.Parameter(p["backbone.ln1.weight"], requires_grad=False)
        self.lnb = nn.Parameter(p["backbone.ln1.bias"], requires_grad=False)
        self.head_w = nn.Parameter(p["head.fc.weight"], requires_grad=False)
        self.head_b = nn.Parameter(p["head.fc.bias"], requires_grad=False)

    def forward(self, img, dist):
        """Classifies an image.

        Args:
            img: Normalized NCHW image, shape [1, 3, 224, 224].
            dist: Token-distance matrix, shape [1, 1, 196, 196].

        Returns:
            Class logits, shape [1, 1000].
        """
        _RT["dist"] = dist
        x = F.conv2d(img, self.patch_w, self.patch_b, stride=PATCH)
        x = x.flatten(2).transpose(1, 2)                # [B,T,C]
        x = x + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = layer_norm(x, self.lnw, self.lnb)
        x = x.mean(dim=1)                               # global avg pool
        return x @ self.head_w.t() + self.head_b        # [B,1000]


def preprocess(path):
    """Loads and ImageNet-normalizes an image to an NCHW tensor.

    Args:
        path: Path to an image file.

    Returns:
        A float32 tensor of shape [1, 3, 224, 224].
    """
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = 256 / min(w, h)
    im = im.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    w, h = im.size
    left, top = (w - IMG_SIZE) // 2, (h - IMG_SIZE) // 2
    im = im.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    arr = (np.asarray(im, np.float32) - IN_MEAN) / IN_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)[None])


def labels():
    """Returns the 1000 ImageNet-1K class names."""
    with open(os.path.join(HERE, "imagenet_classes.txt")) as f:
        return [line.strip() for line in f]


def stage_parity(model):
    """Checks the Bi-WKV oracle and a real-image top-5 prediction.

    Args:
        model: The reconstructed VRWKV classifier.
    """
    torch.manual_seed(0)
    _RT["dist"] = DIST_HOST
    k = torch.randn(1, T, C) * 0.5
    v = torch.randn(1, T, C) * 0.5
    dec = torch.randn(C)
    fst = torch.randn(C)
    a = bi_wkv_matrix(k, v, dec, fst)
    b = bi_wkv_reference(k, v, dec, fst)
    corr = np.corrcoef(a.flatten(), b.flatten())[0, 1]
    print("Bi-WKV matrix vs explicit: max|d| %.3e  corr %.7f"
          % ((a - b).abs().max(), corr))

    x = preprocess(os.path.join(HERE, "dog.jpg"))
    with torch.no_grad():
        logits = model(x, DIST_HOST)[0]
    names = labels()
    top = torch.topk(logits, 5)
    print("dog.jpg top-5:")
    for score, idx in zip(top.values, top.indices):
        print("   %6.2f  %s" % (score.item(), names[idx]))


def stage_convert(model):
    """Converts the classifier to a float32 tflite with litert-torch.

    Args:
        model: The reconstructed VRWKV classifier.
    """
    ex = (torch.zeros(1, 3, IMG_SIZE, IMG_SIZE), DIST_HOST.clone())
    import litert_torch
    litert_torch.convert(model.eval(), ex).export(FP32)
    print("convert: %.1f MB -> %s" % (os.path.getsize(FP32) / 1e6, FP32))


def stage_fp16():
    """Casts the float32 tflite to fp16 via ai_edge_quantizer."""
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
    if os.path.exists(FP16):
        os.remove(FP16)
    qt = quantizer.Quantizer(float_model=FP32)
    qt.load_quantization_recipe(rm.get_quantization_recipe())
    qt.quantize().export_model(FP16)
    print("fp16: %.1f MB -> %s" % (os.path.getsize(FP16) / 1e6, FP16))


def main():
    """Runs the requested conversion stage(s)."""
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    model = VRWKV(load_state()).eval()
    if stage in ("parity", "all"):
        stage_parity(model)
    if stage in ("convert", "all"):
        stage_convert(model)
    if stage in ("fp16", "all"):
        stage_fp16()


if __name__ == "__main__":
    main()
