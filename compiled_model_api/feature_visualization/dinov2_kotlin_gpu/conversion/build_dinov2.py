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

"""DINOv2 ViT-S/14 dense features -> LiteRT CompiledModel GPU.

Emits the 1024 patch tokens (32x32 grid at 448) of DINOv2-small. A host-side
top-3 PCA of those tokens -> RGB gives the classic dense-feature overlay.

Re-authored GPU-clean with the proven ViT recipes: fused-qkv attention
decomposed to 4D, LayerScale gamma baked into the projections, SafeLayerNorm
(fp16 variance-overflow safe), tanh-GELU. timm bakes the pos_embed at the fixed
448 grid at model creation, so there is no runtime interpolation (no GATHER_ND).

Run:  python build_dinov2.py [parity|convert|fp16|all]  (then validate)
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FP32 = os.path.join(HERE, "dinov2_s.tflite")
FP16 = os.path.join(HERE, "dinov2_s_fp16.tflite")
IMG_SIZE = 448
PATCH = 14
GRID = IMG_SIZE // PATCH            # 32
N_PATCH = GRID * GRID              # 1024
N_TOK = N_PATCH + 1               # + cls
C = 384
DEPTH = 12
HEADS = 6
HEAD_DIM = C // HEADS
SCALE = HEAD_DIM ** -0.5
LN_S = 64.0                       # SafeLayerNorm pre-square scale
TIMM_NAME = "vit_small_patch14_dinov2.lvd142m"
IN_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IN_STD = np.array([0.229, 0.224, 0.225], np.float32)


def timm_model():
    """Creates the reference timm DINOv2 at the fixed 448 input.

    Returns:
        The eval-mode timm VisionTransformer.
    """
    import timm
    return timm.create_model(TIMM_NAME, pretrained=True, img_size=IMG_SIZE,
                             num_classes=0).eval()


def safe_layer_norm(x, w, b):
    """LayerNorm over C with an fp16-overflow-safe variance.

    Scales the deviation by 1/LN_S before squaring so the per-token sum of
    squares stays within fp16 range on DINOv2's massive activations, then
    rescales -- algebraically identical to the plain variance.

    Args:
        x: Input tokens, shape [1, N, C].
        w: LayerNorm weight, shape [C].
        b: LayerNorm bias, shape [C].

    Returns:
        The normalized tokens, shape [1, N, C].
    """
    mean = x.mean(-1, keepdim=True)
    d = x - mean
    var = (d * (1.0 / LN_S)).pow(2).mean(-1, keepdim=True) * (LN_S * LN_S)
    return (d * torch.rsqrt(var + 1e-6)) * w + b


def gelu(x):
    """tanh approximation of GELU (delegate-friendly, near-exact).

    Args:
        x: Input tensor.

    Returns:
        The activated tensor.
    """
    inner = 0.7978845608 * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + torch.tanh(inner))


class Block(nn.Module):
    """One DINOv2 transformer block: 4D attention + baked LayerScale."""

    def __init__(self, p, i):
        """Builds block ``i`` from the checkpoint parameters.

        Args:
            p: The timm state dict.
            i: Zero-based block index.
        """
        super().__init__()
        pre = f"blocks.{i}."
        self.n1w = nn.Parameter(p[pre + "norm1.weight"], requires_grad=False)
        self.n1b = nn.Parameter(p[pre + "norm1.bias"], requires_grad=False)
        self.n2w = nn.Parameter(p[pre + "norm2.weight"], requires_grad=False)
        self.n2b = nn.Parameter(p[pre + "norm2.bias"], requires_grad=False)
        self.qkv_w = nn.Parameter(p[pre + "attn.qkv.weight"],
                                  requires_grad=False)
        self.qkv_b = nn.Parameter(p[pre + "attn.qkv.bias"], requires_grad=False)
        g1 = p[pre + "ls1.gamma"]
        g2 = p[pre + "ls2.gamma"]
        self.proj_w = nn.Parameter(g1.view(C, 1) * p[pre + "attn.proj.weight"],
                                   requires_grad=False)
        self.proj_b = nn.Parameter(g1 * p[pre + "attn.proj.bias"],
                                   requires_grad=False)
        self.fc1_w = nn.Parameter(p[pre + "mlp.fc1.weight"],
                                  requires_grad=False)
        self.fc1_b = nn.Parameter(p[pre + "mlp.fc1.bias"], requires_grad=False)
        self.fc2_w = nn.Parameter(g2.view(C, 1) * p[pre + "mlp.fc2.weight"],
                                  requires_grad=False)
        self.fc2_b = nn.Parameter(g2 * p[pre + "mlp.fc2.bias"],
                                  requires_grad=False)

    def forward(self, x):
        """Runs the block on [1, N, C] and returns [1, N, C]."""
        h = safe_layer_norm(x, self.n1w, self.n1b)
        qkv = h @ self.qkv_w.t() + self.qkv_b
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(1, N_TOK, HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(1, N_TOK, HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(1, N_TOK, HEADS, HEAD_DIM).transpose(1, 2)
        attn = ((q * SCALE) @ k.transpose(-2, -1)).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(1, N_TOK, C)
        x = x + (out @ self.proj_w.t() + self.proj_b)
        h2 = safe_layer_norm(x, self.n2w, self.n2b)
        h2 = gelu(h2 @ self.fc1_w.t() + self.fc1_b)
        return x + (h2 @ self.fc2_w.t() + self.fc2_b)


class DINOv2(nn.Module):
    """DINOv2 ViT-S/14 emitting the 1024 patch tokens at a fixed 448 input."""

    def __init__(self, p):
        """Builds the backbone from the timm state dict.

        Args:
            p: The timm state dict.
        """
        super().__init__()
        self.patch_w = nn.Parameter(p["patch_embed.proj.weight"],
                                    requires_grad=False)
        self.patch_b = nn.Parameter(p["patch_embed.proj.bias"],
                                    requires_grad=False)
        self.cls = nn.Parameter(p["cls_token"], requires_grad=False)
        self.pos = nn.Parameter(p["pos_embed"], requires_grad=False)
        self.blocks = nn.ModuleList([Block(p, i) for i in range(DEPTH)])
        self.nw = nn.Parameter(p["norm.weight"], requires_grad=False)
        self.nb = nn.Parameter(p["norm.bias"], requires_grad=False)

    def forward(self, img):
        """Extracts patch tokens.

        Args:
            img: Normalized NCHW image, shape [1, 3, 448, 448].

        Returns:
            Patch tokens, shape [1, 1024, 384].
        """
        x = F.conv2d(img, self.patch_w, self.patch_b, stride=PATCH)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(1, 1, C), x], dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = safe_layer_norm(x, self.nw, self.nb)
        return x[:, 1:]


def load():
    """Loads the DINOv2 state dict (float32) from a fresh timm model."""
    return {k: v.float() for k, v in timm_model().state_dict().items()}


def preprocess(path):
    """Loads an image as an ImageNet-normalized NCHW tensor.

    Args:
        path: Path to an image file.

    Returns:
        A float32 tensor of shape [1, 3, 448, 448].
    """
    im = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE),
                                                Image.BICUBIC)
    arr = (np.asarray(im, np.float32) / 255.0 - IN_MEAN) / IN_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)[None])


def stage_parity(model):
    """Checks the re-authored patch features against stock timm.

    Args:
        model: The reconstructed backbone.
    """
    ref = timm_model()
    x = preprocess(os.path.join(HERE, "test.jpg"))
    with torch.no_grad():
        mine = model(x)[0].numpy()
        gold = ref.forward_features(x)[0, 1:].numpy()
    corr = np.corrcoef(mine.flatten(), gold.flatten())[0, 1]
    print("re-authored vs timm patch features: corr %.6f  max|d| %.4f"
          % (corr, np.abs(mine - gold).max()))


def stage_convert(model):
    """Converts the backbone to a float32 tflite with litert-torch.

    Args:
        model: The reconstructed backbone.
    """
    import litert_torch
    ex = (torch.zeros(1, 3, IMG_SIZE, IMG_SIZE),)
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
    model = DINOv2(load()).eval()
    if stage in ("parity", "all"):
        stage_parity(model)
    if stage in ("convert", "all"):
        stage_convert(model)
    if stage in ("fp16", "all"):
        stage_fp16()


if __name__ == "__main__":
    main()
