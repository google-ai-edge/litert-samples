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

"""Converts Depth Anything 3 (Small) monocular depth to a GPU-clean LiteRT .tflite.

Reproduces litert-community/Depth-Anything-3-Small/da3_small_gpu_fp16.tflite. DA3 is
NOT GPU-clean out of the box; this applies exact, weights-verbatim rewrites so the
DINOv2-RoPE backbone + DPT head ride the ML Drift GPU delegate:

  * checkpoint key-prefix fix (strip the leading "model.")
  * C14  RoPE data-dependent int max_position -> constant table
  * C12  fused-qkv 5D head-split -> separate q/k/v Linears, manual 4D attention
  * LayerScale folded into the preceding Linear (else the token dim is mis-laid-out
    on GPU: fully_connected {1,1,N,C} vs {N,1,1,C} -> compile fails)
  * pos_embed bicubic-resized and baked to a constant (interpolating a constant
    emits a runtime-less RESIZE_BILINEAR the delegate rejects)
  * ConvTranspose2d -> zero-stuff nearest-upsample + Conv2d (exact ~1e-7; Pixel 8a
    rejects TRANSPOSE_CONV)
  * camera-token in-place index-assign -> torch.cat (else SELECT_V2 with a broadcast
    'else' shape the delegate rejects)
  * DPT head resize align_corners=True -> False (banned) + drop the UV sincos
    pos-embed-again (BROADCAST_TO)

Measures depth corr vs the ORIGINAL (all-stock) model, op-checks the graph, then
FP16-quantizes. Needs the upstream DA3 source on the path (clone
github.com/ByteDance-Seed/Depth-Anything-3 and run this from its root so `src/` and
`depth_anything_3` import).

    pip install litert-torch ai-edge-quantizer torch timm safetensors huggingface_hub pillow
    python convert_da3_litert.py [image] [H] [W]   # H,W default to the shipped 896x504
"""

import collections
import json
import math
import os
import sys
import types

import numpy as np


class _Dummy:
    """Absorbs any attribute access and returns a no-op callable."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


# Guards that must run BEFORE torch / depth_anything_3 are imported: stub the
# scipy PROPACK extension (crashes on macOS), restore removed numpy aliases,
# and put the DA3 checkout's `src/` on the path.
_propack = types.ModuleType("scipy.sparse.linalg._propack")
for _name in ("_spropack", "_dpropack", "_cpropack", "_zpropack"):
    setattr(_propack, _name, _Dummy())
sys.modules["scipy.sparse.linalg._propack"] = _propack
for _name, _type in (("bool", bool), ("float", float), ("int", int),
                     ("object", object), ("str", str)):
    if not hasattr(np, _name):
        setattr(np, _name, _type)
sys.path.insert(0, "src")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from depth_anything_3.cfg import create_object  # noqa: E402
from depth_anything_3.model.dinov2.layers.attention import Attention  # noqa: E402
import depth_anything_3.model.dinov2.layers.rope as _rope  # noqa: E402
import depth_anything_3.model.dinov2.vision_transformer as _vt  # noqa: E402
import depth_anything_3.model.utils.head_utils as _hu  # noqa: E402
import depth_anything_3.model.dualdpt as _dd  # noqa: E402
import depth_anything_3.model.dpt as _dpt  # noqa: E402

# Canonical banned-op list for the op-check; SELECT_V2 is NOT banned.
BANNED = {"GATHER_ND", "GATHER", "TOPK_V2", "PACK", "SPLIT", "FLEX_ERF", "ERF",
          "TRANSPOSE_CONV", "BROADCAST_TO"}


def _rope_forward(self, tokens, positions):
    """C14: constant-table RoPE forward.

    Stock RoPE computes max_position = int(positions.max()) + 1, which is
    data-dependent and aborts torch.export. A constant 128 covers the 33 rows a
    518 grid needs; extra table rows are unused -> numerically identical.
    """
    half_dim = tokens.size(-1) // 2
    cos, sin = self._compute_frequency_components(half_dim, 128, tokens.device, tokens.dtype)
    vertical, horizontal = tokens.chunk(2, dim=-1)
    vertical = self._apply_1d_rope(vertical, positions[..., 0], cos, sin)
    horizontal = self._apply_1d_rope(horizontal, positions[..., 1], cos, sin)
    return torch.cat((vertical, horizontal), dim=-1)


def _attention_forward(self, x, pos=None, attn_mask=None):
    """C12: manual 4D attention over the decomposed q/k/v Linears."""
    B, N, C = x.shape
    heads = self.num_heads
    head_dim = C // heads
    q = self.q_lin(x).reshape(B, N, heads, head_dim).permute(0, 2, 1, 3)
    k = self.k_lin(x).reshape(B, N, heads, head_dim).permute(0, 2, 1, 3)
    v = self.v_lin(x).reshape(B, N, heads, head_dim).permute(0, 2, 1, 3)
    q, k = self.q_norm(q), self.k_norm(k)
    if self.rope is not None and pos is not None:
        q = self.rope(q, pos)
        k = self.rope(k, pos)
    q = q * self.scale
    attn = (q @ k.transpose(-2, -1)).softmax(-1)
    return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))


def decompose_qkv(net):
    """C12: split each fused qkv Linear into separate q/k/v Linears (weights verbatim)."""
    for mod in net.modules():
        if isinstance(mod, Attention):
            C = mod.qkv.in_features
            weight = mod.qkv.weight
            bias = mod.qkv.bias
            for name, sl in (("q_lin", slice(0, C)), ("k_lin", slice(C, 2 * C)),
                             ("v_lin", slice(2 * C, 3 * C))):
                lin = nn.Linear(C, C, bias=bias is not None)
                with torch.no_grad():
                    lin.weight.copy_(weight[sl])
                    if bias is not None:
                        lin.bias.copy_(bias[sl])
                setattr(mod, name, lin)


def bake_layerscale(model):
    """Folds ls1/ls2 gamma into attn.proj / mlp.fc2 and replaces them with Identity.

    The LayerScale MUL (FC output [N,C] * gamma [C]) makes ML Drift mis-lay-out the
    token dim ({1,1,1025,384} vs {1025,1,1,384}) -> GPU compile fails. Baking
    eliminates the MUL. (MoGe's fix.)
    """
    count = 0
    for block in model.modules():
        if hasattr(block, "ls1") and hasattr(block.ls1, "gamma") and hasattr(block, "attn"):
            gamma = block.ls1.gamma.data.squeeze()
            with torch.no_grad():
                block.attn.proj.weight.data.mul_(gamma.unsqueeze(1))
                if block.attn.proj.bias is not None:
                    block.attn.proj.bias.data.mul_(gamma)
            block.ls1 = nn.Identity()
            count += 1
        if hasattr(block, "ls2") and hasattr(block.ls2, "gamma") and hasattr(block, "mlp"):
            gamma = block.ls2.gamma.data.squeeze()
            last = None
            for child in reversed(list(block.mlp.children())):
                if isinstance(child, nn.Linear):
                    last = child
                    break
            if last is None and hasattr(block.mlp, "fc2"):
                last = block.mlp.fc2
            if last is not None:
                with torch.no_grad():
                    last.weight.data.mul_(gamma.unsqueeze(1))
                    if last.bias is not None:
                        last.bias.data.mul_(gamma)
            block.ls2 = nn.Identity()
            count += 1
    print(f"baked {count} LayerScale into Linear")
    return count


def bake_pos_embed(net, grid_h, grid_w):
    """C15: precompute the resized pos_embed as a constant buffer.

    Interpolating the constant pos_embed emits RESIZE_BILINEAR with 0 runtime
    inputs, which the GPU delegate rejects.
    """
    for mod in net.modules():
        if hasattr(mod, "interpolate_pos_encoding") and hasattr(mod, "pos_embed"):
            with torch.no_grad():
                n_patches = mod.pos_embed.shape[1] - 1
                side = int(math.sqrt(n_patches))
                dim = mod.pos_embed.shape[-1]
                pe = mod.pos_embed.float()
                cls_token = pe[:, 0]
                patch = pe[:, 1:]
                # bicubic = match official (baked const).
                patch = F.interpolate(patch.reshape(1, side, side, dim).permute(0, 3, 1, 2),
                                      size=(grid_h, grid_w), mode="bicubic", antialias=False)
                patch = patch.permute(0, 2, 3, 1).view(1, -1, dim)
                baked = torch.cat((cls_token.unsqueeze(0), patch),
                                  dim=1).to(mod.pos_embed.dtype)  # [1, N+1, dim]
            mod.register_buffer("_baked_pos", baked)
            mod.interpolate_pos_encoding = types.MethodType(
                lambda self, x, w, h: self._baked_pos, mod)


def _patched_get_intermediate_layers(self, x, n=1, export_feat_layers=(), **kwargs):
    """SELECT_V2 fix: copy of _get_intermediate_layers_not_chunked with the camera-token
    in-place index-assign `x[:, :, 0] = cam_token` replaced by an equivalent torch.cat
    (exact, GPU-clean). alt_start must stay on — the camera-token / global-attn path
    DOES affect mono depth."""
    B, S, _, H, W = x.shape
    x = self.prepare_tokens_with_masks(x)
    output, total_block_len, aux_output = [], len(self.blocks), []
    blocks_to_take = (range(total_block_len - n, total_block_len)
                      if isinstance(n, int) else n)
    pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)
    local_x = x
    for i, blk in enumerate(self.blocks):
        if i < self.rope_start or self.rope is None:
            g_pos, l_pos = None, None
        else:
            g_pos, l_pos = pos_nodiff, pos
        if (self.alt_start != -1 and i == self.alt_start - 1
                and x.shape[1] >= _vt.THRESH_FOR_REF_SELECTION
                and kwargs.get("cam_token", None) is None):
            b_idx = _vt.select_reference_view(
                x, strategy=kwargs.get("ref_view_strategy", "saddle_balanced"))
            x = _vt.reorder_by_reference(x, b_idx)
            local_x = _vt.reorder_by_reference(local_x, b_idx)
        if self.alt_start != -1 and i == self.alt_start:
            if kwargs.get("cam_token", None) is not None:
                cam_token = kwargs.get("cam_token")
            else:
                ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                cam_token = torch.cat([ref_token, src_token], dim=1)
            # was: x[:, :, 0] = cam_token
            x = torch.cat([cam_token.unsqueeze(2), x[:, :, 1:]], dim=2)
        if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
            x = self.process_attention(x, blk, "global", pos=g_pos,
                                       attn_mask=kwargs.get("attn_mask", None))
        else:
            x = self.process_attention(x, blk, "local", pos=l_pos)
            local_x = x
        if i in blocks_to_take:
            out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
            if (x.shape[1] >= _vt.THRESH_FOR_REF_SELECTION and self.alt_start != -1
                    and "b_idx" in locals()):
                out_x = _vt.restore_original_order(out_x, b_idx)
            output.append((out_x[:, :, 0], out_x))
        if i in export_feat_layers:
            aux_output.append(x)
    return output, aux_output


class ZeroStuffConvT(nn.Module):
    """Exact GPU-clean ConvTranspose2d(k=s, stride=s) replacement (Pixel 8a rejects
    TRANSPOSE_CONV): zero-stuff (nearest-upsample x top-left constant mask) + Conv2d
    with the flipped weight. Matches the learned upsampler to ~1e-7, so depth stays
    as sharp as the original (a bilinear approximation blurred it)."""

    def __init__(self, ct, height, width):
        super().__init__()
        self.stride = ct.stride[0]
        self.kernel = ct.kernel_size[0]
        self.register_buffer("w", ct.weight.flip(2, 3).transpose(0, 1).contiguous())
        self.register_buffer("b", ct.bias.detach().clone() if ct.bias is not None
                             else torch.zeros(ct.out_channels))
        s = self.stride
        mask = np.zeros((height * s, width * s), np.float32)
        mask[::s, ::s] = 1.0
        self.register_buffer("mask", torch.from_numpy(mask)[None, None])

    def forward(self, x):
        height, width = x.shape[-2], x.shape[-1]
        s, k = self.stride, self.kernel
        stuffed = F.interpolate(x, size=(height * s, width * s), mode="nearest")
        y = F.conv2d(stuffed * self.mask, self.w, bias=self.b, padding=k - 1)
        return y[:, :, :height * s, :width * s]


def swap_convtranspose(net, run_dummy):
    """Discovers each ConvTranspose2d input size via a dry run, then swaps in the
    exact ZeroStuffConvT equivalent."""
    ct_sizes, hooks = {}, []

    def make_hook(name):
        def hook(module, inputs, output):
            ct_sizes[name] = (inputs[0].shape[-2], inputs[0].shape[-1])
        return hook

    for name, mod in net.named_modules():
        if isinstance(mod, nn.ConvTranspose2d):
            hooks.append(mod.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        run_dummy()
    for hook in hooks:
        hook.remove()

    def swap(module, prefix=""):
        for name, child in module.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.ConvTranspose2d):
                height, width = ct_sizes[full]
                setattr(module, name, ZeroStuffConvT(child, height, width))
            else:
                swap(child, full)

    swap(net)


def patch_dpt_head(net):
    """DPT head: align_corners=True -> False (banned RESIZE_BILINEAR variant) and
    drop the UV sincos pos-embed-again (make_sincos broadcast emits BROADCAST_TO).

    Baking the UV sincos as constants matches official ~0.0002 better but adds
    ~64 MB (full [1,C,H,W] per shape) — not worth it; the ratio-0.1 UV refinement
    is negligible vs the size cost."""
    original_interpolate = _hu.custom_interpolate

    def interpolate_no_align_corners(x, size=None, scale_factor=None, mode="bilinear",
                                     align_corners=True):
        return original_interpolate(x, size=size, scale_factor=scale_factor, mode=mode,
                                    align_corners=False)

    _hu.custom_interpolate = interpolate_no_align_corners
    _dd.custom_interpolate = interpolate_no_align_corners
    _dpt.custom_interpolate = interpolate_no_align_corners

    disabled = 0
    for mod in net.modules():
        if isinstance(mod, _dd.DualDPT) and getattr(mod, "pos_embed", False):
            mod.pos_embed = False
            disabled += 1
    print(f"disabled head pos_embed on {disabled} DualDPT")


def opcheck(path):
    """Prints the banned-op / >4D report for the exported flatbuffer."""
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=path)
    interpreter.allocate_tensors()
    ops = collections.Counter(d.get("op_name", "?")
                              for d in interpreter._get_ops_details())
    bad = {k: v for k, v in ops.items() if k in BANNED}
    over_4d = sum(1 for d in interpreter.get_tensor_details()
                  if len(d.get("shape", [])) > 4)
    print(f"op-check FP32: banned {bad or 'NONE'} | >4D {over_4d} | "
          f"GELU {ops.get('GELU', 0)} | SELECT_V2 {ops.get('SELECT_V2', 0)}")
    print("VERDICT:", "GPU-CLEAN" if not bad and not over_4d else "BLOCKERS REMAIN")


def quantize_fp16(fp32_path, fp16_path):
    """FP16 weights via AI Edge Quantizer FLOAT_CASTING: half size, native on the
    GPU delegate, FP32 == FP16 here."""
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping

    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT,
        ),
        algorithm_key=AlgorithmName.FLOAT_CASTING,
    )
    if os.path.exists(fp16_path):
        os.remove(fp16_path)
    qt = quantizer.Quantizer(float_model=fp32_path)
    qt.load_quantization_recipe(rm.get_quantization_recipe())
    qt.quantize().export_model(fp16_path)


class Mono(nn.Module):
    """Single-image wrapper: (1,3,H,W) -> depth map (backbone + DPT head only)."""

    def __init__(self, net):
        super().__init__()
        self.b = net.backbone
        self.h = net.head

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        features, _ = self.b(x.unsqueeze(1), cam_token=None)
        return self.h(features, H, W, patch_start_idx=0)["depth"]


def main():
    # C14 RoPE patch must be in place before the first forward pass.
    _rope.RotaryPositionEmbedding2D.forward = _rope_forward

    with open(hf_hub_download("depth-anything/DA3-SMALL", "config.json")) as f:
        cfg = json.load(f)
    net = create_object(cfg["config"]).eval()
    sd = load_file(hf_hub_download("depth-anything/DA3-SMALL", "model.safetensors"))
    net.load_state_dict({(k[6:] if k.startswith("model.") else k): v
                         for k, v in sd.items()}, strict=False)
    mono = Mono(net).eval()

    # Input H,W (multiples of 14). argv[2]=H, argv[3]=W. Default = the shipped
    # 896x504 (portrait native aspect; DA3 runs at the image's native aspect, no
    # padding — a square letterbox drops fidelity to corr ~0.977 as padding leaks
    # through global attention).
    height_in = int(sys.argv[2]) if len(sys.argv) > 2 else 896
    width_in = int(sys.argv[3]) if len(sys.argv) > 3 else 504

    # argv[1] image gives a meaningful depth-corr check; without one a random
    # tensor still validates eager parity + the GPU-clean op-check.
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        img = Image.open(sys.argv[1]).convert("RGB").resize((width_in, height_in),
                                                            Image.BILINEAR)
        arr = ((np.asarray(img, np.float32) / 255.0 - [0.485, 0.456, 0.406])
               / [0.229, 0.224, 0.225])
        x_img = torch.from_numpy(np.transpose(arr, (2, 0, 1))[None].astype(np.float32))
    else:
        print("no image given -> random input (op-check + eager parity only; "
              "depth corr not meaningful)")
        x_img = torch.randn(1, 3, height_in, width_in)

    # Reference output from the ORIGINAL (all-stock) model, before any rewrite
    # beyond the numerically-identical RoPE table fix.
    with torch.no_grad():
        depth_original = mono(x_img)[0, 0].numpy()

    # Apply the GPU-clean rewrites (order matters only in that they all precede
    # the parity check below).
    Attention.forward = _attention_forward
    decompose_qkv(net)
    bake_layerscale(net)
    bake_pos_embed(net, height_in // 14, width_in // 14)
    _vt.DinoVisionTransformer._get_intermediate_layers_not_chunked = (
        _patched_get_intermediate_layers)
    swap_convtranspose(net, lambda: mono(x_img))
    patch_dpt_head(net)

    with torch.no_grad():
        depth_clean = mono(x_img)[0, 0].numpy()  # Fully GPU-clean.
    corr = np.corrcoef(depth_original.flatten(), depth_clean.flatten())[0, 1]
    rel_diff = (np.abs(depth_original - depth_clean).mean()
                / (np.abs(depth_original).mean() + 1e-9) * 100)
    print(f"depth corr (original vs full GPU-clean) = {corr:.6f}  "
          f"mean-rel-diff = {rel_diff:.3f}%")

    # Convert + op-check + FP16 quantize.
    import litert_torch
    dummy = torch.rand(1, 3, height_in, width_in)
    litert_torch.convert(mono.eval(), (dummy,)).export("da3_small_gpu.tflite")
    opcheck("da3_small_gpu.tflite")
    print("FP32 size %.1f MB" % (os.path.getsize("da3_small_gpu.tflite") / 1e6))

    fp16_path = "da3_small_gpu_fp16.tflite"  # This is what the app downloads.
    quantize_fp16("da3_small_gpu.tflite", fp16_path)
    print("FP16 size %.1f MB -> %s" % (os.path.getsize(fp16_path) / 1e6, fp16_path))


if __name__ == "__main__":
    main()
