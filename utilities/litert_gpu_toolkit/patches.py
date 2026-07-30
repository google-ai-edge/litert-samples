# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Auto-patching pipeline for GPU-incompatible PyTorch patterns.

Each patch function returns the number of modifications made.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("litert_gpu_toolkit")

# ─── GELU → Sigmoid approximation ────────────────────────────────────────────

class SigmoidGELU(nn.Module):
    """GELU approximation: x * sigmoid(1.702 * x). Max error ~0.01.

    No x^3 term → no fp16 overflow risk; the hardened default for classification /
    feature backbones where a ~0.01 activation error is invisible.
    """
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class TanhGELU(nn.Module):
    """GELU tanh approximation: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))).

    GELU(erf) is banned on GPU (FlexErf); this is the closest GPU-clean form
    (MUL/ADD/TANH, no POW). Use for regression heads (depth, normals, keypoints)
    where SigmoidGELU's ~0.01 error visibly shifts the output. The x^3 term can
    overflow fp16 for |x| > ~35 — check activation magnitudes first.
    """
    C = math.sqrt(2.0 / math.pi)

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(self.C * (x + 0.044715 * x * x * x)))


_original_gelu = F.gelu


def patch_gelu(model: nn.Module, approximation: str = 'sigmoid') -> int:
    """Replace GELU activations (nn.GELU and transformers' GELUActivation).
    Also monkey-patches F.gelu globally.

    approximation='sigmoid' (default): fp16-safe, max error ~0.01.
    approximation='tanh': near-exact shape, required for regression heads.
    Returns number of modules replaced.
    """
    replacement_cls = TanhGELU if approximation == 'tanh' else SigmoidGELU
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.GELU) or type(child).__name__ == 'GELUActivation':
                setattr(module, child_name, replacement_cls())
                count += 1

    if approximation == 'tanh':
        c = TanhGELU.C
        F.gelu = lambda x, approximate='none': 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x * x * x)))
    else:
        F.gelu = lambda x, approximate='none': x * torch.sigmoid(1.702 * x)

    if count:
        log.info(f"Patched {count} GELU → {replacement_cls.__name__}")
    return count


def restore_gelu():
    """Restore original F.gelu."""
    F.gelu = _original_gelu


# ─── DeformableConv2d → Conv2d ────────────────────────────────────────────────

def patch_deformable_conv(model: nn.Module) -> int:
    """Replace DeformableConv2d with its internal regular_conv.
    Expects the module to have a `regular_conv` child (common pattern).
    Returns number of modules replaced.
    """
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if type(child).__name__ == 'DeformableConv2d':
                if not hasattr(child, 'regular_conv'):
                    log.warning(f"DeformableConv2d at {name}.{child_name} has no regular_conv — skipped")
                    continue
                conv = child.regular_conv
                ks = conv.weight.shape[2]
                stride = child.stride[0] if hasattr(child, 'stride') else 1
                new_conv = nn.Conv2d(
                    conv.in_channels, conv.out_channels,
                    kernel_size=ks, stride=stride, padding=ks // 2,
                    bias=conv.bias is not None,
                )
                new_conv.weight.data.copy_(conv.weight.data)
                if conv.bias is not None:
                    new_conv.bias.data.copy_(conv.bias.data)
                setattr(module, child_name, new_conv)
                count += 1

    if count:
        log.info(f"Patched {count} DeformableConv2d → Conv2d")
    return count


# ─── F.interpolate align_corners fix ─────────────────────────────────────────

_original_interpolate = F.interpolate


def patch_interpolate() -> None:
    """Force align_corners=False for bilinear/bicubic interpolation.
    GPU rejects half_pixel_centers=True + align_corners=True.
    """
    def safe_interpolate(input, size=None, scale_factor=None, mode='nearest',
                         align_corners=None, recompute_scale_factor=None, antialias=False):
        if mode in ('bilinear', 'bicubic', 'trilinear'):
            align_corners = False
        return _original_interpolate(
            input, size=size, scale_factor=scale_factor, mode=mode,
            align_corners=align_corners, recompute_scale_factor=recompute_scale_factor,
            antialias=antialias,
        )

    F.interpolate = safe_interpolate
    log.info("Patched F.interpolate → align_corners=False for bilinear/bicubic")


def restore_interpolate():
    """Restore original F.interpolate."""
    F.interpolate = _original_interpolate


# ─── WindowAttention relative position bias pre-compute ──────────────────────

def patch_window_attention(model: nn.Module) -> int:
    """Pre-compute relative_position_bias to eliminate GATHER_ND.
    Detects modules with relative_position_bias_table + relative_position_index.
    Returns number of modules patched.
    """
    count = 0
    wa_class = None

    for name, module in model.named_modules():
        if (hasattr(module, 'relative_position_bias_table') and
            hasattr(module, 'relative_position_index') and
            hasattr(module, 'num_heads') and
            hasattr(module, 'window_size')):

            table = module.relative_position_bias_table
            index = module.relative_position_index.view(-1)
            ws = module.window_size[0] if isinstance(module.window_size, (list, tuple)) else module.window_size
            nH = module.num_heads

            bias = table[index].view(ws * ws, ws * ws, nH).permute(2, 0, 1).contiguous()
            module.register_buffer('_precomputed_pos_bias', bias.unsqueeze(0))

            if wa_class is None:
                wa_class = type(module)
            count += 1

    if wa_class is not None and count > 0:
        _orig_forward = wa_class.forward

        def patched_wa_forward(self, x, mask=None):
            B_, N, C = x.shape
            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn + self._precomputed_pos_bias
            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        wa_class.forward = patched_wa_forward
        log.info(f"Patched {count} WindowAttention modules (pre-computed position bias)")

    return count


# ─── PatchMerging → pixel_unshuffle ──────────────────────────────────────────

def patch_patch_merging(model: nn.Module) -> int:
    """Replace PatchMerging stride-2 slicing with pixel_unshuffle + weight permutation.
    Eliminates GATHER_ND from spatial downsampling.
    Returns number of modules patched.
    """
    count = 0
    pm_class = None

    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if type(child).__name__ == 'PatchMerging' and hasattr(child, 'reduction'):
                C = child.reduction.in_features // 4

                # Build permutation: pixel_unshuffle order → PatchMerging cat order
                perm = torch.zeros(4 * C, dtype=torch.long)
                for c in range(C):
                    perm[c * 4 + 0] = c            # even_h, even_w → x0
                    perm[c * 4 + 1] = 2 * C + c    # even_h, odd_w → x2
                    perm[c * 4 + 2] = C + c         # odd_h, even_w → x1
                    perm[c * 4 + 3] = 3 * C + c     # odd_h, odd_w → x3

                child.norm.weight.data = child.norm.weight.data[perm].clone()
                child.norm.bias.data = child.norm.bias.data[perm].clone()
                child.reduction.weight.data = child.reduction.weight.data[:, perm].clone()

                if pm_class is None:
                    pm_class = type(child)
                count += 1

    if pm_class is not None and count > 0:
        def patched_pm_forward(self, x, H, W):
            B, L, C = x.shape
            x = x.view(B, H, W, C)
            if (H % 2 == 1) or (W % 2 == 1):
                x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
                _, H, W, _ = x.shape
            x_nchw = x.permute(0, 3, 1, 2)
            x_ps = F.pixel_unshuffle(x_nchw, 2)
            x = x_ps.permute(0, 2, 3, 1).reshape(B, -1, 4 * C)
            return self.reduction(self.norm(x))

        pm_class.forward = patched_pm_forward
        log.info(f"Patched {count} PatchMerging → pixel_unshuffle + weight permutation")

    return count


# ─── GroupNorm → 4D manual ops ──────────────────────────────────────────────

class ManualGroupNorm(nn.Module):
    """GPU-friendly GroupNorm using only 4D tensors.
    nn.GroupNorm is not supported by TFLite GPU delegate.
    Reshape to (B*G, C//G, H, W) to stay 4D throughout.
    """
    def __init__(self, gn: nn.GroupNorm):
        super().__init__()
        self.num_groups = gn.num_groups
        self.num_channels = gn.num_channels
        self.eps = gn.eps
        self.weight = nn.Parameter(gn.weight.clone()) if gn.weight is not None else None
        self.bias = nn.Parameter(gn.bias.clone()) if gn.bias is not None else None

    def forward(self, x):
        B, C, H, W = x.shape
        G = self.num_groups
        x = x.reshape(B * G, C // G, H, W)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = ((x - mean) * (x - mean)).mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.reshape(B, C, H, W)
        if self.weight is not None:
            x = x * self.weight.reshape(1, C, 1, 1)
        if self.bias is not None:
            x = x + self.bias.reshape(1, C, 1, 1)
        return x


def patch_groupnorm(model: nn.Module) -> int:
    """Replace all nn.GroupNorm with GPU-friendly ManualGroupNorm.
    Returns number of modules replaced.
    """
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.GroupNorm):
                setattr(module, child_name, ManualGroupNorm(child))
                count += 1
    if count:
        log.info(f"Patched {count} nn.GroupNorm → ManualGroupNorm (4D ops)")
    return count


# ─── Conv2d_WS → baked Conv2d ──────────────────────────────────────────────

def patch_weight_standardization(model: nn.Module) -> int:
    """Replace Conv2d_WS (weight standardization) with regular Conv2d.
    Pre-computes standardized weights at conversion time so the runtime
    forward pass is a standard conv2d (no dynamic weight normalization).
    Returns number of modules replaced.
    """
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if type(child).__name__ == 'Conv2d_WS':
                with torch.no_grad():
                    weight = child.weight.clone()
                    weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
                    weight = weight - weight_mean
                    std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
                    weight = weight / std.expand_as(weight)

                conv = nn.Conv2d(
                    child.in_channels, child.out_channels, child.kernel_size,
                    stride=child.stride, padding=child.padding,
                    dilation=child.dilation, groups=child.groups,
                    bias=child.bias is not None,
                )
                conv.weight = nn.Parameter(weight)
                if child.bias is not None:
                    conv.bias = nn.Parameter(child.bias.clone())
                setattr(module, child_name, conv)
                count += 1
    if count:
        log.info(f"Patched {count} Conv2d_WS → Conv2d (baked weights)")
    return count


# ─── F.normalize → manual sqrt+div ─────────────────────────────────────────

_original_normalize = F.normalize


def patch_normalize() -> None:
    """Replace F.normalize with manual sqrt+div.
    TFLite GPU fails on the div broadcast in F.normalize.
    """
    def safe_normalize(input, p=2.0, dim=1, eps=1e-12, out=None):
        norm = torch.sqrt(torch.sum(input * input, dim=dim, keepdim=True).clamp(min=eps * eps))
        return input / norm

    F.normalize = safe_normalize
    log.info("Patched F.normalize → manual sqrt+div")


def restore_normalize():
    """Restore original F.normalize."""
    F.normalize = _original_normalize


# ─── Swish/SiLU → x * sigmoid(x) ──────────────────────────────────────────

class SigmoidSwish(nn.Module):
    """Swish/SiLU replacement: x * sigmoid(x)."""
    def forward(self, x):
        return x * torch.sigmoid(x)


def patch_swish(model: nn.Module) -> int:
    """Replace Swish/SiLU activations with x * sigmoid(x).
    Returns number of modules replaced.
    """
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.SiLU):
                setattr(module, child_name, SigmoidSwish())
                count += 1
            elif type(child).__name__ in ('Swish', 'SwishMe'):
                setattr(module, child_name, SigmoidSwish())
                count += 1
    if count:
        log.info(f"Patched {count} Swish/SiLU → SigmoidSwish")
    return count


# ─── einops.rearrange replacement ─────────────────────────────────────────────

def patch_einops() -> None:
    """Replace einops.rearrange with manual reshape/permute implementations.
    Covers the 3 patterns used in BiRefNet/ISNet decoders.
    """
    try:
        import einops
    except ImportError:
        return

    def manual_rearrange(x, pattern, **kwargs):
        hg = kwargs.get('hg', 2)
        wg = kwargs.get('wg', 2)
        if pattern == 'b c (hg h) (wg w) -> (b hg wg) c h w':
            B, C, H, W = x.shape
            h, w = H // hg, W // wg
            return x.view(B, C, hg, h, wg, w).permute(0, 2, 4, 1, 3, 5).contiguous().view(B * hg * wg, C, h, w)
        elif pattern == 'b c (hg h) (wg w) -> b (c hg wg) h w':
            B, C, H, W = x.shape
            h, w = H // hg, W // wg
            return x.view(B, C, hg, h, wg, w).permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C * hg * wg, h, w)
        elif pattern == '(b hg wg) c h w -> b c (hg h) (wg w)':
            BHW, C, h, w = x.shape
            B = BHW // (hg * wg)
            return x.view(B, hg, wg, C, h, w).permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, hg * h, wg * w)
        else:
            raise ValueError(f"Unsupported einops pattern: {pattern}")

    einops.rearrange = manual_rearrange
    log.info("Patched einops.rearrange → manual reshape/permute")


# ─── Master apply ────────────────────────────────────────────────────────────

# ─── grid_sample → GATHER/CAST-free tent-matmul (C10) ─────────────────────────

_original_grid_sample = F.grid_sample


def patch_grid_sample() -> None:
    """Replace F.grid_sample with a GATHER/CAST-free bilinear tent-matmul.

    The native lowering of grid_sample is GATHER_ND (GPU-banned). For a fixed-size
    value map the bilinear sample is a linear op: build tent weights
    ``wx = relu(1 - |ix - arange(W)|)`` / ``wy`` and BMM ``value_flat @ W_flat.T``.
    Numerically exact vs F.grid_sample incl. zeros-padding OOB (err ~2e-7), all <=4D.
    Used by RF-DETR Nano's deformable cross-attention (shipped, device-verified).
    Replaces the model's own bilinear sampler too if it calls F.grid_sample.
    """
    def tent_grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=None):
        N, C, H, W = input.shape
        Hg, Wg = grid.shape[1], grid.shape[2]
        ac = bool(align_corners)
        if ac:
            ix = (grid[..., 0] + 1) * (W - 1) / 2; iy = (grid[..., 1] + 1) * (H - 1) / 2
        else:
            ix = (grid[..., 0] + 1) * W / 2 - 0.5; iy = (grid[..., 1] + 1) * H / 2 - 0.5
        ix = ix.reshape(N, Hg * Wg, 1); iy = iy.reshape(N, Hg * Wg, 1)
        xs = torch.arange(W, dtype=input.dtype).reshape(1, 1, W)
        ys = torch.arange(H, dtype=input.dtype).reshape(1, 1, H)
        wx = torch.relu(1 - (ix - xs).abs()); wy = torch.relu(1 - (iy - ys).abs())
        Wm = (wy.unsqueeze(-1) * wx.unsqueeze(-2)).reshape(N, Hg * Wg, H * W)
        return torch.matmul(input.reshape(N, C, H * W), Wm.transpose(1, 2)).reshape(N, C, Hg, Wg)

    F.grid_sample = tent_grid_sample
    log.info("Patched F.grid_sample → tent-matmul (GATHER/CAST-free)")


def restore_grid_sample():
    """Restore original F.grid_sample."""
    F.grid_sample = _original_grid_sample


# ─── SafeLayerNorm — fp16-overflow-safe channel reduction ─────────────────────

def patch_safe_layernorm(scale: str = "adaptive_v2", fixed_s: float = 128.0) -> None:
    """Override nn.LayerNorm.forward with a down-scaled, numerically-exact reduction.

    The Mali ML Drift delegate computes reductions in fp16 even for an fp32 graph, so
    a LayerNorm over a large-magnitude activation overflows the channel sum-of-squares
    (>65504) -> wrong norm at full LITERT_CL residency (residency != correctness).
    Witnessed across deep ViT (PE-Core/SigLIP2/EoMT), deep-residual CNN (NAFNet), and
    transformer detectors (RF-DETR). Fix: reduce in a down-scaled domain ``x/S`` --
    exact because LayerNorm is scale-invariant.

    scale="adaptive_v2" (default): per-row ``S = max(1, amax/8)``, normalize IN the
      scaled domain (``y = dd * rsqrt(mean(dd^2) + eps)``). The two rescale MULs of
      "adaptive" disappear from the graph, so no intermediate ever leaves the safe
      range -- the fp16-safest form (Parakeet; supersedes the others). Caveat: eps
      effectively acts as ``eps * S^2`` in original-domain terms (negligible when
      var >> eps).
    scale="adaptive": same S, but rescale back (``var*(S*S)``, ``d*S``) before the
      rsqrt so eps acts at its true magnitude -- bit-faithful to stock LayerNorm.
    scale="fixed": constant ``S = fixed_s`` (cheapest; fine when all norms are large,
      e.g. NAFNet S=128).

    Both adaptive modes are REQUIRED over "fixed" when one graph mixes norms at very
    different magnitudes (e.g. RF-DETR decoder |x|~8 .. ~1068); a fixed large S
    squashes the small norms. Assumes normalized_shape is the last dim (the common
    case). Opt-in (NOT in apply_all_patches) -- only models that hit the device fp16
    wall need it.
    """
    adaptive = scale in ("adaptive", "adaptive_v2")
    stay_scaled = scale == "adaptive_v2"
    s_const = float(fixed_s)

    def safe_ln_forward(self, x):
        if adaptive:
            S = (x.abs().amax(-1, keepdim=True) * (1.0 / 8.0)).clamp(min=1.0)
        else:
            S = s_const
        xs = x / S
        mu = xs.mean(-1, keepdim=True)
        d = xs - mu
        if stay_scaled:
            var = (d * d).mean(-1, keepdim=True)
        else:
            var = (d * d).mean(-1, keepdim=True) * (S * S)
            d = d * S
        y = d * torch.rsqrt(var + self.eps)
        if self.elementwise_affine:
            y = y * self.weight + self.bias
        return y

    nn.LayerNorm.forward = safe_ln_forward
    log.info(f"Patched nn.LayerNorm → SafeLayerNorm (scale={scale})")


# ─── ConvTranspose → zero-stuff conv (TRANSPOSE_CONV rejected on device) ──────

class ZeroStuffConvT1d(nn.Module):
    """Exact GPU-clean ConvTranspose1d: nearest-upsample × zero-stuff mask + flipped
    conv1d + crop.

    The on-device ML Drift delegate rejects TRANSPOSE_CONV; this lowers to
    RESIZE_NEAREST + MUL + CONV only. Numerically equivalent to nn.ConvTranspose1d
    (grouped convs supported). The input length must be fixed at build time — use
    patch_conv_transpose() to capture it with a dry run. The 1D nearest upsample is
    expressed as a 2D interpolate (unsqueeze/squeeze) to avoid a dynamic 1D resize.
    Device-proven in DAC, Matcha-TTS and Mimi decoders.
    """

    def __init__(self, ct: nn.ConvTranspose1d, input_length: int):
        super().__init__()
        self.stride = ct.stride[0]
        self.kernel_size = ct.kernel_size[0]
        self.pad = ct.padding[0]
        self.output_padding = ct.output_padding[0]
        self.input_length = input_length
        self.groups = ct.groups
        cin, cout, groups = ct.in_channels, ct.out_channels, ct.groups
        # ConvT weight (Cin, Cout//G, K) → grouped conv1d weight (Cout, Cin//G, K), flipped.
        w = ct.weight.detach()
        w = w.view(groups, cin // groups, cout // groups, self.kernel_size)
        w = w.permute(0, 2, 1, 3).reshape(cout, cin // groups, self.kernel_size)
        self.register_buffer('w', w.flip(2).contiguous())
        self.register_buffer('b', ct.bias.detach().clone() if ct.bias is not None
                             else torch.zeros(cout))
        mask = torch.zeros(input_length * self.stride)
        mask[::self.stride] = 1.0
        self.register_buffer('mask', mask[None, None])

    def forward(self, x):
        s, k, p = self.stride, self.kernel_size, self.pad
        up = F.interpolate(x.unsqueeze(2), size=(1, self.input_length * s), mode='nearest').squeeze(2)
        y = F.conv1d(up * self.mask, self.w, bias=self.b, padding=k - 1, groups=self.groups)
        out_len = (self.input_length - 1) * s + k - 2 * p + self.output_padding
        return y[:, :, p:p + out_len]


class ZeroStuffConvT2d(nn.Module):
    """Exact GPU-clean ConvTranspose2d (square k/s/p): nearest-upsample × zero-stuff
    mask + flipped conv2d + crop.

    Same rationale as ZeroStuffConvT1d: TRANSPOSE_CONV is rejected on device; this is
    RESIZE_NEAREST + MUL + CONV_2D. Numerically equivalent to nn.ConvTranspose2d
    (grouped convs supported). Fixed input H/W required — use patch_conv_transpose().
    Device-proven in EDSR, PP-OCR, DewarpNet and TwinLiteNet decoders.
    """

    def __init__(self, ct: nn.ConvTranspose2d, in_h: int, in_w: int):
        super().__init__()
        self.stride = ct.stride[0]
        self.kernel_size = ct.kernel_size[0]
        self.pad = ct.padding[0]
        self.output_padding = ct.output_padding[0]
        self.in_h, self.in_w = in_h, in_w
        self.groups = ct.groups
        cin, cout, groups = ct.in_channels, ct.out_channels, ct.groups
        k = self.kernel_size
        # ConvT weight (Cin, Cout//G, k, k) → grouped conv2d weight (Cout, Cin//G, k, k), flipped.
        w = ct.weight.detach()
        w = w.view(groups, cin // groups, cout // groups, k, k)
        w = w.permute(0, 2, 1, 3, 4).reshape(cout, cin // groups, k, k)
        self.register_buffer('w', w.flip(2).flip(3).contiguous())
        self.register_buffer('b', ct.bias.detach().clone() if ct.bias is not None
                             else torch.zeros(cout))
        mask = torch.zeros(in_h * self.stride, in_w * self.stride)
        mask[::self.stride, ::self.stride] = 1.0
        self.register_buffer('mask', mask[None, None])

    def forward(self, x):
        s, k, p = self.stride, self.kernel_size, self.pad
        up = F.interpolate(x, size=(self.in_h * s, self.in_w * s), mode='nearest')
        y = F.conv2d(up * self.mask, self.w, bias=self.b, padding=k - 1, groups=self.groups)
        out_h = (self.in_h - 1) * s + k - 2 * p + self.output_padding
        out_w = (self.in_w - 1) * s + k - 2 * p + self.output_padding
        return y[:, :, p:p + out_h, p:p + out_w]


def pixelshuffle_to_conv_transpose(upscale_factor: int, out_channels: int) -> nn.ConvTranspose2d:
    """Rewrite nn.PixelShuffle(r) as a fixed one-hot ConvTranspose2d(k=r, stride=r).

    litert-torch lowers PixelShuffle through a 6D reshape (GPU-banned: >4D). A
    stride-r ConvTranspose2d whose kernel one-hot-selects each (channel, sub-pixel)
    position is exact; patch_conv_transpose() then lowers it to the GPU-clean
    zero-stuff form. Device-proven in EDSR x4.

    Swap manually: ``head.upsample = pixelshuffle_to_conv_transpose(r, c_out)`` where
    the PixelShuffle input has ``c_out * r * r`` channels.
    """
    r, c = upscale_factor, out_channels
    ct = nn.ConvTranspose2d(c * r * r, c, kernel_size=r, stride=r, bias=False)
    w = torch.zeros(c * r * r, c, r, r)
    for ch in range(c):
        for py in range(r):
            for px in range(r):
                w[ch * r * r + py * r + px, ch, py, px] = 1.0
    ct.weight.data = w
    return ct


def patch_conv_transpose(model: nn.Module, dummy_input) -> int:
    """Replace every reached ConvTranspose1d/2d with its exact zero-stuff equivalent.

    Runs a dry forward pass with dummy_input to capture each layer's fixed input size
    (forward pre-hooks), then swaps modules in place. Modules NOT reached by the dry
    run (e.g. training-only branches) are left untouched. Skips dilated and
    asymmetric-k/s/p 2D layers with a warning. Returns number of modules replaced.
    """
    captured = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d)):
            def make_hook(layer_name):
                def hook(mod, inputs):
                    captured[layer_name] = inputs[0].shape
                return hook
            hooks.append(module.register_forward_pre_hook(make_hook(name)))
    with torch.no_grad():
        if isinstance(dummy_input, (tuple, list)):
            model(*dummy_input)
        else:
            model(dummy_input)
    for hook in hooks:
        hook.remove()

    count = 0
    for name, module in list(model.named_modules()):
        if name not in captured:
            continue
        if isinstance(module, nn.ConvTranspose2d):
            symmetric = (module.kernel_size[0] == module.kernel_size[1]
                         and module.stride[0] == module.stride[1]
                         and module.padding[0] == module.padding[1])
            if not symmetric or module.dilation != (1, 1):
                log.warning(f"ConvTranspose2d at {name}: asymmetric k/s/p or dilation — skipped")
                continue
            replacement = ZeroStuffConvT2d(module, captured[name][-2], captured[name][-1])
        elif isinstance(module, nn.ConvTranspose1d):
            if module.dilation[0] != 1:
                log.warning(f"ConvTranspose1d at {name}: dilation — skipped")
                continue
            replacement = ZeroStuffConvT1d(module, captured[name][-1])
        else:
            continue
        parent = model
        *path, last = name.split('.')
        for part in path:
            parent = getattr(parent, part)
        setattr(parent, last, replacement)
        count += 1
    if count:
        log.info(f"Patched {count} ConvTranspose → ZeroStuffConvT")
    return count


# ─── MaxPool2d(padding>0) → zero-pad + unpadded maxpool ───────────────────────

class ZeroPadMaxPool(nn.Module):
    """MaxPool2d with explicit zero padding instead of the -inf PADV2 lowering.

    nn.MaxPool2d(padding=p) lowers to PADV2(-inf) + MAX_POOL_2D and the on-device
    delegate rejects the -inf pad. Zero padding is exact when the pooled input is
    non-negative (e.g. the post-ReLU ResNet stem) — the max is unaffected.
    Device-proven in Places365, PlantNet, BiSeNet and SINet-V2 (ResNet stems).
    """

    def __init__(self, kernel_size: int = 3, stride: int = 2, padding: int = 1,
                 ceil_mode: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = padding
        self.ceil_mode = ceil_mode

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), value=0.0)
        return F.max_pool2d(x, self.kernel_size, stride=self.stride, padding=0,
                            ceil_mode=self.ceil_mode)


def patch_maxpool_zeropad(model: nn.Module) -> int:
    """Replace every padded nn.MaxPool2d with ZeroPadMaxPool.

    Opt-in: only exact for non-negative inputs (post-ReLU/post-sigmoid) — verify
    output parity after applying. Skips dilated and asymmetric pools with a warning.
    Returns number of modules replaced.
    """
    def scalar(v):
        return v[0] if isinstance(v, (tuple, list)) else v

    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.MaxPool2d):
                continue
            k, s, p = child.kernel_size, child.stride, child.padding
            if isinstance(k, (tuple, list)) and k[0] != k[1] or scalar(child.dilation) != 1:
                log.warning(f"MaxPool2d at {name}.{child_name}: asymmetric or dilated — skipped")
                continue
            if scalar(p) == 0:
                continue
            setattr(module, child_name,
                    ZeroPadMaxPool(scalar(k), scalar(s), scalar(p), child.ceil_mode))
            count += 1
    if count:
        log.info(f"Patched {count} nn.MaxPool2d → ZeroPadMaxPool")
    return count


# ─── InstanceNorm → hierarchical-mean form (fp16 spatial-sum overflow) ────────

def hierarchical_mean(x):
    """Global spatial mean via a cascade of /2 average-pools, staying magnitude-safe.

    A single global MEAN/SUM over a large map overflows the delegate's fp16
    accumulator (it computes fp16 regardless of tensor dtype); each cascade stage
    averages at most 4 values. The while-loop unrolls at trace time for a fixed input
    size, so it converts to a static chain of AVERAGE_POOL_2D.

    ONLY exact for power-of-two spatial dims: with odd extents the ceil_mode edge
    windows average fewer elements and the bias is significant (max err ~0.2 measured
    on 37x53 normalized outputs). Use with pow2-friendly maps, or pad first.
    """
    while x.shape[-1] > 1 or x.shape[-2] > 1:
        kh = 2 if x.shape[-2] > 1 else 1
        kw = 2 if x.shape[-1] > 1 else 1
        x = F.avg_pool2d(x, (kh, kw), ceil_mode=True)
    return x


class SafeInstanceNorm2d(nn.Module):
    """fp16-safe InstanceNorm2d: mean and variance via hierarchical_mean().

    nn.InstanceNorm2d's full-map reductions overflow the on-device fp16 accumulator
    at large spatial sizes. Device-proven in MODNet (512×512 mattes). Inherits
    hierarchical_mean()'s power-of-two requirement — NOT exact for odd spatial dims.
    """

    def __init__(self, inorm: nn.InstanceNorm2d):
        super().__init__()
        self.eps = inorm.eps
        if inorm.affine:
            self.register_buffer('weight', inorm.weight.detach().clone())
            self.register_buffer('bias', inorm.bias.detach().clone())
        else:
            self.weight = None
            self.bias = None

    def forward(self, x):
        mean = hierarchical_mean(x)
        d = x - mean
        y = d * torch.rsqrt(hierarchical_mean(d * d) + self.eps)
        if self.weight is not None:
            c = y.shape[1]
            y = y * self.weight.reshape(1, c, 1, 1) + self.bias.reshape(1, c, 1, 1)
        return y


def patch_instance_norm(model: nn.Module) -> int:
    """Replace nn.InstanceNorm2d with fp16-safe SafeInstanceNorm2d.

    Opt-in (fp16-wall fix). Skips track_running_stats=True instances (those normalize
    with running statistics, not per-sample ones) with a warning.
    Returns number of modules replaced.
    """
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.InstanceNorm2d):
                if child.track_running_stats:
                    log.warning(f"InstanceNorm2d at {name}.{child_name}: track_running_stats — skipped")
                    continue
                setattr(module, child_name, SafeInstanceNorm2d(child))
                count += 1
    if count:
        log.info(f"Patched {count} nn.InstanceNorm2d → SafeInstanceNorm2d")
    return count


# ─── RMSNorm → fp16-safe max-normalized form ──────────────────────────────────

def safe_rms(x, weight, eps: float = 1e-6):
    """fp16-safe RMSNorm(x) * weight via per-row max-normalization.

    The delegate computes fp16 regardless of dtype; in deep residual stacks the
    stream grows large enough that mean(x^2) overflows (-> inf -> rsqrt = 0 -> the
    whole output collapses to 0). Dividing each row by its own max first bounds x^2
    in [0, 1] so the sum-of-squares never overflows. Mathematically identical to
    standard RMSNorm: with m = max|x|, y = (x/m) * rsqrt(mean((x/m)^2) + eps/m^2) * w.
    Device-proven in the 28-layer Qwen3 embedding/reranking stacks.
    """
    m = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    xs = x / m
    var = (xs * xs).mean(dim=-1, keepdim=True)
    return xs * torch.rsqrt(var + eps / (m * m)) * weight


def patch_rmsnorm(model: nn.Module) -> int:
    """Rebind the forward of every *RMSNorm module to the fp16-safe safe_rms().

    Matches any module whose class name contains "RMSNorm" and that carries a
    ``weight`` plus an eps attribute (``variance_epsilon`` or ``eps`` — covers
    transformers' per-model RMSNorm classes and torch.nn.RMSNorm). Opt-in
    (fp16-wall fix). Returns number of modules patched.
    """
    count = 0
    for name, module in model.named_modules():
        if 'RMSNorm' in type(module).__name__ and hasattr(module, 'weight'):
            eps = getattr(module, 'variance_epsilon', None)
            if eps is None:
                eps = getattr(module, 'eps', None)
            if eps is None:
                eps = 1e-6
            module.forward = (lambda w, e: lambda x: safe_rms(x, w, e))(module.weight, eps)
            count += 1
    if count:
        log.info(f"Patched {count} RMSNorm → safe_rms")
    return count


def apply_all_patches(model: nn.Module, dummy_input=None) -> dict:
    """Apply all always-safe patches. Returns summary of changes made.

    When dummy_input is given, ConvTranspose1d/2d layers are also lowered to their
    exact zero-stuff form (the input sizes are captured with a dry forward pass).

    Opt-in patches NOT applied here (apply directly when the model needs them):
    patch_grid_sample() (deformable attention), patch_safe_layernorm() /
    patch_rmsnorm() / patch_instance_norm() (device fp16-overflow walls),
    patch_maxpool_zeropad() (exact only for non-negative inputs),
    patch_gelu(model, approximation='tanh') (regression heads),
    pixelshuffle_to_conv_transpose() (manual swap).
    """
    summary = {}
    summary['gelu'] = patch_gelu(model)
    summary['swish'] = patch_swish(model)
    summary['groupnorm'] = patch_groupnorm(model)
    summary['weight_standardization'] = patch_weight_standardization(model)
    summary['deformable_conv'] = patch_deformable_conv(model)
    summary['window_attention'] = patch_window_attention(model)
    summary['patch_merging'] = patch_patch_merging(model)
    if dummy_input is not None:
        summary['conv_transpose'] = patch_conv_transpose(model, dummy_input)
    patch_interpolate()
    patch_normalize()
    patch_einops()
    summary['interpolate'] = True
    summary['normalize'] = True
    summary['einops'] = True
    return summary
