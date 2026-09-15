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
"""Unit tests for litert_gpu_toolkit AST and layer patches."""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from litert_gpu_toolkit.patches import (
    ManualGroupNorm,
    SafeInstanceNorm2d,
    SigmoidGELU,
    SigmoidSwish,
    TanhGELU,
    ZeroPadMaxPool,
    ZeroStuffConvT1d,
    ZeroStuffConvT2d,
    apply_all_patches,
    hierarchical_mean,
    patch_conv_transpose,
    patch_deformable_conv,
    patch_einops,
    patch_gelu,
    patch_grid_sample,
    patch_groupnorm,
    patch_instance_norm,
    patch_interpolate,
    patch_maxpool_zeropad,
    patch_normalize,
    patch_patch_merging,
    patch_rmsnorm,
    patch_safe_layernorm,
    patch_swish,
    patch_weight_standardization,
    patch_window_attention,
    pixelshuffle_to_conv_transpose,
    restore_gelu,
    restore_grid_sample,
    restore_interpolate,
    restore_normalize,
    safe_rms,
)


class TestGELU:
    def test_sigmoid_gelu_forward(self):
        mod = SigmoidGELU()
        x = torch.randn(2, 16, 8, 8)
        out = mod(x)
        assert out.shape == x.shape
        # Sigmoid GELU is close to exact GELU within theoretical bound ~0.03
        ref = F.gelu(x)
        assert torch.max(torch.abs(out - ref)) < 0.03

    def test_tanh_gelu_forward(self):
        mod = TanhGELU()
        x = torch.randn(2, 16, 8, 8)
        out = mod(x)
        assert out.shape == x.shape
        ref = F.gelu(x, approximate="tanh")
        assert torch.allclose(out, ref, atol=1e-5)

    def test_patch_gelu_modules(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.g1 = nn.GELU()
                self.sub = nn.Sequential(nn.GELU())

        m = Model()
        count = patch_gelu(m, approximation="sigmoid")
        assert count == 2
        assert isinstance(m.g1, SigmoidGELU)
        assert isinstance(m.sub[0], SigmoidGELU)

        # Test global F.gelu monkeypatch
        x = torch.randn(4, 4)
        out_f = F.gelu(x)
        assert torch.allclose(out_f, SigmoidGELU()(x), atol=1e-6)

        # Test restore
        restore_gelu()
        out_restored = F.gelu(x)
        assert torch.allclose(out_restored, torch.nn.functional.gelu(x), atol=1e-6)


class TestSwish:
    def test_sigmoid_swish(self):
        mod = SigmoidSwish()
        x = torch.randn(2, 8)
        out = mod(x)
        ref = F.silu(x)
        assert torch.allclose(out, ref, atol=1e-6)

    def test_patch_swish(self):
        class Swish(nn.Module):
            def forward(self, x):
                return x * torch.sigmoid(x)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.s1 = nn.SiLU()
                self.s2 = Swish()

        m = Model()
        count = patch_swish(m)
        assert count == 2
        assert isinstance(m.s1, SigmoidSwish)
        assert isinstance(m.s2, SigmoidSwish)


class TestDeformableConv:
    def test_patch_deformable_conv(self):
        class DeformableConv2d(nn.Module):
            def __init__(self):
                super().__init__()
                self.regular_conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.stride = (1, 1)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.dfc = DeformableConv2d()

        m = Model()
        count = patch_deformable_conv(m)
        assert count == 1
        assert isinstance(m.dfc, nn.Conv2d)
        assert m.dfc.in_channels == 3
        assert m.dfc.out_channels == 16


class TestInterpolate:
    def test_patch_interpolate(self):
        patch_interpolate()
        x = torch.randn(1, 3, 16, 16)
        # Bilinear with align_corners=True should be forced to False
        out = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=True)
        ref = torch.nn.functional.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
        assert torch.allclose(out, ref, atol=1e-6)
        restore_interpolate()


class TestGroupNorm:
    def test_manual_group_norm_numerical_parity(self):
        gn = nn.GroupNorm(num_groups=4, num_channels=16, eps=1e-5)
        mgn = ManualGroupNorm(gn)
        x = torch.randn(2, 16, 8, 8)
        out_gn = gn(x)
        out_mgn = mgn(x)
        assert torch.allclose(out_gn, out_mgn, atol=1e-5)

    def test_patch_groupnorm(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.gn = nn.GroupNorm(2, 8)

        m = Model()
        count = patch_groupnorm(m)
        assert count == 1
        assert isinstance(m.gn, ManualGroupNorm)


class TestWeightStandardization:
    def test_patch_weight_standardization(self):
        class Conv2d_WS(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_channels = 3
                self.out_channels = 8
                self.kernel_size = (3, 3)
                self.stride = (1, 1)
                self.padding = (1, 1)
                self.dilation = (1, 1)
                self.groups = 1
                self.weight = nn.Parameter(torch.randn(8, 3, 3, 3))
                self.bias = nn.Parameter(torch.zeros(8))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = Conv2d_WS()

        m = Model()
        count = patch_weight_standardization(m)
        assert count == 1
        assert isinstance(m.conv, nn.Conv2d)
        # Check that the standardized weights have ~0 mean
        w = m.conv.weight
        w_mean = w.mean(dim=(1, 2, 3))
        assert torch.all(w_mean.abs() < 1e-4)


class TestNormalize:
    def test_patch_normalize(self):
        patch_normalize()
        x = torch.randn(2, 10)
        out = F.normalize(x, p=2.0, dim=1)
        ref = x / torch.norm(x, p=2.0, dim=1, keepdim=True)
        assert torch.allclose(out, ref, atol=1e-5)
        restore_normalize()


class TestGridSample:
    def test_tent_grid_sample_parity(self):
        patch_grid_sample()
        x = torch.randn(1, 4, 8, 8)
        grid = torch.rand(1, 4, 4, 2) * 2 - 1  # [-1, 1] range
        out_tent = F.grid_sample(x, grid, mode="bilinear", align_corners=False)
        restore_grid_sample()
        out_ref = F.grid_sample(x, grid, mode="bilinear", align_corners=False)
        assert torch.allclose(out_tent, out_ref, atol=1e-5)


class TestSafeLayerNorm:
    @pytest.mark.parametrize("scale_mode", ["adaptive_v2", "adaptive", "fixed"])
    def test_safe_layernorm_forward(self, scale_mode):
        ln = nn.LayerNorm(16)
        patch_safe_layernorm(scale=scale_mode)
        x = torch.randn(2, 8, 16)
        out = ln(x)
        assert out.shape == x.shape
        # For standard magnitudes, should match original closely
        mu = x.mean(-1, keepdim=True)
        var = ((x - mu) ** 2).mean(-1, keepdim=True)
        ref = (x - mu) / torch.sqrt(var + ln.eps) * ln.weight + ln.bias
        assert torch.allclose(out, ref, atol=1e-3)


class TestConvTranspose:
    def test_zero_stuff_convt_1d_parity(self):
        ct = nn.ConvTranspose1d(in_channels=4, out_channels=8, kernel_size=4, stride=2, padding=1)
        in_len = 16
        zct = ZeroStuffConvT1d(ct, input_length=in_len)
        x = torch.randn(2, 4, in_len)
        out_ct = ct(x)
        out_zct = zct(x)
        assert torch.allclose(out_ct, out_zct, atol=1e-5)

    def test_zero_stuff_convt_2d_parity(self):
        ct = nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=4, stride=2, padding=1)
        in_h, in_w = 8, 8
        zct = ZeroStuffConvT2d(ct, in_h=in_h, in_w=in_w)
        x = torch.randn(2, 4, in_h, in_w)
        out_ct = ct(x)
        out_zct = zct(x)
        assert torch.allclose(out_ct, out_zct, atol=1e-5)

    def test_pixelshuffle_to_conv_transpose(self):
        r, c_out = 2, 4
        c_in = c_out * r * r
        ps = nn.PixelShuffle(r)
        ct = pixelshuffle_to_conv_transpose(upscale_factor=r, out_channels=c_out)
        x = torch.randn(2, c_in, 8, 8)
        out_ps = ps(x)
        out_ct = ct(x)
        assert torch.allclose(out_ps, out_ct, atol=1e-6)

    def test_patch_conv_transpose(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.ct1 = nn.ConvTranspose1d(4, 8, kernel_size=3, stride=1, padding=1)
                self.ct2 = nn.ConvTranspose2d(4, 8, kernel_size=3, stride=1, padding=1)

            def forward(self, x1, x2):
                return self.ct1(x1), self.ct2(x2)

        m = Model()
        x1 = torch.randn(1, 4, 16)
        x2 = torch.randn(1, 4, 8, 8)
        count = patch_conv_transpose(m, dummy_input=(x1, x2))
        assert count == 2
        assert isinstance(m.ct1, ZeroStuffConvT1d)
        assert isinstance(m.ct2, ZeroStuffConvT2d)


class TestMaxPool:
    def test_zero_pad_maxpool_parity_non_negative(self):
        mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        zmp = ZeroPadMaxPool(kernel_size=3, stride=2, padding=1)
        x = torch.relu(torch.randn(2, 4, 16, 16))
        out_mp = mp(x)
        out_zmp = zmp(x)
        assert torch.allclose(out_mp, out_zmp, atol=1e-6)

    def test_patch_maxpool_zeropad(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        m = Model()
        count = patch_maxpool_zeropad(m)
        assert count == 1
        assert isinstance(m.mp, ZeroPadMaxPool)


class TestInstanceNormAndHierarchicalMean:
    def test_hierarchical_mean(self):
        # Exact on power of two spatial dims
        x = torch.randn(2, 4, 16, 16)
        h_mean = hierarchical_mean(x)
        ref_mean = x.mean(dim=(-2, -1), keepdim=True)
        assert torch.allclose(h_mean, ref_mean, atol=1e-5)

    def test_safe_instance_norm_parity(self):
        inorm = nn.InstanceNorm2d(4, affine=True)
        sinorm = SafeInstanceNorm2d(inorm)
        x = torch.randn(2, 4, 16, 16)
        out_in = inorm(x)
        out_sin = sinorm(x)
        assert torch.allclose(out_in, out_sin, atol=1e-4)

    def test_patch_instance_norm(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.inorm = nn.InstanceNorm2d(4)

        m = Model()
        count = patch_instance_norm(m)
        assert count == 1
        assert isinstance(m.inorm, SafeInstanceNorm2d)


class TestRMSNorm:
    def test_safe_rms_parity(self):
        dim = 16
        w = torch.randn(dim)
        x = torch.randn(2, 8, dim)
        eps = 1e-6
        # Reference RMSNorm
        var = (x * x).mean(dim=-1, keepdim=True)
        ref = x * torch.rsqrt(var + eps) * w
        out = safe_rms(x, w, eps=eps)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_patch_rmsnorm(self):
        class CustomRMSNorm(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(dim))
                self.eps = 1e-6

            def forward(self, x):
                return x

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = CustomRMSNorm(16)

        m = Model()
        count = patch_rmsnorm(m)
        assert count == 1
        x = torch.randn(2, 4, 16)
        out = m.norm(x)
        assert out.shape == x.shape


class TestEinops:
    def test_patch_einops_patterns(self):
        try:
            import einops
        except ImportError:
            pytest.skip("einops not installed")

        orig_rearrange = einops.rearrange
        try:
            patch_einops()
            x = torch.randn(2, 4, 16, 16)
            # Pattern 1
            out1 = einops.rearrange(x, 'b c (hg h) (wg w) -> (b hg wg) c h w', hg=2, wg=2)
            assert out1.shape == (8, 4, 8, 8)

            # Pattern 2
            out2 = einops.rearrange(x, 'b c (hg h) (wg w) -> b (c hg wg) h w', hg=2, wg=2)
            assert out2.shape == (2, 16, 8, 8)

            # Pattern 3
            out3 = einops.rearrange(out1, '(b hg wg) c h w -> b c (hg h) (wg w)', hg=2, wg=2)
            assert out3.shape == (2, 4, 16, 16)

            # Unsupported pattern
            with pytest.raises(ValueError, match="Unsupported einops pattern"):
                einops.rearrange(x, 'b c h w -> (b c) h w')
        finally:
            einops.rearrange = orig_rearrange


class TestApplyAllPatches:
    def test_apply_all_patches_composite_model(self):
        class CompositeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.gelu = nn.GELU()
                self.silu = nn.SiLU()
                self.gn = nn.GroupNorm(2, 4)

            def forward(self, x):
                return self.gn(self.silu(self.gelu(x)))

        m = CompositeModel()
        summary = apply_all_patches(m)
        assert summary['gelu'] == 1
        assert summary['swish'] == 1
        assert summary['groupnorm'] == 1
        assert summary['interpolate'] is True
        assert summary['normalize'] is True
        assert summary['einops'] is True
