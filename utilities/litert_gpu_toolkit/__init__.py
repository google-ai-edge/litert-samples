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
litert_gpu_toolkit — Automatic GPU-compatible TFLite conversion for PyTorch models.

Wraps litert-torch with pre-conversion patches that eliminate common
GPU-incompatible patterns (GELU, Swish, GroupNorm, Conv2d_WS, F.normalize,
ConvTranspose/TRANSPOSE_CONV, GATHER_ND, SELECT, align_corners, etc.) and
device fp16-overflow walls (SafeLayerNorm, safe_rms, SafeInstanceNorm2d).

Usage:
    from litert_gpu_toolkit import convert_for_gpu

    tflite_path = convert_for_gpu(
        model,
        dummy_input=torch.randn(1, 3, 1024, 1024),
        output_path="model.tflite",
    )

Opt-in patches (import from litert_gpu_toolkit.patches):
    patch_grid_sample, patch_safe_layernorm, patch_rmsnorm, patch_instance_norm,
    patch_maxpool_zeropad, patch_gelu(model, approximation="tanh"),
    pixelshuffle_to_conv_transpose.
"""

from litert_gpu_toolkit.converter import convert_for_gpu
from litert_gpu_toolkit.patches import (
    SafeInstanceNorm2d,
    SigmoidGELU,
    TanhGELU,
    ZeroPadMaxPool,
    ZeroStuffConvT1d,
    ZeroStuffConvT2d,
    apply_all_patches,
    hierarchical_mean,
    patch_conv_transpose,
    patch_gelu,
    patch_grid_sample,
    patch_instance_norm,
    patch_maxpool_zeropad,
    patch_rmsnorm,
    patch_safe_layernorm,
    pixelshuffle_to_conv_transpose,
    safe_rms,
)
from litert_gpu_toolkit.checker import check_gpu_compatibility

__all__ = [
    "convert_for_gpu",
    "apply_all_patches",
    "check_gpu_compatibility",
    "SafeInstanceNorm2d",
    "SigmoidGELU",
    "TanhGELU",
    "ZeroPadMaxPool",
    "ZeroStuffConvT1d",
    "ZeroStuffConvT2d",
    "hierarchical_mean",
    "patch_conv_transpose",
    "patch_gelu",
    "patch_grid_sample",
    "patch_instance_norm",
    "patch_maxpool_zeropad",
    "patch_rmsnorm",
    "patch_safe_layernorm",
    "pixelshuffle_to_conv_transpose",
    "safe_rms",
]
