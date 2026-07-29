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
Main conversion entry point.
"""

import os
import logging

import torch
import torch.nn as nn

from litert_gpu_toolkit.patches import apply_all_patches, restore_gelu, restore_interpolate, restore_normalize
from litert_gpu_toolkit.checker import check_gpu_compatibility, print_report

log = logging.getLogger("litert_gpu_toolkit")


def convert_for_gpu(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str = "model.tflite",
    check: bool = True,
    verbose: bool = True,
) -> str:
    """Convert a PyTorch model to GPU-compatible TFLite.

    Applies all known patches (GELU, DeformableConv, interpolate, WindowAttention,
    PatchMerging, einops), converts via litert-torch, and validates the result.

    Args:
        model: PyTorch model (must be in eval mode)
        dummy_input: Example input tensor (e.g., torch.randn(1, 3, 1024, 1024))
        output_path: Where to save the .tflite file
        check: Run GPU compatibility check after conversion
        verbose: Print detailed report

    Returns:
        Path to saved .tflite file
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    model.eval()

    # Apply patches (dummy_input lets ConvTranspose layers capture their fixed sizes)
    log.info("Applying GPU compatibility patches...")
    summary = apply_all_patches(model, dummy_input)
    if verbose:
        active = {k: v for k, v in summary.items() if v}
        log.info(f"Patches applied: {active}")

    # Verify forward pass
    log.info("Verifying forward pass...")
    with torch.no_grad():
        out = model(dummy_input)
        if isinstance(out, torch.Tensor):
            log.info(f"Output: {out.shape}")
        elif isinstance(out, (list, tuple)):
            shapes = [o.shape for o in out if isinstance(o, torch.Tensor)]
            log.info(f"Output: {len(out)} tensors, first={shapes[0] if shapes else '?'}")

    # Convert
    log.info("Converting with litert-torch...")
    import litert_torch
    result = litert_torch.convert(model, (dummy_input,))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.export(output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    log.info(f"Saved: {output_path} ({size_mb:.1f} MB)")

    # Restore global patches
    restore_gelu()
    restore_interpolate()
    restore_normalize()

    # Check
    if check:
        log.info("Checking GPU compatibility...")
        report = check_gpu_compatibility(output_path)
        if verbose:
            print_report(report)
        if not report['compatible']:
            log.warning("Model has GPU-incompatible ops. See report above.")

    return output_path
