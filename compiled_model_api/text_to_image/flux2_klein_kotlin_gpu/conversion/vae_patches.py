"""GPU-clean rewrites shared by the FLUX.2-klein VAE encoder and decoder.

`AutoencoderKLFlux2`'s mid-block attention normalizes a 3D `[B, C, L]` tensor,
so a 4D-only manual GroupNorm is not enough. This module keeps one 3D/4D-aware
replacement that both `vae_deploy_klein.py` (decoder) and `vae_encode_klein.py`
(encoder) install before conversion.

The rewrite is numerically exact: it is the GroupNorm definition written with
ops the ML Drift delegate accepts, rather than the fused `GROUP_NORM` it
rejects.
"""

import torch
import torch.nn as nn


class ManualGroupNormND(nn.Module):
    """GroupNorm over a 3D `[B, C, L]` or 4D `[B, C, H, W]` input."""

    def __init__(self, group_norm):
        super().__init__()
        self.num_groups = group_norm.num_groups
        self.num_channels = group_norm.num_channels
        self.eps = group_norm.eps
        self.weight = (nn.Parameter(group_norm.weight.clone())
                       if group_norm.weight is not None else None)
        self.bias = (nn.Parameter(group_norm.bias.clone())
                     if group_norm.bias is not None else None)

    def forward(self, x):
        shape = x.shape
        x4d = x.reshape(shape[0], shape[1], shape[2], 1) if x.dim() == 3 else x
        batch, channels = x4d.shape[0], x4d.shape[1]
        grouped = x4d.reshape(batch * self.num_groups,
                              channels // self.num_groups,
                              x4d.shape[2], x4d.shape[3])
        mean = grouped.mean(dim=(1, 2, 3), keepdim=True)
        centered = grouped - mean
        variance = (centered * centered).mean(dim=(1, 2, 3), keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.eps)
        normalized = normalized.reshape(batch, channels,
                                      x4d.shape[2], x4d.shape[3])
        if self.weight is not None:
            normalized = (normalized * self.weight.reshape(1, channels, 1, 1)
                          + self.bias.reshape(1, channels, 1, 1))
        return normalized.reshape(shape)


def patch_groupnorm_nd(model):
    """Replaces every `nn.GroupNorm` in `model` in place.

    Args:
        model: The module tree to rewrite.

    Returns:
        The number of modules replaced.
    """
    count = 0
    for _, module in list(model.named_modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.GroupNorm):
                setattr(module, name, ManualGroupNormND(child))
                count += 1
    print(f"[patch] {count} GroupNorm -> ManualGroupNormND")
    return count
