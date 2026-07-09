"""FLUX.2-klein VAE decoder -> INTEGER-int8 LiteRT deploy graph.

Fixed 256 px: latent [1,32,32,32] -> image [1,3,256,256]. GroupNorm is replaced
by a manual N-dimensional equivalent, because the mid-block attention normalizes
a 3D tensor and the usual manual rewrite is written for 4D only. Decode parity
is checked against the fp32 VAE before anything is exported.

The host owns the two steps around the graph, exactly as the pipeline does:
  unpack tokens by ids -> BN denorm (x * sqrt(var + eps) + mean) -> unpatchify
  2x2 -> [graph] decode
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = "black-forest-labs/FLUX.2-klein-4B"
LATENT_CH = 32
LATENT_HW = 32       # 256 px


class ManualGroupNormND(nn.Module):
    """GPU-friendly GroupNorm accepting 3D [B,C,L] or 4D [B,C,H,W] input."""

    def __init__(self, group_norm):
        super().__init__()
        self.num_groups = group_norm.num_groups
        self.num_channels = group_norm.num_channels
        self.eps = group_norm.eps
        affine = group_norm.weight is not None
        weight = nn.Parameter(group_norm.weight.clone()) if affine else None
        bias = nn.Parameter(group_norm.bias.clone()) if affine else None
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        shape = x.shape
        if x.dim() == 3:
            batch, channels, length = shape
            x4 = x.reshape(batch, channels, length, 1)
        else:
            x4 = x
        batch, channels = x4.shape[0], x4.shape[1]
        groups = self.num_groups
        grouped = x4.reshape(
            batch * groups, channels // groups, x4.shape[2], x4.shape[3])
        mean = grouped.mean(dim=(1, 2, 3), keepdim=True)
        centered = grouped - mean
        var = (centered * centered).mean(dim=(1, 2, 3), keepdim=True)
        normed = centered * torch.rsqrt(var + self.eps)
        normed = normed.reshape(batch, channels, x4.shape[2], x4.shape[3])
        if self.weight is not None:
            normed = (normed * self.weight.reshape(1, channels, 1, 1)
                      + self.bias.reshape(1, channels, 1, 1))
        return normed.reshape(shape)


def patch_groupnorm_nd(model):
    """Replaces every GroupNorm in a model with ManualGroupNormND, in place.

    Args:
      model: The module to walk.

    Returns:
      The number of modules replaced.
    """
    count = 0
    for _, module in list(model.named_modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.GroupNorm):
                setattr(module, name, ManualGroupNormND(child))
                count += 1
    return count


class DecoderOnly(nn.Module):
    """Wraps the VAE so the exported graph is exactly latent -> RGB."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


def main():
    """Checks the patched decoder against fp32, then exports the int8 graph."""
    from diffusers import AutoencoderKLFlux2

    torch.manual_seed(0)
    print("[load] AutoencoderKLFlux2 (fp32) ...")
    vae = AutoencoderKLFlux2.from_pretrained(
        REPO, subfolder="vae", torch_dtype=torch.float32).eval()
    print(f"[load] bn channels: {tuple(vae.bn.running_mean.shape)}  "
          f"eps: {vae.config.batch_norm_eps}")

    z = torch.randn(1, LATENT_CH, LATENT_HW, LATENT_HW) * 0.5
    model = DecoderOnly(vae).eval()
    with torch.no_grad():
        ref = model(z)
    print(f"[forward] {tuple(z.shape)} -> {tuple(ref.shape)} "
          f"range [{ref.min():.2f},{ref.max():.2f}]")

    n_patched = patch_groupnorm_nd(vae)
    with torch.no_grad():
        patched = model(z)
    pair = torch.stack([ref.flatten(), patched.flatten()])
    corr = torch.corrcoef(pair)[0, 1]
    print(f"[patch] GroupNorm -> ManualGroupNormND: {n_patched} modules, "
          f"corr {corr:.8f}")

    if "--export" in sys.argv:
        import litert_torch
        from litert_torch.generative.quantize import quant_recipes as qrs
        from litert_torch.generative.quantize.quant_attrs import Dtype
        from litert_torch.generative.quantize.quant_attrs import Granularity
        cfg = qrs.full_dynamic_recipe(weight_dtype=Dtype.INT8,
                                      granularity=Granularity.CHANNELWISE)
        out = "kv_vae.tflite"
        litert_torch.convert(model, (z,), quant_config=cfg).export(out)
        print(f"[int8] {out}  {os.path.getsize(out)/1e6:.0f} MB")


if __name__ == "__main__":
    main()
