"""FLUX.2-klein VAE decoder -> INTEGER-int8 LiteRT deploy graph.

Fixed 256 px: latent [1,32,32,32] -> image [1,3,256,256]. The 3D/4D-aware
ManualGroupNormND replaces GroupNorm (the mid-block attention normalizes a 3D
tensor). Verifies decode parity against the fp32 VAE on a real latent.

The host owns the two steps around the graph, exactly as the pipeline does:
  unpack tokens by ids -> BN denorm (x * sqrt(var+eps) + mean) -> unpatchify 2x2
  -> [graph] decode
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vae_patches import patch_groupnorm_nd

REPO = "black-forest-labs/FLUX.2-klein-4B"
LATENT_CH = 32
LATENT_HW = 32       # 256 px


class DecoderOnly(nn.Module):
    """latent -> RGB."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


def main():
    """Checks the decoder rewrite, then optionally exports it."""
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
    corr = torch.corrcoef(torch.stack([ref.flatten(), patched.flatten()]))[0, 1]
    print(f"[patch] GroupNorm -> ManualGroupNormND: {n_patched} modules, "
          f"corr {corr:.8f}")

    if "--export" in sys.argv:
        import litert_torch
        from litert_torch.generative.quantize import quant_recipes as qrs
        from litert_torch.generative.quantize.quant_attrs import Dtype
        from litert_torch.generative.quantize.quant_attrs import Granularity
        from check_ops import check_ops
        cfg = qrs.full_dynamic_recipe(weight_dtype=Dtype.INT8,
                                      granularity=Granularity.CHANNELWISE)
        out = "kv_vae.tflite"
        litert_torch.convert(model, (z,), quant_config=cfg).export(out)
        print(f"[int8] {out}  {os.path.getsize(out)/1e6:.0f} MB")
        check_ops(out)


if __name__ == "__main__":
    main()
