"""FLUX.2-klein VAE encoder -> INTEGER-int8 LiteRT deploy graph.

Image editing feeds a reference image through the VAE encoder and appends its
latent tokens to the noise tokens. This is the encoder half of `kv_vae`, and it
mirrors that graph's host/graph split:

  [graph]  image [1,3,256,256] -> latent mean [1,32,32,32]
  [host]   patchify 2x2 -> [1,128,16,16] -> BN norm -> pack -> [1,256,128]

Two rewrites, both numerically exact:

  * `GroupNorm` -> `ManualGroupNormND` (the mid-block attention normalizes a 3D
    tensor), reused from the decoder.
  * `DiagonalGaussianDistribution.mode()` chunks the 64-channel moments in half,
    which lowers to `SPLIT` -- rejected by the delegate. The mode is just the
    mean, i.e. the first 32 channels, so slice them directly.

Usage:
    python vae_encode_klein.py            # parity check only
    python vae_encode_klein.py --export   # also write kv_vae_enc.tflite
"""

import os
import sys

import torch
import torch.nn as nn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from vae_patches import patch_groupnorm_nd

REPO = "black-forest-labs/FLUX.2-klein-4B"
LATENT_CH = 32
IMAGE_SIZE = 256
OUTPUT_PATH = "kv_vae_enc.tflite"


class EncoderOnly(nn.Module):
    """RGB in [-1, 1] -> the latent mean, without the Gaussian wrapper."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        moments = self.vae.encoder(image)
        if self.vae.quant_conv is not None:
            moments = self.vae.quant_conv(moments)
        # `latent_dist.mode()` == the mean == the first half of the moments.
        # Slicing avoids the SPLIT that `torch.chunk` would emit.
        return moments[:, :LATENT_CH]


def reference_mode(vae, image):
    """The pipeline's own `sample_mode="argmax"` path, for parity.

    Args:
        vae: The stock `AutoencoderKLFlux2`.
        image: A [1, 3, H, W] tensor in [-1, 1].

    Returns:
        The latent mean, as `latent_dist.mode()` returns it.
    """
    with torch.no_grad():
        return vae.encode(image).latent_dist.mode()


def correlation(a, b):
    """Pearson correlation between two tensors, flattened.

    Args:
        a: The first tensor.
        b: The second tensor.

    Returns:
        The correlation coefficient.
    """
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1]


def main():
    """Checks the two rewrites, then optionally exports the graph."""
    from diffusers import AutoencoderKLFlux2

    torch.manual_seed(0)
    print("[load] AutoencoderKLFlux2 (fp32) ...")
    vae = AutoencoderKLFlux2.from_pretrained(
        REPO, subfolder="vae", torch_dtype=torch.float32).eval()

    image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).clamp(-1, 1)
    expected = reference_mode(vae, image)

    model = EncoderOnly(vae).eval()
    with torch.no_grad():
        sliced = model(image)
    print(f"[slice] mode() vs manual slice: "
          f"corr {correlation(expected, sliced):.8f}  "
          f"max|diff| {(expected - sliced).abs().max():.2e}")

    patch_groupnorm_nd(vae)
    with torch.no_grad():
        patched = model(image)
    print(f"[patch] GroupNorm -> ManualGroupNormND: corr "
          f"{correlation(expected, patched):.8f}  "
          f"max|diff| {(expected - patched).abs().max():.2e}")
    print(f"[shape] {tuple(image.shape)} -> {tuple(patched.shape)}  "
          f"range [{patched.min():.2f}, {patched.max():.2f}]")

    if "--export" in sys.argv:
        import litert_torch
        from litert_torch.generative.quantize import quant_recipes as qrs
        from litert_torch.generative.quantize.quant_attrs import Dtype
        from litert_torch.generative.quantize.quant_attrs import Granularity
        from check_ops import check_ops

        config = qrs.full_dynamic_recipe(weight_dtype=Dtype.INT8,
                                         granularity=Granularity.CHANNELWISE)
        converted = litert_torch.convert(model, (image,), quant_config=config)
        converted.export(OUTPUT_PATH)
        print(f"[int8] {OUTPUT_PATH}  "
              f"{os.path.getsize(OUTPUT_PATH) / 1e6:.0f} MB")
        check_ops(OUTPUT_PATH)

        image.numpy().astype("<f4").tofile("kv_vae_enc_in.bin")
        patched.detach().numpy().astype("<f4").tofile("kv_vae_enc_ref.bin")
        print("[dump] kv_vae_enc_in.bin / kv_vae_enc_ref.bin (fp32)")


if __name__ == "__main__":
    main()
