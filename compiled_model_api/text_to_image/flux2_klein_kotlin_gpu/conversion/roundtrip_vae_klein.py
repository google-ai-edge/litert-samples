"""Encode -> decode round trip for the FLUX.2-klein VAE, host and device.

The encoder graph (`kv_vae_enc.tflite`) is only useful for image editing if the
latent it produces is good enough to feed the DiT. The cheapest end-to-end proxy
is a round trip: encode a real image, decode it with the already-shipped decoder
graph (`kv_vae.tflite`), and compare against the fp32 `diffusers` round trip.

    python roundtrip_vae_klein.py --prep --image photo.png   # fp32 references
    python roundtrip_vae_klein.py --compare            # score device dumps

The device half runs between the two, driven by adb:

    ./runner kv_vae_enc.tflite 1 rt_image  rt_latent    # FP32=1
    ./runner kv_vae.tflite     1 rt_latent rt_decoded   # FP32=1
"""

import argparse
import os
import sys

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from vae_patches import patch_groupnorm_nd
from vae_encode_klein import REPO, EncoderOnly

DEFAULT_IMAGE = os.path.join(_SCRIPT_DIR, "..", "docs", "pixel8a_generated.png")
IMAGE_SIZE = 256
PEAK_SIGNAL = 2.0  # images live in [-1, 1]


def load_image(path):
    """Loads the test image as a [1, 3, 256, 256] tensor in [-1, 1].

    Args:
        path: Any image file, resized to IMAGE_SIZE.

    Returns:
        A [1, 3, 256, 256] float tensor in [-1, 1].
    """
    from PIL import Image

    image = Image.open(path).convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)[None]


def psnr(reference, other):
    """PSNR in dB between two arrays that live in [-1, 1].

    Args:
        reference: The ground-truth array.
        other: The array being scored.

    Returns:
        PSNR in dB.
    """
    mse = np.mean((reference - other) ** 2)
    return 10.0 * np.log10(PEAK_SIGNAL**2 / mse) if mse > 0 else float("inf")


def correlation(a, b):
    """Pearson correlation between two arrays, flattened.

    Args:
        a: The first array.
        b: The second array.

    Returns:
        The correlation coefficient.
    """
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def prep(image_path):
    """Writes the device input and the fp32 host references.

    Args:
        image_path: The image to round trip.
    """
    from diffusers import AutoencoderKLFlux2

    torch.manual_seed(0)
    vae = AutoencoderKLFlux2.from_pretrained(
        REPO, subfolder="vae", torch_dtype=torch.float32).eval()

    image = load_image(image_path)
    with torch.no_grad():
        latent = vae.encode(image).latent_dist.mode()
        decoded = vae.decode(latent, return_dict=False)[0]

    patch_groupnorm_nd(vae)
    with torch.no_grad():
        patched_latent = EncoderOnly(vae).eval()(image)

    image.numpy().astype("<f4").tofile("rt_image.0")
    latent.numpy().astype("<f4").tofile("rt_latent_ref.bin")
    decoded.numpy().astype("<f4").tofile("rt_decoded_ref.bin")

    print(f"[prep] image {tuple(image.shape)} -> latent {tuple(latent.shape)}")
    print(f"[prep] rewritten encoder vs stock: corr "
          f"{correlation(latent.numpy(), patched_latent.numpy()):.8f}")
    print(f"[prep] fp32 round trip PSNR: "
          f"{psnr(image.numpy(), decoded.numpy()):.2f} dB   (the ceiling)")
    print("[prep] wrote rt_image.0, rt_latent_ref.bin, rt_decoded_ref.bin")


def compare():
    """Scores whatever the device produced against the fp32 references."""
    image = np.fromfile("rt_image.0", dtype=np.float32)
    latent_ref = np.fromfile("rt_latent_ref.bin", dtype=np.float32)
    decoded_ref = np.fromfile("rt_decoded_ref.bin", dtype=np.float32)

    latent_dev = np.fromfile("rt_latent.0", dtype=np.float32)
    decoded_dev = np.fromfile("rt_decoded.0", dtype=np.float32)

    print(f"[latent ] device int8 GPU vs fp32 host: corr "
          f"{correlation(latent_ref, latent_dev):.6f}  "
          f"max|diff| {np.abs(latent_ref - latent_dev).max():.4f}")
    print(f"[decoded] device round trip vs fp32 round trip: "
          f"{psnr(decoded_ref, decoded_dev):.2f} dB  corr "
          f"{correlation(decoded_ref, decoded_dev):.6f}")
    print(f"[decoded] device round trip vs ORIGINAL image: "
          f"{psnr(image, decoded_dev):.2f} dB")
    print(f"[decoded] fp32 round trip  vs ORIGINAL image: "
          f"{psnr(image, decoded_ref):.2f} dB   (the ceiling)")
    has_nan = bool(np.isnan(decoded_dev).any())
    print(f"[decoded] NaN in device output: {has_nan}")


def main():
    """Stages the references, or scores the device dumps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    args = parser.parse_args()
    if args.prep:
        prep(args.image)
    if args.compare:
        compare()


if __name__ == "__main__":
    main()
