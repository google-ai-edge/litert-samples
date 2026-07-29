"""Export the Bonsai Image VAE DECODER (AutoencoderKLFlux2).

Generation only needs decode: latents -> RGB. The encoder is for img2img/inpaint,
so it is left out and the export is just the decoder path.

Fixed shape: 512x512 output => 64x64 latent at the VAE's 8x downsample.
"""
import os
import time

import torch
from diffusers import AutoencoderKLFlux2

from huggingface_hub import snapshot_download

SNAP = os.environ.get("BONSAI_SNAPSHOT") or snapshot_download(
    "prism-ml/bonsai-image-ternary-4B-unpacked")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "vae_dec_fp32.tflite")
LAT = 64


class Decoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        return self.vae.decode(latents, return_dict=False)[0]


vae = AutoencoderKLFlux2.from_pretrained(f"{SNAP}/vae", torch_dtype=torch.float32).eval()
lc = vae.config.latent_channels
print(f"VAE latent_channels={lc} scaling={getattr(vae.config,'scaling_factor',None)} "
      f"params={sum(p.numel() for p in vae.parameters())/1e6:.1f} M", flush=True)

dec = Decoder(vae).eval()
z = torch.randn(1, lc, LAT, LAT, generator=torch.Generator().manual_seed(0))
with torch.no_grad():
    ref = dec(z)
print(f"decode {tuple(z.shape)} -> {tuple(ref.shape)} "
      f"min={ref.min():.3f} max={ref.max():.3f}", flush=True)
torch.save({"z": z, "ref": ref}, os.path.join(HERE, "vae_ref.pt"))

import litert_torch
t0 = time.time()
m = litert_torch.convert(dec, (z,))
print(f"convert OK in {time.time()-t0:.0f}s", flush=True)
m.export(OUT)
print(f"export OK -> {os.path.getsize(OUT)/2**20:.1f} MiB", flush=True)
