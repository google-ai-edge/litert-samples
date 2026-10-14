#!/usr/bin/env python3
"""Build the 512x512 MLX weight set for the SAM2 MLX-swift benchmark app.

The avbiswas MLX checkpoint bakes `trunk.pos_embed_full` as the fully COMPOSED
positional embedding at the 1024-input token grid (256x256): a bicubically
interpolated global base (7x7) plus the 8x8 window embedding tiled across the
grid. Reusing it at 512 is geometrically wrong either way:

  - slicing the top-left 128x128 keeps the window tiling but misplaces the
    global component (encoder features corr 0.947 vs the PyTorch reference);
  - naively resizing the composed grid to 128x128 aliases the period-8 window
    tiling and is much worse (corr 0.748).

The correct 512 embedding re-composes from the ORIGINAL parts, exactly as the
`transformers` implementation does internally at any input size:

    pos_embed_512 = bicubic(base_7x7 -> 128x128) + tile(window_8x8 -> 128x128)

With this, the MLX-swift pipeline at 512 matches the PyTorch reference at
corr 0.9999 (masks) with iou_scores equal to 4 decimals.

Usage:
    python tools/make_512_weights.py [--size 512] [--out Resources/sam2_tiny_512.safetensors]

Requires: torch, transformers, mlx, huggingface_hub, numpy, safetensors.
"""
import argparse
import sys
import types

import numpy as np

# macOS: stub scipy's broken propack entry point (only needed if scipy is present but broken)
_svdp = types.ModuleType("scipy.sparse.linalg._svdp")
_svdp._svdp = lambda *a, **k: None
sys.modules["scipy.sparse.linalg._svdp"] = _svdp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512, help="Input image size (multiple of 16).")
    parser.add_argument("--out", default="Resources/sam2_tiny_512.safetensors")
    args = parser.parse_args()
    token_grid = args.size // 4

    import torch
    from transformers import Sam2Model

    model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").eval()
    hiera = next(m for m in model.modules() if type(m).__name__ == "Sam2HieraDetModel")
    base = hiera.pos_embed.detach().float()            # [1, 96, 7, 7]
    window = hiera.pos_embed_window.detach().float()   # [1, 96, 8, 8]
    resized = torch.nn.functional.interpolate(
        base, size=(token_grid, token_grid), mode="bicubic")
    tiled = window.tile(1, 1, token_grid // window.shape[2], token_grid // window.shape[3])
    composed = (resized + tiled).permute(0, 2, 3, 1).contiguous()   # NHWC, [1, T, T, 96]

    import mlx.core as mx
    from huggingface_hub import hf_hub_download

    ckpt = hf_hub_download(
        "avbiswas/sam2.1-hiera-tiny-mlx", "sam2.1_hiera_tiny_image_segmenter.safetensors")
    weights = mx.load(ckpt)
    out = {}
    for key, value in weights.items():
        if key == "trunk.pos_embed_full":
            out[key] = mx.array(composed.numpy().astype(np.float32)).astype(mx.float16)
        else:
            out[key] = value.astype(mx.float16) if value.dtype == mx.float32 else value
    mx.save_safetensors(args.out, out)
    print(f"wrote {args.out} (pos_embed re-composed at {token_grid}x{token_grid})")


if __name__ == "__main__":
    main()
