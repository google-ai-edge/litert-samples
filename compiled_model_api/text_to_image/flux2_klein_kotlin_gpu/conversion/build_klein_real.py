"""FLUX.2-klein-4B real-weight parity: GPU-clean DiT vs the diffusers model.

Runs the full 3.876 B transformer (5 double + 20 single blocks) at the real
256 px shapes (512 text tokens + 256 image tokens) and compares the GPU-clean
reimplementation against the stock diffusers forward, on the real checkpoint.

This is the klein analogue of Z-Image's real-weight parity gate: it proves the
baked-permutation RoPE, the max-norm safe norms and the manual SDPA are exact on
trained weights, not just on random ones.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_klein_dit import CTX_DIM, IN_CH, ExportKleinDiT

REPO = "black-forest-labs/FLUX.2-klein-4B"
TOKEN_GRID = 16      # 256 px -> 32x32 latent -> 2x2 pack -> 16x16 tokens
N_TXT = 512


def latent_ids(grid):
    """Builds the (t, h, w, l) position ids the pipeline gives to the DiT.

    Args:
      grid: Side length of the square token map.

    Returns:
      A [grid * grid, 4] float tensor of position ids.
    """
    t = torch.zeros(1)
    h = torch.arange(grid).float()
    w = torch.arange(grid).float()
    l = torch.zeros(1)
    return torch.cartesian_prod(t, h, w, l)


def main():
    """Runs the GPU-clean DiT against the real weights and reports parity."""
    from diffusers import Flux2Transformer2DModel

    torch.manual_seed(0)
    print("[load] real transformer (fp32) ...")
    model = Flux2Transformer2DModel.from_pretrained(
        REPO, subfolder="transformer", torch_dtype=torch.float32).eval()
    n = sum(p.numel() for p in model.parameters())
    n_double = len(model.transformer_blocks)
    n_single = len(model.single_transformer_blocks)
    print(f"[load] {n/1e9:.3f} B params, {n_double} double + "
          f"{n_single} single")

    s_img = TOKEN_GRID * TOKEN_GRID
    img_tokens = torch.randn(1, s_img, IN_CH) * 0.5
    enc_hidden = torch.randn(1, N_TXT, CTX_DIM) * 0.5
    timestep = torch.tensor([0.5])
    img_ids = latent_ids(TOKEN_GRID)
    txt_ids = torch.zeros(N_TXT, 4)

    with torch.no_grad():
        print("[ref] diffusers forward ...")
        ref = model(hidden_states=img_tokens, encoder_hidden_states=enc_hidden,
                    timestep=timestep, img_ids=img_ids, txt_ids=txt_ids,
                    return_dict=False)[0]

        # host prep the graph inputs
        temb = model.time_guidance_embed(timestep * 1000, None)
        img_cos, img_sin = model.pos_embed(img_ids)
        txt_cos, txt_sin = model.pos_embed(txt_ids)
        cos = torch.cat([txt_cos, img_cos], dim=0)[:, 0::2][None, :, None, :]
        sin = torch.cat([txt_sin, img_sin], dim=0)[:, 0::2][None, :, None, :]

        print("[mine] GPU-clean forward ...")
        mine = ExportKleinDiT(model, N_TXT).eval()
        out = mine(img_tokens, enc_hidden, temb, cos, sin)

    corr = torch.corrcoef(torch.stack([ref.flatten(), out.flatten()]))[0, 1]
    diff = (ref - out).abs()
    print(f"[parity] real-weight full DiT {tuple(ref.shape)}: corr {corr:.8f}  "
          f"max|diff| {diff.max():.3e}  mean|diff| {diff.mean():.3e}  "
          f"range [{ref.min():.2f},{ref.max():.2f}]")


if __name__ == "__main__":
    main()
