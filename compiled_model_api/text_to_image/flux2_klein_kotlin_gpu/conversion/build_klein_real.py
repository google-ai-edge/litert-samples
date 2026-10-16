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
REF_TIME_SCALE = 10  # `Flux2KleinPipeline._prepare_image_ids(scale=10)`


def latent_ids(grid):
    """(t, h, w, l) position ids for a grid x grid map, like the pipeline.

    Args:
        grid: Token-map side length, e.g. 16 for a 256 px image.

    Returns:
        A [grid * grid, 4] float tensor of (t, h, w, l) ids.
    """
    t = torch.zeros(1)
    h = torch.arange(grid).float()
    w = torch.arange(grid).float()
    l = torch.zeros(1)
    return torch.cartesian_prod(t, h, w, l)


def reference_ids(grid, index=0):
    """Position ids for a reference, separated from the noise on the T axis.

    The pipeline gives the i-th reference image `T = scale + scale * i`, so the
    noise tokens (T = 0) and every reference occupy disjoint time coordinates.

    Args:
        grid: Token-map side length, e.g. 16 for a 256 px reference.
        index: Which reference image this is, zero based.

    Returns:
        A [grid * grid, 4] float tensor of (t, h, w, l) ids.
    """
    t = torch.full((1,), float(REF_TIME_SCALE * (index + 1)))
    h = torch.arange(grid).float()
    w = torch.arange(grid).float()
    l = torch.zeros(1)
    return torch.cartesian_prod(t, h, w, l)


def edit_ids(grid, num_references=1):
    """Noise ids followed by one block of ids per reference image.

    Args:
        grid: Token-map side length.
        num_references: How many reference images follow the noise.

    Returns:
        A [(1 + num_references) * grid * grid, 4] float tensor of ids.
    """
    blocks = [latent_ids(grid)]
    blocks += [reference_ids(grid, i) for i in range(num_references)]
    return torch.cat(blocks, dim=0)


def check_ids_against_pipeline(grid):
    """Asserts our id tables equal the ones `Flux2KleinPipeline` builds.

    Args:
        grid: Token-map side length to build both tables at.
    """
    from diffusers import Flux2KleinPipeline

    packed = torch.zeros(1, IN_CH, grid, grid)
    expected_noise = Flux2KleinPipeline._prepare_latent_ids(packed)[0].float()
    reference_table = Flux2KleinPipeline._prepare_image_ids([packed])[0]
    expected_reference = reference_table.float()

    noise_ok = torch.equal(latent_ids(grid), expected_noise)
    reference_ok = torch.equal(reference_ids(grid), expected_reference)
    print(f"[ids] noise ids match pipeline: {noise_ok}   "
          f"reference ids match pipeline: {reference_ok}")
    print(f"[ids] noise T={latent_ids(grid)[0, 0]:.0f}  "
          f"reference T={reference_ids(grid)[0, 0]:.0f}")
    if not (noise_ok and reference_ok):
        raise SystemExit("position ids diverge from the pipeline")


def main():
    """Runs the real-weight parity gate for one mode."""
    from diffusers import Flux2Transformer2DModel

    editing = "--edit" in sys.argv
    torch.manual_seed(0)
    check_ids_against_pipeline(TOKEN_GRID)

    print("[load] real transformer (fp32) ...")
    model = Flux2Transformer2DModel.from_pretrained(
        REPO, subfolder="transformer", torch_dtype=torch.float32).eval()
    n = sum(p.numel() for p in model.parameters())
    doubles = len(model.transformer_blocks)
    singles = len(model.single_transformer_blocks)
    print(f"[load] {n/1e9:.3f} B params, {doubles} double + {singles} single")

    # Editing concatenates the reference image's latent tokens onto the noise
    # tokens, exactly as `Flux2KleinPipeline` does before calling the DiT.
    tokens_per_image = TOKEN_GRID * TOKEN_GRID
    s_img = 2 * tokens_per_image if editing else tokens_per_image
    print(f"[mode] {'image editing' if editing else 'text-to-image'}: "
          f"s_img={s_img}, joint sequence={N_TXT + s_img}")
    img_tokens = torch.randn(1, s_img, IN_CH) * 0.5
    enc_hidden = torch.randn(1, N_TXT, CTX_DIM) * 0.5
    timestep = torch.tensor([0.5])
    img_ids = edit_ids(TOKEN_GRID) if editing else latent_ids(TOKEN_GRID)
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
