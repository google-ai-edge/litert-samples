"""Does FLUX.2-klein image editing fit on the phone? Probe the sequence growth.

Editing appends the reference image's latent tokens to the noise tokens:

    latent_model_input = cat([latents, image_latents], dim=1)

At 256x256 that is 256 noise tokens + 256 reference tokens, so the joint
sequence the single-stream blocks see grows from 512 + 256 = 768 to 512 + 512 =
1024. The weights do not change, so chunk sizes stay in the compile sweet spot;
only the activations grow.

This exports the deployment's worst sequence-dependent chunk -- five
single-stream blocks, the `kc_single*` shape -- at both sequence lengths, with
random weights of the real width. Push both to a device and compare compile
time, run time and whether the delegate still takes every node.

Usage:
    python probe_edit_seqlen.py            # export both graphs
    python probe_edit_seqlen.py --seq 1024 # export just one
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_klein_dit import AXES, CTX_DIM, HEAD_DIM, IN_CH, N_HEADS, ROPE_THETA
from chunked_export_klein import DIM, SINGLE_CHUNK, SingleChunk, q_export

N_TXT = 512
SEQ_TEXT_TO_IMAGE = N_TXT + 256    # 512 text + 256 noise tokens
SEQ_IMAGE_EDIT = N_TXT + 512       # 512 text + 256 noise + 256 reference tokens
SEED = 0


def build_chunk():
    """Five single-stream blocks at the real width, randomly initialised.

    Weights are random because this probe measures compile and memory
    behaviour, not numerics. `full_dynamic_recipe` needs no calibration data.

    Returns:
        A `SingleChunk` of `SINGLE_CHUNK` blocks.
    """
    from diffusers import Flux2Transformer2DModel
    from build_klein import ExportKleinSingle

    torch.manual_seed(SEED)
    model = Flux2Transformer2DModel(
        patch_size=1, in_channels=IN_CH, num_layers=1,
        num_single_layers=SINGLE_CHUNK, attention_head_dim=HEAD_DIM,
        num_attention_heads=N_HEADS, joint_attention_dim=CTX_DIM,
        axes_dims_rope=AXES, guidance_embeds=False, rope_theta=ROPE_THETA,
        mlp_ratio=3.0, eps=1e-6, timestep_guidance_channels=256).eval()
    blocks = [ExportKleinSingle(b) for b in model.single_transformer_blocks]
    return SingleChunk(blocks).eval()


def chunk_inputs(seq_len):
    """Fixed-shape inputs for a `SingleChunk` at `seq_len` joint tokens.

    Args:
        seq_len: Joint sequence length the blocks attend over.

    Returns:
        The tuple a `SingleChunk` forward takes.
    """
    joint = torch.randn(1, seq_len, DIM) * 0.5
    cos = torch.randn(1, seq_len, 1, HEAD_DIM // 2)
    sin = torch.randn(1, seq_len, 1, HEAD_DIM // 2)
    mod_s = torch.randn(1, 1, 3 * DIM) * 0.1
    return joint, cos, sin, mod_s


def attention_bytes(seq_len):
    """fp32 bytes of one [1, heads, S, S] attention tensor.

    Args:
        seq_len: Joint sequence length.

    Returns:
        Bytes of one fp32 attention tensor.
    """
    return N_HEADS * seq_len * seq_len * 4


def main():
    """Exports the single-stream chunk at each sequence length."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, action="append",
                        help="joint sequence length; repeatable")
    args = parser.parse_args()
    sequences = args.seq or [SEQ_TEXT_TO_IMAGE, SEQ_IMAGE_EDIT]

    chunk = build_chunk()
    for seq_len in sequences:
        inputs = chunk_inputs(seq_len)
        with torch.no_grad():
            out = chunk(*inputs)
        megabytes = attention_bytes(seq_len) / 1e6
        print(f"[S={seq_len}] out {tuple(out.shape)}   one attention tensor "
              f"{megabytes:.0f} MB fp32")
        q_export(chunk, inputs, f"kc_single_probe_s{seq_len}")


def build_double_chunk():
    """Three double-stream blocks: the largest deployment graph (kc_double0)."""
    from diffusers import Flux2Transformer2DModel
    from build_klein_double import ExportKleinDouble
    from chunked_export_klein import DoubleChunk

    torch.manual_seed(SEED)
    model = Flux2Transformer2DModel(
        patch_size=1, in_channels=IN_CH, num_layers=3, num_single_layers=1,
        attention_head_dim=HEAD_DIM, num_attention_heads=N_HEADS,
        joint_attention_dim=CTX_DIM, axes_dims_rope=AXES, guidance_embeds=False,
        rope_theta=ROPE_THETA, mlp_ratio=3.0, eps=1e-6,
        timestep_guidance_channels=256).eval()
    blocks = [ExportKleinDouble(b) for b in model.transformer_blocks]
    return DoubleChunk(blocks).eval()


def export_double(s_img):
    """Exports kc_double0's shape at `s_img` image tokens.

    Args:
        s_img: Image tokens the double-stream blocks see.
    """
    chunk = build_double_chunk()
    hidden = torch.randn(1, s_img, DIM) * 0.5
    encoder = torch.randn(1, N_TXT, DIM) * 0.5
    seq = N_TXT + s_img
    cos = torch.randn(1, seq, 1, HEAD_DIM // 2)
    sin = torch.randn(1, seq, 1, HEAD_DIM // 2)
    mod_i = torch.randn(1, 1, 6 * DIM) * 0.1
    mod_t = torch.randn(1, 1, 6 * DIM) * 0.1
    inputs = (hidden, encoder, cos, sin, mod_i, mod_t)
    print(f"[double s_img={s_img}] joint seq {seq}")
    q_export(chunk, inputs, f"kc_double_probe_s{seq}")


if __name__ == "__main__":
    main()
