"""FLUX.2-klein-4B full DiT -> GPU-clean LiteRT graph (reduced-depth probe).

The graph consumes host-prepped fixed-shape inputs, exactly like the Z-Image
deploy wrapper. The host does patchify, position ids, the RoPE tables and the
timestep embedding. The graph does the embedders, all transformer blocks, the
adaLN-continuous final norm and the output projection.

  graph(img_tokens[1,Si,128], enc_hidden[1,St,7680], temb[1,3072],
        cos[1,St+Si,1,64], sin[1,St+Si,1,64]) -> patches[1,Si,128]

Verified against diffusers `Flux2Transformer2DModel` at reduced depth (the block
kernels themselves are proven bit-exact at full width by build_klein{,_double}).
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_klein import DIM, HEAD_DIM, ExportKleinSingle, safe_ln_noaffine
from build_klein_double import ExportKleinDouble

IN_CH = 128
CTX_DIM = 7680
N_HEADS = 24
AXES = [32, 32, 32, 32]
ROPE_THETA = 2000


class ExportKleinDiT(nn.Module):
    """GPU-clean reimplementation of the whole Flux2Transformer2DModel."""

    def __init__(self, model, num_text_tokens):
        super().__init__()
        self.n_txt = num_text_tokens
        self.x_embedder = model.x_embedder
        self.context_embedder = model.context_embedder
        self.mod_img = model.double_stream_modulation_img.linear
        self.mod_txt = model.double_stream_modulation_txt.linear
        self.mod_single = model.single_stream_modulation.linear
        self.doubles = nn.ModuleList(
            [ExportKleinDouble(b) for b in model.transformer_blocks])
        self.singles = nn.ModuleList(
            [ExportKleinSingle(b) for b in model.single_transformer_blocks])
        self.norm_out_linear = model.norm_out.linear
        self.proj_out = model.proj_out

    def forward(self, img_tokens, enc_hidden, temb, cos, sin):
        cond = temb * torch.sigmoid(temb)          # SiLU
        mod_i = self.mod_img(cond).unsqueeze(1)    # [1,1,6*DIM]
        mod_t = self.mod_txt(cond).unsqueeze(1)
        mod_s = self.mod_single(cond).unsqueeze(1)  # [1,1,3*DIM]

        hidden = self.x_embedder(img_tokens)
        encoder = self.context_embedder(enc_hidden)
        for block in self.doubles:
            hidden, encoder = block(hidden, encoder, cos, sin, mod_i, mod_t)

        hidden = torch.cat([encoder, hidden], dim=1)
        shift, scale, gate = mod_s.chunk(3, dim=-1)
        for block in self.singles:
            hidden = block(hidden, cos, sin, shift, scale, gate)
        hidden = hidden[:, self.n_txt:]

        emb = self.norm_out_linear(cond)
        out_scale, out_shift = emb.chunk(2, dim=1)
        hidden = (safe_ln_noaffine(hidden) * (1 + out_scale)[:, None, :]
                  + out_shift[:, None, :])
        return self.proj_out(hidden)


def main():
    """Checks the reduced-depth DiT against diffusers, then converts it."""
    from diffusers import Flux2Transformer2DModel

    torch.manual_seed(0)
    n_double = 1
    n_single = 2
    model = Flux2Transformer2DModel(
        patch_size=1, in_channels=IN_CH, num_layers=n_double,
        num_single_layers=n_single, attention_head_dim=HEAD_DIM,
        num_attention_heads=N_HEADS, joint_attention_dim=CTX_DIM,
        axes_dims_rope=AXES, guidance_embeds=False, rope_theta=ROPE_THETA,
        mlp_ratio=3.0, eps=1e-6, timestep_guidance_channels=256).eval()

    s_img, s_txt = 32, 8
    img_tokens = torch.randn(1, s_img, IN_CH) * 0.5
    enc_hidden = torch.randn(1, s_txt, CTX_DIM) * 0.5
    timestep = torch.tensor([0.5])
    img_ids = torch.randint(0, 8, (s_img, len(AXES))).float()
    txt_ids = torch.zeros(s_txt, len(AXES))

    with torch.no_grad():
        ref = model(hidden_states=img_tokens, encoder_hidden_states=enc_hidden,
                    timestep=timestep, img_ids=img_ids, txt_ids=txt_ids,
                    return_dict=False)[0]

        # host prep: timestep embedding + concatenated RoPE tables
        temb = model.time_guidance_embed(timestep * 1000, None)
        img_cos, img_sin = model.pos_embed(img_ids)
        txt_cos, txt_sin = model.pos_embed(txt_ids)
        cos_full = torch.cat([txt_cos, img_cos], dim=0)
        sin_full = torch.cat([txt_sin, img_sin], dim=0)
        # de-interleave: cos/sin come repeat_interleaved in pairs
        cos = cos_full[:, 0::2][None, :, None, :]
        sin = sin_full[:, 0::2][None, :, None, :]

    mine = ExportKleinDiT(model, s_txt).eval()
    with torch.no_grad():
        out = mine(img_tokens, enc_hidden, temb, cos, sin)

    corr = torch.corrcoef(torch.stack([ref.flatten(), out.flatten()]))[0, 1]
    max_diff = (ref - out).abs().max()
    print(f"[parity] full DiT ({n_double} double + {n_single} single) "
          f"{tuple(ref.shape)}: corr {corr:.8f}  max|diff| {max_diff:.2e}")

    if "--convert" in sys.argv:
        import os

        import litert_torch
        path = "klein_dit_probe.tflite"
        inputs = (img_tokens, enc_hidden, temb, cos, sin)
        litert_torch.convert(mine, inputs).export(path)
        print(f"[convert] {path}  {os.path.getsize(path)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
