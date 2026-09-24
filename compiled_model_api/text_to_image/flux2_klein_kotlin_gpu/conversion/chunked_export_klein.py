"""Split the FLUX.2-klein-4B DiT into <1GB int8 chunks (sequential residency).

Chunk plan (all <1 GB int8, the device-proven sweet spot from Z-Image):

  kc_prep      x_embedder + context_embedder + the three modulation FCs
               -> hidden, encoder, mod_img, mod_txt, mod_single        ~165 MB
  kc_double0   3 double-stream blocks                                  ~736 MB
  kc_double1   2 double-stream blocks                                  ~491 MB
  [host]       joint = cat([encoder, hidden], dim=1)   -> [1,768,3072]
  kc_single0-3 5 single-stream blocks each                             ~613 MB
  kc_final     slice(img) + adaLN-continuous norm + proj_out           ~19 MB

Verifies that the sequential composition equals the monolithic GPU-clean DiT
before exporting anything.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_klein import safe_ln_noaffine
from build_klein_dit import CTX_DIM, IN_CH, ExportKleinDiT
from build_klein_real import REPO, TOKEN_GRID, N_TXT, edit_ids, latent_ids

DIM = 3072
DOUBLE_SPLIT = [3, 2]
SINGLE_CHUNK = 5


class Prep(nn.Module):
    """Embedders + the three modulation projections."""

    def __init__(self, dit):
        super().__init__()
        self.x_embedder = dit.x_embedder
        self.context_embedder = dit.context_embedder
        self.mod_img = dit.mod_img
        self.mod_txt = dit.mod_txt
        self.mod_single = dit.mod_single

    def forward(self, img_tokens, enc_hidden, temb):
        cond = temb * torch.sigmoid(temb)
        return (self.x_embedder(img_tokens), self.context_embedder(enc_hidden),
                self.mod_img(cond).unsqueeze(1),
                self.mod_txt(cond).unsqueeze(1),
                self.mod_single(cond).unsqueeze(1))


class DoubleChunk(nn.Module):
    """A run of double-stream blocks."""

    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden, encoder, cos, sin, mod_i, mod_t):
        for block in self.blocks:
            hidden, encoder = block(hidden, encoder, cos, sin, mod_i, mod_t)
        return hidden, encoder


class SingleChunk(nn.Module):
    """A run of single-stream blocks over the joint [txt; img] sequence."""

    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, joint, cos, sin, mod_s):
        shift, scale, gate = mod_s.chunk(3, dim=-1)
        for block in self.blocks:
            joint = block(joint, cos, sin, shift, scale, gate)
        return joint


class Final(nn.Module):
    """Drop the text tokens, adaLN-continuous norm, project to patches."""

    def __init__(self, dit, n_txt):
        super().__init__()
        self.n_txt = n_txt
        self.norm_out_linear = dit.norm_out_linear
        self.proj_out = dit.proj_out

    def forward(self, joint, temb):
        hidden = joint[:, self.n_txt:]
        cond = temb * torch.sigmoid(temb)
        scale, shift = self.norm_out_linear(cond).chunk(2, dim=1)
        normed = safe_ln_noaffine(hidden)
        hidden = normed * (1 + scale)[:, None, :] + shift[:, None, :]
        return self.proj_out(hidden)


def q_export(model, inputs, name):
    """Quantize-exports a module to an int8 LiteRT graph.

    Args:
        model: The module to convert.
        inputs: Example inputs, in forward order.
        name: Output basename; `<name>.tflite` is written.
    """
    import litert_torch
    from litert_torch.generative.quantize import quant_recipes as qrs
    from litert_torch.generative.quantize.quant_attrs import Dtype, Granularity
    cfg = qrs.full_dynamic_recipe(weight_dtype=Dtype.INT8,
                                  granularity=Granularity.CHANNELWISE)
    out = f"{name}.tflite"
    litert_torch.convert(model, inputs, quant_config=cfg).export(out)
    print(f"[{name}] {os.path.getsize(out)/1e6:.0f} MB")


def main():
    """Verifies the chunk composition, then optionally exports it."""
    from diffusers import Flux2Transformer2DModel

    # Image editing appends the reference image's latent tokens to the noise
    # tokens, so `s_img` doubles and every graph is re-exported at that shape.
    # The weights are sequence-independent, so the chunk sizes do not move.
    editing = "--edit" in sys.argv
    prefix = "kce" if editing else "kc"

    torch.manual_seed(0)
    print("[load] real transformer (fp32) ...")
    model = Flux2Transformer2DModel.from_pretrained(
        REPO, subfolder="transformer", torch_dtype=torch.float32).eval()
    dit = ExportKleinDiT(model, N_TXT).eval()

    tokens_per_image = TOKEN_GRID * TOKEN_GRID
    s_img = 2 * tokens_per_image if editing else tokens_per_image
    img_ids = edit_ids(TOKEN_GRID) if editing else latent_ids(TOKEN_GRID)
    print(f"[mode] {'image editing' if editing else 'text-to-image'}: "
          f"s_img={s_img}, joint sequence={N_TXT + s_img}, prefix {prefix}_*")

    img_tokens = torch.randn(1, s_img, IN_CH) * 0.5
    enc_hidden = torch.randn(1, N_TXT, CTX_DIM) * 0.5
    timestep = torch.tensor([0.5])
    with torch.no_grad():
        temb = model.time_guidance_embed(timestep * 1000, None)
        img_cos, img_sin = model.pos_embed(img_ids)
        txt_cos, txt_sin = model.pos_embed(torch.zeros(N_TXT, 4))
        cos = torch.cat([txt_cos, img_cos], dim=0)[:, 0::2][None, :, None, :]
        sin = torch.cat([txt_sin, img_sin], dim=0)[:, 0::2][None, :, None, :]
        ref = dit(img_tokens, enc_hidden, temb, cos, sin)

    prep = Prep(dit).eval()
    doubles, start = [], 0
    for count in DOUBLE_SPLIT:
        blocks = list(dit.doubles[start:start + count])
        doubles.append(DoubleChunk(blocks).eval())
        start += count
    singles = [SingleChunk(list(dit.singles[i:i + SINGLE_CHUNK])).eval()
               for i in range(0, len(dit.singles), SINGLE_CHUNK)]
    final = Final(dit, N_TXT).eval()

    with torch.no_grad():
        hidden, encoder, mod_i, mod_t, mod_s = prep(
            img_tokens, enc_hidden, temb)
        for chunk in doubles:
            hidden, encoder = chunk(hidden, encoder, cos, sin, mod_i, mod_t)
        joint = torch.cat([encoder, hidden], dim=1)      # host-side concat
        for chunk in singles:
            joint = chunk(joint, cos, sin, mod_s)
        out = final(joint, temb)

    corr = torch.corrcoef(torch.stack([ref.flatten(), out.flatten()]))[0, 1]
    n_chunks = 1 + len(doubles) + len(singles) + 1
    print(f"[compose] {n_chunks} chunks sequential vs monolithic: "
          f"corr {corr:.6f}  max|diff| {(ref - out).abs().max():.2e}")

    if "--export" in sys.argv:
        q_export(prep, (img_tokens, enc_hidden, temb), f"{prefix}_prep")
        for i, chunk in enumerate(doubles):
            q_export(chunk, (hidden, encoder, cos, sin, mod_i, mod_t),
                     f"{prefix}_double{i}")
        for i, chunk in enumerate(singles):
            q_export(chunk, (joint, cos, sin, mod_s), f"{prefix}_single{i}")
        q_export(final, (joint, temb), f"{prefix}_final")


if __name__ == "__main__":
    main()
