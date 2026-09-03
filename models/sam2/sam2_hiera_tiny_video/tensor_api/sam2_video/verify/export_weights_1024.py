#!/usr/bin/env python3
"""Export facebook/sam2.1-hiera-tiny (video stack included) for the Tensor API
video example, fp32 at the 1024x1024 native resolution.

Writes ONE safetensors file whose keys follow the original SAM2 checkpoint
naming (trunk./neck./sam_prompt_encoder./sam_mask_decoder./memory_attention./
memory_encoder./obj_ptr_proj./...), the same convention as the 512 image-path
file, plus derived tables under "tables.":

  trunk.pos_embed_full        [1,256,256,96]  bicubic(base 7x7 -> 256) + tiled
                              8x8 window, composed once at the 1024 grid (the
                              network's native size, so no re-composition trick
                              is needed -- this is what the checkpoint encodes).
  memory_attention q/k        stored PERM-PERMUTED: the head-dim permutation
                              [even dims | odd dims] baked into the projection
                              rows turns SAM2's pairwise-interleaved RoPE into
                              the rotate-half form, so q'.k' == q.k exactly.
                              The graph must apply rotate-half RoPE with the
                              deinterleaved tables below.
  tables.rope_cos / rope_sin  [4096,256] deinterleaved ([c_half | c_half]) from
                              the model's own precomputed rotary buffers.
  tables.vision_pos_scaled    [4096,256] 0.1 * top-level vision sine position
                              encoding, token-major (row-major 64x64 grid).
  tables.mem_pos              [4096,64] memory-channel sine position encoding,
                              token-major.
  tables.track_sparse         [2,256] the sparse prompt used on non-prompted
                              tracking frames (label -1 "no point" encoding).

All conv weights are permuted to the TFLite OHWI layout (depthwise stays
per-channel-planes [C,kh,kw,1]; transposed convs to [O,kh,kw,I]).

Cross-check: every key shared with the verified 512 file (which stores the
same weights in fp16) must round-trip: fp16(exported) == stored fp16. The two
expected exceptions are trunk.pos_embed_full (different grid by design) and
the permuted memory-attention q/k rows (compared after inverse permutation).

Run (any venv with torch + transformers>=5 + safetensors + numpy):
  python export_weights_1024.py --out sam2_tiny_1024_video.safetensors \
      [--check_512 sam2_tiny_512_clean.safetensors]
"""
import argparse

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file
from transformers import Sam2VideoModel

CKPT = "facebook/sam2.1-hiera-tiny"
HD = 256  # memory-attention hidden (single head)
PERM = np.concatenate([np.arange(0, HD, 2), np.arange(1, HD, 2)])


def ohwi(w):  # torch conv OIHW -> TFLite OHWI
    return w.permute(0, 2, 3, 1).contiguous()


def t_ohwi(w):  # torch ConvTranspose2d IOHW -> [O,kh,kw,I]
    return w.permute(1, 2, 3, 0).contiguous()


def feedforward(dst, out, hf, prefix, mlx_prefix):
    """Sam2VideoFeedForward(depth 3) -> layers.0/1/2 naming."""
    out[f"{mlx_prefix}.layers.0.weight"] = hf[f"{prefix}.proj_in.weight"]
    out[f"{mlx_prefix}.layers.0.bias"] = hf[f"{prefix}.proj_in.bias"]
    out[f"{mlx_prefix}.layers.1.weight"] = hf[f"{prefix}.layers.0.weight"]
    out[f"{mlx_prefix}.layers.1.bias"] = hf[f"{prefix}.layers.0.bias"]
    out[f"{mlx_prefix}.layers.2.weight"] = hf[f"{prefix}.proj_out.weight"]
    out[f"{mlx_prefix}.layers.2.bias"] = hf[f"{prefix}.proj_out.bias"]
    del dst  # coverage bookkeeping happens on `hf` at the end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sam2_tiny_1024_video.safetensors")
    ap.add_argument("--check_512", default="")
    a = ap.parse_args()

    model = Sam2VideoModel.from_pretrained(CKPT).eval()
    hf = {k: v.detach().clone() for k, v in model.state_dict().items()}
    used = set()

    def take(key):
        used.add(key)
        return hf[key]

    out = {}

    # ---------------------------------------------------------------- trunk
    backbone = model.vision_encoder.backbone
    with torch.no_grad():
        embedded = backbone.patch_embed(torch.randn(1, 3, 1024, 1024))
        # _get_pos_embed returns a permuted VIEW; without .contiguous() the
        # strided buffer round-trips scrambled through numpy(order='K') +
        # safetensors (which writes the raw buffer). Found the hard way.
        pos = backbone._get_pos_embed(embedded.shape[1:3]).detach().contiguous().clone()
    assert pos.shape == (1, 256, 256, 96), pos.shape
    out["trunk.pos_embed_full"] = pos
    used.update({"vision_encoder.backbone.pos_embed",
                 "vision_encoder.backbone.pos_embed_window"})
    out["trunk.patch_embed.proj.weight"] = ohwi(
        take("vision_encoder.backbone.patch_embed.projection.weight"))
    out["trunk.patch_embed.proj.bias"] = take(
        "vision_encoder.backbone.patch_embed.projection.bias")
    n_blocks = sum(1 for k in hf if k.startswith("vision_encoder.backbone.blocks.")
                   and k.endswith(".attn.qkv.weight"))
    for i in range(n_blocks):
        s = f"vision_encoder.backbone.blocks.{i}"
        d = f"trunk.blocks.{i}"
        out[f"{d}.norm1.weight"] = take(f"{s}.layer_norm1.weight")
        out[f"{d}.norm1.bias"] = take(f"{s}.layer_norm1.bias")
        out[f"{d}.norm2.weight"] = take(f"{s}.layer_norm2.weight")
        out[f"{d}.norm2.bias"] = take(f"{s}.layer_norm2.bias")
        for p in ("attn.qkv", "attn.proj"):
            out[f"{d}.{p}.weight"] = take(f"{s}.{p}.weight")
            out[f"{d}.{p}.bias"] = take(f"{s}.{p}.bias")
        if f"{s}.proj.weight" in hf:
            out[f"{d}.proj.weight"] = take(f"{s}.proj.weight")
            out[f"{d}.proj.bias"] = take(f"{s}.proj.bias")
        out[f"{d}.mlp.layers.0.weight"] = take(f"{s}.mlp.proj_in.weight")
        out[f"{d}.mlp.layers.0.bias"] = take(f"{s}.mlp.proj_in.bias")
        out[f"{d}.mlp.layers.1.weight"] = take(f"{s}.mlp.proj_out.weight")
        out[f"{d}.mlp.layers.1.bias"] = take(f"{s}.mlp.proj_out.bias")

    # ----------------------------------------------------------------- neck
    for i in range(4):
        out[f"neck.convs.{i}.weight"] = ohwi(
            take(f"vision_encoder.neck.convs.{i}.weight"))
        out[f"neck.convs.{i}.bias"] = take(f"vision_encoder.neck.convs.{i}.bias")
    out["no_mem_embed"] = take("no_memory_embedding")

    # ------------------------------------------------------- prompt encoder
    out["sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"] = (
        take("shared_image_embedding.positional_embedding"))
    used.add("prompt_encoder.shared_embedding.positional_embedding")  # same tensor
    point_embed = take("prompt_encoder.point_embed.weight")  # (4, 256) fused
    for i in range(4):
        out[f"sam_prompt_encoder.point_embeddings.{i}.weight"] = (
            point_embed[i:i + 1].contiguous())
    out["sam_prompt_encoder.not_a_point_embed.weight"] = take(
        "prompt_encoder.not_a_point_embed.weight")
    out["sam_prompt_encoder.no_mask_embed.weight"] = take(
        "prompt_encoder.no_mask_embed.weight")

    # --------------------------------------------------------- mask decoder
    md = "mask_decoder"
    smd = "sam_mask_decoder"
    out[f"{smd}.conv_s0.weight"] = ohwi(take(f"{md}.conv_s0.weight"))
    out[f"{smd}.conv_s0.bias"] = take(f"{md}.conv_s0.bias")
    out[f"{smd}.conv_s1.weight"] = ohwi(take(f"{md}.conv_s1.weight"))
    out[f"{smd}.conv_s1.bias"] = take(f"{md}.conv_s1.bias")
    for tok in ("iou_token", "mask_tokens", "obj_score_token"):
        out[f"{smd}.{tok}.weight"] = take(f"{md}.{tok}.weight")
    for l in range(2):
        s = f"{md}.transformer.layers.{l}"
        d = f"{smd}.transformer.layers.{l}"
        for attn in ("self_attn", "cross_attn_token_to_image",
                     "cross_attn_image_to_token"):
            for hp, p in (("q_proj", "q_proj"), ("k_proj", "k_proj"),
                          ("v_proj", "v_proj"), ("o_proj", "out_proj")):
                out[f"{d}.{attn}.{p}.weight"] = take(f"{s}.{attn}.{hp}.weight")
                out[f"{d}.{attn}.{p}.bias"] = take(f"{s}.{attn}.{hp}.bias")
        out[f"{d}.mlp.layers.0.weight"] = take(f"{s}.mlp.proj_in.weight")
        out[f"{d}.mlp.layers.0.bias"] = take(f"{s}.mlp.proj_in.bias")
        out[f"{d}.mlp.layers.1.weight"] = take(f"{s}.mlp.proj_out.weight")
        out[f"{d}.mlp.layers.1.bias"] = take(f"{s}.mlp.proj_out.bias")
        for n in range(1, 5):
            out[f"{d}.norm{n}.weight"] = take(f"{s}.layer_norm{n}.weight")
            out[f"{d}.norm{n}.bias"] = take(f"{s}.layer_norm{n}.bias")
    for hp, p in (("q_proj", "q_proj"), ("k_proj", "k_proj"),
                  ("v_proj", "v_proj"), ("o_proj", "out_proj")):
        out[f"{smd}.transformer.final_attn_token_to_image.{p}.weight"] = take(
            f"{md}.transformer.final_attn_token_to_image.{hp}.weight")
        out[f"{smd}.transformer.final_attn_token_to_image.{p}.bias"] = take(
            f"{md}.transformer.final_attn_token_to_image.{hp}.bias")
    out[f"{smd}.transformer.norm_final_attn.weight"] = take(
        f"{md}.transformer.layer_norm_final_attn.weight")
    out[f"{smd}.transformer.norm_final_attn.bias"] = take(
        f"{md}.transformer.layer_norm_final_attn.bias")
    out[f"{smd}.output_upscaling_0.weight"] = t_ohwi(
        take(f"{md}.upscale_conv1.weight"))
    out[f"{smd}.output_upscaling_0.bias"] = take(f"{md}.upscale_conv1.bias")
    out[f"{smd}.output_upscaling_1.weight"] = take(
        f"{md}.upscale_layer_norm.weight")
    out[f"{smd}.output_upscaling_1.bias"] = take(f"{md}.upscale_layer_norm.bias")
    out[f"{smd}.output_upscaling_3.weight"] = t_ohwi(
        take(f"{md}.upscale_conv2.weight"))
    out[f"{smd}.output_upscaling_3.bias"] = take(f"{md}.upscale_conv2.bias")
    for i in range(4):
        feedforward(out, out, hf, f"{md}.output_hypernetworks_mlps.{i}",
                    f"{smd}.output_hypernetworks_mlps.{i}")
        used.update({f"{md}.output_hypernetworks_mlps.{i}.{t}" for t in
                     ("proj_in.weight", "proj_in.bias", "layers.0.weight",
                      "layers.0.bias", "proj_out.weight", "proj_out.bias")})
    for head, mlx in (("iou_prediction_head", f"{smd}.iou_prediction_head"),
                      ("pred_obj_score_head", f"{smd}.pred_obj_score_head")):
        feedforward(out, out, hf, f"{md}.{head}", mlx)
        used.update({f"{md}.{head}.{t}" for t in
                     ("proj_in.weight", "proj_in.bias", "layers.0.weight",
                      "layers.0.bias", "proj_out.weight", "proj_out.bias")})

    # ----------------------------------------------------- memory attention
    n_ma = sum(1 for k in hf if k.startswith("memory_attention.layers.")
               and k.endswith(".self_attn.q_proj.weight"))
    perm_t = torch.from_numpy(PERM)
    for i in range(n_ma):
        s = f"memory_attention.layers.{i}"
        for attn in ("self_attn", "cross_attn_image"):
            for p in ("q_proj", "k_proj"):  # PERM-permuted rows (RoPE bake)
                out[f"{s}.{attn}.{p}.weight"] = take(
                    f"{s}.{attn}.{p}.weight")[perm_t].contiguous()
                out[f"{s}.{attn}.{p}.bias"] = take(
                    f"{s}.{attn}.{p}.bias")[perm_t].contiguous()
            out[f"{s}.{attn}.v_proj.weight"] = take(f"{s}.{attn}.v_proj.weight")
            out[f"{s}.{attn}.v_proj.bias"] = take(f"{s}.{attn}.v_proj.bias")
            out[f"{s}.{attn}.out_proj.weight"] = take(f"{s}.{attn}.o_proj.weight")
            out[f"{s}.{attn}.out_proj.bias"] = take(f"{s}.{attn}.o_proj.bias")
        for p in ("linear1", "linear2"):
            out[f"{s}.{p}.weight"] = take(f"{s}.{p}.weight")
            out[f"{s}.{p}.bias"] = take(f"{s}.{p}.bias")
        for n in range(1, 4):
            out[f"{s}.norm{n}.weight"] = take(f"{s}.layer_norm{n}.weight")
            out[f"{s}.norm{n}.bias"] = take(f"{s}.layer_norm{n}.bias")
    out["memory_attention.norm.weight"] = take("memory_attention.layer_norm.weight")
    out["memory_attention.norm.bias"] = take("memory_attention.layer_norm.bias")

    # ------------------------------------------------------- memory encoder
    me = "memory_encoder"
    for i in range(4):
        out[f"{me}.mask_downsampler.conv{i}.weight"] = ohwi(
            take(f"{me}.mask_downsampler.layers.{i}.conv.weight"))
        out[f"{me}.mask_downsampler.conv{i}.bias"] = take(
            f"{me}.mask_downsampler.layers.{i}.conv.bias")
        out[f"{me}.mask_downsampler.norm{i}.weight"] = take(
            f"{me}.mask_downsampler.layers.{i}.layer_norm.weight")
        out[f"{me}.mask_downsampler.norm{i}.bias"] = take(
            f"{me}.mask_downsampler.layers.{i}.layer_norm.bias")
    out[f"{me}.mask_downsampler.conv4.weight"] = ohwi(
        take(f"{me}.mask_downsampler.final_conv.weight"))
    out[f"{me}.mask_downsampler.conv4.bias"] = take(
        f"{me}.mask_downsampler.final_conv.bias")
    out[f"{me}.pix_feat_proj.weight"] = ohwi(take(f"{me}.feature_projection.weight"))
    out[f"{me}.pix_feat_proj.bias"] = take(f"{me}.feature_projection.bias")
    for i in range(2):
        s = f"{me}.memory_fuser.layers.{i}"
        d = f"{me}.fuser.{i}"
        out[f"{d}.dwconv.weight"] = ohwi(take(f"{s}.depthwise_conv.weight"))
        out[f"{d}.dwconv.bias"] = take(f"{s}.depthwise_conv.bias")
        out[f"{d}.norm.weight"] = take(f"{s}.layer_norm.weight")
        out[f"{d}.norm.bias"] = take(f"{s}.layer_norm.bias")
        out[f"{d}.pwconv1.weight"] = take(f"{s}.pointwise_conv1.weight")
        out[f"{d}.pwconv1.bias"] = take(f"{s}.pointwise_conv1.bias")
        out[f"{d}.pwconv2.weight"] = take(f"{s}.pointwise_conv2.weight")
        out[f"{d}.pwconv2.bias"] = take(f"{s}.pointwise_conv2.bias")
        out[f"{d}.gamma"] = take(f"{s}.scale")
    out[f"{me}.out_proj.weight"] = ohwi(take(f"{me}.projection.weight"))
    out[f"{me}.out_proj.bias"] = take(f"{me}.projection.bias")

    # ---------------------------------------------------- video-path extras
    out["maskmem_tpos_enc"] = take("memory_temporal_positional_encoding")
    out["no_obj_ptr"] = take("no_object_pointer")
    out["no_obj_embed_spatial"] = take("occlusion_spatial_embedding_parameter")
    feedforward(out, out, hf, "object_pointer_proj", "obj_ptr_proj")
    used.update({f"object_pointer_proj.{t}" for t in
                 ("proj_in.weight", "proj_in.bias", "layers.0.weight",
                  "layers.0.bias", "proj_out.weight", "proj_out.bias")})
    out["obj_ptr_tpos_proj.weight"] = take(
        "temporal_positional_encoding_projection_layer.weight")
    out["obj_ptr_tpos_proj.bias"] = take(
        "temporal_positional_encoding_projection_layer.bias")

    # ---------------------------------------------------------------- tables
    ma = model.memory_attention
    cos = ma.rotary_emb.rope_embeddings_cos.detach().clone()  # (4096, 256)
    sin = ma.rotary_emb.rope_embeddings_sin.detach().clone()
    assert cos.shape == (4096, HD), cos.shape
    assert torch.equal(cos[:, 0::2], cos[:, 1::2]), "expected pairwise-equal"

    def deinterleave(c):
        half = c[:, 0::2]
        return torch.cat([half, half], dim=-1).contiguous()

    out["tables.rope_cos"] = deinterleave(cos)
    out["tables.rope_sin"] = deinterleave(sin)

    # Sanity: permuted projections + rotate-half == original + pairwise RoPE.
    def rotate_half(x):
        return torch.cat([-x[..., HD // 2:], x[..., :HD // 2]], dim=-1)

    def rotate_pairwise(x):
        v = x.view(*x.shape[:-1], -1, 2)
        return torch.stack((-v[..., 1], v[..., 0]), dim=-1).flatten(-2)

    g = torch.Generator().manual_seed(0)
    q = torch.randn(8, HD, generator=g)
    k = torch.randn(8, HD, generator=g)
    ref = ((q * cos[:8] + rotate_pairwise(q) * sin[:8]) @
           (k * cos[:8] + rotate_pairwise(k) * sin[:8]).T)
    qp, kp = q[:, PERM], k[:, PERM]
    got = ((qp * out["tables.rope_cos"][:8] + rotate_half(qp) * out["tables.rope_sin"][:8]) @
           (kp * out["tables.rope_cos"][:8] + rotate_half(kp) * out["tables.rope_sin"][:8]).T)
    rope_err = (ref - got).abs().max().item()
    assert rope_err < 1e-4, rope_err

    with torch.no_grad():
        ve = model.vision_encoder(torch.randn(1, 3, 1024, 1024), return_dict=True)
        vision_pos = ve.fpn_position_encoding[-1].detach().clone()  # (1,256,64,64)
        mem_pos = model.memory_encoder.position_encoding(
            (1, 64, 64, 64), torch.device("cpu"), torch.float32).detach().clone()
        ts, _ = model.prompt_encoder(
            input_points=torch.zeros(1, 1, 1, 2),
            input_labels=-torch.ones(1, 1, 1, dtype=torch.int32),
            input_boxes=None, input_masks=None)
    out["tables.vision_pos_scaled"] = (0.1 * vision_pos).reshape(
        HD, 4096).T.contiguous()                       # (4096, 256) token-major
    out["tables.mem_pos"] = mem_pos.reshape(64, 4096).T.contiguous()  # (4096,64)
    out["tables.track_sparse"] = ts.reshape(2, HD).contiguous()

    # ------------------------------------------------------- save + coverage
    # ascontiguousarray: numpy astype(order='K') preserves permuted strides
    # and safetensors then writes the raw buffer in the wrong order.
    arrays = {k: np.ascontiguousarray(v.detach().numpy().astype(np.float32))
              for k, v in out.items()}
    for k, v in arrays.items():
        assert v.flags["C_CONTIGUOUS"], k
    save_file(arrays, a.out)
    unused = sorted(k for k in hf if k not in used)
    print(f"saved {a.out}: {len(arrays)} tensors "
          f"({sum(v.nbytes for v in arrays.values()) / 1e6:.1f} MB fp32)")
    print(f"rope bake check: max |q.k diff| = {rope_err:.2e}")
    print(f"unused HF keys ({len(unused)}):")
    for k in unused:
        print("  ", k, tuple(hf[k].shape))

    # ----------------------------------------------- cross-check vs 512 file
    if a.check_512:
        inv_perm = np.argsort(PERM)
        bad, checked, skipped = [], 0, []
        with safe_open(a.check_512, framework="np") as f:
            keys512 = set(f.keys())
            for key in sorted(keys512):
                if key not in arrays:
                    skipped.append(key)
                    continue
                if key == "trunk.pos_embed_full":  # different grid by design
                    continue
                ref16 = f.get_tensor(key)
                mine = arrays[key]
                if (key.startswith("memory_attention.layers.") and
                        (".q_proj." in key or ".k_proj." in key)):
                    mine = mine[inv_perm]  # stored permuted; undo for compare
                if ref16.shape != mine.shape:
                    bad.append((key, "shape", ref16.shape, mine.shape))
                    continue
                if not np.array_equal(mine.astype(np.float16), ref16):
                    d = np.abs(mine - ref16.astype(np.float32)).max()
                    bad.append((key, "value", float(d), None))
                checked += 1
        print(f"\n512-file cross-check: {checked} keys compared, "
              f"{len(bad)} mismatches, {len(skipped)} keys only in 512 file")
        for b in bad[:20]:
            print("  MISMATCH", b)
        for k in skipped:
            print("  512-only:", k)
        if bad:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
