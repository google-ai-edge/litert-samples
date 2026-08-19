# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Exports the SAM 2.1 Hiera-Tiny VIDEO path to LiteRT CompiledModel GPU.

The tracking loop of ``facebook/sam2.1-hiera-tiny`` becomes four stateless,
fixed-shape per-frame graphs; the rolling memory bank and the orchestration run
host-side (see verify_video.py for the numpy reference of that loop). Each graph
uses a single flat float32 input and output so CompiledModel never has to
disambiguate tensor order.

  encode      image (1,3,1024,1024) -> [pix_raw | hi0 | hi1]
              pix_raw is the Hiera top level WITHOUT the no-memory embedding
              (the image path bakes it in; the video path needs the raw feature
              for both memory attention and the memory encoder).
  memcond{N}  [pix_raw | memory bank | temporal pos | pointers | key mask]
              -> pix_feat.  Memory attention over a FIXED bank of N spatial
              slots (N x 4096 tokens) + 16 object pointers (x4 tokens); unused
              slots are masked, which equals HF's variable-length bank exactly.
              Exported for N in {7, 2} (SAM2 default and a 2-slot variant).
  decode      [pix_feat | hi0 | hi1 | sparse | nomem] -> masks | iou |
              object pointers | object score.  nomem=1 adds the no-memory
              embedding for the initial conditioning frame; the host picks the
              best-IoU mask among tokens 1..3.
  memorize    [pix_raw | mask_for_mem | occ] -> spatial memory (4096 x 64).

The memory attention is re-authored batch-first (rank 4) so it avoids the
ML Drift rank-3 batched-attention miscompute; verify_video.py confirms the
export is exact under fp32 GPU compute.

Usage (convert env: litert-torch, transformers>=5.13, ai-edge-litert,
ai-edge-quantizer):
    SAM2_OUT=out python export_video.py
"""
import os

import numpy as np
import torch
import torch.nn as nn
import transformers.models.sam2_video.modeling_sam2_video as MV
from transformers import Sam2VideoModel

import hiera_gpu_clean as hiera

hiera.stub_scipy_native_leaves()
hiera.install_hiera_patches()

CKPT = os.environ.get("SAM2_CKPT", "facebook/sam2.1-hiera-tiny")
OUT = os.path.abspath(os.environ.get("SAM2_OUT", "out"))
NMM_LIST = [int(x) for x in os.environ.get("SAM2_NMM", "7,2").split(",")]

IE, F0, F1 = hiera.IE, hiera.F0, hiera.F1
HW = 4096            # 64x64 top-level tokens
MEMCH = 64           # memory channel dim
HD = 256             # memory-attention hidden size (single head)
NPTR_FRAMES = 16     # object-pointer frames kept
PTR_SPLIT = 4        # tokens per pointer (256 / 64)
NPTR = NPTR_FRAMES * PTR_SPLIT
DEC_IN = IE + F0 + F1 + 512 + 1
DEC_OUT = 4 * 65536 + 4 + 4 * 256 + 1
MEM_IN = 2 * IE + 1
MEM_OUT = HW * MEMCH

# Head-dim permutation turning SAM2's pairwise-interleaved RoPE into the
# half-split (rotate_half) form: [even dims | odd dims]. Applied to the q/k
# projection rows so the rotated dot product is unchanged.
PERM = torch.tensor(
    [2 * i for i in range(HD // 2)] + [2 * i + 1 for i in range(HD // 2)])


def rotate_half(x):
    """Rotates the last dim by halves: [-second_half | first_half].

    Args:
        x: Tensor whose last dim is even.

    Returns:
        The half-rotated tensor.
    """
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def deinterleave(c):
    """Turns interleaved cos/sin (L, 256) into a 4-D [c_i | c_i] broadcast.

    Args:
        c: Rotary table (L, 256) with c[2i] == c[2i+1].

    Returns:
        A (1, 1, L, 256) tensor holding the de-interleaved half twice.
    """
    half = c[..., 0::2]
    return torch.cat([half, half], -1)[None, None]


class MemoryAttention4D(nn.Module):
    """Batch-first (<= 4-D) re-authoring of Sam2VideoMemoryAttention.

    Query = current pix_raw (4096 tokens); memory = N spatial slots x 4096
    tokens + NPTR pointer tokens, all 64-channel. Keeping the batch dim dodges
    the ML Drift rank-3 miscompute.
    """

    def __init__(self, ma, nmm, vision_pos, mem_pos):
        """Caches the constant rotary, vision and memory position tables.

        Args:
            ma: The Sam2VideoMemoryAttention module (weights already permuted).
            nmm: Number of spatial memory slots (bank size).
            vision_pos: Top-level vision sine position embedding (1,256,64,64).
            mem_pos: Spatial memory position encoding, token-major (1,4096,64).
        """
        super().__init__()
        self.layers = ma.layers
        self.layer_norm = ma.layer_norm
        self.nmm = nmm
        cos, sin = ma.rotary_emb()
        self.register_buffer("cos", deinterleave(cos.detach().clone()))
        self.register_buffer("sin", deinterleave(sin.detach().clone()))
        vpos = (0.1 * vision_pos).reshape(1, HD, HW).transpose(1, 2)
        self.register_buffer("vpos", vpos.reshape(1, 1, HW, HD).contiguous())
        self.register_buffer(
            "mpos", mem_pos.reshape(1, 1, HW, MEMCH).contiguous())

    def rope(self, x):
        """Applies the cached rotary embedding to a query/key tensor.

        Args:
            x: Tensor (1, 1, L, 256).

        Returns:
            The rotary-embedded tensor.
        """
        return x * self.cos + rotate_half(x) * self.sin

    def forward(self, pix_raw, mem, tpe, ptr_tok, ptr_pos, key_mask):
        """Runs the memory attention over the fixed bank.

        Args:
            pix_raw: Current-frame features (1, 256, 64, 64).
            mem: Spatial memory (N*4096*64,) flat.
            tpe: Per-slot temporal position embeddings (N*64,) flat.
            ptr_tok: Object-pointer tokens (NPTR*64,) flat.
            ptr_pos: Object-pointer position embeddings (NPTR*64,) flat.
            key_mask: Additive mask over N*4096 + NPTR keys.

        Returns:
            The memory-conditioned features (1, IE) flat.
        """
        n = self.nmm
        x = pix_raw.reshape(1, HD, HW).transpose(1, 2)
        x = x.reshape(1, 1, HW, HD) + self.vpos
        spatial = mem.reshape(1, n, HW, MEMCH)
        spatial_pos = self.mpos + tpe.reshape(1, n, 1, MEMCH)
        memory = torch.cat([
            spatial.reshape(1, 1, n * HW, MEMCH),
            ptr_tok.reshape(1, 1, NPTR, MEMCH),
        ], 2)
        mem_pos = torch.cat([
            spatial_pos.reshape(1, 1, n * HW, MEMCH),
            ptr_pos.reshape(1, 1, NPTR, MEMCH),
        ], 2)
        keys_in = memory + mem_pos
        km = key_mask.reshape(1, 1, 1, n * HW + NPTR)
        for layer in self.layers:
            h = layer.layer_norm1(x)
            sa = layer.self_attn
            q = self.rope(sa.q_proj(h))
            k = self.rope(sa.k_proj(h))
            v = sa.v_proj(h)
            a = torch.softmax((q @ k.transpose(-1, -2)) * sa.scaling, dim=-1)
            x = x + sa.o_proj(a @ v)
            h = layer.layer_norm2(x)
            ca = layer.cross_attn_image
            q = self.rope(ca.q_proj(h))
            k = ca.k_proj(keys_in)
            k_sp = k[..., :n * HW, :].reshape(1, n, HW, HD)
            k_sp = k_sp * self.cos + rotate_half(k_sp) * self.sin
            k_sp = k_sp.reshape(1, 1, n * HW, HD)
            k = torch.cat([k_sp, k[..., n * HW:, :]], 2)
            v = ca.v_proj(memory)
            aw = (q @ k.transpose(-1, -2)) * ca.scaling + km
            a = torch.softmax(aw, dim=-1)
            x = x + ca.o_proj(a @ v)
            h = layer.layer_norm3(x)
            x = x + layer.linear2(layer.activation(layer.linear1(h)))
        x = self.layer_norm(x)
        return x.reshape(1, HW, HD).transpose(1, 2).reshape(1, IE)


def _vdec4d(self, image_embeddings, image_positional_embeddings,
            sparse_prompt_embeddings, dense_prompt_embeddings,
            multimask_output, high_resolution_features,
            attention_similarity=None, target_embedding=None, **kwargs):
    """Sam2VideoMaskDecoder.forward kept <= 4-D, returning all 4 mask tokens.

    Args:
        self: The Sam2VideoMaskDecoder module.
        image_embeddings: pix_feat (1, 256, 64, 64).
        image_positional_embeddings: Image-wide sine embedding.
        sparse_prompt_embeddings: Sparse prompt tokens.
        dense_prompt_embeddings: Dense (no-mask) prompt.
        multimask_output: Unused; all mask tokens are emitted.
        high_resolution_features: [feat_s0, feat_s1].
        attention_similarity: Optional PerSAM term.
        target_embedding: Optional PerSAM term.
        **kwargs: Forwarded to the two-way transformer.

    Returns:
        A tuple (masks, iou, mask_tokens, object_score).
    """
    bs, nc, h, w = image_embeddings.shape
    pb = sparse_prompt_embeddings.shape[1]
    output_tokens = torch.cat([
        self.obj_score_token.weight, self.iou_token.weight,
        self.mask_tokens.weight,
    ], 0).repeat(bs, pb, 1, 1)
    point = torch.cat((output_tokens, sparse_prompt_embeddings), 2)
    point = point.to(self.iou_token.weight.dtype)
    image_emb = (image_embeddings + dense_prompt_embeddings)
    image_emb = image_emb.repeat_interleave(pb, 0)
    image_pos = image_positional_embeddings.repeat_interleave(pb, 0)
    point, image_emb = self.transformer(
        point_embeddings=point, image_embeddings=image_emb,
        image_positional_embeddings=image_pos,
        attention_similarity=attention_similarity,
        target_embedding=target_embedding, **kwargs)
    iou_token_out = point[:, :, 1, :]
    mask_tokens_out = point[:, :, 2:(2 + self.num_mask_tokens), :]
    image_emb = image_emb.transpose(2, 3).view(bs * pb, nc, h, w)
    feat_s0, feat_s1 = high_resolution_features
    feat_s0 = feat_s0.repeat_interleave(pb, 0)
    feat_s1 = feat_s1.repeat_interleave(pb, 0)
    upscaled = self.activation(
        self.upscale_layer_norm(self.upscale_conv1(image_emb) + feat_s1))
    upscaled = self.activation(self.upscale_conv2(upscaled) + feat_s0)
    hyper = torch.stack([
        self.output_hypernetworks_mlps[i](mask_tokens_out[:, :, i, :])
        for i in range(self.num_mask_tokens)
    ], 2)
    _, nc2, h2, w2 = upscaled.shape
    batch = bs * pb
    masks = (hyper.view(batch, self.num_mask_tokens, nc2)
             @ upscaled.view(batch, nc2, h2 * w2))
    masks = masks.view(batch, self.num_mask_tokens, h2, w2)
    iou = self.iou_prediction_head(iou_token_out)
    obj = self.pred_obj_score_head(point[:, :, 0, :])
    return masks, iou, mask_tokens_out, obj


MV.Sam2VideoMaskDecoder.forward = _vdec4d


class ConstPos(nn.Module):
    """Returns a constant position encoding regardless of the input shape."""

    def __init__(self, const):
        """Stores the constant tensor.

        Args:
            const: The precomputed position encoding to return.
        """
        super().__init__()
        self.register_buffer("c", const)

    def forward(self, *args, **kwargs):
        """Returns the stored constant."""
        return self.c


def permute_rope_projections(model) -> None:
    """Bakes PERM into every memory-attention q/k projection, once.

    Args:
        model: The Sam2VideoModel whose weights are permuted in place.
    """
    assert not getattr(model, "_rope_permuted", False)
    for layer in model.memory_attention.layers:
        for attn in (layer.self_attn, layer.cross_attn_image):
            for proj in (attn.q_proj, attn.k_proj):
                proj.weight.data = proj.weight.data[PERM].contiguous()
                proj.bias.data = proj.bias.data[PERM].contiguous()
    model._rope_permuted = True


def memcond_in_size(nmm):
    """Returns the flat memcond input length for a bank of ``nmm`` slots.

    Args:
        nmm: Number of spatial memory slots.

    Returns:
        The total float count of the memcond input.
    """
    return (IE + nmm * HW * MEMCH + nmm * MEMCH + 2 * NPTR * MEMCH
            + nmm * HW + NPTR)


def build(model):
    """Builds the four flat-IO graph modules over one loaded model.

    Args:
        model: The Sam2VideoModel (patched, weights permuted).

    Returns:
        A tuple of module classes (Encode, MemCond, Decode, Memorize).
    """
    permute_rope_projections(model)
    md = model.mask_decoder
    md.upscale_conv1 = hiera.ZeroStuffConvT(md.upscale_conv1, 64)
    md.upscale_conv2 = hiera.ZeroStuffConvT(md.upscale_conv2, 128)
    with torch.no_grad():
        ve = model.vision_encoder(
            torch.randn(1, 3, 1024, 1024), return_dict=True)
        vision_pos = ve.fpn_position_encoding[-1].detach().clone()
        mem_pos = model.memory_encoder.position_encoding(
            (1, MEMCH, 64, 64), torch.device("cpu"), torch.float32)
        mem_pos = mem_pos.detach().clone()
        mem_pos_tok = mem_pos.reshape(1, MEMCH, HW).transpose(1, 2)
        mem_pos_tok = mem_pos_tok.contiguous()
        img_pos = model.get_image_wide_positional_embeddings().detach().clone()
    model.memory_encoder.position_encoding = ConstPos(mem_pos)

    class Encode(nn.Module):
        """Hiera encoder -> [pix_raw | hi0 | hi1] flat."""

        def __init__(self):
            super().__init__()
            self.ve = model.vision_encoder
            self.cs0, self.cs1 = md.conv_s0, md.conv_s1

        def forward(self, x):
            fpn = self.ve(x, return_dict=True).fpn_hidden_states
            return torch.cat([
                fpn[2].reshape(-1),
                self.cs0(fpn[0]).reshape(-1),
                self.cs1(fpn[1]).reshape(-1),
            ])[None]

    class MemCond(nn.Module):
        """Memory attention over a fixed bank -> pix_feat flat."""

        def __init__(self, nmm):
            super().__init__()
            self.nmm = nmm
            self.attn = MemoryAttention4D(
                model.memory_attention, nmm, vision_pos, mem_pos_tok)

        def forward(self, flat):
            n = self.nmm
            f = flat[0]
            o = 0
            pix = f[o:o + IE].reshape(1, HD, 64, 64)
            o += IE
            mem = f[o:o + n * HW * MEMCH]
            o += n * HW * MEMCH
            tpe = f[o:o + n * MEMCH]
            o += n * MEMCH
            ptr = f[o:o + NPTR * MEMCH]
            o += NPTR * MEMCH
            ppos = f[o:o + NPTR * MEMCH]
            o += NPTR * MEMCH
            km = f[o:o + n * HW + NPTR]
            return self.attn(pix, mem, tpe, ptr, ppos, km)

    class Decode(nn.Module):
        """Prompt-conditioned video mask decoder + object-pointer proj."""

        def __init__(self):
            super().__init__()
            self.d = md
            self.optr = model.object_pointer_proj
            self.register_buffer("ipe", img_pos)
            dense = model.prompt_encoder.no_mask_embed.weight
            dense = dense.reshape(1, -1, 1, 1).expand(1, 256, 64, 64)
            self.register_buffer("dense", dense.contiguous())
            no_mem = model.no_memory_embedding.detach()
            self.register_buffer("no_mem", no_mem.reshape(1, HD, 1, 1))

        def forward(self, flat):
            f = flat[0]
            pix = f[:IE].reshape(1, HD, 64, 64)
            h0 = f[IE:IE + F0].reshape(1, 32, 256, 256)
            h1 = f[IE + F0:IE + F0 + F1].reshape(1, 64, 128, 128)
            sparse = f[IE + F0 + F1:IE + F0 + F1 + 512]
            sparse = sparse.reshape(1, 1, 2, 256)
            nomem = f[IE + F0 + F1 + 512:].reshape(1, 1, 1, 1)
            pix = pix + nomem * self.no_mem
            masks, iou, tok, obj = self.d(
                pix, self.ipe, sparse, self.dense, multimask_output=True,
                high_resolution_features=[h0, h1])
            ptr = self.optr(tok)
            return torch.cat([
                masks[0].reshape(-1), iou[0].reshape(-1),
                ptr[0].reshape(-1), obj[0].reshape(-1),
            ])[None]

    class Memorize(nn.Module):
        """Memory encoder -> spatial memory (4096 x 64) flat."""

        def __init__(self):
            super().__init__()
            self.e = model.memory_encoder
            occ = model.occlusion_spatial_embedding_parameter.detach()
            self.register_buffer("occ", occ.reshape(1, MEMCH))

        def forward(self, flat):
            f = flat[0]
            pix = f[:IE].reshape(1, HD, 64, 64)
            mfm = f[IE:2 * IE].reshape(1, 1, 1024, 1024)
            occ = f[2 * IE:].reshape(1, 1)
            mem, _ = self.e(pix, mfm)
            mem = mem.reshape(1, MEMCH, HW).transpose(1, 2)
            mem = mem + occ * self.occ
            return mem.reshape(1, HW * MEMCH)

    return Encode, MemCond, Decode, Memorize


def save_constants(model) -> None:
    """Writes the host-side constants (prompt, temporal PE, projections).

    Args:
        model: The loaded Sam2VideoModel.
    """
    pe = model.prompt_encoder
    gauss = pe.shared_embedding.positional_embedding
    prompt = np.concatenate([
        gauss.detach().numpy().flatten(),
        pe.point_embed.weight[1].detach().numpy(),
        pe.point_embed.weight[0].detach().numpy(),
        pe.not_a_point_embed.weight[0].detach().numpy(),
    ]).astype(np.float32)
    prompt.tofile(f"{OUT}/sam2v_prompt.bin")
    with torch.no_grad():
        track, _ = pe(
            input_points=torch.zeros(1, 1, 1, 2),
            input_labels=-torch.ones(1, 1, 1, dtype=torch.int32),
            input_boxes=None, input_masks=None)
    track.numpy().astype(np.float32).tofile(f"{OUT}/sam2v_track_sparse.bin")
    mtpe = model.memory_temporal_positional_encoding.detach().numpy()
    mtpe.astype(np.float32).tofile(f"{OUT}/sam2v_mtpe.bin")
    no_obj = model.no_object_pointer.detach().numpy()
    no_obj.astype(np.float32).tofile(f"{OUT}/sam2v_no_obj_ptr.bin")
    lin = model.temporal_positional_encoding_projection_layer
    proj = np.concatenate([
        lin.weight.detach().numpy().flatten(),
        lin.bias.detach().numpy(),
    ]).astype(np.float32)
    proj.tofile(f"{OUT}/sam2v_tpos_proj.bin")
    print("constants: prompt track_sparse mtpe no_obj_ptr tpos_proj")


def convert(module, example, name) -> None:
    """Traces, exports, op-checks and fp16-quantizes one graph.

    Args:
        module: The nn.Module to export.
        example: A tuple of example inputs for tracing.
        name: The output basename (no extension).
    """
    import litert_torch
    fp32 = f"{OUT}/{name}_fp32.tflite"
    with torch.no_grad():
        module(*example)
    args = tuple(t.detach().clone() for t in example)
    litert_torch.convert(module, args).export(fp32)
    hiera.opcheck(fp32, name)
    size = hiera.fp16(fp32, f"{OUT}/{name}.tflite")
    print(f"{name} fp16 {size:.1f} MB")
    os.remove(fp32)


def main() -> None:
    """Exports every graph plus the host constants."""
    os.makedirs(OUT, exist_ok=True)
    model = Sam2VideoModel.from_pretrained(CKPT).eval()
    print("baked pos_embed:", hiera.bake_pos_embed(model))
    encode_cls, memcond_cls, decode_cls, memorize_cls = build(model)
    save_constants(model)
    which = os.environ.get(
        "SAM2_GRAPHS", "encode,memcond,decode,memorize").split(",")
    if "encode" in which:
        convert(
            encode_cls().eval(),
            (torch.randn(1, 3, 1024, 1024),), "sam2v_encode")
    if "memcond" in which:
        for n in NMM_LIST:
            convert(
                memcond_cls(n).eval(),
                (torch.randn(1, memcond_in_size(n)),), f"sam2v_memcond{n}")
    if "decode" in which:
        convert(
            decode_cls().eval(), (torch.randn(1, DEC_IN),), "sam2v_decode")
    if "memorize" in which:
        convert(
            memorize_cls().eval(), (torch.randn(1, MEM_IN),), "sam2v_memorize")


if __name__ == "__main__":
    main()
