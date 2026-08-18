# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SAM 2.1 (hiera-tiny) mask decoder -> LiteRT GPU-clean .tflite.

Bucket 1: model-side re-authoring only.

Phase-2 companion to convert_sam2.py (the image encoder). This converts
the prompt-conditioned mask DECODER. The tiny prompt-encoder (point ->
sparse tokens, sin/cos) is done HOST-SIDE in Kotlin (see emit at the
end) so the GPU graph stays sin/cos-free; the decoder takes `sparse` as
an input.

Walls re-authored (all model-side; no converter patch):
  1. Sam2Attention (7x: 2 blocks x 3 + 1 final)  : 4D fused attn ->
     rank-4 batched SDPA [1, heads, N, d]. The leading batch dim MUST
     be kept -- see "GPU correctness" below.
  2. ConvTranspose2d (upscale_conv1/2)           : -> ZeroStuffConvT
     (exact zero-stuff + Conv2d), TRANSPOSE_CONV-free
  3. mask head (hyper_in @ upscaled)             : kept <=4D (no
     [1,1,4,256,256] 5D tensor); batched bmm [1,4,32] @ [1,32,65536]
  4. LayerNorm (9x)                              : SafeLayerNorm
     (scale-before-square), fp16-overflow-safe, exact
  5. image_positional_embeddings + no-mask dense : baked CONSTANT
     buffers (host doesn't supply them)
  6. multimask_output=True path                  : static slice [1:],
     no dynamic-stability argmax/gather/where

GPU correctness (Pixel 8a, device A/B vs the CPU reference):
  Writing the attention with the batch dim collapsed -- q/k/v shaped
  [heads, N, d] (rank 3) -- produces a graph that compiles, delegates
  fully (banned ops NONE, >4-D tensors 0) and runs without error, yet
  returns SILENTLY WRONG masks: corr 0.265 vs CPU. It is not an fp16
  problem (forcing fp32 GPU compute still gives corr 0.473) and not
  LayerNorm (plain and overflow-safe LN give the same wrong result).
  Keeping the batch dim (rank 4) restores corr 0.9998 and is ~20%
  faster (6.8 ms vs 8.5 ms). The image encoder's rank-3 SDPA happens to
  be GPU-correct, so a healthy sibling graph proves nothing: only a
  numeric GPU-vs-CPU check on device catches this.

Decoder I/O (single point prompt):
  inputs : image_embeddings [1,256,64,64], sparse [1,2,256],
           feat_s1 [1,64,128,128], feat_s0 [1,32,256,256]
  outputs: pred_masks [1,3,256,256] (logits, 3 multimask),
           iou_scores [1,3]

Run:
  python convert_sam2_decoder.py            # eager parity vs reference
  python convert_sam2_decoder.py --convert  # + convert + gate + fp16
"""
import os
import sys
import types
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# macOS scipy stub (same as convert_sam2.py)
_svdp = types.ModuleType("scipy.sparse.linalg._svdp")
_svdp._svdp = lambda *a, **k: None
sys.modules["scipy.sparse.linalg._svdp"] = _svdp
_opt = types.ModuleType("scipy.optimize")
_opt.linear_sum_assignment = lambda *a, **k: (None, None)
sys.modules["scipy.optimize"] = _opt

from transformers import Sam2Model

MODEL_ID = "facebook/sam2.1-hiera-tiny"
SCRATCH = os.environ.get("SAM2_OUT", "/tmp/sam2_out")


# ----- LayerNorm. SafeLayerNorm (scale-before-square) protects the
#       encoder's huge deep-stage activations from fp16 variance
#       overflow, but the decoder's activations are normal-scale, where
#       the down-scaling instead HURTS GPU fp16 (device A/B: SafeLN
#       decoder masks the background). PLAIN_LN=1 uses stock LayerNorm
#       for the decoder. -----
_PLAIN_LN = os.environ.get("PLAIN_LN") == "1"


def safe_ln(x, weight, bias, eps, sc=0.03125):
    """Overflow-safe LayerNorm over the last dim (or stock if PLAIN_LN).

    Args:
        x: Input tensor; normalization is over the last dimension.
        weight: Per-channel scale tensor.
        bias: Per-channel shift tensor.
        eps: Numerical epsilon added to the variance.
        sc: Pre-square down-scale factor for fp16 overflow safety.

    Returns:
        The normalized tensor, same shape as x.
    """
    if _PLAIN_LN:
        xc = x - x.mean(-1, keepdim=True)
        var = (xc * xc).mean(-1, keepdim=True)
        return xc * torch.rsqrt(var + eps) * weight + bias
    xc = x - x.mean(-1, keepdim=True)
    xs = xc * sc
    var = (xs * xs).mean(-1, keepdim=True) / (sc * sc)
    return xc * torch.rsqrt(var + eps) * weight + bias


# ----- ZeroStuffConvT: ConvTranspose2d(k=s,stride=s) ==
#       zero-stuff(nearest x top-left mask) + Conv2d(flipped w) -----
class ZeroStuffConvT(nn.Module):
    def __init__(self, ct, H, W):
        super().__init__()
        self.s = ct.stride[0]
        self.k = ct.kernel_size[0]
        weight = ct.weight.flip(2, 3).transpose(0, 1).contiguous()
        self.register_buffer("w", weight)
        if ct.bias is not None:
            bias = ct.bias.detach().clone()
        else:
            bias = torch.zeros(ct.out_channels)
        self.register_buffer("b", bias)
        s = self.s
        mk = torch.zeros(H * s, W * s)
        mk[::s, ::s] = 1.0
        self.register_buffer("mask", mk[None, None])

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        s, k = self.s, self.k
        xn = F.interpolate(x, size=(H * s, W * s), mode="nearest")
        y = F.conv2d(xn * self.mask, self.w, bias=self.b, padding=k - 1)
        return y[:, :, :H * s, :W * s]


class CleanMaskDecoder(nn.Module):
    """Static single-point SAM2 mask decoder, GPU-clean.

    batch==1 and point_batch==1 are collapsed away.
    """

    def __init__(self, model: Sam2Model):
        super().__init__()
        dec = model.mask_decoder
        self.dec = dec
        self.layers = dec.transformer.layers
        self.final_attn = dec.transformer.final_attn_token_to_image
        self.ln_final = dec.transformer.layer_norm_final_attn
        self.mlps = dec.output_hypernetworks_mlps
        self.iou_head = dec.iou_prediction_head
        self.act = dec.activation
        # Sam2LayerNorm channels_first (32ch).
        self.upscale_ln = dec.upscale_layer_norm

        # ConvTranspose2d -> ZeroStuffConvT (input sizes are static:
        # 64x64 -> 128 -> 256).
        # 256->64, 64x64 -> 128x128.
        self.upscale_conv1 = ZeroStuffConvT(dec.upscale_conv1, 64, 64)
        # 64->32, 128x128 -> 256x256.
        self.upscale_conv2 = ZeroStuffConvT(dec.upscale_conv2, 128, 128)

        # baked constants
        with torch.no_grad():
            # [1,256,64,64].
            image_pos = model.get_image_wide_positional_embeddings()
            # [4096,256].
            image_pos_flat = image_pos.flatten(2).transpose(1, 2)[0]
            self.register_buffer(
                "image_pos_flat", image_pos_flat.contiguous())
            dense = model.prompt_encoder.no_mask_embed.weight
            dense = dense.reshape(1, -1, 1, 1).expand(1, 256, 64, 64)
            # [1,256,64,64].
            self.register_buffer("dense", dense.contiguous())
            out_tokens = torch.cat(
                [dec.obj_score_token.weight, dec.iou_token.weight,
                 dec.mask_tokens.weight], 0)
            # [6,256].
            self.register_buffer("output_tokens", out_tokens.contiguous())

    def _ln(self, ln_module, x):
        return safe_ln(x, ln_module.weight, ln_module.bias, ln_module.eps)

    def _attn(self, mod, query, key, value):
        """Batched SDPA that keeps the leading batch dim (rank 4).

        The rank-3 form ([heads, N, d]) is silently mis-computed by the
        GPU delegate. See "GPU correctness" in the module docstring.

        Args:
          mod: Sam2Attention module supplying the q/k/v/o projections.
          query: query tokens, [1, Nq, C].
          key: key tokens, [1, Nk, C].
          value: value tokens, [1, Nk, C].

        Returns:
          Attention output, [1, Nq, C].
        """
        B, Nq = query.shape[0], query.shape[1]
        Nk = key.shape[1]
        H, hd = mod.num_attention_heads, mod.head_dim
        q = mod.q_proj(query).reshape(B, Nq, H, hd).transpose(1, 2)
        k = mod.k_proj(key).reshape(B, Nk, H, hd).transpose(1, 2)
        v = mod.v_proj(value).reshape(B, Nk, H, hd).transpose(1, 2)
        # [B,H,Nq,hd].
        o = F.scaled_dot_product_attention(q, k, v, scale=mod.scaling)
        o = o.transpose(1, 2).reshape(B, Nq, H * hd)
        return mod.o_proj(o)                                   # [1,Nq,C]

    def _block(self, layer, queries, keys, qpe, kpe, skip):
        if skip:
            queries = self._attn(
                layer.self_attn, queries, queries, queries)
        else:
            qq = queries + qpe
            queries = queries + self._attn(
                layer.self_attn, qq, qq, queries)
        queries = self._ln(layer.layer_norm1, queries)
        qq = queries + qpe
        kk = keys + kpe
        queries = queries + self._attn(
            layer.cross_attn_token_to_image, qq, kk, keys)
        queries = self._ln(layer.layer_norm2, queries)
        queries = queries + layer.mlp(queries)
        queries = self._ln(layer.layer_norm3, queries)
        qq = queries + qpe
        kk = keys + kpe
        keys = keys + self._attn(
            layer.cross_attn_image_to_token, kk, qq, queries)
        keys = self._ln(layer.layer_norm4, keys)
        return queries, keys

    def forward(self, image_embeddings, sparse, feat_s1, feat_s0):
        # keys/kpe: [1,4096,256]. The batch dim is kept throughout; see
        # "GPU correctness" in the module docstring.
        keys = (image_embeddings + self.dense).flatten(2).transpose(1, 2)
        kpe = self.image_pos_flat.unsqueeze(0)
        # queries: [1,8,256].
        queries = torch.cat([self.output_tokens.unsqueeze(0), sparse], 1)
        # query_point_embedding (constant across layers).
        qpe = queries

        q, k = queries, keys
        q, k = self._block(self.layers[0], q, k, qpe, kpe, skip=True)
        q, k = self._block(self.layers[1], q, k, qpe, kpe, skip=False)
        fq, fk = q + qpe, k + kpe
        q = q + self._attn(self.final_attn, fq, fk, k)
        q = self._ln(self.ln_final, q)

        iou_tok = q[:, 1:2]                                        # [1,1,256]
        mask_toks = q[:, 2:6]                                      # [1,4,256]

        # [1,256,64,64].
        img = k.transpose(1, 2).reshape(1, 256, 64, 64)
        u = self.upscale_conv1(img) + feat_s1                 # [1,64,128,128]
        # upscale_layer_norm: channels_first SafeLN over the 64 channels
        u = u.permute(0, 2, 3, 1)
        u = safe_ln(u, self.upscale_ln.weight, self.upscale_ln.bias,
                    self.upscale_ln.eps)
        u = u.permute(0, 3, 1, 2)
        u = self.act(u)
        # [1,32,256,256].
        u = self.act(self.upscale_conv2(u) + feat_s0)

        # [1,4,32].
        hyper = torch.cat(
            [self.mlps[j](mask_toks[:, j:j + 1]) for j in range(4)], 1)
        uf = u.reshape(1, 32, 256 * 256)                         # [1,32,65536]
        # [1,3,256,256].
        masks = torch.bmm(hyper, uf).reshape(1, 4, 256, 256)[:, 1:]
        iou = self.iou_head(iou_tok)[:, :, 1:].reshape(1, 3)        # [1,3]
        return masks, iou


def main():
    """Runs the eager parity gate and, optionally, converts to .tflite."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert", action="store_true")
    args = ap.parse_args()

    m = Sam2Model.from_pretrained(MODEL_ID).eval()
    ref = torch.load(f"{SCRATCH}/ref_decoder.pt")
    image_embeddings = ref["image_embeddings"]            # list of 3
    img_emb = image_embeddings[-1]                        # [1,256,64,64]
    feat_s0, feat_s1 = image_embeddings[0], image_embeddings[1]
    # [1,1,2,256] -> [1,2,256].
    sparse = ref["sparse"][0]
    # [1,1,3,256,256], [1,1,3].
    ref_masks, ref_iou = ref["masks"], ref["iou"]

    net = CleanMaskDecoder(m).eval()
    with torch.no_grad():
        masks, iou = net(img_emb, sparse, feat_s1, feat_s0)

    rm = ref_masks.reshape(3, -1)
    gm = masks.reshape(3, -1)
    cos = F.cosine_similarity(gm.flatten(), rm.flatten(), dim=0).item()
    mae = (gm - rm).abs().mean().item()
    # mask agreement (binary IoU at threshold 0)
    inter = ((gm > 0) & (rm > 0)).float().sum().item()
    union = ((gm > 0) | (rm > 0)).float().sum().item()
    iou_mask = inter / max(union, 1.0)
    print(f"[eager] masks cos={cos:.6f} mae={mae:.3e} | "
          f"binary-IoU(thr0)={iou_mask:.5f}")
    print(f"[eager] iou ref={ref_iou.flatten().tolist()}")
    print(f"[eager] iou got={iou.flatten().tolist()}")
    assert cos > 0.9999, f"re-authoring changed the math! cos={cos}"
    print("  -> re-authoring is numerically exact ✓")

    if args.convert:
        import os
        import collections
        import numpy as np
        import litert_torch
        from ai_edge_litert import schema_py_generated as schema
        from ai_edge_litert.compiled_model import CompiledModel
        BANNED = {"GATHER_ND", "GATHER", "TOPK_V2", "FLEX_ERF", "ERF",
                  "BROADCAST_TO", "TRANSPOSE_CONV"}
        FP32 = f"{SCRATCH}/sam2_tiny_dec_fp32.tflite"
        FP16 = f"{SCRATCH}/sam2_tiny_mask_decoder_fp16.tflite"
        ex = (img_emb, sparse, feat_s1, feat_s0)

        with torch.no_grad():
            ref_out = [t.detach().numpy().astype("float64").reshape(-1)
                       for t in net(*ex)]

        print("converting (litert_torch) ...")
        litert_torch.convert(net, ex).export(FP32)

        def gate(path, tag):
            # Static GPU-compat scan: read the op set straight from the
            # .tflite flatbuffer.
            with open(path, "rb") as f:
                fb = schema.ModelT.InitFromPackedBuf(f.read(), 0)
            names = {v: k
                     for k, v in vars(schema.BuiltinOperator).items()
                     if not k.startswith("_")}
            hist = collections.Counter()
            over4d = 0
            for g in fb.subgraphs:
                for op in g.operators:
                    c = fb.operatorCodes[op.opcodeIndex]
                    code = max(c.builtinCode, c.deprecatedBuiltinCode)
                    key = (c.customCode.decode() if c.customCode
                           else names.get(code, str(code)))
                    hist[key] += 1
                over4d += sum(1 for t in g.tensors
                              if t.shape is not None and len(t.shape) > 4)
            bad = {k: v for k, v in hist.items() if k in BANNED}
            ordered = dict(sorted(hist.items(), key=lambda kv: -kv[1]))
            print(f"[{tag}] ops: {ordered}")
            print(f"[{tag}] banned: {bad or 'NONE'} | >4D tensors: {over4d}")
            return bad, over4d

        def parity(path, tag):
            # Inference through the LiteRT CompiledModel API.
            model = CompiledModel.from_file(path)
            sig = model.get_signature_list()
            key = list(sig)[0]
            details = model.get_input_tensor_details(key)
            order = [img_emb, sparse, feat_s1, feat_s0]
            ins = model.create_input_buffers(0)
            obufs = model.create_output_buffers(0)
            # match each model input slot to our tensors by shape
            for i, name in enumerate(sig[key]["inputs"]):
                shape = details[name]["shape"]
                want = next(t for t in order
                            if tuple(t.shape) == tuple(shape))
                ins[i].write(
                    np.ascontiguousarray(want.numpy(), dtype=np.float32))
            model.run_by_index(0, ins, obufs)
            item = np.dtype(np.float32).itemsize
            outs = []
            for j in range(len(obufs)):
                reqs = model.get_output_buffer_requirements(j, 0)
                count = reqs["buffer_size"] // item
                outs.append(
                    obufs[j].read(count, np.float32).astype("float64"))
            for ro in ref_out:
                cand = [o for o in outs if o.size == ro.size]
                if cand:
                    c = max(np.corrcoef(ro, o)[0, 1] for o in cand)
                    print(f"[{tag}] parity corr={c:.6f} (len {ro.size})")

        bad, over4d = gate(FP32, "FP32")
        parity(FP32, "FP32")

        print("quantizing fp16 (FLOAT_CASTING) ...")
        from ai_edge_quantizer import quantizer, recipe_manager
        from ai_edge_quantizer.recipe import AlgorithmName, qtyping
        rmgr = recipe_manager.RecipeManager()
        weight_config = qtyping.TensorQuantizationConfig(
            num_bits=16, dtype=qtyping.TensorDataType.FLOAT)
        rmgr.add_quantization_config(
            regex=".*",
            operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
            op_config=qtyping.OpQuantizationConfig(
                weight_tensor_config=weight_config,
                compute_precision=qtyping.ComputePrecision.FLOAT),
            algorithm_key=AlgorithmName.FLOAT_CASTING)
        if os.path.exists(FP16):
            os.remove(FP16)
        qt = quantizer.Quantizer(float_model=FP32)
        qt.load_quantization_recipe(rmgr.get_quantization_recipe())
        qt.quantize().export_model(FP16)
        size_fp32 = os.path.getsize(FP32) / 1e6
        size_fp16 = os.path.getsize(FP16) / 1e6
        print(f"SIZE fp32 {size_fp32:.1f} MB -> fp16 {size_fp16:.1f} MB")
        bad16, over4d16 = gate(FP16, "FP16")
        parity(FP16, "FP16")
        status = ("OK -> GPU-clean" if not bad16 and over4d16 == 0
                  else "BLOCKERS REMAIN")
        print(f"\n{status}: {FP16}")


if __name__ == "__main__":
    main()
