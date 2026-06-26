"""
SAM 2.1 (hiera-tiny) mask decoder -> LiteRT GPU-clean .tflite  (Bucket 1: model-side re-authoring only)

Phase-2 companion to convert_sam2.py (the image encoder). This converts the prompt-conditioned
mask DECODER. The tiny prompt-encoder (point -> sparse tokens, sin/cos) is done HOST-SIDE in Kotlin
(see emit at the end) so the GPU graph stays sin/cos-free; the decoder takes `sparse` as an input.

Walls re-authored (all model-side; no converter patch):
  1. Sam2Attention (7x: 2 blocks x 3 + 1 final)  : 4D fused attn -> 3D batched SDPA  [heads, N, d]
  2. ConvTranspose2d (upscale_conv1/2)           : -> ZeroStuffConvT (exact zero-stuff + Conv2d), TRANSPOSE_CONV-free
  3. mask head (hyper_in @ upscaled)             : kept <=4D (no [1,1,4,256,256] 5D tensor); collapse batch==1
  4. LayerNorm (9x)                              : SafeLayerNorm (scale-before-square), fp16-overflow-safe, exact
  5. image_positional_embeddings + no-mask dense : baked CONSTANT buffers (host doesn't supply them)
  6. multimask_output=True path                  : static slice [1:], no dynamic-stability argmax/gather/where

Decoder I/O (single point prompt):
  inputs : image_embeddings [1,256,64,64], sparse [1,2,256], feat_s1 [1,64,128,128], feat_s0 [1,32,256,256]
  outputs: pred_masks [1,3,256,256] (logits, 3 multimask), iou_scores [1,3]

Run:
  python convert_sam2_decoder.py            # eager parity vs transformers reference (correctness gate)
  python convert_sam2_decoder.py --convert  # + litert_torch convert + op-gate + fp16
"""
import sys, types, argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F

# macOS scipy stub (same as convert_sam2.py)
_svdp = types.ModuleType("scipy.sparse.linalg._svdp"); _svdp._svdp = lambda *a, **k: None
sys.modules["scipy.sparse.linalg._svdp"] = _svdp
_opt = types.ModuleType("scipy.optimize"); _opt.linear_sum_assignment = lambda *a, **k: (None, None)
sys.modules["scipy.optimize"] = _opt

from transformers import Sam2Model

MODEL_ID = "facebook/sam2.1-hiera-tiny"
SCRATCH = os.environ.get("SAM2_OUT", "/tmp/sam2_out")


# ----- LayerNorm. SafeLayerNorm (scale-before-square) protects the encoder's huge deep-stage
#       activations from fp16 variance overflow, but the decoder's activations are normal-scale, where
#       the down-scaling instead HURTS GPU fp16 (device A/B: SafeLN decoder masks the background).
#       PLAIN_LN=1 uses stock LayerNorm for the decoder. -----
import os
_PLAIN_LN = os.environ.get("PLAIN_LN") == "1"


def safe_ln(x, weight, bias, eps, sc=0.03125):
    if _PLAIN_LN:
        xc = x - x.mean(-1, keepdim=True)
        var = (xc * xc).mean(-1, keepdim=True)
        return xc * torch.rsqrt(var + eps) * weight + bias
    xc = x - x.mean(-1, keepdim=True)
    xs = xc * sc
    var = (xs * xs).mean(-1, keepdim=True) / (sc * sc)
    return xc * torch.rsqrt(var + eps) * weight + bias


# ----- ZeroStuffConvT: ConvTranspose2d(k=s,stride=s) == zero-stuff(nearest x top-left mask) + Conv2d(flipped w) -----
class ZeroStuffConvT(nn.Module):
    def __init__(self, ct, H, W):
        super().__init__()
        self.s = ct.stride[0]; self.k = ct.kernel_size[0]
        self.register_buffer("w", ct.weight.flip(2, 3).transpose(0, 1).contiguous())
        self.register_buffer("b", ct.bias.detach().clone() if ct.bias is not None else torch.zeros(ct.out_channels))
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
    """Static single-point SAM2 mask decoder, GPU-clean. batch==1, point_batch==1 collapsed away."""
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
        self.upscale_ln = dec.upscale_layer_norm   # Sam2LayerNorm channels_first (32ch)

        # ConvTranspose2d -> ZeroStuffConvT (input sizes are static: 64x64 -> 128 -> 256)
        self.upscale_conv1 = ZeroStuffConvT(dec.upscale_conv1, 64, 64)     # 256->64, 64x64 -> 128x128
        self.upscale_conv2 = ZeroStuffConvT(dec.upscale_conv2, 128, 128)   # 64->32, 128x128 -> 256x256

        # baked constants
        with torch.no_grad():
            image_pos = model.get_image_wide_positional_embeddings()       # [1,256,64,64]
            self.register_buffer("image_pos_flat", image_pos.flatten(2).transpose(1, 2)[0].contiguous())  # [4096,256]
            dense = model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(1, 256, 64, 64).contiguous()
            self.register_buffer("dense", dense)                            # [1,256,64,64]
            out_tokens = torch.cat([dec.obj_score_token.weight, dec.iou_token.weight, dec.mask_tokens.weight], 0)
            self.register_buffer("output_tokens", out_tokens.contiguous())  # [6,256]

    def _ln(self, ln_module, x):
        return safe_ln(x, ln_module.weight, ln_module.bias, ln_module.eps)

    def _attn(self, mod, query, key, value):
        """3D batched SDPA. query [Nq,C], key/value [Nk,C] -> [Nq,C]."""
        Nq, Nk = query.shape[0], key.shape[0]
        H, hd = mod.num_attention_heads, mod.head_dim
        q = mod.q_proj(query).reshape(Nq, H, hd).transpose(0, 1)    # [H,Nq,hd]
        k = mod.k_proj(key).reshape(Nk, H, hd).transpose(0, 1)      # [H,Nk,hd]
        v = mod.v_proj(value).reshape(Nk, H, hd).transpose(0, 1)    # [H,Nk,hd]
        o = F.scaled_dot_product_attention(q, k, v, scale=mod.scaling)  # [H,Nq,hd]
        o = o.transpose(0, 1).reshape(Nq, H * hd)                   # [Nq, internal]
        return mod.o_proj(o)                                        # [Nq, C]

    def _block(self, layer, queries, keys, qpe, kpe, skip):
        if skip:
            queries = self._attn(layer.self_attn, queries, queries, queries)
        else:
            qq = queries + qpe
            queries = queries + self._attn(layer.self_attn, qq, qq, queries)
        queries = self._ln(layer.layer_norm1, queries)
        qq = queries + qpe; kk = keys + kpe
        queries = queries + self._attn(layer.cross_attn_token_to_image, qq, kk, keys)
        queries = self._ln(layer.layer_norm2, queries)
        queries = queries + layer.mlp(queries)
        queries = self._ln(layer.layer_norm3, queries)
        qq = queries + qpe; kk = keys + kpe
        keys = keys + self._attn(layer.cross_attn_image_to_token, kk, qq, queries)
        keys = self._ln(layer.layer_norm4, keys)
        return queries, keys

    def forward(self, image_embeddings, sparse, feat_s1, feat_s0):
        keys = (image_embeddings + self.dense).flatten(2).transpose(1, 2)[0]   # [4096,256]
        kpe = self.image_pos_flat                                              # [4096,256]
        queries = torch.cat([self.output_tokens, sparse[0]], 0)                # [8,256]
        qpe = queries                                                          # query_point_embedding (constant across layers)

        q, k = queries, keys
        q, k = self._block(self.layers[0], q, k, qpe, kpe, skip=True)
        q, k = self._block(self.layers[1], q, k, qpe, kpe, skip=False)
        fq, fk = q + qpe, k + kpe
        q = q + self._attn(self.final_attn, fq, fk, k)
        q = self._ln(self.ln_final, q)

        iou_tok = q[1:2]                                                       # [1,256]
        mask_toks = q[2:6]                                                     # [4,256]

        img = k.transpose(0, 1).reshape(1, 256, 64, 64)                        # [1,256,64,64]
        u = self.upscale_conv1(img) + feat_s1                                  # [1,64,128,128]
        # upscale_layer_norm: channels_first SafeLN over the 64 channels
        u = u.permute(0, 2, 3, 1)
        u = safe_ln(u, self.upscale_ln.weight, self.upscale_ln.bias, self.upscale_ln.eps)
        u = u.permute(0, 3, 1, 2)
        u = self.act(u)
        u = self.act(self.upscale_conv2(u) + feat_s0)                          # [1,32,256,256]

        hyper = torch.cat([self.mlps[j](mask_toks[j:j + 1]) for j in range(4)], 0)  # [4,32]
        uf = u.reshape(32, 256 * 256)                                          # [32,65536]
        masks = (hyper @ uf).reshape(4, 256, 256)[1:].unsqueeze(0)             # [1,3,256,256]
        iou = self.iou_head(iou_tok)[:, 1:]                                    # [1,3]
        return masks, iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert", action="store_true")
    args = ap.parse_args()

    m = Sam2Model.from_pretrained(MODEL_ID).eval()
    ref = torch.load(f"{SCRATCH}/ref_decoder.pt")
    image_embeddings = ref["image_embeddings"]            # list of 3
    img_emb = image_embeddings[-1]                        # [1,256,64,64]
    feat_s0, feat_s1 = image_embeddings[0], image_embeddings[1]
    sparse = ref["sparse"][0]                             # [1,1,2,256] -> [1,2,256]
    ref_masks, ref_iou = ref["masks"], ref["iou"]         # [1,1,3,256,256], [1,1,3]

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
    print(f"[eager] masks cos={cos:.6f} mae={mae:.3e} | binary-IoU(thr0)={iou_mask:.5f}")
    print(f"[eager] iou ref={ref_iou.flatten().tolist()}")
    print(f"[eager] iou got={iou.flatten().tolist()}")
    assert cos > 0.9999, f"re-authoring changed the math! cos={cos}"
    print("  -> re-authoring is numerically exact ✓")

    if args.convert:
        import os, collections, numpy as np, litert_torch
        from ai_edge_litert.interpreter import Interpreter
        BANNED = {"GATHER_ND", "GATHER", "TOPK_V2", "FLEX_ERF", "ERF", "BROADCAST_TO", "TRANSPOSE_CONV"}
        FP32 = f"{SCRATCH}/sam2_tiny_dec_fp32.tflite"
        FP16 = f"{SCRATCH}/sam2_tiny_mask_decoder_fp16.tflite"
        ex = (img_emb, sparse, feat_s1, feat_s0)

        with torch.no_grad():
            ref_out = [t.detach().numpy().astype("float64").reshape(-1) for t in net(*ex)]

        print("converting (litert_torch) ...")
        litert_torch.convert(net, ex).export(FP32)

        def gate(path, tag):
            it = Interpreter(model_path=path); it.allocate_tensors()
            hist = collections.Counter(d["op_name"] for d in it._get_ops_details())
            over4d = sum(1 for d in it.get_tensor_details() if len(d.get("shape", [])) > 4)
            bad = {k: v for k, v in hist.items() if k in BANNED}
            print(f"[{tag}] ops: {dict(sorted(hist.items(), key=lambda kv: -kv[1]))}")
            print(f"[{tag}] banned: {bad or 'NONE'} | >4D tensors: {over4d}")
            return it, bad, over4d

        def parity(it, tag):
            ins = it.get_input_details()
            order = [img_emb, sparse, feat_s1, feat_s0]
            # match each model input slot to our tensors by shape
            for d in ins:
                want = next(t for t in order if tuple(t.shape) == tuple(d["shape"]))
                it.set_tensor(d["index"], want.numpy().astype(d["dtype"]))
            it.invoke()
            outs = [it.get_tensor(o["index"]).astype("float64").reshape(-1) for o in it.get_output_details()]
            for ro in ref_out:
                cand = [o for o in outs if o.size == ro.size]
                if cand:
                    c = max(np.corrcoef(ro, o)[0, 1] for o in cand)
                    print(f"[{tag}] parity corr={c:.6f} (len {ro.size})")

        it32, bad, over4d = gate(FP32, "FP32")
        parity(it32, "FP32")

        print("quantizing fp16 (FLOAT_CASTING) ...")
        from ai_edge_quantizer import quantizer, recipe_manager
        from ai_edge_quantizer.recipe import AlgorithmName, qtyping
        rmgr = recipe_manager.RecipeManager()
        rmgr.add_quantization_config(
            regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
            op_config=qtyping.OpQuantizationConfig(
                weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
                compute_precision=qtyping.ComputePrecision.FLOAT),
            algorithm_key=AlgorithmName.FLOAT_CASTING)
        if os.path.exists(FP16):
            os.remove(FP16)
        qt = quantizer.Quantizer(float_model=FP32)
        qt.load_quantization_recipe(rmgr.get_quantization_recipe())
        qt.quantize().export_model(FP16)
        print(f"SIZE fp32 {os.path.getsize(FP32)/1e6:.1f} MB -> fp16 {os.path.getsize(FP16)/1e6:.1f} MB")
        it16, bad16, over4d16 = gate(FP16, "FP16")
        parity(it16, "FP16")
        print(f"\n{'OK -> GPU-clean' if not bad16 and over4d16 == 0 else 'BLOCKERS REMAIN'}: {FP16}")


if __name__ == "__main__":
    main()
