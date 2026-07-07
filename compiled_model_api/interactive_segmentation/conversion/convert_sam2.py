# Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

"""
SAM 2.1 (hiera-tiny) image encoder -> LiteRT GPU-clean .tflite  (Bucket 1: model-side re-authoring only)

Walls re-authored (all model-side; no converter patch):
  1. window_partition / window_unpartition : 6D reshape+permute  ->  <=4D sequence (verified exact)
  2. Sam2MultiScaleAttention.forward        : 5D qkv reshape      ->  4D-clean (split q/k/v from 3D) + SDPA
  3. Sam2HieraDetModel._get_pos_embed       : bicubic-interp + tile of a constant -> baked buffer (add only)
  4. output                                 : return the 3 FPN feature maps only (pos-encodings are constant -> host-side)

Run:
  python convert_sam2.py            # eager numerical self-test vs reference (correctness gate)
  python convert_sam2.py --convert  # + litert_torch convert to fp16 .tflite
"""
import os
import sys
import types
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# scipy 1.15.3's _propack/_cobyla compiled .so fail dlopen on macOS 27 (Darwin 27), and py3.10 caps
# scipy at 1.15.x so we can't upgrade past the broken wheels. Both are UNUSED by our path; the one
# thing litert_torch genuinely needs (scipy.sparse.csgraph.maximum_flow, for the layout min-cut) has a
# healthy .so. So stub the two broken entrypoints: _svdp (pulls _propack) lets scipy.sparse import for
# csgraph; a fake scipy.optimize gives transformers the linear_sum_assignment symbol (D-FINE loss, unused).
_svdp = types.ModuleType("scipy.sparse.linalg._svdp")
_svdp._svdp = lambda *a, **k: None
sys.modules["scipy.sparse.linalg._svdp"] = _svdp
_opt = types.ModuleType("scipy.optimize")
_opt.linear_sum_assignment = lambda *a, **k: (None, None)
sys.modules["scipy.optimize"] = _opt

import transformers.models.sam2.modeling_sam2 as M
from transformers import Sam2Model

MODEL_ID = "facebook/sam2.1-hiera-tiny"
SCRATCH = os.environ.get("SAM2_OUT", "/tmp/sam2_out")


# ----- 0. overflow-safe LayerNorm (the ML Drift GPU delegate reduces variance in fp16; the deepest
#          SAM2 FPN feature drifts to cos 0.9925 without this. SC=1/32, mathematically identical) -----
class SafeLayerNorm(nn.Module):
    SC = 0.03125

    def __init__(self, ln):
        super().__init__()
        self.weight, self.bias, self.eps = ln.weight, ln.bias, ln.eps

    def forward(self, x):
        xc = x - x.mean(-1, keepdim=True)
        xs = xc * self.SC
        var = (xs * xs).mean(-1, keepdim=True) / (self.SC * self.SC)
        return xc * torch.rsqrt(var + self.eps) * self.weight + self.bias


def patch_layernorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, SafeLayerNorm(child))
        else:
            patch_layernorm(child)


# ----- 1. window partition / unpartition in <=4D (B==1, static shapes) -----
def window_partition_4d(x, window_size):
    """x:[B,H,W,C] (B==1) -> windows:[nW, ws, ws, C], pad_hw=(Hp,Wp). Stays <=4D throughout."""
    B, H, W, C = x.shape
    ws = window_size
    ph = (ws - H % ws) % ws
    pw = (ws - W % ws) % ws
    x = F.pad(x, (0, 0, 0, pw, 0, ph))            # [B,Hp,Wp,C]
    Hp, Wp = H + ph, W + pw
    Hg, Wg = Hp // ws, Wp // ws
    x = x.reshape(Hp, Wp, C)                       # drop B==1 -> 3D
    x = x.reshape(Hp, Wg, ws * C)                  # split W=Wg*ws, merge (ws_col,C) -> 3D
    x = x.permute(1, 0, 2)                         # [Wg, Hp, ws*C]
    x = x.reshape(Wg, Hg, ws, ws * C)              # split Hp=Hg*ws -> 4D
    x = x.permute(1, 0, 2, 3)                      # [Hg, Wg, ws_row, ws*C]
    windows = x.reshape(Hg * Wg, ws, ws, C)        # [nW, ws_row, ws_col, C]
    return windows, (Hp, Wp)


def window_unpartition_4d(windows, window_size, pad_hw, hw):
    """inverse of window_partition_4d, then crop to (H,W). Stays <=4D."""
    ws = window_size
    Hp, Wp = pad_hw
    H, W = hw
    Hg, Wg = Hp // ws, Wp // ws
    C = windows.shape[-1]
    x = windows.reshape(Hg, Wg, ws, ws * C)        # [Hg, Wg, ws_row, ws*C]
    x = x.permute(1, 0, 2, 3)                      # [Wg, Hg, ws_row, ws*C]
    x = x.reshape(Wg, Hp, ws * C)                  # merge (Hg,ws_row)=Hp
    x = x.permute(1, 0, 2)                         # [Hp, Wg, ws*C]
    x = x.reshape(Hp, Wp, C)                       # merge (Wg,ws_col,C)=Wp,C
    x = x.reshape(1, Hp, Wp, C)                    # restore B==1
    x = x[:, :H, :W, :].contiguous()
    return x


# ----- 2. 4D-clean MultiScale attention (no 5D qkv reshape) -----
def attn_forward_4d(self, hidden_states, **kwargs):
    B, H, W, _ = hidden_states.shape
    nH = self.num_attention_heads
    do = self.dim_out
    hd = do // nH
    N = H * W
    qkv = self.qkv(hidden_states).reshape(B, N, 3 * do)     # 3D (avoids the [B,N,3,nH,hd] 5D)
    q, k, v = qkv.split(do, dim=-1)                          # each [B, N, do]
    q = q.reshape(B, N, nH, hd)
    k = k.reshape(B, N, nH, hd)
    v = v.reshape(B, N, nH, hd)

    if self.query_stride:
        # pool the queries spatially (stage downsample)
        q = q.reshape(B, H, W, do).permute(0, 3, 1, 2)       # [B, do, H, W]
        q = F.max_pool2d(q, kernel_size=self.query_stride[0], stride=self.query_stride[0])
        Hq, Wq = q.shape[2], q.shape[3]
        q = q.permute(0, 2, 3, 1).reshape(B, Hq * Wq, nH, hd)
    else:
        Hq, Wq = H, W

    # collapse (batch, heads) -> one dim: 3D BMM is GPU-clean; a 4D SDPA with batch>1 (windowed
    # blocks) makes the delegate emit a [C,C]->[nW,ws,C,C] BROADCAST_TO. See gpu-delegate rules.
    q = q.transpose(1, 2).reshape(B * nH, Hq * Wq, hd)       # 3D [B*nH, Nq, hd]
    k = k.transpose(1, 2).reshape(B * nH, N, hd)
    v = v.transpose(1, 2).reshape(B * nH, N, hd)
    out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)   # 3D SDPA, materialized transpose
    out = out.reshape(B, nH, Hq * Wq, hd).transpose(1, 2).reshape(B, Hq, Wq, do)
    return self.proj(out)


# ----- 3. baked positional embedding (constant) -----
def make_baked_pos_embed(backbone, hw):
    with torch.no_grad():
        h, w = hw
        pe = F.interpolate(backbone.pos_embed, size=(h, w), mode="bicubic")
        win = backbone.pos_embed_window
        pe = pe + win.tile([x // y for x, y in zip(pe.shape, win.shape)])
        pe = pe.permute(0, 2, 3, 1).contiguous()             # [1,h,w,C]
    return pe


# ----- 4. neck without the (unused, constant) sine position encoding -----
def neck_forward_nopos(self, hidden_states):
    """FPN feature pyramid only; drop fpn_position_encoding (it's shape-only constant -> host-side).
    Removes the sine-embed BROADCAST_TO ops that otherwise block the GPU delegate."""
    fpn_hidden_states = ()
    n = len(self.convs) - 1
    for i in range(n, -1, -1):
        lateral = hidden_states[i].permute(0, 3, 1, 2)
        lateral = self.convs[n - i](lateral.to(self.convs[i].weight.dtype))
        if i not in self.fpn_top_down_levels or i == n:
            prev = lateral
        else:
            td = F.interpolate(prev.to(torch.float32), scale_factor=2.0, mode="nearest").to(lateral.dtype)
            prev = lateral + td
        fpn_hidden_states += (prev,)
    return fpn_hidden_states, ()  # empty position encodings


def patch_model(m):
    bb = m.vision_encoder.backbone
    # 1 + 2: globals + attention
    M.window_partition = window_partition_4d
    M.window_unpartition = window_unpartition_4d
    M.Sam2MultiScaleAttention.forward = attn_forward_4d
    # 3: bake pos embed (patch_embed output is 256x256 for 1024 input)
    baked = make_baked_pos_embed(bb, (256, 256))
    bb.register_buffer("_baked_pos_embed", baked)
    bb._get_pos_embed = types.MethodType(lambda self, hw: self._baked_pos_embed, bb)
    # 4: neck without the constant sine position encoding (kills BROADCAST_TO)
    neck = m.vision_encoder.neck
    neck.forward = types.MethodType(neck_forward_nopos, neck)
    # 0: overflow-safe LayerNorm (deepest FPN feature drifts to cos 0.9925 on GPU fp16 without this)
    patch_layernorm(bb)
    return m


class ImageEncoder(nn.Module):
    """Wrapper: pixel_values[1,3,1024,1024] -> 3 FPN feature maps (NCHW), GPU-clean."""
    def __init__(self, vision_encoder):
        super().__init__()
        self.ve = vision_encoder

    def forward(self, x):
        out = self.ve(x)
        # fpn_hidden_states: high->low res [(1,256,256,256?),...]; already NCHW from neck
        f = out.fpn_hidden_states
        return f[0], f[1], f[2]


class ImageEncoderV2(nn.Module):
    """Phase-2 wrapper: folds the mask-decoder's conv_s0/conv_s1 projections + the no_memory
    embedding into the encoder so it directly emits DECODER-READY features (one model = one run):
      pixel_values[1,3,1024,1024] -> (image_embeddings[1,256,64,64], feat_s1[1,64,128,128], feat_s0[1,32,256,256])
    Matches Sam2Model.get_image_embeddings() exactly (conv_s0/s1 live on the mask decoder, the
    no_memory add lands on the coarsest FPN level)."""
    def __init__(self, model):
        super().__init__()
        self.ve = model.vision_encoder
        self.conv_s0 = model.mask_decoder.conv_s0   # 256 -> 32  (1x1)
        self.conv_s1 = model.mask_decoder.conv_s1   # 256 -> 64  (1x1)
        self.register_buffer("no_mem", model.no_memory_embedding.detach().reshape(1, -1, 1, 1).contiguous())

    def forward(self, x):
        f = self.ve(x).fpn_hidden_states            # NCHW high->low: [1,256,256,256],[1,256,128,128],[1,256,64,64]
        feat_s0 = self.conv_s0(f[0])                 # [1,32,256,256]
        feat_s1 = self.conv_s1(f[1])                 # [1,64,128,128]
        img_emb = f[2] + self.no_mem                 # [1,256,64,64]
        return img_emb, feat_s1, feat_s0


DEC_SCRATCH = os.environ.get("SAM2_OUT", "/tmp/sam2_out")


def convert_v2(m, x):
    """Encoder v2: decoder-ready outputs (folds conv_s0/s1 + no_memory). Verify vs the reference
    image_embeddings captured by recon_sam2_decoder.py, then convert + op-gate + fp16."""
    import os
    import collections
    import numpy as np
    import litert_torch
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.compiled_model import CompiledModel
    BANNED = {"GATHER_ND", "GATHER", "TOPK_V2", "FLEX_ERF", "ERF", "BROADCAST_TO", "TRANSPOSE_CONV"}
    FP32 = f"{DEC_SCRATCH}/sam2_tiny_enc_v2_fp32.tflite"
    FP16 = f"{DEC_SCRATCH}/sam2_tiny_image_encoder_v2_fp16.tflite"

    enc = ImageEncoderV2(m).eval()
    with torch.no_grad():
        got = enc(x)
    ref = torch.load(f"{DEC_SCRATCH}/ref_decoder.pt")
    ie = ref["image_embeddings"]                          # [feat_s0(256x256), feat_s1(128x128), img_emb(64x64)]
    targets = {"img_emb": ie[-1], "feat_s1": ie[1], "feat_s0": ie[0]}
    for (name, tref), tgot in zip(targets.items(), got):
        cos = torch.nn.functional.cosine_similarity(tgot.flatten(), tref.flatten(), dim=0).item()
        print(f"[v2 eager] {name}: cos={cos:.6f} shape={tuple(tgot.shape)}")
        assert cos > 0.9999, f"{name} re-authoring drift cos={cos}"
    print("  -> encoder-v2 decoder-ready outputs are numerically exact ✓")

    with torch.no_grad():
        ref_fp = [t.detach().numpy().astype("float64").reshape(-1) for t in enc(x)]
    print("converting v2 (litert_torch) ...")
    litert_torch.convert(enc, (x,)).export(FP32)

    def gate(path, tag):
        # Static GPU-compat scan: read the op set straight from the .tflite flatbuffer.
        with open(path, "rb") as f:
            fb = schema.ModelT.InitFromPackedBuf(f.read(), 0)
        names = {v: k for k, v in vars(schema.BuiltinOperator).items() if not k.startswith("_")}
        hist = collections.Counter()
        over4d = 0
        for g in fb.subgraphs:
            for op in g.operators:
                c = fb.operatorCodes[op.opcodeIndex]
                code = max(c.builtinCode, c.deprecatedBuiltinCode)
                hist[c.customCode.decode() if c.customCode else names.get(code, str(code))] += 1
            over4d += sum(1 for t in g.tensors if t.shape is not None and len(t.shape) > 4)
        bad = {k: v for k, v in hist.items() if k in BANNED}
        print(f"[{tag}] banned: {bad or 'NONE'} | >4D tensors: {over4d}")
        return bad, over4d

    def parity(path, tag):
        # Inference through the LiteRT CompiledModel API.
        model = CompiledModel.from_file(path)
        ins = model.create_input_buffers(0)
        obufs = model.create_output_buffers(0)
        ins[0].write(np.ascontiguousarray(x.numpy(), dtype=np.float32))
        model.run_by_index(0, ins, obufs)
        item = np.dtype(np.float32).itemsize
        outs = [obufs[j].read(model.get_output_buffer_requirements(j, 0)["buffer_size"] // item,
                              np.float32).astype("float64") for j in range(len(obufs))]
        for ro in ref_fp:
            cand = [o for o in outs if o.size == ro.size]
            if cand:
                c = max(np.corrcoef(ro, o)[0, 1] for o in cand)
                print(f"[{tag}] parity corr={c:.6f} (len {ro.size})")

    bad, over4d = gate(FP32, "FP32")
    parity(FP32, "FP32")
    print("quantizing fp16 (FLOAT_CASTING) ...")
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(
        regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT),
        algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(FP16):
        os.remove(FP16)
    qt = quantizer.Quantizer(float_model=FP32)
    qt.load_quantization_recipe(rm.get_quantization_recipe())
    qt.quantize().export_model(FP16)
    print(f"SIZE fp32 {os.path.getsize(FP32)/1e6:.1f} MB -> fp16 {os.path.getsize(FP16)/1e6:.1f} MB")
    bad16, over4d16 = gate(FP16, "FP16")
    parity(FP16, "FP16")
    print(f"\n{'OK -> GPU-clean' if not bad16 and over4d16 == 0 else 'BLOCKERS REMAIN'}: {FP16}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--v2", action="store_true", help="export decoder-ready encoder (Phase 2)")
    args = ap.parse_args()

    m = Sam2Model.from_pretrained(MODEL_ID).eval()
    ref = torch.load(f"{SCRATCH}/ref_tiny.pt")
    x = ref["x"]
    ref_lhs = ref["last_hidden_state"]

    patch_model(m)

    if args.v2:
        convert_v2(m, x)
        return
    with torch.no_grad():
        out = m.vision_encoder(x)
    got = out.last_hidden_state

    cos = F.cosine_similarity(got.flatten(), ref_lhs.flatten(), dim=0).item()
    mae = (got - ref_lhs).abs().mean().item()
    print(f"[eager self-test] last_hidden_state  cos={cos:.6f}  mae={mae:.3e}  shape={tuple(got.shape)}")
    assert cos > 0.9999, f"re-authoring changed the math! cos={cos}"
    print("  -> re-authoring is numerically exact ✓")

    if args.convert:
        import os
        import collections
        import numpy as np
        import litert_torch
        from ai_edge_litert import schema_py_generated as schema
        from ai_edge_litert.compiled_model import CompiledModel
        BANNED = {"GATHER_ND", "GATHER", "TOPK_V2", "FLEX_ERF", "ERF", "BROADCAST_TO"}
        FP32 = f"{SCRATCH}/sam2_tiny_enc_fp32.tflite"
        FP16 = f"{SCRATCH}/sam2_tiny_image_encoder_fp16.tflite"

        enc = ImageEncoder(m.vision_encoder).eval()
        with torch.no_grad():
            ref_fp = [t.detach().numpy().astype("float64").reshape(-1) for t in enc(x)]
        print(f"[wrapper] fpn out shapes = {[t.shape for t in enc(x)]}")

        print("converting (litert_torch) ...")
        litert_torch.convert(enc, (x,)).export(FP32)

        def gate(path, tag):
            # Static GPU-compat scan: read the op set straight from the .tflite flatbuffer.
            with open(path, "rb") as f:
                fb = schema.ModelT.InitFromPackedBuf(f.read(), 0)
            names = {v: k for k, v in vars(schema.BuiltinOperator).items() if not k.startswith("_")}
            hist = collections.Counter()
            over4d = 0
            for g in fb.subgraphs:
                for op in g.operators:
                    c = fb.operatorCodes[op.opcodeIndex]
                    code = max(c.builtinCode, c.deprecatedBuiltinCode)
                    hist[c.customCode.decode() if c.customCode else names.get(code, str(code))] += 1
                over4d += sum(1 for t in g.tensors if t.shape is not None and len(t.shape) > 4)
            bad = {k: v for k, v in hist.items() if k in BANNED}
            print(f"[{tag}] ops: {dict(sorted(hist.items(), key=lambda kv: -kv[1]))}")
            print(f"[{tag}] banned: {bad or 'NONE'} | >4D tensors: {over4d}")
            return bad, over4d

        def parity(path, tag):
            # Inference through the LiteRT CompiledModel API.
            model = CompiledModel.from_file(path)
            ins = model.create_input_buffers(0)
            obufs = model.create_output_buffers(0)
            ins[0].write(np.ascontiguousarray(x.numpy(), dtype=np.float32))
            model.run_by_index(0, ins, obufs)
            item = np.dtype(np.float32).itemsize
            outs = [obufs[j].read(model.get_output_buffer_requirements(j, 0)["buffer_size"] // item,
                                  np.float32).astype("float64") for j in range(len(obufs))]
            # match each tflite output to a ref by length, report corr
            for ro in ref_fp:
                cand = [o for o in outs if o.size == ro.size]
                if cand:
                    c = max(np.corrcoef(ro, o)[0, 1] for o in cand)
                    print(f"[{tag}] parity corr={c:.6f} (len {ro.size})")

        bad, over4d = gate(FP32, "FP32")
        parity(FP32, "FP32")

        print("quantizing fp16 (FLOAT_CASTING) ...")
        from ai_edge_quantizer import quantizer, recipe_manager
        from ai_edge_quantizer.recipe import AlgorithmName, qtyping
        rm = recipe_manager.RecipeManager()
        rm.add_quantization_config(
            regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
            op_config=qtyping.OpQuantizationConfig(
                weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
                compute_precision=qtyping.ComputePrecision.FLOAT),
            algorithm_key=AlgorithmName.FLOAT_CASTING)
        if os.path.exists(FP16):
            os.remove(FP16)
        qt = quantizer.Quantizer(float_model=FP32)
        qt.load_quantization_recipe(rm.get_quantization_recipe())
        qt.quantize().export_model(FP16)
        print(f"SIZE fp32 {os.path.getsize(FP32)/1e6:.1f} MB -> fp16 {os.path.getsize(FP16)/1e6:.1f} MB")
        bad16, over4d16 = gate(FP16, "FP16")
        parity(FP16, "FP16")
        print(f"\n{'OK -> GPU-clean' if not bad16 and over4d16 == 0 else 'BLOCKERS REMAIN'}: {FP16}")


if __name__ == "__main__":
    main()
