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

"""Phase 1a: re-author RAM++ Swin-L encoder GPU-clean.

Verifies torch parity and converts to LiteRT fp16.

Usage: python build_swin.py [parity|convert]
"""
import os
import sys
import collections
import numpy as np
import torch
import torch.nn as nn

from ram_load import load_ram_plus, REPO
WORK = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(WORK, "out")
os.makedirs(OUT, exist_ok=True)
REF = np.load(os.path.join(WORK, "ref", "ref_demo1.npz"))
BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT",
          "SELECT_V2", "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST",
          "EMBEDDING_LOOKUP", "PACK", "RFFT2D", "FFT", "STFT", "COMPLEX",
          "RFFT", "IRFFT", "CUMSUM", "SPLIT", "SPLIT_V"}


def opcheck(path, label):
    """Statically scans a .tflite flatbuffer for GPU-hostile ops.

    Args:
        path: Path to the .tflite file.
        label: Tag prepended to the printed report lines.

    Returns:
        True when no banned op and no >4-D tensor is present.
    """
    from ai_edge_litert import schema_py_generated as schema
    with open(path, "rb") as f:
        model = schema.ModelT.InitFromPackedBuf(f.read(), 0)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items()
             if not k.startswith("_")}
    ops = collections.Counter()
    over = 0
    for g in model.subgraphs:
        for op in g.operators:
            c = model.operatorCodes[op.opcodeIndex]
            code = max(c.builtinCode, c.deprecatedBuiltinCode)
            if c.customCode:
                op_name = c.customCode.decode()
            else:
                op_name = names.get(code, str(code))
            ops[op_name] += 1
        over += sum(1 for t in g.tensors
                    if t.shape is not None and len(t.shape) > 4)
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
    print(f"[{label}] nodes:{sum(ops.values())} banned:{bad or 'NONE'} "
          f">4D:{over} size {os.path.getsize(path) / 1e6:.1f}MB")
    print(f"[{label}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    if not bad and not over:
        verdict = "GPU-CLEAN"
    else:
        verdict = f"BLOCKERS {bad} >4D:{over}"
    print(f"[{label}] VERDICT:", verdict)
    return not bad and not over


def to_fp16(fp32, fp16):
    """Quantizes a .tflite to fp16 weights via ai-edge-quantizer.

    Args:
        fp32: Source fp32 .tflite path.
        fp16: Destination fp16 .tflite path.

    Returns:
        The fp16 path.
    """
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(
        regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT),
        algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(fp16):
        os.remove(fp16)
    qt = quantizer.Quantizer(float_model=fp32)
    qt.load_quantization_recipe(rm.get_quantization_recipe())
    qt.quantize().export_model(fp16)
    return fp16


# ------------------- GPU-clean building blocks -------------------
def tanh_gelu(self, x):
    """Tanh-approximation GELU (GPU-clean, no Erf op).

    Args:
        self: Unused module slot (installed as nn.GELU.forward).
        x: Input tensor.

    Returns:
        GELU(x) computed via the tanh approximation.
    """
    return 0.5 * x * (1.0 + torch.tanh(
        0.7978845608028654 * (x + 0.044715 * x * x * x)))

def safe_ln(self, x):
    """Adaptive SafeLayerNorm v2 (installed as nn.LayerNorm.forward).

    Scales each token by its own max|x-mean| BEFORE squaring, so the
    fp16 sum-of-squares stays O(N) at ANY activation magnitude (Swin
    stage-3 hits absmax ~847, at which a fixed /16 scale still
    overflows fp16 in the reduction). eps in the scaled domain is
    negligible vs the real variance, so this stays ~exact while being
    overflow-free.

    Args:
        self: The nn.LayerNorm module supplying weight/bias.
        x: Input tensor [..., C].

    Returns:
        The normalized tensor, same shape as x.
    """
    mu = x.mean(-1, keepdim=True)
    dd = x - mu
    s = dd.abs().amax(-1, keepdim=True).clamp_min(1e-4)   # per-token scale
    e = dd / s
    v = (e * e).mean(-1, keepdim=True)  # <= 1, never overflows
    y = e * torch.rsqrt(v + 1e-6)
    if self.weight is not None:
        y = y * self.weight
    if self.bias is not None:
        y = y + self.bias
    return y

def wp4d(x, ws, H, W):
    """Window partition: [1,H,W,C] -> [nW, ws, ws, C], order [hi,wi,c].

    Args:
        x: Input tensor [1,H,W,C].
        ws: Window size.
        H: Feature-map height.
        W: Feature-map width.

    Returns:
        Tensor [nW, ws, ws, C] of the partitioned windows.
    """
    C = x.shape[-1]
    x = x.reshape(H // ws, ws, W // ws, ws * C)
    x = x.permute(0, 2, 1, 3)
    return x.reshape((H // ws) * (W // ws), ws, ws, C)

def wr4d(x, ws, H, W):
    """Window reverse: [nW, ws, ws, C] -> [1,H,W,C].

    Args:
        x: Partitioned windows [nW, ws, ws, C].
        ws: Window size.
        H: Feature-map height.
        W: Feature-map width.

    Returns:
        Tensor [1,H,W,C] with the windows stitched back.
    """
    C = x.shape[-1]
    x = x.reshape(H // ws, W // ws, ws, ws * C)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(1, H, W, C)

def attn_forward(self, x, mask=None):
    """GPU-clean WindowAttention.forward with baked relative-pos bias.

    Args:
        self: The WindowAttention module (expects the baked rpb buffer).
        x: Window tokens [Bn, N, C].
        mask: Optional additive attention mask [nW, N, N].

    Returns:
        Attention output [Bn, N, C] after the output projection.
    """
    Bn, N, C = x.shape
    nH = self.num_heads
    hd = C // nH
    qkv = self.qkv(x)                                  # [Bn,N,3C]
    q = qkv[:, :, :C].reshape(Bn, N, nH, hd).permute(0, 2, 1, 3)
    k = qkv[:, :, C:2 * C].reshape(Bn, N, nH, hd).permute(0, 2, 1, 3)
    v = qkv[:, :, 2 * C:].reshape(Bn, N, nH, hd).permute(0, 2, 1, 3)
    attn = (q * self.scale) @ k.transpose(-2, -1)      # [Bn,nH,N,N]
    attn = attn + self.rpb.unsqueeze(0)                # baked [1,nH,N,N]
    if mask is not None:                               # Bn == nW (B=1)
        attn = attn + mask.unsqueeze(1)                # [nW,1,N,N]
    attn = torch.softmax(attn, dim=-1)
    x = (attn @ v).permute(0, 2, 1, 3).reshape(Bn, N, C)
    return self.proj(x)

def block_forward(self, x):
    """GPU-clean SwinTransformerBlock.forward (4D shift + window ops).

    Args:
        self: The SwinTransformerBlock module.
        x: Input tokens [1, H*W, C].

    Returns:
        Output tokens [1, H*W, C].
    """
    H, W = self.input_resolution
    C = x.shape[-1]
    ws = self.window_size
    s = self.shift_size
    shortcut = x
    x = self.norm1(x).view(1, H, W, C)
    if s > 0:
        x = torch.cat([x[:, s:, :, :], x[:, :s, :, :]], dim=1)
        x = torch.cat([x[:, :, s:, :], x[:, :, :s, :]], dim=2)
    xw = wp4d(x, ws, H, W).reshape(-1, ws * ws, C)
    aw = self.attn(xw, mask=self.attn_mask).reshape(-1, ws, ws, C)
    x = wr4d(aw, ws, H, W)
    if s > 0:
        x = torch.cat([x[:, -s:, :, :], x[:, :-s, :, :]], dim=1)
        x = torch.cat([x[:, :, -s:, :], x[:, :, :-s, :]], dim=2)
    x = x.reshape(1, H * W, C)
    x = shortcut + x
    x = x + self.mlp(self.norm2(x))
    return x

def merge_forward(self, x):
    """GPU-clean PatchMerging.forward (no strided slicing).

    Args:
        self: The PatchMerging module.
        x: Input tokens [1, H*W, C].

    Returns:
        Merged tokens [1, (H/2)*(W/2), 2C] after norm + reduction.
    """
    H, W = self.input_resolution
    C = x.shape[-1]
    x = x.view(1, H, W, C).reshape(H // 2, 2, W, C)
    r0, r1 = x[:, 0, :, :], x[:, 1, :, :]  # even/odd rows [H//2,W,C]
    def sc(t):
        t = t.reshape(H // 2, W // 2, 2, C)
        return t[:, :, 0, :], t[:, :, 1, :]
    x00, x01 = sc(r0)
    x10, x11 = sc(r1)
    x = torch.cat([x00, x10, x01, x11], dim=-1)        # [H//2,W//2,4C]
    x = x.reshape(1, (H // 2) * (W // 2), 4 * C)
    return self.reduction(self.norm(x))

def swin_forward(self, x, **kw):
    """GPU-clean SwinTransformer.forward with mean-pooled cls token.

    Args:
        self: The SwinTransformer module.
        x: Input image [1, 3, H, W].
        **kw: Ignored keyword arguments from the stock call site.

    Returns:
        Tokens [1, 145, 1536] (mean-pooled cls prepended).
    """
    x = self.patch_embed(x)
    x = self.pos_drop(x)
    for layer in self.layers:
        x = layer(x)
    x = self.norm(x)
    x_cls = x.mean(dim=1, keepdim=True)
    return torch.cat([x_cls, x], dim=1)                # [1,145,1536]

def patch_gpu_clean(model):
    """Installs the GPU-clean forwards and bakes relative-pos biases.

    Args:
        model: The loaded RAM++ model to patch in place.

    Returns:
        The same model with patched Swin forwards and baked rpb buffers.
    """
    import ram.models.swin_transformer as S
    nn.LayerNorm.forward = safe_ln
    nn.GELU.forward = tanh_gelu
    S.WindowAttention.forward = attn_forward
    S.SwinTransformerBlock.forward = block_forward
    S.PatchMerging.forward = merge_forward
    S.SwinTransformer.forward = swin_forward
    # bake relative_position_bias per window-attention
    for m in model.modules():
        if isinstance(m, S.WindowAttention):
            idx = m.relative_position_index.view(-1)
            n = m.window_size[0] * m.window_size[1]
            rpb = (m.relative_position_bias_table[idx]
                   .view(n, n, -1).permute(2, 0, 1).contiguous())
            m.register_buffer("rpb", rpb)
    return model

class SwinEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.visual_encoder = model.visual_encoder
        self.image_proj = model.image_proj
    def forward(self, image):
        return self.image_proj(self.visual_encoder(image))   # [1,145,512]

class SwinEncoderTapped(nn.Module):
    """Bisect probe: outputs each of the 4 Swin stage activations.

    Used to localize the fp16 drop.
    """
    def __init__(self, model):
        super().__init__()
        self.ve = model.visual_encoder
    def forward(self, image):
        ve = self.ve
        x = ve.pos_drop(ve.patch_embed(image))
        outs = []
        for layer in ve.layers:
            x = layer(x)
            outs.append(x)
        # [1,2304,384][1,576,768][1,144,1536][1,144,1536]
        return outs[0], outs[1], outs[2], outs[3]

def corr(a, b):
    """Pearson correlation of two arrays, flattened to float64.

    Args:
        a: First array.
        b: Second array.

    Returns:
        The scalar correlation coefficient.
    """
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1])

def main():
    """Runs the requested mode: parity, tapped, or convert."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "parity"
    model = load_ram_plus(384)
    patch_gpu_clean(model)
    enc = SwinEncoder(model).eval()
    image = torch.from_numpy(REF["image"])

    with torch.no_grad():
        image_embeds = enc(image).numpy()
    ref_ie = REF["image_embeds"]
    print("image_embeds", image_embeds.shape,
          "corr vs ref:", corr(image_embeds, ref_ie),
          "maxabsdiff:", float(np.abs(image_embeds - ref_ie).max()))
    # also compare raw swin vis
    with torch.no_grad():
        vis = model.visual_encoder(image).numpy()
    print("vis         ", vis.shape, "corr vs ref:", corr(vis, REF["vis"]),
          "maxabsdiff:", float(np.abs(vis - REF["vis"]).max()))

    if mode == "tapped":
        import litert_torch
        tap = SwinEncoderTapped(model).eval()
        image = torch.from_numpy(REF["image"])
        with torch.no_grad():
            outs = tap(image)
        for i, o in enumerate(outs):
            o.numpy().astype("<f4").tofile(
                os.path.join(OUT, f"swin_tap{i}.bin"))
            print(f"tap{i} shape {tuple(o.shape)} "
                  f"absmax {float(o.abs().max()):.2f} -> swin_tap{i}.bin")
        fp32 = os.path.join(OUT, "ram_swin_tapped.tflite")
        litert_torch.convert(tap.eval(), (image,)).export(fp32)
        clean = opcheck(fp32, "tapped fp32")
        if clean:
            to_fp16(fp32, os.path.join(OUT, "ram_swin_tapped_fp16.tflite"))
            print("wrote tapped fp16 + per-stage refs")
        return

    if mode == "convert":
        import litert_torch
        image = torch.from_numpy(REF["image"])
        fp32 = os.path.join(OUT, "ram_swin_enc.tflite")
        print("converting Swin encoder ...")
        litert_torch.convert(enc.eval(), (image,)).export(fp32)
        clean = opcheck(fp32, "swin fp32")
        if clean:
            fp16 = to_fp16(fp32, os.path.join(OUT, "ram_swin_enc_fp16.tflite"))
            opcheck(fp16, "swin fp16")
            # fixtures for on-device probe
            image.numpy().astype(np.float32).tofile(
                os.path.join(OUT, "swin_input.bin"))
            np.save(os.path.join(OUT, "swin_ref.npy"),
                    REF["image_embeds"].astype(np.float32))
            print("wrote fp16 + fixtures (swin_input.bin, swin_ref.npy)")

if __name__ == "__main__":
    main()
