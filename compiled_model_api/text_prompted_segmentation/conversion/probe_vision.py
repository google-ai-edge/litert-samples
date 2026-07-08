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

"""Phase-1 probe: re-authored CLIP ViT-B/16 vision encoder -> fp16 corr.

Runs the encoder @352 (484+1 tokens) and measures fp16 device corr. This
decides CLIPSeg feasibility (between CLIP-B/32 49-token OK and DA-V2
784-token wall).
"""
import os
import sys
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 352
EPS = 1e-4
SAFE = 16.0
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


def safe_ln(x, w, b):
    """fp16-safe LayerNorm, vision variant (large-variance inputs).

    Scaled-sum variance + eps 1e-4 (device-proven 0.998).

    Args:
        x: Input tensor [..., C].
        w: LayerNorm weight [C].
        b: LayerNorm bias [C].

    Returns:
        The normalized tensor, same shape as x.
    """
    m = x.mean(dim=-1, keepdim=True)
    d = (x - m) * (1.0 / SAFE)
    v = (d * d).mean(dim=-1, keepdim=True)
    return (x - m) / torch.sqrt(v * (SAFE * SAFE) + EPS) * w + b


def safe_ln_up(x, w, b, up=8.0, eps=1e-5):
    """fp16-safe LayerNorm, small-variance variant.

    For CLIP text / CLIPSeg decoder activations: up-scales x so the eps
    lands in fp16-normal range (var(8x) + 64*eps = var*64 + 6.4e-4),
    exact vs torch to ~1e-5 relative.

    Args:
        x: Input tensor [..., C].
        w: LayerNorm weight [C].
        b: LayerNorm bias [C].
        up: Pre-scale factor applied to x.
        eps: LayerNorm epsilon (scaled by up*up inside).

    Returns:
        The normalized tensor, same shape as x.
    """
    y = x * up
    m = y.mean(dim=-1, keepdim=True)
    d = (y - m) * (1.0 / SAFE)
    v = (d * d).mean(dim=-1, keepdim=True)
    return (y - m) / torch.sqrt(v * (SAFE * SAFE) + eps * up * up) * w + b


class VisionGPU(nn.Module):
    """Functional re-author of CLIPSegVisionTransformer.

    Batch-1; taps the hidden states after layers 3/6/9 + final.
    """

    def __init__(self, vm):
        super().__init__()
        self.vm = vm
        emb = vm.embeddings
        # bake pos-embed interpolated 14x14 -> 22x22 (bicubic,
        # align_corners=False as HF)
        pe = emb.position_embedding.weight  # [197, 768]
        cls_pe, patch_pe = pe[:1], pe[1:]
        g = int(patch_pe.shape[0] ** 0.5)
        p2 = patch_pe.T.reshape(1, -1, g, g)
        p2 = F.interpolate(p2, size=(SIZE // 16, SIZE // 16),
                           mode="bicubic", align_corners=False)
        p2 = p2.reshape(768, -1).T
        # pos: [1, 485, 768]
        self.register_buffer("pos", torch.cat([cls_pe, p2], 0).unsqueeze(0))

    def forward(self, x):
        vm = self.vm
        emb = vm.embeddings
        # p: [1, 768, 22, 22]
        p = F.conv2d(x, emb.patch_embedding.weight, None, stride=16)
        p = p.flatten(2).transpose(1, 2)  # [1, 484, 768]
        cls = emb.class_embedding.view(1, 1, -1)
        # h: [1, 485, 768]
        h = torch.cat([cls.expand(1, -1, -1), p], dim=1) + self.pos
        h = safe_ln(h, vm.pre_layrnorm.weight, vm.pre_layrnorm.bias)
        taps = []
        for li, layer in enumerate(vm.encoder.layers):
            r = h
            y = safe_ln(h, layer.layer_norm1.weight, layer.layer_norm1.bias)
            a = layer.self_attn
            q = F.linear(y, a.q_proj.weight, a.q_proj.bias) * a.scale
            k = F.linear(y, a.k_proj.weight, a.k_proj.bias)
            v = F.linear(y, a.v_proj.weight, a.v_proj.bias)
            B, N, C = q.shape
            H = a.num_heads
            d = C // H
            q3 = q.reshape(N, H, d).permute(1, 0, 2)  # [12, 485, 64]
            k3 = k.reshape(N, H, d).permute(1, 0, 2)
            v3 = v.reshape(N, H, d).permute(1, 0, 2)
            att = torch.matmul(q3, k3.transpose(1, 2))  # [12, 485, 485]
            att = F.softmax(att, dim=-1)
            o = torch.matmul(att, v3)  # [12, 485, 64]
            o = o.permute(1, 0, 2).reshape(1, N, C)
            h = r + F.linear(o, a.out_proj.weight, a.out_proj.bias)
            r = h
            y = safe_ln(h, layer.layer_norm2.weight, layer.layer_norm2.bias)
            y = F.linear(y, layer.mlp.fc1.weight, layer.mlp.fc1.bias)
            y = y * torch.sigmoid(1.702 * y)  # quick_gelu
            h = r + F.linear(y, layer.mlp.fc2.weight, layer.mlp.fc2.bias)
            # CLIPSeg: hidden_states[i+1] (+1 for embeddings)
            if li in (3, 6, 9):
                taps.append(h)
        return taps[0], taps[1], taps[2], h


def main():
    """Probes the re-authored vision encoder and exports fp16 + fixtures."""
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    m = CLIPSegForImageSegmentation.from_pretrained(
        "CIDAS/clipseg-rd64-refined").eval()
    vm = m.clip.vision_model
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    from PIL import Image
    img = Image.open(os.path.join(HERE, "cats.jpg"))
    x = proc(images=img, return_tensors="pt")["pixel_values"]
    print("input", tuple(x.shape))

    with torch.no_grad():
        ref = vm(x, interpolate_pos_encoding=True, output_hidden_states=True)
        refs = ([ref.hidden_states[i] for i in (3, 6, 9)]
                + [ref.hidden_states[12]])
        g = VisionGPU(vm).eval()
        mine = g(x)
    for nm, a, b in zip(["tap3", "tap6", "tap9", "final"], mine, refs):
        c = np.corrcoef(a.numpy().ravel(), b.numpy().ravel())[0, 1]
        print(f"[torch] {nm}: corr {c:.7f} absmax {a.abs().max():.1f}")

    import litert_torch
    fp32 = os.path.join(HERE, "clipseg_vis.tflite")
    litert_torch.convert(g, (x,)).export(fp32)
    clean = opcheck(fp32, "fp32")
    if clean:
        fp16 = to_fp16(fp32, os.path.join(HERE, "clipseg_vis_fp16.tflite"))
        opcheck(fp16, "fp16")
        x.numpy().astype(np.float32).tofile(
            os.path.join(HERE, "vis_input.bin"))
        np.save(os.path.join(HERE, "vis_ref.npy"),
                np.concatenate([t.numpy().ravel() for t in mine]))
        print("wrote fp16 + fixtures")


if __name__ == "__main__":
    main()
