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

"""XFeat (Accelerated Features, Apache, ~1.5M pure CNN) -> CompiledModel GPU.

Lightweight local feature extraction for image matching (SLAM/AR/registration). The CNN emits dense
descriptors + keypoint logits + reliability; keypoint NMS, descriptor sampling, and mutual-NN matching
run host-side (the raw-head pattern). Two re-authoring fixes:
  - input gray + InstanceNorm -> HOST (graph takes normalized grayscale [1,1,H,W]); the InstanceNorm
    spatial reduction over H*W would overflow fp16 on the GPU (Gotcha 7).
  - `_unfold2d(x, 8)` (space-to-depth via unfold -> >4D / GATHER_ND) -> a one-hot Conv2d(1,64,k=8,s=8)
    (exact, single CONV_2D, GPU-clean; the YOLOX/PixelShuffle one-hot trick).

Run: python convert_xfeat.py [--nhwc]   (480x640)
"""
import sys
import os
import types
import collections
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class _D:
    def __getattr__(s, n):
        return lambda *a, **k: None
_pp = types.ModuleType("scipy.sparse.linalg._propack")
for n in ("_spropack", "_dpropack", "_cpropack", "_zpropack"): setattr(_pp, n, _D())
sys.modules["scipy.sparse.linalg._propack"] = _pp

H, W = 480, 640
BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP", "PRELU"}


def space_to_depth_conv(ws=8):
    """space-to-depth(ws) on 1-ch == Conv2d(1, ws*ws, k=ws, s=ws) one-hot. Channel order i*ws+j
    matches XFeat._unfold2d (unfold rows/cols)."""
    conv = nn.Conv2d(1, ws * ws, kernel_size=ws, stride=ws, bias=False)
    w = torch.zeros(ws * ws, 1, ws, ws)
    for i in range(ws):
        for j in range(ws):
            w[i * ws + j, 0, i, j] = 1.0
    conv.weight.data.copy_(w)
    return conv.eval()


def reauthor(net):
    net._s2d = space_to_depth_conv(8)

    def fwd(self, x):
        # x = pre-normalized grayscale [B,1,H,W] (gray + InstanceNorm done host-side)
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        feats = self.block_fusion(x3 + x4 + x5)
        heatmap = self.heatmap_head(feats)
        keypoints = self.keypoint_head(self._s2d(x))   # one-hot space-to-depth, GPU-clean
        return feats, keypoints, heatmap
    net.forward = types.MethodType(fwd, net)
    return net


def host_normalize(x_gray):
    """Reference host preprocessing: per-image InstanceNorm (affine=False) on grayscale."""
    mu = x_gray.mean(dim=(2, 3), keepdim=True)
    var = x_gray.var(dim=(2, 3), keepdim=True, unbiased=False)
    return (x_gray - mu) / torch.sqrt(var + 1e-5)


class Wrap(nn.Module):
    def __init__(self, net, nhwc=False):
        super().__init__()
        self.net = net
        self.nhwc = nhwc
    def forward(self, x):
        if self.nhwc:
            x = x.permute(0, 3, 1, 2)
        return self.net(x)


def opcheck(path, label):
    """Static GPU-compat scan: read the op set straight from the .tflite flatbuffer."""
    from ai_edge_litert import schema_py_generated as schema
    with open(path, "rb") as f:
        model = schema.ModelT.InitFromPackedBuf(f.read(), 0)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items() if not k.startswith("_")}
    ops = collections.Counter()
    over = 0
    for g in model.subgraphs:
        for op in g.operators:
            c = model.operatorCodes[op.opcodeIndex]
            code = max(c.builtinCode, c.deprecatedBuiltinCode)
            ops[c.customCode.decode() if c.customCode else names.get(code, str(code))] += 1
        over += sum(1 for t in g.tensors if t.shape is not None and len(t.shape) > 4)
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
    print(f"[{label}] banned:{bad or 'NONE'} >4D:{over} size {os.path.getsize(path)/1e6:.1f}MB ops:{len(ops)}")
    print(f"[{label}] top:", dict(sorted(ops.items(), key=lambda kv: -kv[1])[:10]))


def to_fp16(fp32, fp16):
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT), algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(fp16): os.remove(fp16)
    q = quantizer.Quantizer(float_model=fp32)
    q.load_quantization_recipe(rm.get_quantization_recipe())
    q.quantize().export_model(fp16)
    return fp16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nhwc", action="store_true")
    args = ap.parse_args()
    xf = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)
    net = xf.net.eval()
    raw = torch.rand(1, 1, H, W)            # raw grayscale image (gray = mean over 1 ch = identity)
    with torch.no_grad():
        ref = net(raw)                       # STOCK (does gray + InstanceNorm internally) — capture BEFORE reauthor
    xn = host_normalize(raw)                  # replicate the host-side gray+InstanceNorm
    reauthor(net)
    inp = xn.permute(0, 2, 3, 1).contiguous() if args.nhwc else xn
    with torch.no_grad():
        ra = Wrap(net, args.nhwc)(inp)
    for nm, a, b in zip(["feats", "keypoints", "heatmap"], ref, ra):
        c = np.corrcoef(a.numpy().ravel(), b.numpy().ravel())[0, 1]
        print(f"re-auth {nm}: {tuple(b.shape)} corr {c:.6f}")

    tag = "xfeat" + ("_nhwc" if args.nhwc else "")
    import litert_torch
    litert_torch.convert(Wrap(net, args.nhwc).eval(), (inp,)).export(f"{tag}.tflite")
    opcheck(f"{tag}.tflite", "xfeat-fp32")
    to_fp16(f"{tag}.tflite", f"{tag}_fp16.tflite")
    opcheck(f"{tag}_fp16.tflite", "xfeat-fp16")
    # parity run through the LiteRT CompiledModel API (same API the sample app uses)
    from ai_edge_litert.compiled_model import CompiledModel
    cm = CompiledModel.from_file(f"{tag}.tflite")
    sig = cm.get_signature_list()
    key = list(sig)[0]
    dets = cm.get_output_tensor_details(key)
    ins = cm.create_input_buffers(0)
    outs = cm.create_output_buffers(0)
    ins[0].write(np.ascontiguousarray(inp.numpy(), dtype=np.float32))
    cm.run_by_index(0, ins, outs)
    res = [outs[i].read(int(np.prod(dets[n]["shape"])), np.float32).reshape(dets[n]["shape"])
           for i, n in enumerate(sig[key]["outputs"])]
    print("outputs:", [(dets[n]["name"][-20:], tuple(o.shape)) for n, o in zip(sig[key]["outputs"], res)])
    inp.numpy().astype(np.float32).tofile(f"{tag}_input.bin")
    # dump feats (output 0-ish) expected for device compare
    big = max(res, key=lambda o: o.size)
    big.astype(np.float32).tofile(f"{tag}_expected.bin")


if __name__ == "__main__":
    main()
