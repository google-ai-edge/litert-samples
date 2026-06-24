"""SSDlite320-MobileNetV3-Large (detection) -> LiteRT, PATCH-FREE clean path.

NCHW input + 4D-head-tap: return each FPN level's RAW conv outputs
(cls [N, A*91, H, W], box [N, A*4, H, W]); anchor-decode + multiclass-NMS go to Kotlin.
This rewrites NO model-internal op (= U2-Net d0 / YOLOX raw-head technique), so it stays
clean / community-eligible.

NOTE: do NOT use to_channel_last_io here. Its channel-last pass turns MobileNetV3's 8
SqueezeExcitation global-avg-pools into GATHER_ND + 5D. Keeping NCHW IO converts stock-clean
with zero model patches. (Clean NHWC would require a converter-side fix for channel-last x
global-pool, not a model monkeypatch.)

Requires: torch>=2.11, torchvision, litert-torch, ai-edge-litert, ai-edge-quantizer.

    python conversion/convert_ssdlite.py
"""
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import litert_torch
from ai_edge_litert.interpreter import Interpreter
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights)

BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "CAST", "GELU", "ERF", "WHERE",
          "SELECT", "SELECT_V2", "BROADCAST_TO", "POW", "NON_MAX_SUPPRESSION_V5"}


class SSDRaw4D(nn.Module):
    """Return per-level raw head conv outputs (4D); decode + NMS done in app code.

    Output order (12 tensors): for each of the 6 FPN levels, (cls, box):
      cls[i] = [N, A*91, H, W]   box[i] = [N, A*4, H, W]
    with A=6 anchors/loc, levels H=W in {20,10,5,3,2,1}.
    """
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        feats = list(self.m.backbone(x).values())
        ch = self.m.head.classification_head.module_list
        rh = self.m.head.regression_head.module_list
        outs = []
        for i, f in enumerate(feats):
            outs.append(ch[i](f))   # [N, A*91, H, W]
            outs.append(rh[i](f))   # [N, A*4,  H, W]
        return tuple(outs)


def fp16_recipe():
    """The known-good FLOAT_CASTING fp16 recipe (same path as MiDaS/U2Net/ZeroDCE)."""
    from ai_edge_quantizer import recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    op_config = qtyping.OpQuantizationConfig(
        weight_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
        compute_precision=qtyping.ComputePrecision.FLOAT,
    )
    rm.add_quantization_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=op_config,
        algorithm_key=AlgorithmName.FLOAT_CASTING,
    )
    return rm.get_quantization_recipe()


def op_report(path):
    itp = Interpreter(model_path=path)
    itp.allocate_tensors()
    ops = Counter(o["op_name"] for o in itp._get_ops_details())
    maxnd = max(len(t["shape"]) for t in itp.get_tensor_details())
    over4 = sum(1 for t in itp.get_tensor_details() if len(t["shape"]) > 4)
    banned = sorted(k for k in ops if k.upper() in BANNED)
    flex = sorted(k for k in ops if "flex" in k.lower() or k.lower() == "custom")
    return itp, dict(ops), banned, flex, maxnd, over4


def parity(itp, x, ref):
    """Run tflite, match each output to a torch ref by unique shape, return corrs."""
    ins, outs = itp.get_input_details(), itp.get_output_details()
    itp.set_tensor(ins[0]["index"], x.numpy())
    itp.invoke()
    tfl_outs = [itp.get_tensor(o["index"]) for o in outs]
    used, corrs = set(), []
    for to in tfl_outs:
        for j, ro in enumerate(ref):
            if j not in used and tuple(ro.shape) == tuple(to.shape):
                corrs.append(float(np.corrcoef(ro.flatten(), to.flatten())[0, 1]))
                used.add(j)
                break
    return corrs


def main():
    out_dir = "out/ssdlite"
    os.makedirs(out_dir, exist_ok=True)
    m = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT).eval()
    model = SSDRaw4D(m).eval()
    x = torch.rand(1, 3, 320, 320)
    with torch.no_grad():
        ref = [o.numpy() for o in model(x)]

    fp32 = f"{out_dir}/ssdlite_320_4dtap_clean.tflite"
    litert_torch.convert(model, (x,)).export(fp32)
    print(f"FP32 exported: {round(os.path.getsize(fp32)/1e6, 2)} MB")

    itp, ops, banned, flex, maxnd, over4 = op_report(fp32)
    print(f"  ops={ops}")
    print(f"  >>> BANNED={banned or 'NONE'} FLEX/CUSTOM={flex or 'NONE'} "
          f"max_ndim={maxnd} over4D={over4}")
    c32 = parity(itp, x, ref)
    print(f"  >>> FP32 parity: {len(c32)}/{len(ref)} matched, min corr={round(min(c32), 6)}")

    # FP16 (float_casting) — the GPU-deployment recipe (same as all shipped samples)
    from ai_edge_quantizer import quantizer
    fp16 = f"{out_dir}/ssdlite_320_4dtap_clean_fp16.tflite"
    if os.path.exists(fp16):
        os.remove(fp16)
    qt = quantizer.Quantizer(float_model=fp32)
    qt.load_quantization_recipe(fp16_recipe())
    qt.quantize().export_model(fp16)
    print(f"FP16 exported: {round(os.path.getsize(fp16)/1e6, 2)} MB")
    itp16, _, b16, f16, nd16, o16 = op_report(fp16)
    print(f"  >>> FP16 BANNED={b16 or 'NONE'} FLEX/CUSTOM={f16 or 'NONE'} "
          f"max_ndim={nd16} over4D={o16}")
    c16 = parity(itp16, x, ref)
    print(f"  >>> FP16 parity vs torch: {len(c16)}/{len(ref)} matched, "
          f"min corr={round(min(c16), 6)}")


if __name__ == "__main__":
    main()
