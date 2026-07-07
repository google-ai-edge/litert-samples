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

"""Convert the 3DDFA_V2 MobileNetV1 regressor (120x120x3 -> 62 params) to fp16 tflite.

Expects the 3DDFA_V2 repository checked out next to this script and the
reference fixtures produced by ref_tddfa.py in ./ref. Writes the fp32/fp16
.tflite models plus the Kotlin fixture and BFM recon-asset binaries to ./out.
"""
import collections
import os
import sys
import types

import numpy as np
import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3DDFA_V2")
sys.path.insert(0, REPO)
os.chdir(REPO)

# Stub the Cython NMS extension so TDDFA imports without building FaceBoxes.
_m = types.ModuleType("FaceBoxes.utils.nms.cpu_nms")
_m.cpu_nms = lambda *a, **k: []
_m.cpu_soft_nms = lambda *a, **k: []
sys.modules["FaceBoxes.utils.nms.cpu_nms"] = _m

import litert_torch
import yaml
from TDDFA import TDDFA

BANNED = {"GATHER_ND", "GATHER", "SELECT", "SELECT_V2", "NOT_EQUAL", "EQUAL",
          "GREATER", "LESS", "TOPK_V2", "CAST", "PACK", "SPLIT"}


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
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED or k.startswith("Flex")}
    ok = not bad and not over
    print(f"[{label}] {sum(ops.values())} ops {dict(ops)} banned:{bad or 'NONE'} "
          f">4D:{over} -> {'GPU-CLEAN' if ok else 'BLOCKERS'}")
    return ok


def run_tflite(path, x):
    """Single inference through the LiteRT CompiledModel API; returns the flat fp32 output."""
    from ai_edge_litert.compiled_model import CompiledModel
    model = CompiledModel.from_file(path)
    inputs = model.create_input_buffers(0)
    outputs = model.create_output_buffers(0)
    inputs[0].write(np.ascontiguousarray(x, dtype=np.float32))
    model.run_by_index(0, inputs, outputs)
    n = model.get_output_buffer_requirements(0, 0)["buffer_size"] // np.dtype(np.float32).itemsize
    return outputs[0].read(n, np.float32)


def to_fp16(fp32, fp16):
    """Weight-only fp16 cast via ai-edge-quantizer (compute stays float)."""
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT), algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(fp16): os.remove(fp16)
    qt = quantizer.Quantizer(float_model=fp32)
    qt.load_quantization_recipe(rm.get_quantization_recipe())
    qt.quantize().export_model(fp16)
    return fp16


def main():
    work = os.path.dirname(REPO)
    out_dir = os.path.join(work, "out")
    os.makedirs(out_dir, exist_ok=True)
    ref = np.load(os.path.join(work, "ref", "ref_emma.npz"))

    cfg = yaml.load(open("configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
    tddfa = TDDFA(gpu_mode=False, **cfg)
    model = tddfa.model.eval()

    inp = torch.from_numpy(ref["inp"].astype(np.float32))          # [1,3,120,120]
    with torch.no_grad():
        out = model(inp).numpy()[0]                                # 62 (normalized)
    # de-normalize to compare to the reference param
    param = out * tddfa.param_std + tddfa.param_mean
    print("torch param corr vs ref:", float(np.corrcoef(param, ref["param"])[0, 1]),
          "maxdiff", float(np.abs(param - ref["param"]).max()))

    fp32 = os.path.join(out_dir, "tddfa_mb1.tflite")
    litert_torch.convert(model, (inp,)).export(fp32)
    clean = opcheck(fp32, "tddfa fp32")
    if clean:
        fp16 = to_fp16(fp32, os.path.join(out_dir, "tddfa_mb1_fp16.tflite"))
        opcheck(fp16, "tddfa fp16")
        # tflite parity
        o = run_tflite(fp16, ref["inp"].astype(np.float32))
        p16 = o * tddfa.param_std + tddfa.param_mean
        print("tflite fp16 param corr vs ref:", float(np.corrcoef(p16, ref["param"])[0, 1]),
              "maxdiff", float(np.abs(p16 - ref["param"]).max()))
        # what matters: reconstruct the 68 landmarks from the fp16 params, compare 2D pixel error
        ver16 = tddfa.recon_vers([p16.astype(np.float32)], [ref["roi"]], dense_flag=False)[0]  # [3,68]
        d2d = np.sqrt((ver16[0] - ref["ver"][0]) ** 2 + (ver16[1] - ref["ver"][1]) ** 2)
        print(f"landmark 2D pixel error (img coords, face ~{int(ref['roi'][2]-ref['roi'][0])}px): "
              f"mean {d2d.mean():.2f}px max {d2d.max():.2f}px")
        ref["inp"].astype("<f4").tofile(os.path.join(out_dir, "tddfa_input.bin"))
        ref["param"].astype("<f4").tofile(os.path.join(out_dir, "tddfa_param_ref.bin"))
        # host recon assets as raw f32 bins for Kotlin
        ra = np.load(os.path.join(work, "ref", "recon_assets.npz"))
        for k in ["u_base", "w_shp_base", "w_exp_base", "param_mean", "param_std"]:
            ra[k].astype("<f4").tofile(os.path.join(out_dir, f"tddfa_{k}.bin"))
        print("wrote fp16 + fixtures + recon assets")


if __name__ == "__main__":
    main()
