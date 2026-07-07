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

"""op-check the fp32 graph, cast to fp16, op-check again, emit device fixtures.

Run: python check_qwen3emb.py
"""
import collections
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FP32 = os.path.join(HERE, "qwen3emb_gpu.tflite")
FP16 = os.path.join(HERE, "qwen3emb_gpu_fp16.tflite")

BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP", "PACK",
          "RFFT2D", "FFT", "STFT", "COMPLEX", "RFFT", "IRFFT", "CUMSUM", "SPLIT", "SPLIT_V",
          "NOT_EQUAL", "EQUAL", "GREATER", "LESS"}


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
    print(f"[{label}] nodes:{sum(ops.values())} banned:{bad or 'NONE'} >4D:{over} "
          f"size {os.path.getsize(path)/1e6:.1f}MB")
    print(f"[{label}] ops: {dict(sorted(ops.items(), key=lambda kv: -kv[1]))}")
    print(f"[{label}] VERDICT:", "GPU-CLEAN" if not bad and not over else f"BLOCKERS {bad} >4D:{over}")
    return not bad and not over


def to_fp16(fp32, fp16):
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


if __name__ == "__main__":
    print("=== op-check fp32 ===", flush=True)
    opcheck(FP32, "fp32")
    print("=== casting to fp16 ===", flush=True)
    to_fp16(FP32, FP16)
    print("=== op-check fp16 ===", flush=True)
    opcheck(FP16, "fp16")
    print("done", flush=True)
