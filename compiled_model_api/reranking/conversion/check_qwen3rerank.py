#!/usr/bin/env python3
"""op-check the fp32 graph, cast to fp16, op-check again, emit device fixtures.

Run: ~/clipconv/bin/python check_qwen3emb.py
"""
import os, sys, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FP32 = os.path.join(HERE, "qwen3rerank_gpu.tflite")
FP16 = os.path.join(HERE, "qwen3rerank_gpu_fp16.tflite")

BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP", "PACK",
          "RFFT2D", "FFT", "STFT", "COMPLEX", "RFFT", "IRFFT", "CUMSUM", "SPLIT", "SPLIT_V",
          "NOT_EQUAL", "EQUAL", "GREATER", "LESS"}


def opcheck(path, label):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    ops = collections.Counter(d.get("op_name", "?") for d in it._get_ops_details())
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
    over = sum(1 for d in it.get_tensor_details() if len(d.get("shape", [])) > 4)
    print(f"[{label}] nodes:{sum(ops.values())} banned:{bad or 'NONE'} >4D:{over} "
          f"size {os.path.getsize(path)/1e6:.1f}MB")
    print(f"[{label}] ops: {dict(sorted(ops.items(), key=lambda kv: -kv[1]))}")
    print(f"[{label}] VERDICT:", "GPU-CLEAN" if not bad and not over else f"BLOCKERS {bad} >4D:{over}")
    return it, (not bad and not over)


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
