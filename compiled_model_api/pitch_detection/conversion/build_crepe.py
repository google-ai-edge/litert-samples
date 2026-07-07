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

"""CREPE pitch detection (marl/crepe via torchcrepe, MIT) -> LiteRT CompiledModel GPU.

The whole model rides one GPU graph: 6x {zero-pad -> Conv2d -> ReLU -> BatchNorm -> MaxPool} +
permute/reshape (<=4D) + Linear + sigmoid -> 360 pitch-bin activations. Pure CNN, no banned ops,
per-frame-normalized input (values ~O(1)) so no fp16 wall. Host-side (Kotlin): frame the 16 kHz
audio into 1024-sample windows, per-frame normalize (mean/std), and decode 360 bins -> cents -> Hz.

Run: ~/clipconv/bin/python build_crepe.py
"""
import os
import collections
import numpy as np
import torch
import torch.nn as nn
import torchcrepe

HERE = os.path.dirname(os.path.abspath(__file__))
SR, WIN, BINS, CENTS_PER_BIN = 16000, 1024, 360, 20
CENTS_OFFSET = 1997.3794084376191   # torchcrepe.convert.bins_to_cents intercept
CAP = "full"
BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "PACK", "SPLIT"}


def load_model():
    m = torchcrepe.Crepe(CAP)
    f = os.path.join(os.path.dirname(torchcrepe.__file__), "assets", f"{CAP}.pth")
    m.load_state_dict(torch.load(f, map_location="cpu", weights_only=True))
    return m.eval()


def norm_frame(frame):
    frame = frame - frame.mean()
    return frame / max(frame.std(), 1e-10)


def decode(act):
    """360 activations -> Hz (local weighted average around argmax, torchcrepe 'weighted_argmax')."""
    c = int(np.argmax(act))
    s, e = max(0, c - 4), min(BINS, c + 5)
    w = act[s:e]
    b = np.arange(s, e)
    cents = CENTS_PER_BIN * (np.sum(w * b) / np.sum(w)) + CENTS_OFFSET
    return 10.0 * 2 ** (cents / 1200.0)


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
    print(f"[{label}] banned:{bad or 'NONE'} >4D:{over} size {os.path.getsize(path)/1e6:.1f}MB "
          f"VERDICT {'GPU-CLEAN' if not bad and not over else bad}")


def tfl(path, x):
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
    m = load_model()

    # self-test: synth a 440 Hz (A4) sine, take a centered 1024-frame, normalize
    for f0 in (220.0, 440.0, 880.0):
        t = np.arange(WIN) / SR
        frame = np.sin(2 * np.pi * f0 * t).astype(np.float32)
        x = torch.from_numpy(norm_frame(frame))[None, :]   # [1,1024]
        with torch.no_grad():
            act = m(x).numpy().ravel()
        print(f"torch: f0={f0:.1f} Hz -> predicted {decode(act):.1f} Hz")

    # use the 440 Hz frame as the conversion fixture
    t = np.arange(WIN) / SR
    frame = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    x = torch.from_numpy(norm_frame(frame))[None, :]
    with torch.no_grad():
        ref = m(x).numpy().ravel()

    import litert_torch
    fp32 = os.path.join(HERE, "crepe_full.tflite")
    litert_torch.convert(m, (x,)).export(fp32)
    opcheck(fp32, "fp32")
    o32 = tfl(fp32, x.numpy()).ravel()
    print(f"[fp32] tflite-vs-torch corr {np.corrcoef(o32, ref)[0,1]:.6f}  decoded {decode(o32):.1f} Hz")

    fp16 = to_fp16(fp32, os.path.join(HERE, "crepe_full_fp16.tflite"))
    opcheck(fp16, "fp16")
    o16 = tfl(fp16, x.numpy()).ravel()
    print(f"[fp16] tflite-vs-torch corr {np.corrcoef(o16, ref)[0,1]:.6f}  decoded {decode(o16):.1f} Hz")

    x.numpy().astype(np.float32).tofile(os.path.join(HERE, "crepe_input.bin"))
    np.save(os.path.join(HERE, "crepe_ref.npy"), o16)
    print("wrote crepe_full_fp16.tflite + crepe_input.bin + crepe_ref.npy")


if __name__ == "__main__":
    main()
