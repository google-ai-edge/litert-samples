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

"""Basic Pitch parity check: torch reimpl (gpu_bp.py) vs official ONNX.

Runs the re-implemented BasicPitchGPU against the official ONNX model on a
short audio clip, then converts the torch model to a .tflite, op-checks it,
quantizes to fp16, and compares tflite vs torch through the LiteRT
CompiledModel API.
"""
import os
import sys
import collections
import numpy as np
import torch
import torchaudio
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gpu_bp import BasicPitchGPU, N_SAMPLES, SR  # noqa: E402

# Validation fixture: any speech/music clip works. Pass a path as argv[1] or
# drop sample_speech.wav next to this script.
SAMPLE_WAV = (sys.argv[1] if len(sys.argv) > 1
              else os.path.join(HERE, "sample_speech.wav"))
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
    if not bad and not over:
        verdict = "GPU-CLEAN"
    else:
        verdict = f"BLOCKERS {bad} >4D:{over}"
    print(f"[{label}] VERDICT:", verdict)
    return not bad and not over


def tfl_run(path, x_in):
    """Runs one inference through the LiteRT CompiledModel API.

    Args:
        path: Path to the .tflite file.
        x_in: Input array written to the first input buffer.

    Returns:
        A list of flat fp32 output arrays in buffer order.
    """
    from ai_edge_litert.compiled_model import CompiledModel
    model = CompiledModel.from_file(path)
    inputs = model.create_input_buffers(0)
    outputs = model.create_output_buffers(0)
    x_in = x_in.numpy() if hasattr(x_in, "numpy") else x_in
    inputs[0].write(np.ascontiguousarray(x_in, dtype=np.float32))
    model.run_by_index(0, inputs, outputs)
    result = []
    for i, buf in enumerate(outputs):
        req = model.get_output_buffer_requirements(0, i)
        n = req["buffer_size"] // np.dtype(np.float32).itemsize
        result.append(buf.read(n, np.float32))
    return result


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


wav, sr = torchaudio.load(SAMPLE_WAV)
x = torchaudio.functional.resample(wav.mean(0, keepdim=True), sr, SR)[0]
# make it musical-ish: add a 220+440 Hz tone so notes exist
t = torch.arange(N_SAMPLES) / SR
tone = (0.3 * torch.sin(2 * np.pi * 220 * t)
        + 0.2 * torch.sin(2 * np.pi * 440 * t))
sig = torch.zeros(N_SAMPLES)
n = min(x.shape[0], N_SAMPLES)
sig[:n] = x[:n] * 0.5
sig = sig + tone
sig = (sig / sig.abs().max() * 0.7).float()

sess = ort.InferenceSession(os.path.join(HERE, "nmp_official.onnx"),
                            providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
ref = sess.run(None, {iname: sig.numpy().reshape(1, N_SAMPLES, 1)})
onames = [o.name for o in sess.get_outputs()]
print("onnx outs:", [(nm, r.shape) for nm, r in zip(onames, ref)])

g = BasicPitchGPU(os.path.join(HERE, "bp_weights.npz")).eval()
with torch.no_grad():
    mine = g(sig[None])
print("torch outs:", [tuple(m.shape) for m in mine])

# match by shape (264 -> contour, 88 x2 -> note/onset by best corr)
by_shape = {r.shape[-1]: [] for r in ref}
for nm, r in zip(onames, ref):
    by_shape[r.shape[-1]].append((nm, r))
contour_ref = by_shape[264][0][1]
for m, label in zip(mine, ["contour", "note", "onset"]):
    best = None
    for nm, r in zip(onames, ref):
        if r.shape[-1] != m.shape[-1]:
            continue
        c = np.corrcoef(m.numpy().ravel(), r.ravel())[0, 1]
        if best is None or c > best[1]:
            best = (nm, c, np.abs(m.numpy() - r).max())
    print(f"{label:>8}: best-match {best[0][-25:]} corr {best[1]:.6f} "
          f"max|d| {best[2]:.4f}")

# ---- convert + opcheck + fp16 (run when parity passes)
ok = all(np.corrcoef(m.numpy().ravel(), r.ravel())[0, 1] > 0.9999
         for m, r in zip(mine, [ref[2], ref[1], ref[0]]))
print("parity gate:", ok)
if ok:
    import litert_torch
    fp32 = os.path.join(HERE, "basicpitch.tflite")
    litert_torch.convert(g, (sig[None],)).export(fp32)
    clean = opcheck(fp32, "fp32")
    outs32 = tfl_run(fp32, sig[None])
    for o32, mm, nm in zip(outs32, mine, ["contour", "note", "onset"]):
        corr32 = np.corrcoef(o32.ravel(), mm.numpy().ravel())[0, 1]
        print(f"[fp32 tflite] {nm} corr {corr32:.7f}")
    if clean:
        fp16 = to_fp16(fp32, os.path.join(HERE, "basicpitch_fp16.tflite"))
        opcheck(fp16, "fp16")
        outs16 = tfl_run(fp16, sig[None])
        for o16, mm, nm in zip(outs16, mine, ["contour", "note", "onset"]):
            corr16 = np.corrcoef(o16.ravel(), mm.numpy().ravel())[0, 1]
            print(f"[fp16 tflite] {nm} corr {corr16:.7f}")
        sig[None].numpy().tofile(os.path.join(HERE, "bp_input.bin"))
        np.save(os.path.join(HERE, "bp_ref.npy"),
                np.concatenate([m.numpy().ravel() for m in mine]))
        print("wrote basicpitch_fp16.tflite + fixtures")
