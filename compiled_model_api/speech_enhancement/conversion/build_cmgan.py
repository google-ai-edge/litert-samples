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

"""CMGAN -> LiteRT CompiledModel GPU conversion and parity checks.

Pipeline: torch parity -> convert -> op-check -> fp16 parity.

Run: python build_cmgan.py
"""
import os
import sys
import collections
import types
import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "CMGAN", "src"))
sys.path.insert(0, HERE)
for mname in ("pesq", "joblib"):
    sys.modules.setdefault(mname, types.ModuleType(mname))
sys.modules["pesq"].pesq = lambda *a, **k: 0.0
sys.modules["joblib"].Parallel = object
sys.modules["joblib"].delayed = lambda f: f

from models.generator import TSCNet  # noqa: E402
from utils import power_compress  # noqa: E402
from gpu_cmgan import GPUCMGAN  # noqa: E402

SR, NFFT, HOP = 16000, 400, 100
SECONDS = 2.0
S = int(SR * SECONDS)          # 32000 -> T = 321 frames
BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE",
          "SELECT", "SELECT_V2", "BROADCAST_TO", "POW", "TRANSPOSE_CONV",
          "CAST", "EMBEDDING_LOOKUP", "PACK", "RFFT2D", "FFT", "STFT",
          "COMPLEX", "RFFT", "IRFFT", "CUMSUM", "SPLIT", "SPLIT_V"}


def opcheck(path, label):
    """Scans a .tflite flatbuffer for GPU-incompatible ops and tensors.

    Args:
        path: Path to the .tflite model file.
        label: Short label used in the printed report.

    Returns:
        True if the model has no banned ops and no tensor with more than
        four dimensions.
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
            key = (c.customCode.decode() if c.customCode
                   else names.get(code, str(code)))
            ops[key] += 1
        over += sum(1 for t in g.tensors
                    if t.shape is not None and len(t.shape) > 4)
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
    print(f"[{label}] nodes:{sum(ops.values())} "
          f"banned:{bad or 'NONE'} >4D:{over} "
          f"size {os.path.getsize(path)/1e6:.1f}MB")
    print(f"[{label}] ops: {dict(sorted(ops.items(), key=lambda kv: -kv[1]))}")
    verdict = ("GPU-CLEAN" if not bad and not over
               else f"BLOCKERS {bad} >4D:{over}")
    print(f"[{label}] VERDICT:", verdict)
    return not bad and not over


def tfl_run(path, x):
    """Runs one inference through the LiteRT CompiledModel API.

    Args:
        path: Path to the .tflite model file.
        x: Input tensor or array; cast to contiguous float32.

    Returns:
        A list of flat float32 output arrays, one per model output.
    """
    from ai_edge_litert.compiled_model import CompiledModel
    model = CompiledModel.from_file(path)
    inputs = model.create_input_buffers(0)
    outputs = model.create_output_buffers(0)
    x = x.numpy() if hasattr(x, "numpy") else x
    inputs[0].write(np.ascontiguousarray(x, dtype=np.float32))
    model.run_by_index(0, inputs, outputs)
    result = []
    for i, buf in enumerate(outputs):
        req = model.get_output_buffer_requirements(0, i)
        n = req["buffer_size"] // np.dtype(np.float32).itemsize
        result.append(buf.read(n, np.float32))
    return result


def to_fp16(fp32, fp16):
    """Quantizes an fp32 .tflite model to fp16 weights.

    Args:
        fp32: Path to the source float32 model.
        fp16: Output path for the float16 model.

    Returns:
        The output path fp16.
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


def corr(a, b):
    """Pearson correlation between two flattened arrays.

    Args:
        a: First array-like input.
        b: Second array-like input.

    Returns:
        The correlation coefficient as a float.
    """
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return np.corrcoef(a, b)[0, 1]


def main():
    """Runs torch parity, conversion, op-check and fp16 parity."""
    model = TSCNet(num_channel=64, num_features=NFFT // 2 + 1).eval()
    model.load_state_dict(torch.load(
        os.path.join(HERE, "CMGAN", "src", "best_ckpt", "ckpt"),
        map_location="cpu", weights_only=False))

    # real noisy chunk (RMS-normalized, as evaluation.py does)
    noisy, _ = torchaudio.load(os.path.join(HERE, "noisy.wav"))
    noisy = noisy[:, :S]
    c = torch.sqrt(noisy.shape[-1] / noisy.pow(2).sum(-1))
    x = noisy * c

    # ---- original reference on the chunk
    spec = torch.view_as_real(torch.stft(
        x, NFFT, HOP, window=torch.hamming_window(NFFT),
        onesided=True, return_complex=True))
    spec_c = power_compress(spec).permute(0, 1, 3, 2)
    with torch.no_grad():
        ref_r, ref_i = model(spec_c)               # [1, 1, T, F]

    # ---- re-authored torch parity (graph input = reflect-padded wav)
    g = GPUCMGAN(model, S, NFFT, HOP).eval()
    xp = torch.nn.functional.pad(x, (NFFT // 2, NFFT // 2), mode="reflect")
    with torch.no_grad():
        est_r, est_i = g(xp)
    cr, ci = corr(est_r, ref_r), corr(est_i, ref_i)
    md = max(float((est_r - ref_r).abs().max()),
             float((est_i - ref_i).abs().max()))
    print(f"[torch-vs-torch] corr real {cr:.7f} imag {ci:.7f}  max|d| {md:.3e}")
    if min(cr, ci) < 0.9999:
        print("PARITY FAIL — bisect: encoder-out / TSCB1-out taps")
        return

    # ---- convert
    import litert_torch
    fp32 = os.path.join(HERE, "cmgan.tflite")
    litert_torch.convert(g, (xp,)).export(fp32)
    clean = opcheck(fp32, "fp32")
    o = tfl_run(fp32, xp)
    print(f"[fp32 tflite] corr real {corr(o[0], ref_r):.7f} "
          f"imag {corr(o[1], ref_i):.7f}")

    if clean:
        fp16 = to_fp16(fp32, os.path.join(HERE, "cmgan_fp16.tflite"))
        opcheck(fp16, "fp16")
        o = tfl_run(fp16, xp)
        print(f"[fp16 tflite] corr real {corr(o[0], ref_r):.7f} "
              f"imag {corr(o[1], ref_i):.7f}")
        xp.numpy().astype(np.float32).tofile(
            os.path.join(HERE, "cmgan_input.bin"))
        np.save(os.path.join(HERE, "cmgan_ref.npy"),
                np.stack([ref_r.numpy(), ref_i.numpy()]))
        print("wrote cmgan_fp16.tflite + device fixtures")


if __name__ == "__main__":
    main()
