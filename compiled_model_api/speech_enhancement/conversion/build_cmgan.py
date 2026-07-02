#!/usr/bin/env python3
"""CMGAN -> LiteRT CompiledModel GPU: torch parity -> convert -> op-check -> fp16 parity.

Run: ~/clipconv/bin/python build_cmgan.py
"""
import os, sys, collections, types
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
BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP", "PACK",
          "RFFT2D", "FFT", "STFT", "COMPLEX", "RFFT", "IRFFT", "CUMSUM", "SPLIT", "SPLIT_V"}


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


def tfl_run(it, x):
    d = it.get_input_details()[0]
    it.set_tensor(d["index"], x.numpy().astype(np.float32))
    it.invoke()
    return [it.get_tensor(o["index"]) for o in
            sorted(it.get_output_details(), key=lambda o: o["index"])]


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


def corr(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return np.corrcoef(a, b)[0, 1]


def main():
    model = TSCNet(num_channel=64, num_features=NFFT // 2 + 1).eval()
    model.load_state_dict(torch.load(os.path.join(HERE, "CMGAN", "src", "best_ckpt", "ckpt"),
                                     map_location="cpu", weights_only=False))

    # real noisy chunk (RMS-normalized, as evaluation.py does)
    noisy, _ = torchaudio.load(os.path.join(HERE, "noisy.wav"))
    noisy = noisy[:, :S]
    c = torch.sqrt(noisy.shape[-1] / noisy.pow(2).sum(-1))
    x = noisy * c

    # ---- original reference on the chunk
    spec = torch.view_as_real(torch.stft(x, NFFT, HOP, window=torch.hamming_window(NFFT),
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
    md = max(float((est_r - ref_r).abs().max()), float((est_i - ref_i).abs().max()))
    print(f"[torch-vs-torch] corr real {cr:.7f} imag {ci:.7f}  max|d| {md:.3e}")
    if min(cr, ci) < 0.9999:
        print("PARITY FAIL — bisect: encoder-out / TSCB1-out taps")
        return

    # ---- convert
    import litert_torch
    fp32 = os.path.join(HERE, "cmgan.tflite")
    litert_torch.convert(g, (xp,)).export(fp32)
    it32, clean = opcheck(fp32, "fp32")
    o = tfl_run(it32, xp)
    print(f"[fp32 tflite] corr real {corr(o[0], ref_r):.7f} imag {corr(o[1], ref_i):.7f}")

    if clean:
        fp16 = to_fp16(fp32, os.path.join(HERE, "cmgan_fp16.tflite"))
        it16, _ = opcheck(fp16, "fp16")
        o = tfl_run(it16, xp)
        print(f"[fp16 tflite] corr real {corr(o[0], ref_r):.7f} imag {corr(o[1], ref_i):.7f}")
        xp.numpy().astype(np.float32).tofile(os.path.join(HERE, "cmgan_input.bin"))
        np.save(os.path.join(HERE, "cmgan_ref.npy"),
                np.stack([ref_r.numpy(), ref_i.numpy()]))
        print("wrote cmgan_fp16.tflite + device fixtures")


if __name__ == "__main__":
    main()
