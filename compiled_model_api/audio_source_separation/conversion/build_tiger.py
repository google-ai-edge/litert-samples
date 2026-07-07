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

"""TIGER-DnR (dialog sub-model first) -> LiteRT CompiledModel GPU.

Phases:
  1. torch-vs-torch: GPUTiger (re-authored, exact) vs original look2hear TIGER on a real 12 s
     chunk of the repo test mixture -> waveform corr must be ~1.0 BEFORE any conversion.
  2. litert-torch convert -> op histogram (banned / >4D / FFT) -> tflite CPU parity.
  3. fp16 (FLOAT_CASTING) -> parity again -> device fixtures.

Run: ~/clipconv/bin/python build_tiger.py [dialog|effect|music]
"""
import sys
import os
import collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stub_tiger  # noqa: F401
import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "TIGER"))
import look2hear.models  # noqa: E402
from gpu_tiger import GPUTiger, host_istft, host_pad  # noqa: E402

SUB = sys.argv[1] if len(sys.argv) > 1 else "dialog"
SR = 44100
WIN, HOP = 2048, 512
T_FRAMES = int(os.environ.get("TIGER_T", "1040"))  # divisible by 16 -> uniform fast path
S = (T_FRAMES - 1) * HOP       # 1040 -> 531968 samples = 12.06 s

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
    it.set_tensor(d["index"], x.astype(np.float32))
    it.invoke()
    outs = sorted(it.get_output_details(), key=lambda o: o["index"])
    return [it.get_tensor(o["index"]) for o in outs]


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
    dnr = look2hear.models.TIGERDNR.from_pretrained(os.path.join(HERE, "ckpt-dnr")).eval()
    sub = getattr(dnr, SUB)
    print(f"sub-model '{SUB}': {sum(p.numel() for p in sub.parameters())/1e6:.2f}M params, "
          f"{sub.nband} bands, {sub.num_output} sources, iters={sub.separator.iter}")

    wav, sr = torchaudio.load(os.path.join(HERE, "TIGER", "test", "test_mixture_466.wav"))
    assert sr == SR
    x = wav[:, :S]
    if x.shape[-1] < S:
        x = torch.cat([x, torch.zeros(1, S - x.shape[-1])], -1)

    # ---- original reference on the chunk
    with torch.no_grad():
        ref = sub(x)                                   # [1, K, S]
    print("orig chunk out", tuple(ref.shape), "absmax", ref.abs().max().item())

    # ---- re-authored torch parity
    g = GPUTiger(sub, S).eval()
    xp = host_pad(x, WIN)
    with torch.no_grad():
        est_r, est_i = g(xp)
    K = est_r.shape[1]
    est = host_istft(est_r.view(K, sub.enc_dim, -1), est_i.view(K, sub.enc_dim, -1),
                     WIN, HOP, S)                      # [K, S]
    c = corr(est.numpy(), ref[0].numpy())
    md = np.abs(est.numpy() - ref[0].numpy()).max()
    print(f"[torch-vs-torch] corr {c:.7f}  max|d| {md:.3e}")
    if c < 0.9999:
        print("PARITY FAIL — fix re-authoring before converting")
        # per-source diagnostics
        for k in range(K):
            print(f"  src{k}: corr {corr(est[k].numpy(), ref[0,k].numpy()):.6f}")
        return

    # ---- convert
    import litert_torch
    fp32 = os.path.join(HERE, f"tiger_{SUB}.tflite")
    litert_torch.convert(g, (xp,)).export(fp32)
    it32, clean = opcheck(fp32, "fp32")
    o = tfl_run(it32, xp.numpy())
    est32 = host_istft(torch.from_numpy(o[0]).view(K, sub.enc_dim, -1),
                       torch.from_numpy(o[1]).view(K, sub.enc_dim, -1), WIN, HOP, S)
    print(f"[fp32 tflite] wav corr {corr(est32.numpy(), ref[0].numpy()):.7f}")

    if clean:
        fp16 = to_fp16(fp32, os.path.join(HERE, f"tiger_{SUB}_fp16.tflite"))
        it16, _ = opcheck(fp16, "fp16")
        o = tfl_run(it16, xp.numpy())
        est16 = host_istft(torch.from_numpy(o[0]).view(K, sub.enc_dim, -1),
                           torch.from_numpy(o[1]).view(K, sub.enc_dim, -1), WIN, HOP, S)
        print(f"[fp16 tflite] wav corr {corr(est16.numpy(), ref[0].numpy()):.7f}")
        xp.numpy().astype(np.float32).tofile(os.path.join(HERE, f"tiger_input.bin"))
        np.save(os.path.join(HERE, f"tiger_{SUB}_ref.npy"), ref[0].numpy())
        print("wrote fp16 + device fixtures")


if __name__ == "__main__":
    main()
