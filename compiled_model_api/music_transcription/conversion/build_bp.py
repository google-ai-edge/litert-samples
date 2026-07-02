#!/usr/bin/env python3
"""Basic Pitch parity check: torch reimpl (gpu_bp.py) vs official ONNX on real audio."""
import os, sys
import numpy as np
import torch
import torchaudio
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gpu_bp import BasicPitchGPU, N_SAMPLES, SR

wav, sr = torchaudio.load(os.path.expanduser("~/Downloads/meeting/wav2vec2-work/sample_speech.wav"))
x = torchaudio.functional.resample(wav.mean(0, keepdim=True), sr, SR)[0]
# make it musical-ish: add a 220+440 Hz tone so notes exist
t = torch.arange(N_SAMPLES) / SR
tone = 0.3 * torch.sin(2 * np.pi * 220 * t) + 0.2 * torch.sin(2 * np.pi * 440 * t)
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
    print(f"{label:>8}: best-match {best[0][-25:]} corr {best[1]:.6f} max|d| {best[2]:.4f}")

# ---- convert + opcheck + fp16 (run when parity passes)
import collections
sys.path.insert(0, os.path.expanduser("~/Downloads/meeting/cmgan-work"))
from build_cmgan import opcheck, to_fp16

ok = all(np.corrcoef(m.numpy().ravel(), r.ravel())[0, 1] > 0.9999
         for m, r in zip(mine, [ref[2], ref[1], ref[0]]))
print("parity gate:", ok)
if ok:
    import litert_torch
    fp32 = os.path.join(HERE, "basicpitch.tflite")
    litert_torch.convert(g, (sig[None],)).export(fp32)
    it32, clean = opcheck(fp32, "fp32")
    d = it32.get_input_details()[0]
    it32.set_tensor(d["index"], sig[None].numpy())
    it32.invoke()
    outs32 = [it32.get_tensor(o["index"]) for o in sorted(it32.get_output_details(), key=lambda o: o["index"])]
    for o32, mm, nm in zip(outs32, mine, ["contour", "note", "onset"]):
        print(f"[fp32 tflite] {nm} corr {np.corrcoef(o32.ravel(), mm.numpy().ravel())[0,1]:.7f}")
    if clean:
        fp16 = to_fp16(fp32, os.path.join(HERE, "basicpitch_fp16.tflite"))
        it16, _ = opcheck(fp16, "fp16")
        d = it16.get_input_details()[0]
        it16.set_tensor(d["index"], sig[None].numpy())
        it16.invoke()
        outs16 = [it16.get_tensor(o["index"]) for o in sorted(it16.get_output_details(), key=lambda o: o["index"])]
        for o16, mm, nm in zip(outs16, mine, ["contour", "note", "onset"]):
            print(f"[fp16 tflite] {nm} corr {np.corrcoef(o16.ravel(), mm.numpy().ravel())[0,1]:.7f}")
        sig[None].numpy().tofile(os.path.join(HERE, "bp_input.bin"))
        np.save(os.path.join(HERE, "bp_ref.npy"), np.concatenate([m.numpy().ravel() for m in mine]))
        print("wrote basicpitch_fp16.tflite + fixtures")
