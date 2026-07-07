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

"""PANNs CNN14 (AudioSet 527-tag) -> LiteRT CompiledModel GPU.

Deployment = host-side log-mel + a single GPU graph for the CNN body:

    waveform[320000] --[Kotlin log-mel]--> logmel[1,1,1001,64] --[GPU CNN14]--> probs[527]

Why the log-mel is host-side and not in the graph: PANNs builds the spectrogram with torchlibrosa's
STFT (a DFT-as-Conv1d, so there is NO FFT op and the full raw-audio graph IS op-clean — only the
center reflect-pad emits a single GATHER_ND, removable with pad_mode='constant'). BUT the converted
spectral front-end is numerically wrong (fp32 corr 0.19) and the power spectrum |STFT|^2 (~1e6)
overflows fp16 on Mali -> NaN. The CNN body alone (logmel -> tags) converts at corr 1.000000 in both
fp32 and fp16, so we keep it on the GPU and compute the log-mel on the CPU (Whisper/Kokoro pattern),
matched to torchlibrosa exactly (validated here in numpy at corr 1.0).

Deps: the qiuqiangkong model defs (`models.py` as panns_models.py + `pytorch_utils.py`) from
github.com/qiuqiangkong/audioset_tagging_cnn, and Cnn14_mAP=0.431.pth (Zenodo, CC-BY-4.0):
    https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth
Run: python build_panns.py
"""
import _stub_propack  # noqa: F401  (narrow scipy _propack shim; keeps librosa real)
import sys
import os
import csv
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, ".")
from panns_models import Cnn14

HERE = os.path.dirname(os.path.abspath(__file__))
SR, NFFT, HOP, NMEL = 32000, 1024, 320, 64
PAD = NFFT // 2
CLIP_SAMPLES = 320000          # 10 s @ 32 kHz (PANNs canonical eval window)
CFG = dict(sample_rate=SR, window_size=NFFT, hop_size=HOP, mel_bins=NMEL, fmin=50, fmax=14000, classes_num=527)
CKPT = os.path.join(HERE, "Cnn14_mAP=0.431.pth")
SAMPLE_WAV = os.path.join(HERE, "sample.wav")   # any mono clip; resampled to 32 kHz, padded/cropped to 10 s
LABELS_CSV = os.path.join(HERE, "class_labels_indices.csv")
BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP",
          "RFFT2D", "FFT", "STFT", "COMPLEX", "RFFT", "IRFFT", "CUMSUM"}


class LogmelCNN(nn.Module):
    """The GPU graph: logmel[B,1,T,64] -> clipwise_output[B,527]. Mirrors Cnn14.forward after logmel."""
    def __init__(s, m):
        super().__init__()
        s.m = m
    def forward(s, x):
        m = s.m
        x = x.transpose(1, 3)
        x = m.bn0(x)
        x = x.transpose(1, 3)
        x = m.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = m.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = m.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = m.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = m.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = m.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.relu_(m.fc1(x))
        return torch.sigmoid(m.fc_audioset(x))


def numpy_logmel(wav, melW):
    """What the Kotlin front-end computes: reflect-pad center, periodic Hann, FFT, power, melW, 10log10."""
    padded = np.pad(wav, (PAD, PAD), mode="reflect")
    n = 1 + (len(padded) - NFFT) // HOP
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / NFFT)
    power = np.empty((n, NFFT // 2 + 1), np.float64)
    for t in range(n):
        s = np.fft.rfft(padded[t * HOP: t * HOP + NFFT] * win, n=NFFT)
        power[t] = s.real ** 2 + s.imag ** 2
    return (10.0 * np.log10(np.maximum(power @ melW, 1e-10))).astype(np.float32)


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
    labels = [r[2] for r in list(csv.reader(open(LABELS_CSV)))[1:]]
    m = Cnn14(**CFG).eval()
    m.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])
    melW = m.logmel_extractor.melW.detach().numpy()      # [513,64]

    import librosa
    wav, _ = librosa.load(SAMPLE_WAV, sr=SR, mono=True)
    wav = np.pad(wav, (0, max(0, CLIP_SAMPLES - len(wav))))[:CLIP_SAMPLES].astype(np.float32)

    with torch.no_grad():
        logmel = m.logmel_extractor(m.spectrogram_extractor(torch.from_numpy(wav)[None]))  # [1,1,T,64]
        y = LogmelCNN(m).eval()(logmel).numpy().ravel()

    # validate the host-side (Kotlin) log-mel against torch
    lm_np = numpy_logmel(wav, melW)
    lm_t = logmel.numpy().reshape(-1, NMEL)
    print(f"host log-mel vs torch: corr {np.corrcoef(lm_t.ravel(), lm_np.ravel())[0,1]:.6f} "
          f"max|d| {np.abs(lm_t-lm_np).max():.4f}")

    # convert the CNN body
    import litert_torch
    fp32 = os.path.join(HERE, "cnn14_audioset.tflite")
    litert_torch.convert(LogmelCNN(m).eval(), (logmel,)).export(fp32)
    opcheck(fp32, "fp32")
    o32 = run_tflite(fp32, logmel.numpy())
    print(f"[fp32] tflite-vs-torch corr {np.corrcoef(o32, y)[0,1]:.6f}")
    fp16 = to_fp16(fp32, os.path.join(HERE, "cnn14_audioset_fp16.tflite"))
    opcheck(fp16, "fp16")
    o16 = run_tflite(fp16, logmel.numpy())
    print(f"[fp16] tflite-vs-torch corr {np.corrcoef(o16, y)[0,1]:.6f}")

    # assets + fixtures
    melW.T.astype(np.float32).tofile(os.path.join(HERE, "mel_basis.bin"))   # [64,513] mel-major
    wav.tofile(os.path.join(HERE, "panns_input.bin"))
    np.save(os.path.join(HERE, "panns_ref.npy"), o16)
    top = np.argsort(o16)[::-1][:8]
    print("top tags:", [(labels[i], round(float(o16[i]), 3)) for i in top])
    print("wrote cnn14_audioset_fp16.tflite + mel_basis.bin + fixtures")


if __name__ == "__main__":
    main()
