#!/usr/bin/env python3
"""Speaker diarization on-device build:
  A) WeSpeaker ResNet34 embedding -> LiteRT GPU tflite (input: CMN'd kaldi-fbank [1, 500, 80])
  B) PyanNet segmentation-3.0 (SincNet+BiLSTM, GPU-hostile) -> ONNX for onnxruntime CPU

Run: ~/clipconv/bin/python build_diar.py
"""
import os, sys, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 16000
T_FRAMES = 500                     # 5 s fbank window
N_SAMPLES = 400 + (T_FRAMES - 1) * 160   # 80240

_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
torchaudio.AudioMetaData = getattr(torchaudio, "AudioMetaData", type("AudioMetaData", (), {}))
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from pyannote.audio import Model  # noqa: E402
import torchaudio.compliance.kaldi as kaldi  # noqa: E402

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


class GPUWeSpeaker(nn.Module):
    """CMN'd fbank [1, T, 80] -> embedding [1, 256]. Mirrors WeSpeakerResNet34.forward after
    compute_fbank; StatsPool std re-authored (unbiased var via down-scaled sum, fp16-safe)."""

    SAFE = 16.0

    def __init__(self, m):
        super().__init__()
        self.r = m.resnet

    def forward(self, fbank):
        r = self.r
        x = fbank.permute(0, 2, 1).unsqueeze(1)          # [1, 1, 80, T]
        x = F.relu(r.bn1(r.conv1(x)))
        x = r.layer1(x)
        x = r.layer2(x)
        x = r.layer3(x)
        x = r.layer4(x)                                   # [1, C, 10, T/8]
        B, C, D, Tp = x.shape
        seq = x.reshape(1, C * D, Tp)                     # match TSTP rearrange (C-major)
        m = seq.mean(dim=2, keepdim=True)
        d = (seq - m) * (1.0 / self.SAFE)
        var = (d * d).sum(dim=2) * (self.SAFE * self.SAFE / (Tp - 1))
        std = torch.sqrt(var + 1e-12)
        stats = torch.cat([m.squeeze(2), std], dim=1)     # [1, 2*C*D]
        embed_a = r.seg_1(stats)
        if r.two_emb_layer:
            out = F.relu(embed_a)
            out = r.seg_bn_1(out)
            return r.seg_2(out)
        return embed_a


def main():
    emb = Model.from_pretrained(os.path.join(HERE, "wespeaker", "pytorch_model.bin")).eval()
    print("two_emb_layer:", emb.resnet.two_emb_layer)

    # fixture: 5.015 s of real speech -> fbank + CMN
    wav, sr = torchaudio.load(os.path.expanduser("~/Downloads/meeting/wav2vec2-work/sample_speech.wav"))
    wav = torchaudio.functional.resample(wav.mean(0, keepdim=True), sr, SR)[:, :N_SAMPLES]
    assert wav.shape[1] == N_SAMPLES, wav.shape
    with torch.no_grad():
        ref = emb(wav[None, 0:1])                        # pyannote wrapper output
        ref = ref[1] if isinstance(ref, tuple) else ref
        fb = kaldi.fbank(wav * (1 << 15), num_mel_bins=80, frame_length=25.0, frame_shift=10.0,
                         dither=0.0, window_type="hamming", use_energy=False,
                         sample_frequency=SR)
        fb = fb - fb.mean(dim=0, keepdim=True)           # CMN
        g = GPUWeSpeaker(emb).eval()
        mine = g(fb[None])
    print("fbank", fb.shape, "ref emb", ref.shape)
    cos = F.cosine_similarity(mine, ref).item()
    print(f"[torch-vs-torch] cosine {cos:.7f}  max|d| {(mine-ref).abs().max():.3e}")
    if cos < 0.9999:
        print("PARITY FAIL")
        return
    np.save(os.path.join(HERE, "emb_fb_fix.npy"), fb.numpy())
    np.save(os.path.join(HERE, "emb_ref_fix.npy"), ref.numpy())

    import litert_torch
    fp32 = os.path.join(HERE, "wespeaker_emb.tflite")
    litert_torch.convert(g, (fb[None],)).export(fp32)
    it32, clean = opcheck(fp32, "fp32")
    d = it32.get_input_details()[0]
    it32.set_tensor(d["index"], fb[None].numpy().astype(np.float32))
    it32.invoke()
    o32 = it32.get_tensor(it32.get_output_details()[0]["index"])
    print(f"[fp32 tflite] cosine {F.cosine_similarity(torch.from_numpy(o32), ref).item():.7f}")

    if clean:
        fp16 = to_fp16(fp32, os.path.join(HERE, "wespeaker_emb_fp16.tflite"))
        it16, _ = opcheck(fp16, "fp16")
        d = it16.get_input_details()[0]
        it16.set_tensor(d["index"], fb[None].numpy().astype(np.float32))
        it16.invoke()
        o16 = it16.get_tensor(it16.get_output_details()[0]["index"])
        print(f"[fp16 tflite] cosine {F.cosine_similarity(torch.from_numpy(o16), ref).item():.7f}")
        fb[None].numpy().astype(np.float32).tofile(os.path.join(HERE, "emb_input.bin"))
        np.save(os.path.join(HERE, "emb_ref16.npy"), o16)

    # ---- B) segmentation -> ONNX
    seg = Model.from_pretrained(os.path.join(HERE, "seg30", "pytorch_model.bin")).eval()
    x = torch.from_numpy(np.load(os.path.join(HERE, "seg_in.npy")))
    ref_ps = np.load(os.path.join(HERE, "seg_out.npy"))
    onnx_path = os.path.join(HERE, "pyannote_seg30.onnx")
    torch.onnx.export(seg, (x,), onnx_path, input_names=["waveform"], output_names=["powerset"],
                      opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    o = sess.run(None, {"waveform": x.numpy()})[0]
    corr = np.corrcoef(o.ravel(), ref_ps.ravel())[0, 1]
    agree = (o[0].argmax(1) == ref_ps[0].argmax(1)).mean()
    print(f"[seg onnx] corr {corr:.6f}  argmax agree {agree*100:.2f}%  size "
          f"{os.path.getsize(onnx_path)/1e6:.1f}MB")


if __name__ == "__main__":
    main()
