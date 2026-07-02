#!/usr/bin/env python3
"""CMGAN reference: load TSCNet ckpt, denoise a constructed noisy sample (clean speech + noise
at ~5 dB SNR), report SNR improvement, save wavs + fixtures for conversion parity."""
import os, sys
import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "CMGAN", "src"))
SR, NFFT, HOP = 16000, 400, 100

# --- stub the pesq/joblib imports pulled by models.generator's neighbors
import types
for m in ("pesq", "joblib"):
    sys.modules.setdefault(m, types.ModuleType(m))
sys.modules["pesq"].pesq = lambda *a, **k: 0.0
sys.modules["joblib"].Parallel = object
sys.modules["joblib"].delayed = lambda f: f

from models.generator import TSCNet  # noqa: E402
from utils import power_compress, power_uncompress  # noqa: E402

model = TSCNet(num_channel=64, num_features=NFFT // 2 + 1).eval()
ckpt = torch.load(os.path.join(HERE, "CMGAN", "src", "best_ckpt", "ckpt"),
                  map_location="cpu", weights_only=False)
model.load_state_dict(ckpt)
print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M")

# --- noisy sample: clean speech + white+pink noise @ ~5 dB SNR, 4 s
clean, sr = torchaudio.load(os.path.expanduser("~/Downloads/meeting/wav2vec2-work/sample_speech.wav"))
clean = torchaudio.functional.resample(clean.mean(0, keepdim=True), sr, SR)[:, :4 * SR]
rng = np.random.default_rng(0)
white = rng.standard_normal(clean.shape[1]).astype(np.float32)
pink = np.cumsum(rng.standard_normal(clean.shape[1])).astype(np.float32)
pink /= np.abs(pink).max()
noise = torch.from_numpy(0.7 * white / np.abs(white).max() + 0.5 * pink)[None]
snr_db = 5.0
p_c = clean.pow(2).mean()
p_n = noise.pow(2).mean()
noise = noise * torch.sqrt(p_c / (p_n * 10 ** (snr_db / 10)))
noisy = clean + noise
torchaudio.save(os.path.join(HERE, "noisy.wav"), noisy / noisy.abs().max() * 0.9, SR)

# --- reference pipeline (evaluation.py semantics)
c = torch.sqrt(noisy.shape[-1] / noisy.pow(2).sum(-1))
x = noisy * c
spec = torch.stft(x, NFFT, HOP, window=torch.hamming_window(NFFT), onesided=True,
                  return_complex=True)
spec = torch.view_as_real(spec)                      # [1, F, T, 2]
spec_c = power_compress(spec).permute(0, 1, 3, 2)    # [1, 2, T, F]
with torch.no_grad():
    est_real, est_imag = model(spec_c)
est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
est = power_uncompress(est_real, est_imag).squeeze(1)  # [1, F, T, 2]
wav = torch.istft(torch.view_as_complex(est.contiguous()), NFFT, HOP,
                  window=torch.hamming_window(NFFT), onesided=True)
wav = (wav / c)[:, :clean.shape[1]]
torchaudio.save(os.path.join(HERE, "enhanced.wav"), wav / wav.abs().max() * 0.9, SR)


def snr(ref, est):
    e = ref - est
    return 10 * torch.log10(ref.pow(2).mean() / e.pow(2).mean()).item()


print(f"input SNR  {snr(clean, noisy):5.2f} dB")
print(f"output SNR {snr(clean, wav):5.2f} dB   (improvement {snr(clean, wav)-snr(clean, noisy):+.1f} dB)")
np.save(os.path.join(HERE, "ref_in_spec.npy"), spec_c.numpy())
np.save(os.path.join(HERE, "ref_out.npy"),
        np.stack([est_real.numpy(), est_imag.numpy()]))
noisy.numpy().tofile(os.path.join(HERE, "ref_noisy.bin"))
print("saved fixtures: ref_in_spec.npy / ref_out.npy / noisy.wav / enhanced.wav")
