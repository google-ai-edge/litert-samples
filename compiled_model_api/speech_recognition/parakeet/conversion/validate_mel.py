# Validate a from-scratch (numpy) NeMo-equivalent mel vs ref_mel, using the EXACT model buffers.
import numpy as np, soundfile as sf

SR, NFFT, WIN, HOP, NMEL = 16000, 512, 400, 160, 80
GUARD = 2.0**-24
PREEMPH = 0.97

audio, sr = sf.read("sample.wav"); assert sr == SR
x = audio.astype(np.float64)

# preemphasis: y[0]=x[0]; y[t]=x[t]-0.97*x[t-1]
y = np.empty_like(x); y[0] = x[0]; y[1:] = x[1:] - PREEMPH * x[:-1]
x = y

fb = np.load("prep_fb.npy")[0].astype(np.float64)      # [80,257] exact
hann = np.load("prep_win.npy").astype(np.float64)      # [400] exact
win = np.zeros(NFFT); off = (NFFT - WIN) // 2; win[off:off+WIN] = hann   # center-pad to 512

pad = NFFT // 2
xp = np.pad(x, (pad, pad), mode="constant")
nfr = (len(xp) - NFFT) // HOP + 1
frames = np.stack([xp[i*HOP:i*HOP+NFFT] * win for i in range(nfr)], 0)   # [nfr,512]
spec = np.fft.rfft(frames, n=NFFT, axis=1)                              # [nfr,257]
power = spec.real**2 + spec.imag**2                                     # |.|^2 (mag_power=2)
mel = fb @ power.T                                                      # [80,nfr]
mel = np.log(mel + GUARD)
mu = mel.mean(1, keepdims=True)
sd = np.sqrt(((mel - mu)**2).sum(1, keepdims=True) / (mel.shape[1] - 1)) + 1e-5
mel = (mel - mu) / sd

ref = np.load("ref_mel.npy")[0]
T = min(mel.shape[1], ref.shape[1])
corr = np.corrcoef(mel[:, :T].ravel(), ref[:, :T].ravel())[0,1]
mad = np.abs(mel[:, :T] - ref[:, :T]).max()
print("from-scratch mel", mel.shape, "ref", ref.shape, "corr %.6f maxabsdiff %.5f" % (corr, mad))
