# Extract the exact preprocessor buffers (mel filterbank + STFT window) from the NeMo model.
import _stub, torch, numpy as np, os
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
m = EncDecHybridRNNTCTCBPEModel.restore_from("parakeet110m.nemo", map_location="cpu").eval()
fe = m.preprocessor.featurizer
fb = fe.fb.detach().numpy()                 # [80, 257]
win = fe.window.detach().numpy()            # [400] or [512]?
print("fb", fb.shape, "win", win.shape, "win sum %.4f" % float(win.sum()), flush=True)
print("n_fft", fe.n_fft, "win_length", fe.win_length, "hop", fe.hop_length,
      "mag_power", fe.mag_power, "log", fe.log, "normalize", fe.normalize,
      "guard", fe.log_zero_guard_value, "dither", fe.dither, "preemph", fe.preemph, flush=True)
np.save("prep_fb.npy", fb.astype(np.float32))
np.save("prep_win.npy", win.astype(np.float32))
print("saved prep_fb.npy + prep_win.npy", flush=True)
os._exit(0)
