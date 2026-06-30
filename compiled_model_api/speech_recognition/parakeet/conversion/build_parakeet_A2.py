# Process A2 (NeMo only): REAL speech -> mel/enc/ctc reference + torch greedy-CTC transcript (ground truth).
import _stub, torch, numpy as np, os, soundfile as sf
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
m = EncDecHybridRNNTCTCBPEModel.restore_from("parakeet110m.nemo", map_location="cpu").eval()
enc, ctc, prep, tok = m.encoder, m.ctc_decoder, m.preprocessor, m.tokenizer
prep.featurizer.dither = 0.0          # deterministic mel (device gets this exact mel)
prep.featurizer.pad_to = 0

audio, sr = sf.read("sample.wav")
assert sr == 16000, sr
audio = torch.from_numpy(audio).float().unsqueeze(0)
alen = torch.tensor([audio.shape[1]])
with torch.no_grad():
    mel, mlen = prep(input_signal=audio, length=alen)
    enc_out, enc_len = enc(audio_signal=mel, length=mlen)
    logp = ctc(encoder_output=enc_out)                  # [1,T',1025]
print("mel", tuple(mel.shape), "enc_out", tuple(enc_out.shape), "logp", tuple(logp.shape), flush=True)
print("mel absmax %.2f enc_out absmax %.2f" % (float(mel.abs().max()), float(enc_out.abs().max())), flush=True)

ids = logp[0].argmax(-1).tolist()
blank = logp.shape[-1] - 1
collapsed, prev = [], -1
for i in ids:
    if i != blank and i != prev: collapsed.append(i)
    prev = i
text = tok.ids_to_text(collapsed)
print("TORCH greedy-CTC transcript:", repr(text), flush=True)
print("collapsed ids:", collapsed, flush=True)

np.save("ref_mel.npy", mel.numpy()); np.save("ref_encout.npy", enc_out.numpy()); np.save("ref_logp.npy", logp.numpy())
np.save("ref_collapsed.npy", np.array(collapsed))
print("SAVED real-speech reference.", flush=True)
os._exit(0)
