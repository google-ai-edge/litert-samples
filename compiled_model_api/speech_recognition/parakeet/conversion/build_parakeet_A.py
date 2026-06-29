# Process A (NeMo only, NO litert_torch): load model, save encoder+ctc modules + reference I/O.
import _stub, torch, numpy as np, os
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
m = EncDecHybridRNNTCTCBPEModel.restore_from("parakeet110m.nemo", map_location="cpu").eval()
enc, ctc, prep = m.encoder, m.ctc_decoder, m.preprocessor
# reference: 5s random audio -> mel -> encoder -> ctc logits
torch.manual_seed(0)
audio = torch.randn(1, 16000*5) * 0.1
alen = torch.tensor([audio.shape[1]])
with torch.no_grad():
    mel, mlen = prep(input_signal=audio, length=alen)          # [1,80,T]
    enc_out, enc_len = enc(audio_signal=mel, length=mlen)      # [1,512,T']
    logp = ctc(encoder_output=enc_out)                          # [1,T',1025] log-probs
print("mel", tuple(mel.shape), "enc_out", tuple(enc_out.shape), "logp", tuple(logp.shape), flush=True)
torch.save({"encoder": enc, "ctc": ctc}, "parakeet_modules.pt")
np.save("ref_mel.npy", mel.numpy()); np.save("ref_encout.npy", enc_out.numpy()); np.save("ref_logp.npy", logp.numpy())
print("SAVED modules + reference. mel absmax", float(mel.abs().max()), "encout absmax", float(enc_out.abs().max()), flush=True)
os._exit(0)
