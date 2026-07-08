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

"""Hybrid deployment graph split for Mimi.

The transformer is fp16-limited on Mali -> CPU; the convs -> GPU.
Produces the exact tflite the Android module loads + validates the full
round-trip:

  ENCODE: audio --[GPU enc_conv]--> feat
          --[CPU enc_tx: enc_transformer+downsample]--> emb
          --[CPU RVQ encode]--> codes
  DECODE: codes --[CPU RVQ decode]--> emb
          --[CPU dec_tx: upsample+dec_transformer]--> conv_in
          --[GPU deconly: SEANet decoder]--> audio

Graphs (fixed length; module rebuilds per duration or pads):
  mimi_enc_conv.tflite   (GPU)  audio(1,1,L) -> feat(1,512,Se)
  mimi_enc_tx.tflite     (CPU)  feat(1,Se,512) -> emb(1,512,Tc)
  mimi_dec_tx.tflite     (CPU)  emb(1,512,Tc) -> conv_in(1,512,seq)
  mimi_deconly.tflite    (GPU)  conv_in(1,512,seq) -> audio(1,1,L)
  mimi_rvq.bin           (CPU/Kotlin)  RVQ encode+decode weights

  python build_hybrid_graphs.py
"""
import _stub
import os
import sys
import wave
import numpy as np
import torch
import torch.nn as nn
import build_mimi as B
from transformers import MimiModel
from transformers.models.mimi.modeling_mimi import MimiConv1d as MC

HERE = B.HERE
# Validation clip: any short 24 kHz mono 16-bit PCM WAV works. Pass a path
# as argv[1] or drop sample_24k.wav next to this script.
WAV = (sys.argv[1] if len(sys.argv) > 1
       else os.path.join(HERE, "sample_24k.wav"))
SECS = 2.0


class EncConv(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, audio):
        return self.enc(audio)  # (1,512,Se)


class EncTx(nn.Module):
    def __init__(self, fwd, downsample):
        super().__init__()
        self.fn = fwd
        self.ds = downsample

    def forward(self, feat):
        # feat (1,Se,512) -> emb (1,512,Tc)
        return self.ds(self.fn(feat).transpose(1, 2))


class DecTx(nn.Module):
    def __init__(self, upsample, fwd):
        super().__init__()
        self.up = upsample
        self.fn = fwd

    def forward(self, emb):
        # emb (1,512,Tc) -> conv_in (1,512,seq)
        return self.fn(self.up(emb).transpose(1, 2)).transpose(1, 2)


class DecOnlyCF(nn.Module):
    """SEANet decoder, channel-first input (1,512,seq) -> audio.

    The input layout matches the dec_tx output.
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(x)


def cap_lengths(model, emb_or_audio, kind):
    """Records per-conv input lengths via forward pre-hooks.

    Args:
        model: MimiModel whose conv input lengths are captured.
        emb_or_audio: Example input tensor for the traced path.
        kind: "enc" traces the encode path, anything else the decode
            path.

    Returns:
        A tuple (L, LC) of dicts mapping module name -> input length,
        for ConvTranspose1d and MimiConv1d modules respectively.
    """
    L = {}
    LC = {}
    hk = []
    for n, mo in model.named_modules():
        if isinstance(mo, nn.ConvTranspose1d):
            hk.append(mo.register_forward_pre_hook(
                (lambda nm: (lambda md, i: L.__setitem__(
                    nm, i[0].shape[-1])))(n)))
        elif isinstance(mo, MC):
            hk.append(mo.register_forward_pre_hook(
                (lambda nm: (lambda md, i: LC.__setitem__(
                    nm, i[0].shape[-1])))(n)))
    with torch.no_grad():
        if kind == "enc":
            f = model.encoder(emb_or_audio)
            edt = model.encoder_transformer(
                f.transpose(1, 2), return_dict=False)[0]
            model.downsample(edt.transpose(1, 2))
        else:
            model.decoder(model.upsample(emb_or_audio))
    for h in hk:
        h.remove()
    return L, LC


def main():
    """Builds the 4 hybrid graphs and validates the full round-trip."""
    m = MimiModel.from_pretrained("kyutai/mimi").eval()
    cfg = m.config
    w = wave.open(WAV, "rb")
    frames = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    audio_np = (frames.astype(np.float32) / 32768.0)[:int(24000 * SECS)]
    audio = torch.from_numpy(audio_np).reshape(1, 1, -1)

    with torch.no_grad():
        feat = m.encoder(audio)
        Se = feat.shape[-1]
        edt = m.encoder_transformer(feat.transpose(1, 2), return_dict=False)[0]
        emb = m.downsample(edt.transpose(1, 2))
        Tc = emb.shape[-1]
        seq = 2 * Tc
        up = m.upsample(emb)
        ddt = m.decoder_transformer(up.transpose(1, 2), return_dict=False)[0]
        ref_audio = m.decoder(ddt.transpose(1, 2))
    print(f"audio {audio.shape[-1]} -> feat Se={Se} -> emb Tc={Tc} "
          f"seq={seq} -> audio {ref_audio.shape[-1]}")

    # ---------- GPU enc_conv ----------
    me = MimiModel.from_pretrained("kyutai/mimi").eval()
    _, LCe = cap_lengths(me, audio, "enc")
    B.bake_mimi_convs(me, LCe)
    B.swap_elu(me.encoder)
    enc_conv = EncConv(me.encoder).eval()
    # ---------- CPU enc_tx (transformer + downsample) ----------
    etx_fwd = B.reauth_transformer(me.encoder_transformer, cfg, Se)
    # me.downsample already baked by cap/bake above
    enc_tx = EncTx(etx_fwd, me.downsample).eval()
    # ---------- GPU deconly + CPU dec_tx ----------
    md = MimiModel.from_pretrained("kyutai/mimi").eval()
    Ld, LCd = cap_lengths(md, emb, "dec")
    B.bake_mimi_convs(md, LCd)
    B.swap_convtranspose(md, Ld)
    B.swap_elu(md.decoder)
    dtx_fwd = B.reauth_transformer(md.decoder_transformer, cfg, seq)
    dec_tx = DecTx(md.upsample, dtx_fwd).eval()
    deconly = DecOnlyCF(md.decoder).eval()

    # convert
    p_ec = B.convert(enc_conv, (audio,),
                     os.path.join(HERE, "mimi_enc_conv.tflite"))
    p_et = B.convert(enc_tx, (feat.transpose(1, 2),),
                     os.path.join(HERE, "mimi_enc_tx.tflite"))
    p_dt = B.convert(dec_tx, (emb,), os.path.join(HERE, "mimi_dec_tx.tflite"))
    # deconly input is (1,512,seq)
    p_do = B.convert(deconly, (ddt.transpose(1, 2),),
                     os.path.join(HERE, "mimi_deconly.tflite"))
    for p, lab, gpu in [(p_ec, "enc_conv", 1), (p_et, "enc_tx", 0),
                        (p_dt, "dec_tx", 0), (p_do, "deconly", 1)]:
        B.opcheck(p, lab + (" [GPU]" if gpu else " [CPU]"))
        B.to_fp16(p, p.replace(".tflite", "_fp16.tflite"))

    # ---------- full round-trip via tflite (CPU interp) + numpy RVQ ----------
    import mimi_rvq_validate_export as R
    sem = m.quantizer.semantic_residual_vector_quantizer
    aco = m.quantizer.acoustic_residual_vector_quantizer
    sem_CB = [R.codebook_embed(l.codebook) for l in sem.layers]
    aco_CB = [R.codebook_embed(l.codebook) for l in aco.layers]
    sem_Win = sem.input_proj.weight.detach().numpy()[:, :, 0]
    aco_Win = aco.input_proj.weight.detach().numpy()[:, :, 0]
    sem_Wout = sem.output_proj.weight.detach().numpy()[:, :, 0]
    aco_Wout = aco.output_proj.weight.detach().numpy()[:, :, 0]

    def tfl(path, x):
        # Inference through the LiteRT CompiledModel API; returns the
        # shaped output.
        from ai_edge_litert.compiled_model import CompiledModel
        model = CompiledModel.from_file(path)
        key = list(model.get_signature_list())[0]
        oname = list(model.get_signature_list()[key]["outputs"])[0]
        shp = tuple(model.get_output_tensor_details(key)[oname]["shape"])
        ins = model.create_input_buffers(0)
        outs = model.create_output_buffers(0)
        ins[0].write(np.ascontiguousarray(x, dtype=np.float32))
        model.run_by_index(0, ins, outs)
        return outs[0].read(int(np.prod(shp)), np.float32).reshape(shp)

    feat_t = tfl(p_ec, audio.numpy())
    emb_t = tfl(p_et, np.transpose(feat_t, (0, 2, 1)))
    codes_t = R.rvq_encode(emb_t, sem_Win, sem_CB, aco_Win, aco_CB)
    emb_back = R.rvq_decode(codes_t, sem_CB, sem_Wout, aco_CB, aco_Wout)
    conv_in = tfl(p_dt, emb_back.astype(np.float32))
    audio_t = tfl(p_do, conv_in)

    with torch.no_grad():
        # quantized reference
        ref_q = m.decode(torch.from_numpy(codes_t)[None])[0].numpy().ravel()
    n = min(audio_t.size, ref_q.size)
    c = np.corrcoef(audio_t.ravel()[:n], ref_q[:n])[0, 1]
    match = float(
        (codes_t == m.encode(audio).audio_codes[0].numpy()).mean()) * 100
    print(f"\n>>> FULL HYBRID ROUND-TRIP (tflite CPU + numpy RVQ) vs "
          f"torch: audio corr {c:.6f}  codes match {match:.1f}%")
    B._wav = lambda p, x: None
    a = (np.clip(audio_t.ravel(), -1, 1) * 32767).astype("<i2")
    wv = wave.open(os.path.join(HERE, "real_roundtrip_audio.wav"), "wb")
    wv.setnchannels(1)
    wv.setsampwidth(2)
    wv.setframerate(24000)
    wv.writeframes(a.tobytes())
    wv.close()
    print("wrote real_roundtrip_audio.wav + 4 graphs (fp32+fp16). "
          "GPU: enc_conv, deconly. CPU: enc_tx, dec_tx.")


if __name__ == "__main__":
    main()
