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

"""Validate a numpy reimplementation of Mimi's SPLIT RVQ decode (the host/Kotlin glue) against
torch, and export the weights. RVQ is the only non-GPU decode part (int64 CAST + EMBEDDING_LOOKUP
fail on Mali, cf. DAC). Split = 1 semantic + 31 acoustic codebooks; each group sums its codebook
lookups in the 256-dim VQ space, then a shared output_proj (Conv1d 256->512, no bias) lifts to
hidden; emb = semantic_out + acoustic_out.

  decode: codes[1,32,T] -> emb[1,512,T]   (== m.quantizer.decode; feeds mimi_decode.tflite)

Run: ~/clipconv/bin/python mimi_rvq_validate_export.py
"""
import _stub  # noqa: F401
import os
import numpy as np
import torch
from transformers import MimiModel

HERE = os.path.dirname(os.path.abspath(__file__))


def codebook_embed(cb):
    """MimiEuclideanCodebook.embed property: embed_sum / cluster_usage.clamp(min=eps).
    Unused rows (cluster_usage~0) blow up to inf; they are never indexed by real codes, so
    zero them to keep the exported blob clean (does not touch any used/finite row)."""
    e = (cb.embed_sum / cb.cluster_usage.clamp(min=cb.epsilon)[:, None]).detach().numpy().astype("<f4")
    return np.nan_to_num(e, posinf=0.0, neginf=0.0)


def rvq_decode(codes, sem_CB, sem_Wout, aco_CB, aco_Wout):
    """codes [32,T] int -> emb [1,512,T]. Mirrors MimiSplitResidualVectorQuantizer.decode."""
    T = codes.shape[-1]
    q_sem = sem_CB[0][codes[0]].T                       # (256,T)
    sem_out = sem_Wout @ q_sem                          # (512,T)
    q_aco = np.zeros((aco_CB[0].shape[1], T), np.float32)
    for i in range(len(aco_CB)):
        q_aco += aco_CB[i][codes[1 + i]].T
    aco_out = aco_Wout @ q_aco                          # (512,T)
    return (sem_out + aco_out)[None]                    # (1,512,T)


def nearest(res_T, CB):
    """Euclidean argmin (MimiEuclideanCodebook): argmin_c ||r-c||^2 == argmin_c (||c||^2 - 2 r.c)."""
    cc = (CB * CB).sum(1)                               # (2048,)
    return (cc[None, :] - 2.0 * (res_T @ CB.T)).argmin(1)


def rvq_encode(emb, sem_Win, sem_CB, aco_Win, aco_CB):
    """emb [1,512,T] -> codes [32,T]. Split: semantic + acoustic each encode emb independently
    via their own input_proj; residual loop inside each group. Mirrors Split RVQ encode."""
    x = emb[0]                                          # (512,T)
    T = x.shape[-1]
    codes = np.zeros((1 + len(aco_CB), T), np.int64)
    res = sem_Win @ x                                  # (256,T) semantic
    codes[0] = nearest(res.T, sem_CB[0])
    res = aco_Win @ x                                  # (256,T) acoustic
    for i in range(len(aco_CB)):
        c = nearest(res.T, aco_CB[i])
        codes[1 + i] = c
        res = res - aco_CB[i][c].T
    return codes


def main():
    m = MimiModel.from_pretrained("kyutai/mimi").eval()
    sem = m.quantizer.semantic_residual_vector_quantizer
    aco = m.quantizer.acoustic_residual_vector_quantizer
    sem_CB = [codebook_embed(l.codebook) for l in sem.layers]          # 1 x (2048,256)
    aco_CB = [codebook_embed(l.codebook) for l in aco.layers]          # 31 x (2048,256)
    assert sem.output_proj.bias is None and aco.output_proj.bias is None
    sem_Wout = sem.output_proj.weight.detach().numpy()[:, :, 0].astype("<f4")  # (512,256)
    aco_Wout = aco.output_proj.weight.detach().numpy()[:, :, 0].astype("<f4")
    sem_Win = sem.input_proj.weight.detach().numpy()[:, :, 0].astype("<f4")    # (256,512)
    aco_Win = aco.input_proj.weight.detach().numpy()[:, :, 0].astype("<f4")
    print(f"semantic codebooks={len(sem_CB)} acoustic codebooks={len(aco_CB)} "
          f"dim={sem_CB[0].shape[1]} size={sem_CB[0].shape[0]}")

    # real codes
    sr = m.config.sampling_rate
    t = torch.linspace(0, 1, sr)
    audio = (0.3 * torch.sin(2 * np.pi * 220 * t) + 0.2 * torch.sin(2 * np.pi * 440 * t)).reshape(1, 1, -1)
    with torch.no_grad():
        codes = m.encode(audio).audio_codes            # (1,32,T)
        emb_ref = m.quantizer.decode(codes).numpy()    # (1,512,T)
        enc_emb = m.encoder(audio)
        enc_emb = m.encoder_transformer(enc_emb.transpose(1, 2), return_dict=False)[0]
        enc_emb = m.downsample(enc_emb.transpose(1, 2)).numpy()   # (1,512,T) = RVQ encode input

    emb_mine = rvq_decode(codes[0].numpy(), sem_CB, sem_Wout, aco_CB, aco_Wout)
    c = np.corrcoef(emb_mine.flatten(), emb_ref.flatten())[0, 1]
    print(f">>> RVQ decode vs torch corr: {c:.6f}  max|diff|={np.abs(emb_mine-emb_ref).max():.2e}")

    codes_mine = rvq_encode(enc_emb, sem_Win, sem_CB, aco_Win, aco_CB)
    match = float((codes_mine == codes[0].numpy()).mean())
    print(f">>> RVQ encode codes match torch: {match*100:.2f}%")

    # export weights for Kotlin: contiguous float32 LE.
    #   sem_Win (256*512), aco_Win (256*512), sem_Wout (512*256), aco_Wout (512*256),
    #   sem_CB[0] (2048*256), aco_CB[0..30] (2048*256 each)
    blobs = [sem_Win.tobytes(), aco_Win.tobytes(), sem_Wout.tobytes(), aco_Wout.tobytes(),
             sem_CB[0].tobytes()] + [cb.tobytes() for cb in aco_CB]
    path = os.path.join(HERE, "mimi_rvq.bin")
    with open(path, "wb") as f:
        f.write(b"".join(blobs))
    print(f"exported -> {path} ({os.path.getsize(path)} bytes; header-less; sem_Win, aco_Win, "
          f"sem_Wout, aco_Wout, sem_CB, 31x aco_CB ; dim=256 size=2048)")


if __name__ == "__main__":
    main()
