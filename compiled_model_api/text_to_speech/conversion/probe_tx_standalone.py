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

#!/usr/bin/env python3
"""Decisive graph-split feasibility test: extract the up1 BasicTransformerBlock as a
STANDALONE LiteRT graph and check whether it computes correctly on Mali in isolation
(at large activation magnitude). If isolated => correct, the decoder bug is a fusion
artifact and graph-splitting fixes it. If isolated => still wrong, the transformer is
fundamentally broken on Mali and splitting won't help.

Outputs: artifacts/tx_probe.tflite, tx_input.bin (fixed ±MAG input), tx_expected.npy.
Run: ~/clipconv/bin/python probe_tx_standalone.py [MAG]   (default 60)
"""
import _stub, os, sys, types as _t, numpy as np, torch, torch.nn as nn
import build_matcha as B
from einops import rearrange

ART = os.path.join(B.HERE, "artifacts")
SEQ, CH = 512, 256
MAG = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
YLEN = 268


def main():
    sd = B.load_sd()
    dec = B.reauth_decoder_masked(B.build_decoder(sd), 512).d   # has manual-LN + patched attn
    # grab the up_blocks[1] transformer block (the one that collapses on device)
    blk = dec.up_blocks[1][1][0]

    class TXWrap(nn.Module):
        def __init__(s, b): super().__init__(); s.b = b
        def forward(s, hidden, mask):     # hidden (1,SEQ,CH); mask (1,SEQ) float 0/1
            return s.b(hidden_states=hidden, attention_mask=mask, timestep=None)
    w = TXWrap(blk).eval()

    # deterministic large-magnitude input (reproducible in Kotlin): hidden[0,t,c] = MAG*sin(...)
    idx = np.arange(SEQ * CH, dtype=np.float32)
    hidden = (MAG * np.sin(idx * 0.013) * np.cos(idx * 0.0007)).reshape(1, SEQ, CH).astype(np.float32)
    mask = np.zeros((1, SEQ), np.float32); mask[:, :YLEN] = 1.0
    with torch.no_grad():
        exp = w(torch.from_numpy(hidden), torch.from_numpy(mask)).numpy()
    print(f"input range [{hidden.min():.1f},{hidden.max():.1f}]  expected out range [{exp.min():.2f},{exp.max():.2f}] std {exp.std():.3f}")

    import litert_torch
    out = os.path.join(ART, "tx_probe.tflite")
    litert_torch.convert(w, (torch.from_numpy(hidden), torch.from_numpy(mask))).export(out)
    # bundle input + expected
    hidden.tofile(os.path.join(ART, "tx_input.bin"))
    mask.tofile(os.path.join(ART, "tx_mask.bin"))
    np.save(os.path.join(ART, "tx_expected.npy"), exp)

    # desktop tflite (CPU) sanity
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=out); it.allocate_tensors()
    ins = it.get_input_details()
    for d, v in zip(ins, [hidden, mask]): it.set_tensor(d["index"], v.astype(d["dtype"]))
    it.invoke()
    o = it.get_tensor(it.get_output_details()[0]["index"])
    print(f"desktop-CPU tflite vs torch corr {np.corrcoef(o.ravel(), exp.ravel())[0,1]:.6f}")
    print("wrote tx_probe.tflite, tx_input.bin, tx_mask.bin, tx_expected.npy")


if __name__ == "__main__":
    main()
