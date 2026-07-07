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

"""Decisive graph-split feasibility test for the Mali decoder-transformer bug.

Extracts the up1 BasicTransformerBlock as a STANDALONE LiteRT graph and checks
whether it computes correctly on Mali in isolation (at large activation
magnitude). If isolated => correct, the decoder bug is a fusion artifact and
graph-splitting fixes it. If isolated => still wrong, the transformer is
fundamentally broken on Mali and splitting won't help.

Outputs: artifacts/tx_probe.tflite, tx_input.bin (fixed ±MAG input),
tx_mask.bin, tx_expected.npy.

Run: python probe_tx_standalone.py [MAG]   (default 60)
"""

import _stub  # noqa: F401  (must be first: scipy / getsourcefile guards)

import os
import sys

import numpy as np
import torch
import torch.nn as nn

import build_matcha as B

ART = os.path.join(B.HERE, "artifacts")
SEQ = 512
CH = 256
MAG = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
YLEN = 268


class TXWrap(nn.Module):
    """Standalone wrapper around one BasicTransformerBlock."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden, mask):
        # hidden: (1, SEQ, CH); mask: (1, SEQ) float 0/1.
        return self.block(hidden_states=hidden, attention_mask=mask, timestep=None)


def main():
    sd = B.load_sd()
    # Has manual-LN + patched attention already applied.
    decoder = B.reauth_decoder_masked(B.build_decoder(sd), 512).d
    # Grab the up_blocks[1] transformer block (the one that collapses on device).
    block = decoder.up_blocks[1][1][0]
    wrapped = TXWrap(block).eval()

    # Deterministic large-magnitude input (reproducible in Kotlin):
    # hidden[0, t, c] = MAG * sin(...) * cos(...).
    idx = np.arange(SEQ * CH, dtype=np.float32)
    hidden = (MAG * np.sin(idx * 0.013) * np.cos(idx * 0.0007)).reshape(
        1, SEQ, CH).astype(np.float32)
    mask = np.zeros((1, SEQ), np.float32)
    mask[:, :YLEN] = 1.0
    with torch.no_grad():
        expected = wrapped(torch.from_numpy(hidden), torch.from_numpy(mask)).numpy()
    print(f"input range [{hidden.min():.1f},{hidden.max():.1f}]  "
          f"expected out range [{expected.min():.2f},{expected.max():.2f}] "
          f"std {expected.std():.3f}")

    import litert_torch
    out = os.path.join(ART, "tx_probe.tflite")
    litert_torch.convert(wrapped, (torch.from_numpy(hidden), torch.from_numpy(mask))).export(out)
    # Bundle input + expected output for the on-device check.
    hidden.tofile(os.path.join(ART, "tx_input.bin"))
    mask.tofile(os.path.join(ART, "tx_mask.bin"))
    np.save(os.path.join(ART, "tx_expected.npy"), expected)

    # Desktop tflite (CPU) sanity check.
    from ai_edge_litert.compiled_model import CompiledModel
    model = CompiledModel.from_file(out)
    input_buffers = model.create_input_buffers(0)
    output_buffers = model.create_output_buffers(0)
    for buffer, value in zip(input_buffers, [hidden, mask]):
        buffer.write(np.ascontiguousarray(value))
    model.run_by_index(0, input_buffers, output_buffers)
    n = (model.get_output_buffer_requirements(0, 0)["buffer_size"]
         // np.dtype(np.float32).itemsize)
    got = output_buffers[0].read(n, np.float32)
    print(f"desktop-CPU tflite vs torch corr "
          f"{np.corrcoef(got, expected.ravel())[0, 1]:.6f}")
    print("wrote tx_probe.tflite, tx_input.bin, tx_mask.bin, tx_expected.npy")


if __name__ == "__main__":
    main()
