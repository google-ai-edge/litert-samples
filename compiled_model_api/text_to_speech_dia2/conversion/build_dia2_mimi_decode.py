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

"""Rebuild the Dia2 Mimi decode graph as a single fixed-length window.

The decode path is upsample -> **causal** decoder transformer -> SEANet. The
transformer attends over the whole prefix, so its receptive field is unbounded:
decoding disjoint 13-frame windows starts each one with no history and costs
~16% relative error per window (overall corr 0.991 vs the torch decode).

Because the path is causal, decoding one window that spans the entire utterance
-- with the unused tail zero-padded -- reproduces the full-sequence output
exactly for every real frame. Export that window once at DECODE_FRAMES and drop
the chunk loop on device.

Needs the Mimi conversion helpers (`build_mimi.py`, which bakes the causal convs
and re-authors the transformer). Point MIMI_WORK at the directory holding it:

    MIMI_WORK=/path/to/mimi-work python scripts/build_dia2_mimi_decode.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

MIMI_WORK = os.environ.get("MIMI_WORK")
if not MIMI_WORK or not os.path.isdir(MIMI_WORK):
    raise SystemExit(
        "set MIMI_WORK to the directory containing build_mimi.py")

# transformers must be imported before build_mimi: build_mimi pulls in `_stub`,
# which replaces scipy.optimize with a dummy and breaks sklearn's import chain.
from transformers import MimiModel
from transformers.models.mimi.modeling_mimi import MimiConv1d as MC

sys.path.insert(0, MIMI_WORK)
import build_mimi as B  # noqa: E402

OUT = os.environ.get("DIA2_OUT", "out")
DECODE_FRAMES = 256          # >= max undelayed frames the app can produce
SAMPLES_PER_FRAME = 1920


class MimiDecode(nn.Module):
    """emb (1,512,T) -> audio (1,1,T*1920): upsample + transformer + SEANet."""

    def __init__(self, upsample, transformer_fwd, decoder):
        super().__init__()
        self.upsample = upsample
        self.transformer_fwd = transformer_fwd
        self.decoder = decoder

    def forward(self, emb):
        up = self.upsample(emb).transpose(1, 2)
        conv_in = self.transformer_fwd(up).transpose(1, 2)
        return self.decoder(conv_in)


def cap_lengths(model, emb):
    """Records the fixed input length every conv sees, so it can be baked.

    Args:
      model: The Mimi model.
      emb: A probe latent of the exported window shape.

    Returns:
      A (conv_transpose_lengths, conv_lengths) pair keyed by module name.
    """
    conv_t, conv_c, hooks = {}, {}, []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ConvTranspose1d):
            hooks.append(mod.register_forward_pre_hook(
                (lambda n: (lambda m, i: conv_t.__setitem__(
                    n, i[0].shape[-1])))(name)))
        elif isinstance(mod, MC):
            hooks.append(mod.register_forward_pre_hook(
                (lambda n: (lambda m, i: conv_c.__setitem__(
                    n, i[0].shape[-1])))(name)))
    with torch.no_grad():
        model.decoder(model.upsample(emb))
    for h in hooks:
        h.remove()
    return conv_t, conv_c


def main():
    """Exports the one-shot Mimi decode window."""
    model = MimiModel.from_pretrained("kyutai/mimi").eval()
    cfg = model.config
    emb = torch.zeros(1, 512, DECODE_FRAMES)
    seq = 2 * DECODE_FRAMES              # upsample doubles the frame rate

    with torch.no_grad():
        up = model.upsample(emb)
        ddt = model.decoder_transformer(
            up.transpose(1, 2), return_dict=False)[0]
        ref = model.decoder(ddt.transpose(1, 2))
    print(f"emb {tuple(emb.shape)} -> seq {seq} -> audio {tuple(ref.shape)} "
          f"({ref.shape[-1] // DECODE_FRAMES} samples/frame)")
    assert ref.shape[-1] == DECODE_FRAMES * SAMPLES_PER_FRAME

    conv_t, conv_c = cap_lengths(model, emb)
    B.bake_mimi_convs(model, conv_c)
    B.swap_convtranspose(model, conv_t)
    B.swap_elu(model.decoder)
    transformer_fwd = B.reauth_transformer(model.decoder_transformer, cfg, seq)

    mod = MimiDecode(model.upsample, transformer_fwd, model.decoder).eval()
    with torch.no_grad():
        probe = mod(emb)
    print("re-authored output:", tuple(probe.shape))

    path = os.path.join(OUT, f"dia2_mimi_decode_t{DECODE_FRAMES}.tflite")
    B.convert(mod, (emb,), path)
    print("wrote", path, f"{os.path.getsize(path) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
