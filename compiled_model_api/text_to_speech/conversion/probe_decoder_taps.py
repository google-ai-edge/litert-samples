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

"""Builds a DEBUG decoder graph with intermediate taps to localize the Mali NaN.

Same re-authoring as the real decoder, but forward() returns
[v, tap_resnet0, tap_attn0, tap_upsample] so one on-device run isolates the
bad op (GN/Mish vs attention/SnakeBeta vs ZeroStuffConvT1d).

Run: python probe_decoder_taps.py -> artifacts/matcha_decoder_dbg.tflite
"""

import _stub  # noqa: F401  (must be first: scipy / getsourcefile guards)

import os
import types

import numpy as np  # noqa: F401  (kept for parity with sibling probes)
import torch
import torch.nn as nn
from einops import pack, rearrange

import build_matcha as B

T = 512


def dbg_forward(self, x, mask, mu, t):
    """Decoder.forward copy that also returns intermediate taps.

    Args:
        self: The matcha Decoder this function is bound to.
        x: Noisy mel (B, 80, T).
        mask: Float mel mask (B, 1, T).
        mu: Expanded encoder output (B, 80, T).
        t: Time embedding input.

    Returns:
        (v, tap_up1_resnet, tap_up1_attn, tap_up1_upsample).
    """

    def dec2(m):
        b, c, length = m.shape
        return m.reshape(b, c, length // 2, 2)[:, :, :, 0]

    taps = {}
    t = self.time_embeddings(t)
    t = self.time_mlp(t)
    x = pack([x, mu], "b * t")[0]
    hiddens = []
    masks = [mask]
    down_blocks = enumerate(self.down_blocks)
    for block_index, (resnet, transformer_blocks, downsample) in down_blocks:
        mask_down = masks[-1]
        x = resnet(x, mask_down, t)
        if block_index == 0:
            taps["resnet0"] = x
        x = rearrange(x, "b c t -> b t c")
        mask_down = rearrange(mask_down, "b 1 t -> b t")
        for tb in transformer_blocks:
            x = tb(hidden_states=x, attention_mask=mask_down, timestep=t)
        if block_index == 0:
            taps["attn0"] = rearrange(x, "b t c -> b c t")
        x = rearrange(x, "b t c -> b c t")
        mask_down = rearrange(mask_down, "b t -> b 1 t")
        hiddens.append(x)
        x = downsample(x * mask_down)
        masks.append(dec2(mask_down))
    masks = masks[:-1]
    mask_mid = masks[-1]
    for resnet, transformer_blocks in self.mid_blocks:
        x = resnet(x, mask_mid, t)
        x = rearrange(x, "b c t -> b t c")
        mask_mid = rearrange(mask_mid, "b 1 t -> b t")
        for tb in transformer_blocks:
            x = tb(hidden_states=x, attention_mask=mask_mid, timestep=t)
        x = rearrange(x, "b t c -> b c t")
        mask_mid = rearrange(mask_mid, "b t -> b 1 t")
    up_blocks = enumerate(self.up_blocks)
    for block_index, (resnet, transformer_blocks, upsample) in up_blocks:
        mask_up = masks.pop()
        x = resnet(pack([x, hiddens.pop()], "b * t")[0], mask_up, t)
        if block_index == 1:
            taps["up1_resnet"] = x
        x = rearrange(x, "b c t -> b t c")
        mask_up = rearrange(mask_up, "b 1 t -> b t")
        for tb in transformer_blocks:
            x = tb(hidden_states=x, attention_mask=mask_up, timestep=t)
        if block_index == 1:
            taps["up1_attn"] = rearrange(x, "b t c -> b c t")
        x = rearrange(x, "b t c -> b c t")
        mask_up = rearrange(mask_up, "b t -> b 1 t")
        x = upsample(x * mask_up)
    out = self.final_block(x, mask_up)
    out = self.final_proj(out * mask_up) * mask
    return out, taps["up1_resnet"], taps["up1_attn"], x  # x = up1 upsample out


class DecoderWrap(nn.Module):
    """Reorders inputs to (x, mu, t_emb, mask) to match the shipped graph."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, x, mu, t_emb, mask):
        return self.decoder(x, mask, mu, t_emb)


def main():
    """Converts the tapped debug decoder graph and lists its outputs."""
    sd = B.load_sd()
    # Unwrap DecWrapM -> the Decoder itself.
    decoder = B.reauth_decoder_masked(B.build_decoder(sd), T).d
    decoder.forward = types.MethodType(dbg_forward, decoder)
    wrapped = DecoderWrap(decoder).eval()

    x = torch.randn(1, 80, T)
    mu = torch.randn(1, 80, T)
    t_emb = B.sin_pos_emb(torch.zeros(1), 160)
    mask = torch.ones(1, 1, T)
    out = os.path.join(B.HERE, "artifacts", "matcha_decoder_dbg.tflite")

    import litert_torch
    litert_torch.convert(wrapped, (x, mu, t_emb, mask)).export(out)

    from ai_edge_litert.compiled_model import CompiledModel
    model = CompiledModel.from_file(out)
    key = list(model.get_signature_list())[0]
    print("DBG decoder outputs (index, name, shape):")
    for detail in model.get_output_tensor_details(key).values():
        print("  ", detail["index"], detail["name"], list(detail["shape"]))
    print("wrote", out, os.path.getsize(out) // 1_000_000, "MB")


if __name__ == "__main__":
    main()
