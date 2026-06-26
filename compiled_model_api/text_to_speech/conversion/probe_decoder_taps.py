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
"""Build a DEBUG decoder graph that outputs intermediate taps, to pinpoint where the
NaN first appears on the Mali ML Drift delegate. Same re-authoring as the real decoder,
but forward() returns [v, tap_resnet0, tap_attn0, tap_upsample] so one on-device run
isolates the bad op (GN/Mish vs attention/SnakeBeta vs ZeroStuffConvT1d).
Run: ~/clipconv/bin/python probe_decoder_taps.py   ->  artifacts/matcha_decoder_dbg.tflite
"""
import _stub, os, types as _t, numpy as np, torch, torch.nn as nn
import build_matcha as B
from einops import pack, rearrange

T = 512


def dbg_forward(self, x, mask, mu, t):
    def dec2(m):
        b, c, L = m.shape
        return m.reshape(b, c, L // 2, 2)[:, :, :, 0]
    taps = {}
    t = self.time_embeddings(t); t = self.time_mlp(t)
    x = pack([x, mu], "b * t")[0]
    hiddens = []; masks = [mask]
    for bi, (resnet, tbs, downsample) in enumerate(self.down_blocks):
        md = masks[-1]
        x = resnet(x, md, t)
        if bi == 0: taps["resnet0"] = x
        x = rearrange(x, "b c t -> b t c"); md = rearrange(md, "b 1 t -> b t")
        for tb in tbs:
            x = tb(hidden_states=x, attention_mask=md, timestep=t)
        if bi == 0: taps["attn0"] = rearrange(x, "b t c -> b c t")
        x = rearrange(x, "b t c -> b c t"); md = rearrange(md, "b t -> b 1 t")
        hiddens.append(x); x = downsample(x * md); masks.append(dec2(md))
    masks = masks[:-1]; mm = masks[-1]
    for resnet, tbs in self.mid_blocks:
        x = resnet(x, mm, t); x = rearrange(x, "b c t -> b t c"); mm = rearrange(mm, "b 1 t -> b t")
        for tb in tbs: x = tb(hidden_states=x, attention_mask=mm, timestep=t)
        x = rearrange(x, "b t c -> b c t"); mm = rearrange(mm, "b t -> b 1 t")
    for ui, (resnet, tbs, upsample) in enumerate(self.up_blocks):
        mu_ = masks.pop()
        x = resnet(pack([x, hiddens.pop()], "b * t")[0], mu_, t)
        if ui == 1: taps["up1_resnet"] = x
        x = rearrange(x, "b c t -> b t c"); mu_ = rearrange(mu_, "b 1 t -> b t")
        for tb in tbs: x = tb(hidden_states=x, attention_mask=mu_, timestep=t)
        if ui == 1: taps["up1_attn"] = rearrange(x, "b t c -> b c t")
        x = rearrange(x, "b t c -> b c t"); mu_ = rearrange(mu_, "b t -> b 1 t")
        x = upsample(x * mu_)
    out = self.final_block(x, mu_); out = self.final_proj(out * mu_) * mask
    return out, taps["up1_resnet"], taps["up1_attn"], x  # x = up1 upsample out


def main():
    sd = B.load_sd()
    dec = B.reauth_decoder_masked(B.build_decoder(sd), T).d   # unwrap DecWrapM -> the Decoder
    dec.forward = _t.MethodType(dbg_forward, dec)

    class W(nn.Module):
        def __init__(s, d): super().__init__(); s.d = d
        def forward(s, x, mu, t_emb, mask): return s.d(x, mask, mu, t_emb)
    w = W(dec).eval()
    x = torch.randn(1, 80, T); mu = torch.randn(1, 80, T); te = B.sin_pos_emb(torch.zeros(1), 160)
    mask = torch.ones(1, 1, T)
    out = os.path.join(B.HERE, "artifacts", "matcha_decoder_dbg.tflite")
    import litert_torch
    litert_torch.convert(w, (x, mu, te, mask)).export(out)
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=out); it.allocate_tensors()
    print("DBG decoder outputs (index, name, shape):")
    for d in it.get_output_details(): print("  ", d["index"], d["name"], list(d["shape"]))
    print("wrote", out, os.path.getsize(out)//1_000_000, "MB")


if __name__ == "__main__":
    main()
