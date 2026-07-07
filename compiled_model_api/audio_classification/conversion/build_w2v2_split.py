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

"""wav2vec2 KWS deployment = 2 GPU graphs (the full 1008-node graph exceeds the Mali shader-compile
limit: each half compiles 134/134 + 868/868 LITERT_CL, but fused fails). Both graphs all-GPU.

  frontend: waveform[1,16000] -> feat[1,T,768]          (feature_extractor + feature_projection)
  head    : feat[1,T,768] -> logits[1,12]               (encoder 12L w/ output_hidden_states ->
            weighted-layer-sum over all 13 hidden states -> projector -> mean-pool -> classifier)

The head MUST include the weighted-layer-sum (this checkpoint has use_weighted_layer_sum=True:
logits use a softmax-weighted combination of ALL 13 layer outputs, not just the last) — dropping it
flips predictions (corr 0.54). Re-authoring (GELU->tanh, GroupNorm->GN4D, weight_norm fold, mask->None)
from build_w2v2.

Run: python build_w2v2_split.py
"""
import _stub  # noqa: F401  (must import first: env guards)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import build_w2v2 as B
from transformers import Wav2Vec2ForSequenceClassification

HERE = B.HERE


class FrontEnd(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w
    def forward(self, x):
        ef = self.w.feature_extractor(x)
        f, _ = self.w.feature_projection(ef.transpose(1, 2))
        return f


class Head(nn.Module):
    """feat -> logits, replicating Wav2Vec2ForSequenceClassification with use_weighted_layer_sum.
    The weighted-layer-sum is accumulated INCREMENTALLY (acc += w[i]*hidden after each layer) rather
    than stack-all-13-then-sum: the 13-way stack keeps every encoder layer output live at once, which
    splits the Mali partition and fails to compile. Incremental keeps one accumulator -> delegates."""
    def __init__(self, m):
        super().__init__()
        self.m = m
        # bake softmax(layer_weights) as Python-float constants -> no runtime softmax + no w[i] gather
        # (those 13 scalar gathers off a runtime tensor break Mali delegation into partitions).
        self.w = F.softmax(m.layer_weights.detach(), dim=-1).tolist()
    def forward(self, feat):
        enc = self.m.wav2vec2.encoder
        h = feat + enc.pos_conv_embed(feat)
        h = enc.layer_norm(h)
        h = enc.dropout(h)                                   # eval no-op
        acc = self.w[0] * h                                     # hidden_states[0] = pre-layer-0
        for i, layer in enumerate(enc.layers):
            h = layer(h, attention_mask=None)[0]
            acc = acc + self.w[i + 1] * h
        acc = self.m.projector(acc)
        pooled = acc.mean(dim=1)
        return self.m.classifier(pooled)


def main():
    B.patch_mask()
    m = Wav2Vec2ForSequenceClassification.from_pretrained(B.MID).eval()
    B.swap(m)
    B.fold_weight_norm(m)
    id2 = m.config.id2label
    use_wls = m.config.use_weighted_layer_sum
    print(f"use_weighted_layer_sum={use_wls}")

    x = torch.randn(1, 16000)
    with torch.no_grad():
        full = m(x, attention_mask=None).logits
        fe = FrontEnd(m.wav2vec2).eval()
        hd = Head(m).eval()
        feat = fe(x)
        split = hd(feat)
    print(f"torch full-vs-split corr {np.corrcoef(full.numpy().ravel(), split.numpy().ravel())[0,1]:.6f} "
          f"max|d| {(full-split).abs().max():.2e}  full={full.argmax().item()} split={split.argmax().item()}")

    import litert_torch
    fa = os.path.join(HERE, "w2v2_frontend.tflite")
    fb = os.path.join(HERE, "w2v2_head.tflite")
    litert_torch.convert(fe, (x,)).export(fa)
    litert_torch.convert(hd, (feat,)).export(fb)
    ia, ca = B.opcheck(fa, "frontend")
    ib, cb = B.opcheck(fb, "head")
    oa = B.tfl_run(ia, x.numpy())
    ob = B.tfl_run(ib, feat.numpy())
    print(f"frontend tflite vs torch corr {np.corrcoef(oa.ravel(), feat.numpy().ravel())[0,1]:.6f}")
    print(f"head tflite vs torch corr {np.corrcoef(ob.ravel(), split.numpy().ravel())[0,1]:.6f}  "
          f"argmax {ob.argmax()} ({id2[int(ob.argmax())]})")

    if ca and cb:
        B.to_fp16(fa, fa.replace(".tflite", "_fp16.tflite"))
        B.to_fp16(fb, fb.replace(".tflite", "_fp16.tflite"))
        print("wrote w2v2_frontend_fp16.tflite + w2v2_head_fp16.tflite")


if __name__ == "__main__":
    main()
