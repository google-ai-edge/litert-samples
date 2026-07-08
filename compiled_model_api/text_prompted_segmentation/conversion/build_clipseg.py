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

"""CLIPSeg -> LiteRT GPU, 2 graphs:
  A) vision+decoder: pixel_values [1,3,352,352] + cond [1,512]
     -> logits [1,352,352]
  B) text encoder: token embeddings [1,77,512] (host lookup)
     -> final hidden [1,77,512]
     (host: EOT row -> text_projection matmul -> cond; projection
     exported as .bin)

Run: python build_clipseg.py
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# Import the REAL joblib before build_cmgan's stub can shadow it.
import joblib  # noqa: F401
sys.path.insert(0, os.path.expanduser("~/Downloads/meeting/cmgan-work"))
from probe_vision import (  # noqa: E402
    VisionGPU, safe_ln as safe_ln_vis, safe_ln_up as safe_ln)
from build_cmgan import opcheck, to_fp16  # noqa: E402

SIZE = 352


def interleave4_last(x):
    """[1, C*4, H, W] channels (g, c4) -> [1, C, H, W*4].

    The TIGER width-interleave; every intermediate stays 4D.

    Args:
        x: Input tensor [1, C*4, H, W] with channels ordered (g, c4).

    Returns:
        Tensor [1, C, H, W*4] with the 4 groups interleaved along width.
    """
    b, c4, hh, ww = x.shape
    c = c4 // 4
    return (x.reshape(b, c, 4, hh * ww).permute(0, 1, 3, 2)
            .reshape(b, c, hh, ww * 4))


def convT4x4(x, w, bias):
    """Exact non-overlapping ConvTranspose2d k4 s4, GPU-clean.

    Implemented as a 1x1 conv + two 4D interleaves.

    Args:
        x: Input tensor [1, Cin, H, W].
        w: ConvTranspose2d weight [Cin, Cout, 4, 4].
        bias: ConvTranspose2d bias [Cout].

    Returns:
        Tensor [1, Cout, 4H, 4W].
    """
    cin, cout = w.shape[0], w.shape[1]
    # 1x1 conv producing channels ordered (cout, r, c)
    w1 = w.permute(1, 2, 3, 0).reshape(cout * 16, cin, 1, 1)
    y = F.conv2d(x, w1)  # [1, cout*16, H, W] (cout, r, c)
    # step 1: interleave c along W -> [1, cout*4 (cout,r), H, 4W]
    y = interleave4_last(y)
    # step 2: interleave r along H: transpose H/W, interleave, transpose back
    y = y.transpose(2, 3)                                  # [1, cout*4, 4W, H]
    y = interleave4_last(y)                                # [1, cout, 4W, 4H]
    y = y.transpose(2, 3)                                  # [1, cout, 4H, 4W]
    return y + bias.view(1, -1, 1, 1)


class ClipSegGPU(nn.Module):
    """Graph A: vision encoder taps + CLIPSegDecoder.

    FiLM conditioning is applied via NCHW channel-scale.
    """

    def __init__(self, model):
        super().__init__()
        self.vis = VisionGPU(model.clip.vision_model)
        self.dec = model.decoder

    def dec_layer(self, layer, h):
        # CLIPSegDecoderLayer is POST-norm:
        # attn -> +res -> LN1 -> mlp -> +res -> LN2
        a = layer.self_attn
        q = F.linear(h, a.q_proj.weight, a.q_proj.bias) * a.scale
        k = F.linear(h, a.k_proj.weight, a.k_proj.bias)
        v = F.linear(h, a.v_proj.weight, a.v_proj.bias)
        B, N, C = q.shape
        H = a.num_heads
        d = C // H
        q3 = q.reshape(N, H, d).permute(1, 0, 2)
        k3 = k.reshape(N, H, d).permute(1, 0, 2)
        v3 = v.reshape(N, H, d).permute(1, 0, 2)
        att = F.softmax(torch.matmul(q3, k3.transpose(1, 2)), dim=-1)
        o = torch.matmul(att, v3).permute(1, 0, 2).reshape(1, N, C)
        h = h + F.linear(o, a.out_proj.weight, a.out_proj.bias)
        h = safe_ln(h, layer.layer_norm1.weight, layer.layer_norm1.bias)
        y = F.relu(F.linear(h, layer.mlp.fc1.weight, layer.mlp.fc1.bias))
        h = h + F.linear(y, layer.mlp.fc2.weight, layer.mlp.fc2.bias)
        return safe_ln(h, layer.layer_norm2.weight, layer.layer_norm2.bias)

    def forward(self, pixel_values, cond):
        # NOTE: the intermediate tensors are ALSO graph outputs on
        # purpose — they act as fusion barriers: without them the Mali
        # delegate mis-fuses the decoder (device logits corr 0.858);
        # with them it computes correctly (0.99998). The app ignores
        # them.
        t3, t6, t9, _ = self.vis(pixel_values)
        dec = self.dec
        barriers = [t9.clone()]
        out = None
        for i, (act, layer, reduce) in enumerate(
                zip((t9, t6, t3), dec.layers, dec.reduces)):
            red = F.linear(act, reduce.weight, reduce.bias)  # [1, 485, 64]
            out = red if out is None else red + out
            if i == dec.conditional_layer:
                # gamma, beta: [1, 64]
                gamma = F.linear(cond, dec.film_mul.weight,
                                 dec.film_mul.bias)
                beta = F.linear(cond, dec.film_add.weight,
                                dec.film_add.bias)
                o4 = out.transpose(1, 2).unsqueeze(3)  # [1, 64, 485, 1]
                o4 = o4 * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
                out = o4.squeeze(3).transpose(1, 2)  # [1, 485, 64]
                barriers.append(out.clone())
            out = self.dec_layer(layer, out)
            barriers.append(out.clone())
        out = out[:, 1:, :].transpose(1, 2).reshape(1, 64, 22, 22)
        tc = dec.transposed_convolution
        y = F.conv2d(out, tc[0].weight, tc[0].bias, padding=1)
        y = F.relu(y)
        y = F.relu(convT4x4(y, tc[2].weight, tc[2].bias))  # [1, 32, 88, 88]
        y = convT4x4(y, tc[4].weight, tc[4].bias)  # [1, 1, 352, 352]
        # logits LAST (device fusion is output-ORDER-sensitive:
        # logits-first miscomputes)
        return tuple(barriers) + (y.squeeze(1),)


class TextGPU(nn.Module):
    """Graph B: token embeddings + pos -> 12 causal layers.

    Emits the final LN hidden state [1,77,512].
    """

    def __init__(self, tm):
        super().__init__()
        self.tm = tm
        self.register_buffer(
            "pos", tm.embeddings.position_embedding.weight[:77].unsqueeze(0))
        # fp16-safe large-negative (C38 spirit)
        mask = torch.full((77, 77), -60.0)
        self.register_buffer("causal",
                             torch.triu(mask, diagonal=1).unsqueeze(0))

    def forward(self, tok_emb):
        h = tok_emb + self.pos
        for layer in self.tm.encoder.layers:
            r = h
            y = safe_ln_up(h, layer.layer_norm1.weight, layer.layer_norm1.bias)
            a = layer.self_attn
            q = F.linear(y, a.q_proj.weight, a.q_proj.bias) * a.scale
            k = F.linear(y, a.k_proj.weight, a.k_proj.bias)
            v = F.linear(y, a.v_proj.weight, a.v_proj.bias)
            B, N, C = q.shape
            H = a.num_heads
            d = C // H
            q3 = q.reshape(N, H, d).permute(1, 0, 2)
            k3 = k.reshape(N, H, d).permute(1, 0, 2)
            v3 = v.reshape(N, H, d).permute(1, 0, 2)
            att = torch.matmul(q3, k3.transpose(1, 2)) + self.causal
            att = F.softmax(att, dim=-1)
            o = torch.matmul(att, v3).permute(1, 0, 2).reshape(1, N, C)
            h = r + F.linear(o, a.out_proj.weight, a.out_proj.bias)
            r = h
            y = safe_ln_up(h, layer.layer_norm2.weight, layer.layer_norm2.bias)
            y = F.linear(y, layer.mlp.fc1.weight, layer.mlp.fc1.bias)
            y = y * torch.sigmoid(1.702 * y)
            h = r + F.linear(y, layer.mlp.fc2.weight, layer.mlp.fc2.bias)
        return safe_ln_up(h, self.tm.final_layer_norm.weight,
                          self.tm.final_layer_norm.bias)


def corr(a, b):
    """Pearson correlation of two arrays, flattened.

    Args:
        a: First array-like.
        b: Second array-like.

    Returns:
        The scalar correlation coefficient.
    """
    return np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1]


def main():
    """Verifies torch parity, converts both graphs, and dumps fixtures."""
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    from PIL import Image
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained(
        "CIDAS/clipseg-rd64-refined").eval()
    img = Image.open(os.path.join(HERE, "cats.jpg"))
    prompts = ["a cat", "a remote control", "a blanket"]
    inputs = proc(text=prompts, images=[img] * 3, padding="max_length",
                  return_tensors="pt")
    ref = torch.load(os.path.join(HERE, "ref_logits.pt"))["logits"]

    with torch.no_grad():
        cond = model.get_conditional_embeddings(
            batch_size=3, input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"])
        g = ClipSegGPU(model).eval()
        x1 = inputs["pixel_values"][0:1]
        logits = [g(x1, cond[i:i + 1])[-1][0] for i in range(3)]
    for i, p in enumerate(prompts):
        print(f"[graphA torch] '{p}': corr "
              f"{corr(logits[i].numpy(), ref[i].numpy()):.6f}")

    # ---- text graph parity
    tm = model.clip.text_model
    with torch.no_grad():
        tg = TextGPU(tm).eval()
        # host lookup
        emb = tm.embeddings.token_embedding(inputs["input_ids"])
        hid = torch.cat([tg(emb[i:i + 1]) for i in range(3)])
        eot = inputs["input_ids"].argmax(dim=-1)  # EOT = max token id
        pooled = hid[torch.arange(3), eot]
        my_cond = pooled @ model.clip.text_projection.weight.T
    print(f"[graphB torch] cond corr {corr(my_cond.numpy(), cond.numpy()):.7f}")

    if min(corr(logits[i].numpy(), ref[i].numpy()) for i in range(3)) < 0.999:
        print("PARITY FAIL")
        return

    import litert_torch
    a32 = os.path.join(HERE, "clipseg_visdec.tflite")
    litert_torch.convert(g, (x1, cond[0:1])).export(a32)
    itA, cleanA = opcheck(a32, "A fp32")
    b32 = os.path.join(HERE, "clipseg_text.tflite")
    litert_torch.convert(tg, (emb[0:1],)).export(b32)
    itB, cleanB = opcheck(b32, "B fp32")
    if cleanA and cleanB:
        a16 = to_fp16(a32, os.path.join(HERE, "clipseg_visdec_fp16.tflite"))
        opcheck(a16, "A fp16")
        b16 = to_fp16(b32, os.path.join(HERE, "clipseg_text_fp16.tflite"))
        opcheck(b16, "B fp16")
        # fixtures
        x1.numpy().astype(np.float32).tofile(os.path.join(HERE, "cs_img.bin"))
        cond[0:1].numpy().astype(np.float32).tofile(
            os.path.join(HERE, "cs_cond.bin"))
        emb[0:1].numpy().astype(np.float32).tofile(
            os.path.join(HERE, "cs_tokemb.bin"))
        np.save(os.path.join(HERE, "cs_ref.npy"), ref.numpy())
        np.save(os.path.join(HERE, "cs_hid_ref.npy"), hid[0:1].numpy())
        # host-side assets: token embedding table + text projection
        tm.embeddings.token_embedding.weight.detach().numpy().astype(
            np.float32).tofile(os.path.join(HERE, "token_embedding_f32.bin"))
        text_proj = model.clip.text_projection.weight.T.detach().contiguous()
        text_proj.numpy().astype(np.float32).tofile(
            os.path.join(HERE, "text_projection_f32.bin"))
        print("wrote fp16 graphs + fixtures + host assets")


if __name__ == "__main__":
    main()
