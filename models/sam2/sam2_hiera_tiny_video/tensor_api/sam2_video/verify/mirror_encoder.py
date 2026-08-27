#!/usr/bin/env python3
"""Torch mirror of sam2_graph.cc's encoder at 1024, block by block, compared
against hooks on the HF video vision encoder. Finds WHERE the authored
encoder diverges without rebuilding C++.

  python mirror_encoder.py
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
WPATH = os.environ["SAM2V_WEIGHTS"]  # sam2_tiny_1024_video.safetensors
CKPT = os.environ.get("SAM2_CKPT", "facebook/sam2.1-hiera-tiny")

# Block specs mirroring Sam2Config::Blocks() at image_size=1024.
STAGES = [1, 2, 7, 2]
WINDOW_SPEC = [8, 4, 14, 7]
GLOBAL_BLOCKS = {5, 7, 9}
EMBED, HEADS, TOKEN_GRID = 96, 1, 256


def block_specs():
    ends, acc = [], 0
    for s in STAGES:
        acc += s
        ends.append(acc - 1)
    pool_blocks = {ends[i] + 1 for i in range(len(ends) - 1)}
    specs = []
    ed, nh, cur_stage, grid = EMBED, HEADS, 1, TOKEN_GRID
    for i in range(sum(STAGES)):
        dim_out = ed
        ws = WINDOW_SPEC[cur_stage - 1]
        if i in GLOBAL_BLOCKS:
            ws = 0
        if i > 0 and (i - 1) in ends:
            dim_out, nh, cur_stage = ed * 2, nh * 2, cur_stage + 1
        qs = 2 if i in pool_blocks else 0
        grid_out = grid // qs if qs else grid
        specs.append(dict(dim=ed, dim_out=dim_out, heads=nh, window=ws,
                          q_stride=qs, grid_in=grid, grid_out=grid_out))
        ed, grid = dim_out, grid_out
    return specs


class W:
    def __init__(self, path):
        self.f = safe_open(path, framework="pt")

    def __call__(self, name):
        return self.f.get_tensor(name).float()


def layer_norm(x, w, b, eps=1e-6):
    return F.layer_norm(x, (x.shape[-1],), w, b, eps)


def window_partition(x, ws):
    """[1,H,W,C] -> [nH*nW, ws, ws, C], split-H-then-W (my graph's form)."""
    _, h, w, c = x.shape
    ph, pw = (ws - h % ws) % ws, (ws - w % ws) % ws
    if ph or pw:
        x = F.pad(x, (0, 0, 0, pw, 0, ph))
    hp, wp = h + ph, w + pw
    nh, nw = hp // ws, wp // ws
    t = x.reshape(nh, ws, wp, c).permute(0, 2, 1, 3)     # [nH, Wp, ws, C]
    return t.reshape(nh * nw, ws, ws, c), nh, nw


def window_unpartition(t, nh, nw, ws2, h, w, c):
    t = t.reshape(nh, nw * ws2, ws2, c).permute(0, 2, 1, 3)
    t = t.reshape(1, nh * ws2, nw * ws2, c)
    return t[:, :h, :w, :]


def mha(q, k, v, heads):
    b, nq, c = q.shape
    nk = k.shape[1]
    hd = c // heads
    q4 = q.reshape(b, nq, heads, hd).permute(0, 2, 1, 3)
    k4 = k.reshape(b, nk, heads, hd).permute(0, 2, 1, 3)
    v4 = v.reshape(b, nk, heads, hd).permute(0, 2, 1, 3)
    a = torch.softmax(q4 @ k4.transpose(-1, -2) / hd**0.5, dim=-1)
    return (a @ v4).permute(0, 2, 1, 3).reshape(b, nq, c)


def block(x, spec, p, w):
    dim, dim_out, h = spec["dim"], spec["dim_out"], spec["grid_in"]
    shortcut = x
    xn = layer_norm(x, w(p + ".norm1.weight"), w(p + ".norm1.bias"))
    if dim != dim_out:
        shortcut = xn @ w(p + ".proj.weight").T + w(p + ".proj.bias")
        if spec["q_stride"]:
            shortcut = F.max_pool2d(shortcut.permute(0, 3, 1, 2), 2).permute(
                0, 2, 3, 1)
    tokens, nh, nw, batch, n, win = xn, 1, 1, 1, h * h, spec["window"]
    if win:
        tokens, nh, nw = window_partition(tokens, win)
        batch, n = nh * nw, win * win
    tokens = tokens.reshape(batch, n, dim)
    qkv = tokens @ w(p + ".attn.qkv.weight").T + w(p + ".attn.qkv.bias")
    q, k, v = qkv[..., :dim_out], qkv[..., dim_out:2 * dim_out], qkv[..., 2 * dim_out:]
    nq = n
    if spec["q_stride"]:
        side = win if win else h
        q = q.reshape(batch, side, side, dim_out)
        q = F.max_pool2d(q.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
        side2 = side // 2
        nq = side2 * side2
        q = q.reshape(batch, nq, dim_out)
    attn = mha(q, k, v, spec["heads"])
    attn = attn @ w(p + ".attn.proj.weight").T + w(p + ".attn.proj.bias")
    go = spec["grid_out"]
    if win:
        ws2 = win // 2 if spec["q_stride"] else win
        attn = attn.reshape(batch, ws2, ws2, dim_out)
        attn = window_unpartition(attn, nh, nw, ws2, go, go, dim_out)
    else:
        attn = attn.reshape(1, go, go, dim_out)
    merged = shortcut + attn
    m = layer_norm(merged, w(p + ".norm2.weight"), w(p + ".norm2.bias"))
    m = F.gelu(m @ w(p + ".mlp.layers.0.weight").T + w(p + ".mlp.layers.0.bias"))
    m = m @ w(p + ".mlp.layers.1.weight").T + w(p + ".mlp.layers.1.bias")
    return merged + m


def corr(a, b):
    a = a.detach().reshape(-1).double().numpy()
    b = b.detach().reshape(-1).double().numpy()
    return float(np.corrcoef(a, b)[0, 1])


def main():
    import sys
    sys.path.insert(0, HERE)
    from verify_video_1024 import frame
    from transformers import Sam2VideoModel

    w = W(WPATH)
    model = Sam2VideoModel.from_pretrained(CKPT).eval()
    backbone = model.vision_encoder.backbone

    hooks = {}
    for i, blk in enumerate(backbone.blocks):
        blk.register_forward_hook(
            lambda mod, args, out, i=i: hooks.__setitem__(i, (args[0].detach(),
                                                              out.detach())))
    px = torch.from_numpy(frame(3))
    with torch.no_grad():
        ve = model.vision_encoder(px, return_dict=True)

    # --- my patch embed + pos ---
    with torch.no_grad():
        x = F.pad(px.permute(0, 2, 3, 1), (0, 0, 3, 3, 3, 3))       # NHWC pad3
        pw = w("trunk.patch_embed.proj.weight")                      # [96,7,7,3]
        x = F.conv2d(x.permute(0, 3, 1, 2), pw.permute(0, 3, 1, 2),
                     w("trunk.patch_embed.proj.bias"), stride=4)
        x = x.permute(0, 2, 3, 1)                                    # [1,256,256,96]
        pos = w("trunk.pos_embed_full")
        print("block0 input corr (patch+pos vs HF):",
              corr(x + pos, hooks[0][0]),
              " patch-only corr:", corr(x, hooks[0][0]))
        x = x + pos
        specs = block_specs()
        for i, spec in enumerate(specs):
            x = block(x, spec, f"trunk.blocks.{i}", w)
            ref = hooks[i][1]
            print(f"block {i:2d} (dim{spec['dim']}->{spec['dim_out']} "
                  f"win{spec['window']:2d} qs{spec['q_stride']} "
                  f"grid{spec['grid_in']}->{spec['grid_out']}): "
                  f"out corr={corr(x, ref):.6f} "
                  f"max|d|={float((x - ref).abs().max()):.5f} "
                  f"shapes {tuple(x.shape)} vs {tuple(ref.shape)}")


if __name__ == "__main__":
    main()
