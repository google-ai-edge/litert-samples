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

"""
MoViNet-A0 streaming, re-authored as a single-frame, 4D-only functional forward
for LiteRT CompiledModel GPU. Reuses the pretrained submodules of the original
Atze00/MoViNet-pytorch model but drives them one frame at a time with explicit
state I/O (no 5D tensors, no mutable buffers).

State (all 4D or scalar):
  - 28 stream-buffer frames  [1, C, H, W]  (11 temporal convs, dim_pad in {2,4})
  - 16 pool running-sums     [1, C, 1, 1]  (15 SE + 1 head)
  -  1 frame counter (scalar); every pool advances once per frame -> shared
"""
import torch
import torch.nn.functional as F
from movinets.models import same_padding, tfAvgPool3D


# ---- primitive block ops on 4D NCHW tensors -------------------------------

def spatial(cb, x):
    """ConvBlock3D spatial part (conv_1) with tf 'same' padding. 4D in/out.

    Args:
        cb: The ConvBlock3D submodule.
        x: 4D NCHW input tensor.

    Returns:
        The conv_1 output tensor.
    """
    if cb.tf_like:
        x = same_padding(x, x.shape[-2], x.shape[-1],
                         cb.stride[-2], cb.stride[-1],
                         cb.kernel_size[-2], cb.kernel_size[-1])
    return cb.conv_1(x)


def temporal(cb, s, buf):
    """Causal temporal depthwise conv (conv_2, kernel (kt,1)) as a weighted
    sum of the buffered post-spatial frames. buf = list of dim_pad prior
    frames (oldest first); s = current post-spatial frame. Returns (out, s).

    The returned `s` (the current post-spatial frame) is emitted as a graph
    output so the host can maintain the shift register; the buffered input
    frames are consumed ONLY here (never passed through to an output). This
    avoids the Mali/litert-torch bug that drops the compute-side use of an
    input that is also passed through to a graph output (a shifted stream
    buffer).

    Args:
        cb: The ConvBlock3D whose conv_2 depthwise kernel is applied.
        s: Current post-spatial frame [1, C, H, W].
        buf: List of dim_pad prior frames (oldest first).

    Returns:
        A tuple (out, s): the temporal-conv output and the current frame.
    """
    frames = buf + [s]                       # length kt = dim_pad + 1
    w = cb.conv_2.conv2d.weight              # [C, 1, kt, 1] depthwise
    C = s.shape[1]
    out = w[:, 0, 0, 0].view(1, C, 1, 1) * frames[0]
    for i in range(1, len(frames)):
        out = out + w[:, 0, i, 0].view(1, C, 1, 1) * frames[i]
    out = cb.conv_2.norm(out)
    out = cb.conv_2.act(out)
    return out, s                            # emit current frame; host shifts


def tf_avg_pool_2d(x):
    """4D equivalent of tfAvgPool3D (1,3,3)/(1,2,2).

    Trace-safe: the last-row/col 9/6 edge correction is a constant
    outer-product mask (no in-place scatter).

    Args:
        x: 4D NCHW input tensor.

    Returns:
        The pooled 4D tensor.
    """
    odd = x.shape[-1] % 2 != 0
    if odd:
        # count_include_pad=False (symmetric pad 1) -> True + boundary mask, so
        # it lowers to AVERAGE_POOL_2D + MUL (no STABLEHLO_COMPOSITE for GPU).
        x = F.avg_pool2d(x, (3, 3), stride=(2, 2), padding=(1, 1),
                         count_include_pad=True)          # divides by 9
        h, w = x.shape[-2], x.shape[-1]
        rs = torch.ones(h, device=x.device)
        rs[0] = 1.5
        rs[-1] = 1.5
        cs = torch.ones(w, device=x.device)
        cs[0] = 1.5
        cs[-1] = 1.5
        return x * torch.outer(rs, cs).view(1, 1, h, w)   # 9/valid at edges
    x = F.pad(x, (0, 1, 0, 1))
    x = F.avg_pool2d(x, (3, 3), stride=(2, 2))   # count_include_pad=True -> /9
    h, w = x.shape[-2], x.shape[-1]
    rs = torch.ones(h, device=x.device)
    rs[-1] = 9.0 / 6.0
    cs = torch.ones(w, device=x.device)
    cs[-1] = 9.0 / 6.0
    mask = torch.outer(rs, cs).view(1, 1, h, w)  # corner = (9/6)^2, edges = 9/6
    return x * mask


def res_forward(bb, x):
    """Runs the residual branch (spatial convs + tf avg pool) of a block.

    Args:
        bb: The MoViNet block whose `res` branch is evaluated.
        x: 4D NCHW input tensor.

    Returns:
        The residual-branch output tensor.
    """
    r = x
    for layer in bb.res:
        if isinstance(layer, tfAvgPool3D):
            r = tf_avg_pool_2d(r)
        else:
            r = spatial(layer, r)
    return r


def se_forward(se, x, pool_sum, inv_count):
    """Squeeze-Excite with streaming cumulative temporal avg.

    inv_count = 1/N for the current frame number N (host-computed, shape
    [1,1,1,1]). pool_sum = running sum of x_space from PREVIOUS frames.
    Returns (out, x_space) — the running-sum accumulation is done host-side,
    so the graph only emits the fresh per-frame mean. The average uses MUL
    by 1/N (not DIV by a [1] scalar): the Mali GPU delegate mishandles a
    broadcast DIVIDE by a rank-1 count input.

    Args:
        se: The squeeze-excite submodule.
        x: 4D NCHW input tensor.
        pool_sum: Running sum of x_space from previous frames [1, C, 1, 1].
        inv_count: 1/N for the current frame number N, shape [1, 1, 1, 1].

    Returns:
        A tuple (out, x_space): the scaled output and the fresh per-frame
        mean.
    """
    x_space = x.mean(dim=[2, 3], keepdim=True)      # [1,C,1,1]
    avg = (pool_sum + x_space) * inv_count            # running temporal avg
    # [1,2C,1,1] (causal se_mult=2)
    scale = torch.cat([avg, x_space], dim=1)
    scale = se.fc1.conv_1(scale)
    scale = se.activation_1(scale)
    scale = se.fc2.conv_1(scale)
    scale = se.activation_2(scale)
    return scale * x, x_space


# ---- full single-frame step ----------------------------------------------

def block_names(model):
    """Returns the ordered list of block names in the model.

    Args:
        model: The MoViNet model.

    Returns:
        A list of block-name strings.
    """
    return list(model.blocks._modules.keys())


def init_state(model, device="cpu"):
    """Return an ordered dict of zero-initialised states.

    Args:
        model: The MoViNet model.
        device: Torch device for the created tensors.

    Returns:
        A dict with "count", "stream", and "pool" entries.
    """
    st = {"count": torch.zeros(1, device=device)}
    st["stream"] = {}   # name -> list of dim_pad frames
    st["pool"] = {}     # name -> running sum
    return st


def step(model, frame4d, st):
    """One streaming frame. frame4d = [1,3,172,172].

    Args:
        model: The MoViNet model.
        frame4d: Single input frame [1, 3, 172, 172].
        st: State dict from init_state or a previous step.

    Returns:
        A tuple (logits, new_st): the frame logits and the updated state.
    """
    count = st["count"] + 1.0
    new_stream = {}
    new_pool = {}

    x = spatial(model.conv1, frame4d)

    for name, bb in model.blocks._modules.items():
        residual = res_forward(bb, x) if bb.res is not None else x
        h = spatial(bb.expand, x)
        s = spatial(bb.deep, h)
        if bb.deep.conv_2 is not None:
            dp = bb.deep.dim_pad
            buf = st["stream"].get(name)
            if buf is None:
                buf = [torch.zeros_like(s) for _ in range(dp)]
            s, nb = temporal(bb.deep, s, buf)
            new_stream[name] = nb
        ps = st["pool"].get(name)
        if ps is None:
            ps = torch.zeros(1, bb.se.fc1.conv_1.conv2d.in_channels // 2,
                             1, 1, device=frame4d.device)
        s, nsum = se_forward(bb.se, s, ps, count)
        new_pool[name] = nsum
        s = spatial(bb.project, s)
        x = residual + bb.alpha * s

    x = spatial(model.conv7, x)

    x_space = x.mean(dim=[2, 3], keepdim=True)       # [1,480,1,1]
    head_sum = st["pool"].get("head")
    if head_sum is None:
        head_sum = torch.zeros_like(x_space)
    new_head = head_sum + x_space
    pooled = new_head / count
    new_pool["head"] = new_head

    y = model.classifier[0].conv_1(pooled)           # dense9 1x1 + bias
    y = model.classifier[1](y)                       # Swish
    y = model.classifier[3].conv_1(y)                # dense10 1x1 + bias
    logits = y.flatten(1)

    return logits, {"count": count, "stream": new_stream, "pool": new_pool}


# ---- traceable nn.Module (flat positional state I/O) ----------------------

import torch.nn as nn

STREAM_SPEC = [("b1_l0", 2), ("b1_l1", 2), ("b1_l2", 2),
               ("b2_l0", 4), ("b2_l1", 2), ("b2_l2", 2),
               ("b3_l0", 4), ("b3_l1", 2), ("b3_l2", 2), ("b3_l3", 2),
               ("b4_l0", 4)]
POOL_SPEC = ["b0_l0", "b1_l0", "b1_l1", "b1_l2",
             "b2_l0", "b2_l1", "b2_l2",
             "b3_l0", "b3_l1", "b3_l2", "b3_l3",
             "b4_l0", "b4_l1", "b4_l2", "b4_l3", "head"]


class MoViNetA0Stream(nn.Module):
    """forward(frame, *states) -> (logits, *stream_out, *xspace_out).

    Inputs  (46): frame, 28 stream frames (STREAM_SPEC), 16 pool running-sums
                  (POOL_SPEC), 1 frame count (== current frame number, >=1).
    Outputs (28): logits, 11 current per-temporal-conv frames
                  (STREAM_SPEC order), 16 fresh per-frame means
                  (POOL_SPEC order).

    Both the stream-buffer shift register and the pool running-sum accumulation
    are done HOST-SIDE. The graph consumes the stream frames / running sums (for
    the temporal convs / SE averages) but only EMITS fresh tensors (the current
    frame and the per-frame means). This avoids three Mali/litert-torch bugs:
      * an input that is ALSO passed through to an output (a shifted stream
        buffer) has its compute-side use silently dropped;
      * a graph output of the form `state_input + tensor` is silently zeroed;
      * a conv-output tensor that is BOTH consumed and emitted has its emitted
        copy corrupted (~2.5x) — so each emitted stream frame is decoupled from
        its compute use by a multiply against a runtime `one` (== 1.0) input."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, frame, *states):
        m = self.m
        idx = 0
        stream = {}
        for name, dp in STREAM_SPEC:
            stream[name] = [states[idx + i] for i in range(dp)]
            idx += dp
        pool = {}
        for name in POOL_SPEC:
            pool[name] = states[idx]
            idx += 1
        # 1 / current frame number, [1,1,1,1]
        inv_count = states[idx]
        idx += 1
        # constant 1.0 [1,1,1,1] (decoupler)
        one = states[idx]

        # current post-spatial frame / conv
        cur = {}
        xspace = {}
        x = spatial(m.conv1, frame)
        for name, bb in m.blocks._modules.items():
            residual = res_forward(bb, x) if bb.res is not None else x
            h = spatial(bb.expand, x)
            s = spatial(bb.deep, h)
            if bb.deep.conv_2 is not None:
                s, s_cur = temporal(bb.deep, s, stream[name])
                # decouple emitted copy from compute
                cur[name] = s_cur * one
            s, xs = se_forward(bb.se, s, pool[name], inv_count)
            xspace[name] = xs
            s = spatial(bb.project, s)
            x = residual + bb.alpha * s

        x = spatial(m.conv7, x)
        xs_head = x.mean(dim=[2, 3], keepdim=True)
        xspace["head"] = xs_head
        pooled = (pool["head"] + xs_head) * inv_count

        y = m.classifier[0].conv_1(pooled)
        y = m.classifier[1](y)
        y = m.classifier[3].conv_1(y)
        logits = y.flatten(1)

        out = [logits]
        for name, dp in STREAM_SPEC:
            out.append(cur[name])                # 11 current frames
        for name in POOL_SPEC:
            out.append(xspace[name])             # 16 fresh means
        return tuple(out)


def make_dummy_states(device="cpu"):
    """Zero states matching the flat forward signature (excluding the frame).

    Args:
        device: Torch device for the created tensors.

    Returns:
        A list of state tensors in the flat forward order.
    """
    shapes = {
        "b1_l0": (80, 22, 22), "b1_l1": (80, 22, 22), "b1_l2": (80, 22, 22),
        "b2_l0": (184, 11, 11), "b2_l1": (112, 11, 11), "b2_l2": (184, 11, 11),
        "b3_l0": (184, 11, 11), "b3_l1": (184, 11, 11), "b3_l2": (184, 11, 11),
        "b3_l3": (184, 11, 11), "b4_l0": (384, 6, 6),
    }
    pool_c = {"b0_l0": 24, "b1_l0": 80, "b1_l1": 80, "b1_l2": 80,
              "b2_l0": 184, "b2_l1": 112, "b2_l2": 184,
              "b3_l0": 184, "b3_l1": 184, "b3_l2": 184, "b3_l3": 184,
              "b4_l0": 384, "b4_l1": 280, "b4_l2": 280, "b4_l3": 344,
              "head": 480}
    st = []
    for name, dp in STREAM_SPEC:
        c, h, w = shapes[name]
        for _ in range(dp):
            st.append(torch.zeros(1, c, h, w, device=device))
    for name in POOL_SPEC:
        st.append(torch.zeros(1, pool_c[name], 1, 1, device=device))
    # inv_count (1/N), [1,1,1,1]
    st.append(torch.ones(1, 1, 1, 1, device=device))
    st.append(torch.ones(1, 1, 1, 1, device=device))   # one (==1.0) decoupler
    return st
