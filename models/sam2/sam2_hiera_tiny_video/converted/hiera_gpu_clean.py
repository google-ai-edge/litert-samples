# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Shared GPU-clean rewrites for the SAM 2.1 Hiera-Tiny image path.

The video-path exporter (export_video.py) reuses the same image encoder and
the same mask-decoder upsampling, so the rewrites that make them compile on the
LiteRT CompiledModel GPU (ML Drift) delegate live here and are installed by
calling ``install_hiera_patches()`` before the model is traced.

Every rewrite is numerically identical to the upstream forward (parity holds at
corr 1.0 in the verifier); they exist only because ML Drift rejects tensors of
rank > 4 and a few ops. The rewrites are: bake the windowed positional
embedding (constant for a fixed 1024x1024 input, so the bicubic GATHER_ND and
the tiled BROADCAST_TO disappear), re-express window partition / unpartition
with <= 4-D tensors, replace the fused 5-D qkv reshape with a channel-wise
q/k/v slice, and swap ConvTranspose2d for a zero-stuffed conv (TRANSPOSE_CONV
is unsupported).
"""
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers.models.sam2.modeling_sam2 as M
from ai_edge_litert import schema_py_generated as schema

# Flat-IO element counts, independent of the backbone size: image embedding
# (256x64x64), high-res feature s0 (32x256x256), high-res feature s1
# (64x128x128).
IE = 256 * 64 * 64
F0 = 32 * 256 * 256
F1 = 64 * 128 * 128

# Ops the ML Drift GPU delegate cannot lower; an export that emits any of these
# has not been made GPU-clean yet.
GPU_BAD = {
    "GATHER_ND", "GATHER", "SELECT_V2", "SELECT", "PACK", "SPLIT", "CAST",
    "TOPK_V2", "BROADCAST_TO", "WHILE", "TRANSPOSE_CONV",
}

# fp16 weight-only quantization recipe for ai_edge_quantizer.
FP16_RECIPE = [{
    "regex": ".*",
    "operation": "*",
    "algorithm_key": "float_casting",
    "op_config": {
        "weight_tensor_config": {"num_bits": 16, "dtype": "FLOAT"}
    },
}]


class _Dummy:
    """A no-op stand-in for a broken native scipy leaf module attribute."""

    def __getattr__(self, name):
        """Returns another dummy for any non-dunder attribute access.

        Args:
            name: The attribute being looked up.

        Returns:
            A nested ``_Dummy`` (or raises for dunder names).
        """
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Dummy()

    def __call__(self, *args, **kwargs):
        """Returns a dummy so a stubbed callable can be invoked."""
        return _Dummy()


class _Leaf(types.ModuleType):
    """A module whose attributes resolve to ``_Dummy`` placeholders."""

    def __getattr__(self, name):
        """Returns a dummy for any non-dunder attribute.

        Args:
            name: The attribute being looked up.

        Returns:
            A ``_Dummy`` placeholder (or raises for dunder names).
        """
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Dummy()


def stub_scipy_native_leaves() -> None:
    """Stubs scipy native leaf modules that some conda builds fail to load.

    Importing ``transformers``' SAM 2 stack pulls scipy; a broken native build
    aborts the process, so replace the optional leaves with harmless stubs.
    """
    for name in [
        "scipy.sparse.linalg._propack", "scipy.optimize._cobyla",
        "scipy.optimize._slsqp", "scipy.optimize._minpack",
        "scipy.optimize._lbfgsb", "scipy.optimize._zeros",
        "scipy.optimize._highs", "scipy.optimize._direct",
        "scipy.optimize._trlib", "scipy.optimize._group_columns",
        "scipy.optimize._bglu_dense",
    ]:
        sys.modules[name] = _Leaf(name)


class ZeroStuffConvT(nn.Module):
    """ConvTranspose2d as nearest interpolate + stride mask + conv2d.

    TRANSPOSE_CONV is rejected by the ML Drift delegate. The stuffing mask is a
    constant buffer built in __init__; building it at runtime with ``.repeat()``
    lowers to BROADCAST_TO and 8-D tensors.
    """

    def __init__(self, conv_transpose, in_hw):
        """Bakes the flipped weight and the stride mask.

        Args:
            conv_transpose: The nn.ConvTranspose2d to replace.
            in_hw: The (square) spatial size of this layer's input.
        """
        super().__init__()
        self.stride = conv_transpose.stride[0]
        self.kernel = conv_transpose.kernel_size[0]
        self.out_hw = in_hw * self.stride
        self.register_buffer(
            "weight",
            conv_transpose.weight.flip(2, 3).transpose(0, 1).contiguous())
        self.bias = conv_transpose.bias
        mask = torch.zeros(1, 1, self.out_hw, self.out_hw)
        mask[:, :, ::self.stride, ::self.stride] = 1.0
        self.register_buffer("mask", mask)

    def forward(self, x):
        """Runs the zero-stuffed transposed convolution.

        Args:
            x: Input feature map (N, C, H, W).

        Returns:
            The upsampled feature map (N, C, H*stride, W*stride).
        """
        up = F.interpolate(
            x, size=(self.out_hw, self.out_hw), mode="nearest")
        out = F.conv2d(
            up * self.mask, self.weight, self.bias, padding=self.kernel - 1)
        return out[:, :, :self.out_hw, :self.out_hw]


def _window_partition_4d(hidden_state, window_size):
    """Partitions [B,H,W,C] into windows using only <= 4-D reshapes.

    The upstream version uses a 6-D view+permute. Splitting H into the batch,
    transposing, then splitting W transposes row/col within each window, which
    ``_window_unpartition_4d`` reverses exactly; window attention is
    order-equivariant, so the result is numerically identical.

    Args:
        hidden_state: Feature map (B, H, W, C).
        window_size: Square window edge length.

    Returns:
        A tuple of (windows, (padded_height, padded_width)).
    """
    batch_size, height, width, num_channels = hidden_state.shape
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    hidden_state = F.pad(hidden_state, (0, 0, 0, pad_w, 0, pad_h))
    padded_h, padded_w = height + pad_h, width + pad_w
    n_h, n_w = padded_h // window_size, padded_w // window_size
    x = hidden_state.reshape(
        batch_size * n_h, window_size, padded_w, num_channels)
    x = x.transpose(1, 2)
    windows = x.reshape(
        batch_size * n_h * n_w, window_size, window_size, num_channels)
    return windows, (padded_h, padded_w)


def _window_unpartition_4d(windows, window_size, pad_hw, hw):
    """Exact inverse of ``_window_partition_4d``, cropping any padding.

    Args:
        windows: The partitioned windows.
        window_size: Square window edge length.
        pad_hw: The (padded_height, padded_width) from partitioning.
        hw: The original (height, width) to crop back to.

    Returns:
        The reassembled feature map (B, H, W, C).
    """
    padded_h, padded_w = pad_hw
    height, width = hw
    num_channels = windows.shape[-1]
    n_h, n_w = padded_h // window_size, padded_w // window_size
    batch_size = windows.shape[0] // (n_h * n_w)
    x = windows.reshape(
        batch_size * n_h, n_w * window_size, window_size, num_channels)
    x = x.transpose(1, 2)
    x = x.reshape(batch_size, padded_h, padded_w, num_channels)
    if padded_h > height or padded_w > width:
        x = x[:, :height, :width, :].contiguous()
    return x


def _msa_forward_4d(self, hidden_states, **kwargs):
    """Sam2MultiScaleAttention.forward with every tensor kept <= 4-D.

    Upstream reshapes qkv to a 5-D [B, H*W, 3, nHead, C] and unbinds. Slicing
    the fused projection along the channel dim instead keeps every tensor 4-D;
    the discarded pre-pool attention weights do not affect the pooled-query
    result, so the computation is reproduced exactly.

    Args:
        self: The Sam2MultiScaleAttention module.
        hidden_states: Input (B, H, W, C).
        **kwargs: Unused, accepted for signature compatibility.

    Returns:
        The attention output projected back to (B, H, W, C).
    """
    batch_size, height, width, _ = hidden_states.shape
    dim_out = self.dim_out
    num_heads = self.num_attention_heads
    head_dim = dim_out // num_heads
    qkv = self.qkv(hidden_states).reshape(
        batch_size, height * width, 3 * dim_out)
    query = qkv[..., :dim_out].reshape(
        batch_size, height * width, num_heads, head_dim)
    key = qkv[..., dim_out:2 * dim_out].reshape(
        batch_size, height * width, num_heads, head_dim)
    value = qkv[..., 2 * dim_out:].reshape(
        batch_size, height * width, num_heads, head_dim)
    if self.query_stride:
        query = M.do_pool(
            query.reshape(batch_size, height, width, -1), self.query_stride)
        height, width = query.shape[1:3]
        query = query.reshape(
            batch_size, height * width, num_heads, head_dim)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    attn = (query * self.scale) @ key.transpose(-2, -1)
    attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = (attn @ value).transpose(1, 2).reshape(
        batch_size, height, width, -1)
    return self.proj(attn_output)


def install_hiera_patches() -> None:
    """Installs the module-level Hiera rewrites onto the transformers stack."""
    M.window_partition = _window_partition_4d
    M.window_unpartition = _window_unpartition_4d
    M.Sam2MultiScaleAttention.forward = _msa_forward_4d


def bake_pos_embed(model):
    """Replaces ``_get_pos_embed`` with a precomputed constant.

    For a fixed 1024x1024 input the Hiera position embedding is a pure function
    of learned parameters (bicubic interpolate of ``pos_embed`` plus a tiled
    ``pos_embed_window``), so it is constant. Baking it removes the GATHER_ND
    from the bicubic sampling and the BROADCAST_TO from the tiling.

    Args:
        model: The loaded Sam2Model / Sam2VideoModel.

    Returns:
        The baked embedding shape, or None if no Hiera module was found.
    """
    for module in model.modules():
        if type(module).__name__ != "Sam2HieraDetModel":
            continue
        with torch.no_grad():
            embedded = module.patch_embed(torch.randn(1, 3, 1024, 1024))
            baked = module._get_pos_embed(embedded.shape[1:3])
            baked = baked.detach().clone()
        module.register_buffer("baked_pos_embed", baked)
        module._get_pos_embed = types.MethodType(
            lambda self, hw: self.baked_pos_embed, module)
        return baked.shape
    return None


def _builtin_names():
    """Builds an int -> name map of the TFLite builtin operator codes.

    Returns:
        A dict mapping each BuiltinOperator enum value to its name.
    """
    return {
        v: k for k, v in vars(schema.BuiltinOperator).items()
        if isinstance(v, int)
    }


def opcheck(path, tag):
    """Prints GPU-hostile op counts and any > 4-D tensors for one graph.

    Parses the flatbuffer directly (no Interpreter) so the check does not need
    a runtime.

    Args:
        path: Path to the .tflite file.
        tag: A short label printed with the result.

    Returns:
        The number of GPU-hostile ops found (0 means GPU-clean).
    """
    with open(path, "rb") as handle:
        buf = bytearray(handle.read())
    model = schema.Model.GetRootAsModel(buf, 0)
    names = _builtin_names()
    codes = [
        model.OperatorCodes(i) for i in range(model.OperatorCodesLength())
    ]
    bad = {}
    over = 0
    for s in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(s)
        for o in range(subgraph.OperatorsLength()):
            code = codes[subgraph.Operators(o).OpcodeIndex()]
            name = names.get(code.BuiltinCode(), "?")
            if name in GPU_BAD:
                bad[name] = bad.get(name, 0) + 1
        for t in range(subgraph.TensorsLength()):
            if subgraph.Tensors(t).ShapeLength() > 4:
                over += 1
    print(f"{tag}: GPU_BAD={bad or 'NONE'} >4D={over}")
    return sum(bad.values())


def fp16(src, dst):
    """Quantizes a graph's weights to fp16 and returns its size in MB.

    Args:
        src: Path to the fp32 .tflite input.
        dst: Path to write the fp16 .tflite output.

    Returns:
        The fp16 file size in megabytes.
    """
    import os
    from ai_edge_quantizer import quantizer
    if os.path.exists(dst):
        os.remove(dst)
    quant = quantizer.Quantizer(src)
    quant.load_quantization_recipe(FP16_RECIPE)
    quant.quantize().export_model(dst)
    return os.path.getsize(dst) / 1e6
