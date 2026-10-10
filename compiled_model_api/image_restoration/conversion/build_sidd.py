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

"""NAFNet (image restoration) -> LiteRT CompiledModel GPU.

NAFNet (Nonlinear Activation Free Network, ECCV 2022, MIT) = a U-Net of
NAFBlocks. No activation functions at all (SimpleGate = channel-split
multiply). Pure CNN -> Bucket-1. Three GPU re-authorings, all numerically
exact:
  - LayerNorm2d (custom autograd Function) -> standard channel LayerNorm
    (x*x, not pow; POW is banned)
  - Simplified Channel Attention `AdaptiveAvgPool2d(1)` ->
    `mean(3).mean(2)` (Mali mis-computes the multi-axis global pool ->
    two single-axis means)
  - upsample `Conv2d(1x1) + PixelShuffle(2)` -> Conv2d + depth-to-space
    `ZeroStuffConvT2d` (PixelShuffle lowers to a 6D reshape;
    ZeroStuffConvT2d is RESIZE_NEAREST + MUL + CONV_2D)

Default = NAFNet-SIDD-width32 (denoise).
Run: python build_sidd.py [opcheck|parity|all]
"""
import sys
import os
import collections
import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "NAFNet-SIDD-width32.pth")
WIDTH, ENC, MID, DEC = 32, [2, 2, 4, 8], 12, [2, 2, 2, 2]
# multiple of 2^len(ENC)=16
SIZE = int(os.environ.get("NAF_SIZE", "256"))
BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE",
          "SELECT", "SELECT_V2", "BROADCAST_TO", "POW", "TRANSPOSE_CONV",
          "CAST", "EMBEDDING_LOOKUP", "RFFT2D", "FFT", "STFT", "COMPLEX",
          "RFFT", "IRFFT", "CUMSUM"}


# ─── ZeroStuffConvT2d (shipped DAC/PP-OCR/Metric3D pattern) ───────────────────
class ZeroStuffConvT2d(nn.Module):
    """Exact GPU-clean ConvTranspose2d (k,s,p=0,op=0): nearest-upsample x
    stride zero-stuff mask + flipped conv2d + crop. RESIZE_NEAREST + MUL +
    CONV_2D (no TRANSPOSE_CONV)."""
    def __init__(self, ct, Hin, Win):
        super().__init__()
        self.s = ct.stride[0]
        self.k = ct.kernel_size[0]
        self.p = ct.padding[0]
        self.op = ct.output_padding[0]
        self.Hin = Hin
        self.Win = Win
        w = ct.weight.detach().flip(2).flip(3).permute(1, 0, 2, 3).contiguous()
        self.register_buffer("w", w)
        self.register_buffer(
            "b", ct.bias.detach().clone() if ct.bias is not None
            else torch.zeros(ct.out_channels))
        mh = np.zeros((Hin * self.s, Win * self.s), np.float32)
        mh[::self.s, ::self.s] = 1.0
        self.register_buffer("mask", torch.from_numpy(mh)[None, None])

    def forward(self, x):
        xn = F.interpolate(x, size=(self.Hin * self.s, self.Win * self.s),
                           mode="nearest") * self.mask
        y = F.conv2d(xn, self.w, bias=self.b, padding=self.k - 1)
        olH = (self.Hin - 1) * self.s + self.k - 2 * self.p + self.op
        olW = (self.Win - 1) * self.s + self.k - 2 * self.p + self.op
        return y[:, :, self.p:self.p + olH, self.p:self.p + olW]


def pixelshuffle_d2s(conv, r, Hin, Win):
    """Conv2d(1x1) + PixelShuffle(r) -> Conv2d(1x1) + depth-to-space
    ZeroStuffConvT2d (exact).

    Args:
        conv: The 1x1 Conv2d producing Cout * r * r channels.
        r: PixelShuffle upscale factor.
        Hin: Input height at this stage.
        Win: Input width at this stage.

    Returns:
        nn.Sequential(conv, ZeroStuffConvT2d) equivalent to
        conv + PixelShuffle(r).
    """
    Cin = conv.out_channels                      # = Cout * r*r
    Cout = Cin // (r * r)
    ct = nn.ConvTranspose2d(Cin, Cout, r, stride=r, bias=False)
    with torch.no_grad():
        wt = torch.zeros(Cin, Cout, r, r)
        for c in range(Cout):
            for i in range(r):
                for j in range(r):
                    # PyTorch PixelShuffle channel order
                    wt[c * r * r + i * r + j, c, i, j] = 1.0
        ct.weight.copy_(wt)
    return nn.Sequential(conv, ZeroStuffConvT2d(ct, Hin, Win))


# ─── NAFNet (keys match the official checkpoint exactly) ──────────────────────
class LayerNorm2d(nn.Module):
    """Channel LayerNorm, fp16-SAFE. NAFNet's residual stream grows large
    (|x|~175 at the bottleneck), so the channel reduction `sum_c x` (~90k)
    and `sum_c (x-mu)^2` (~15M) OVERFLOW fp16 (max 65504) on the Mali GPU
    delegate (which computes in fp16 regardless of model dtype) ->
    garbage, compounding over the deep blocks. Fix: do the reductions in a
    down-scaled domain (x/S) so the sums stay < 65504, then rescale var
    and (x-mu) back to the original domain -> numerically EXACT (LayerNorm
    is scale-invariant), eps in the original domain. S=128 keeps sums safe
    up to ~3x the observed magnitude."""
    S = 128.0

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(c)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(c)))
        self.eps = eps

    def forward(self, x):
        xs = x * (1.0 / self.S)  # down-scale before any reduction
        mu = xs.mean(1, keepdim=True)                 # sum_c (x/S): fp16-safe
        d = xs - mu                                   # (x - mu_orig)/S
        # sum_c (x/S)^2 safe, then rescale to true var
        var = (d * d).mean(1, keepdim=True) * (self.S * self.S)
        d = d * self.S  # back to (x - mu_orig), representable
        # exact original LayerNorm, eps in original domain
        y = d * torch.rsqrt(var + self.eps)
        return (y * self.weight[None, :, None, None]
                + self.bias[None, :, None, None])


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class GlobalAvgPool(nn.Module):
    """AdaptiveAvgPool2d(1) -> two single-axis means (Mali rejects the
    multi-axis global pool)."""
    def forward(self, x):
        return x.mean(3, keepdim=True).mean(2, keepdim=True)


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2):
        super().__init__()
        dw = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        self.sca = nn.Sequential(GlobalAvgPool(),
                                 nn.Conv2d(dw // 2, dw // 2, 1))
        self.sg = SimpleGate()
        ffn = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=WIDTH, middle_blk_num=MID,
                 enc_blk_nums=ENC, dec_blk_nums=DEC):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2
        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            # placeholder Sequential(Conv2d, PixelShuffle); PixelShuffle
            # is swapped after a size probe
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        x = self.intro(inp)
        encs = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for dec, up, skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + skip
            x = dec(x)
        x = self.ending(x)
        return x + inp


def swap_pixelshuffle(model, size):
    """Probes up-input sizes, then replaces each Conv2d+PixelShuffle with
    Conv2d+depth-to-space.

    Args:
        model: NAFNet instance (modified in place).
        size: Square input size used for the probe forward.
    """
    L = {}
    hks = []
    for n, mo in model.named_modules():
        if isinstance(mo, nn.PixelShuffle):
            par = ".".join(n.split(".")[:-1])
            conv = dict(model.named_modules())[par][0]
            hks.append(conv.register_forward_pre_hook(
                (lambda nm: (lambda mod, i: L.__setitem__(
                    nm, i[0].shape[-2:])))(par)))
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size))
    for h in hks:
        h.remove()
    n = 0
    for name, mo in list(model.named_modules()):
        if (isinstance(mo, nn.Sequential) and len(mo) == 2
                and isinstance(mo[1], nn.PixelShuffle)):
            hh, ww = L[name]
            par = model
            *pth, last = name.split(".")
            for q in pth:
                par = getattr(par, q) if not q.isdigit() else par[int(q)]
            r = mo[1].upscale_factor
            par[int(last)] = (
                pixelshuffle_d2s(mo[0], r, int(hh), int(ww))
                if last.isdigit()
                else setattr(par, last,
                             pixelshuffle_d2s(mo[0], r, int(hh), int(ww))))
            n += 1
    print(f"  swapped {n} PixelShuffle -> depth-to-space ZeroStuffConvT2d")


def build():
    """Builds NAFNet, loads the checkpoint, and swaps PixelShuffle.

    Returns:
        The eval-mode NAFNet with GPU-clean upsampling.
    """
    m = NAFNet()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck.get("params", ck.get("state_dict", ck))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert not missing, f"missing keys: {missing[:5]}"
    m.eval()
    swap_pixelshuffle(m, SIZE)
    return m


def opcheck(path, label):
    """Static GPU-compat scan of the op set in a .tflite flatbuffer.

    Args:
        path: Path to the .tflite file to scan.
        label: Tag printed with the results.
    """
    from ai_edge_litert import schema_py_generated as schema
    with open(path, "rb") as f:
        model = schema.ModelT.InitFromPackedBuf(f.read(), 0)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items()
             if not k.startswith("_")}
    ops = collections.Counter()
    over = 0
    for g in model.subgraphs:
        for op in g.operators:
            c = model.operatorCodes[op.opcodeIndex]
            code = max(c.builtinCode, c.deprecatedBuiltinCode)
            ops[c.customCode.decode() if c.customCode
                else names.get(code, str(code))] += 1
        over += sum(1 for t in g.tensors
                    if t.shape is not None and len(t.shape) > 4)
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
    print(f"[{label}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    print(f"[{label}] banned:{bad or 'NONE'} >4D:{over} "
          f"size {os.path.getsize(path)/1e6:.1f}MB",
          "GPU-CLEAN" if not bad and not over else "BLOCKERS")


def run_tflite(path, x):
    """Runs a single inference through the LiteRT CompiledModel API.

    Args:
        path: Path to the .tflite model.
        x: Input array, written as fp32 into the first input buffer.

    Returns:
        The flat fp32 output array.
    """
    from ai_edge_litert.compiled_model import CompiledModel
    model = CompiledModel.from_file(path)
    inputs = model.create_input_buffers(0)
    outputs = model.create_output_buffers(0)
    inputs[0].write(np.ascontiguousarray(x, dtype=np.float32))
    model.run_by_index(0, inputs, outputs)
    n = (model.get_output_buffer_requirements(0, 0)["buffer_size"]
         // np.dtype(np.float32).itemsize)
    return outputs[0].read(n, np.float32)


def to_fp16(fp32, fp16):
    """Quantizes an fp32 .tflite model to fp16 weights.

    Args:
        fp32: Path to the source fp32 .tflite model.
        fp16: Output path for the fp16 model (overwritten if present).

    Returns:
        The fp16 output path.
    """
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT),
        algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(fp16):
        os.remove(fp16)
    q = quantizer.Quantizer(float_model=fp32)
    q.load_quantization_recipe(rm.get_quantization_recipe())
    q.quantize().export_model(fp16)
    return fp16


def main():
    """Builds the model, converts to .tflite, and verifies parity.

    argv[1] selects the stage: "forward" stops after the torch forward
    pass; "all" (default) additionally quantizes to fp16.
    """
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    m = build()
    img = torch.randn(1, 3, SIZE, SIZE)
    with torch.no_grad():
        ref = m(img)
    print(f"forward: out {tuple(ref.shape)} "
          f"range [{ref.min():.3f},{ref.max():.3f}]")
    if stage == "forward":
        return
    import litert_torch
    fp32 = os.path.join(HERE, "sidd.tflite")
    litert_torch.convert(m, (img,)).export(fp32)
    opcheck(fp32, "sidd")
    o = run_tflite(fp32, img.numpy()).reshape(tuple(ref.shape))
    print(f"tflite vs torch: corr "
          f"{np.corrcoef(o.ravel(), ref.numpy().ravel())[0,1]:.6f} "
          f"max|d| {np.abs(o-ref.numpy()).max():.3e}")
    if stage == "all":
        to_fp16(fp32, os.path.join(HERE, "sidd_fp16.tflite"))
        opcheck(os.path.join(HERE, "sidd_fp16.tflite"), "sidd_fp16")


if __name__ == "__main__":
    main()
