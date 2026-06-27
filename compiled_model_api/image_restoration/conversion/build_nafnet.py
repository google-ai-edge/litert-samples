#!/usr/bin/env python3
"""NAFNet (image restoration) -> LiteRT CompiledModel GPU.

NAFNet (Nonlinear Activation Free Network, ECCV 2022, MIT) = a U-Net of NAFBlocks. No activation
functions at all (SimpleGate = channel-split multiply). Pure CNN -> Bucket-1. Three GPU re-authorings,
all numerically exact:
  - LayerNorm2d (custom autograd Function) -> standard channel LayerNorm (x*x, not pow; POW is banned)
  - Simplified Channel Attention `AdaptiveAvgPool2d(1)` -> `mean(3).mean(2)` (Mali mis-computes the
    multi-axis global pool -> two single-axis means)
  - upsample `Conv2d(1x1) + PixelShuffle(2)` -> Conv2d + depth-to-space `ZeroStuffConvT2d`
    (PixelShuffle lowers to a 6D reshape; ZeroStuffConvT2d is RESIZE_NEAREST + MUL + CONV_2D)

Default = NAFNet-GoPro-width32 (deblur, 17M). Run: ~/clipconv/bin/python build_nafnet.py [opcheck|parity|all]
"""
import sys, os, collections, types
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "NAFNet-GoPro-width32.pth")
WIDTH, ENC, MID, DEC = 32, [1, 1, 1, 28], 1, [1, 1, 1, 1]
SIZE = int(os.environ.get("NAF_SIZE", "256"))                  # multiple of 2^len(ENC)=16
BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP",
          "RFFT2D", "FFT", "STFT", "COMPLEX", "RFFT", "IRFFT", "CUMSUM"}


# ─── ZeroStuffConvT2d (shipped DAC/PP-OCR/Metric3D pattern) ───────────────────
class ZeroStuffConvT2d(nn.Module):
    """Exact GPU-clean ConvTranspose2d (k,s,p=0,op=0): nearest-upsample x stride zero-stuff mask
    + flipped conv2d + crop. RESIZE_NEAREST + MUL + CONV_2D (no TRANSPOSE_CONV)."""
    def __init__(s, ct, Hin, Win):
        super().__init__()
        s.s = ct.stride[0]; s.k = ct.kernel_size[0]; s.p = ct.padding[0]
        s.op = ct.output_padding[0]; s.Hin = Hin; s.Win = Win
        w = ct.weight.detach().flip(2).flip(3).permute(1, 0, 2, 3).contiguous()
        s.register_buffer("w", w)
        s.register_buffer("b", ct.bias.detach().clone() if ct.bias is not None else torch.zeros(ct.out_channels))
        mh = np.zeros((Hin * s.s, Win * s.s), np.float32); mh[::s.s, ::s.s] = 1.0
        s.register_buffer("mask", torch.from_numpy(mh)[None, None])

    def forward(s, x):
        xn = F.interpolate(x, size=(s.Hin * s.s, s.Win * s.s), mode="nearest") * s.mask
        y = F.conv2d(xn, s.w, bias=s.b, padding=s.k - 1)
        olH = (s.Hin - 1) * s.s + s.k - 2 * s.p + s.op
        olW = (s.Win - 1) * s.s + s.k - 2 * s.p + s.op
        return y[:, :, s.p:s.p + olH, s.p:s.p + olW]


def pixelshuffle_d2s(conv, r, Hin, Win):
    """Conv2d(1x1) + PixelShuffle(r)  ->  Conv2d(1x1) + depth-to-space ZeroStuffConvT2d (exact)."""
    Cin = conv.out_channels                      # = Cout * r*r
    Cout = Cin // (r * r)
    ct = nn.ConvTranspose2d(Cin, Cout, r, stride=r, bias=False)
    with torch.no_grad():
        wt = torch.zeros(Cin, Cout, r, r)
        for c in range(Cout):
            for i in range(r):
                for j in range(r):
                    wt[c * r * r + i * r + j, c, i, j] = 1.0   # PyTorch PixelShuffle channel order
        ct.weight.copy_(wt)
    return nn.Sequential(conv, ZeroStuffConvT2d(ct, Hin, Win))


# ─── NAFNet (keys match the official checkpoint exactly) ──────────────────────
class LayerNorm2d(nn.Module):
    """Channel LayerNorm, fp16-SAFE. NAFNet's residual stream grows large (|x|~175 at the bottleneck),
    so the channel reduction `sum_c x` (~90k) and `sum_c (x-mu)^2` (~15M) OVERFLOW fp16 (max 65504) on the
    Mali GPU delegate (which computes in fp16 regardless of model dtype) -> garbage, compounding over the
    deep blocks. Fix: do the reductions in a down-scaled domain (x/S) so the sums stay < 65504, then rescale
    var and (x-mu) back to the original domain -> numerically EXACT (LayerNorm is scale-invariant), eps in
    the original domain. S=128 keeps sums safe up to ~3x the observed magnitude."""
    S = 128.0

    def __init__(s, c, eps=1e-6):
        super().__init__()
        s.register_parameter("weight", nn.Parameter(torch.ones(c)))
        s.register_parameter("bias", nn.Parameter(torch.zeros(c)))
        s.eps = eps

    def forward(s, x):
        xs = x * (1.0 / s.S)                          # down-scale before any reduction
        mu = xs.mean(1, keepdim=True)                 # sum_c (x/S): fp16-safe
        d = xs - mu                                   # (x - mu_orig)/S
        var = (d * d).mean(1, keepdim=True) * (s.S * s.S)   # sum_c (x/S)^2 safe, then rescale to true var
        d = d * s.S                                   # back to (x - mu_orig), representable
        y = d * torch.rsqrt(var + s.eps)              # exact original LayerNorm, eps in original domain
        return y * s.weight[None, :, None, None] + s.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(s, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class GlobalAvgPool(nn.Module):
    """AdaptiveAvgPool2d(1) -> two single-axis means (Mali rejects the multi-axis global pool)."""
    def forward(s, x):
        return x.mean(3, keepdim=True).mean(2, keepdim=True)


class NAFBlock(nn.Module):
    def __init__(s, c, DW_Expand=2, FFN_Expand=2):
        super().__init__()
        dw = c * DW_Expand
        s.conv1 = nn.Conv2d(c, dw, 1)
        s.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        s.conv3 = nn.Conv2d(dw // 2, c, 1)
        s.sca = nn.Sequential(GlobalAvgPool(), nn.Conv2d(dw // 2, dw // 2, 1))
        s.sg = SimpleGate()
        ffn = FFN_Expand * c
        s.conv4 = nn.Conv2d(c, ffn, 1)
        s.conv5 = nn.Conv2d(ffn // 2, c, 1)
        s.norm1 = LayerNorm2d(c)
        s.norm2 = LayerNorm2d(c)
        s.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        s.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(s, inp):
        x = s.norm1(inp)
        x = s.conv1(x); x = s.conv2(x); x = s.sg(x); x = x * s.sca(x); x = s.conv3(x)
        y = inp + x * s.beta
        x = s.conv4(s.norm2(y)); x = s.sg(x); x = s.conv5(x)
        return y + x * s.gamma


class NAFNet(nn.Module):
    def __init__(s, img_channel=3, width=WIDTH, middle_blk_num=MID, enc_blk_nums=ENC, dec_blk_nums=DEC):
        super().__init__()
        s.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        s.ending = nn.Conv2d(width, img_channel, 3, padding=1)
        s.encoders = nn.ModuleList(); s.decoders = nn.ModuleList()
        s.ups = nn.ModuleList(); s.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            s.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            s.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2)); chan *= 2
        s.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            # placeholder Sequential(Conv2d, PixelShuffle); PixelShuffle is swapped after a size probe
            s.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            s.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        s.padder_size = 2 ** len(s.encoders)

    def forward(s, inp):
        x = s.intro(inp)
        encs = []
        for enc, down in zip(s.encoders, s.downs):
            x = enc(x); encs.append(x); x = down(x)
        x = s.middle_blks(x)
        for dec, up, skip in zip(s.decoders, s.ups, encs[::-1]):
            x = up(x); x = x + skip; x = dec(x)
        x = s.ending(x)
        return x + inp


def swap_pixelshuffle(model, size):
    """Probe up-input sizes, then replace each Conv2d+PixelShuffle with Conv2d+depth-to-space."""
    L = {}; hks = []
    for n, mo in model.named_modules():
        if isinstance(mo, nn.PixelShuffle):
            par = ".".join(n.split(".")[:-1])
            conv = dict(model.named_modules())[par][0]
            hks.append(conv.register_forward_pre_hook(
                (lambda nm: (lambda mod, i: L.__setitem__(nm, i[0].shape[-2:])))(par)))
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size))
    for h in hks: h.remove()
    n = 0
    for name, mo in list(model.named_modules()):
        if isinstance(mo, nn.Sequential) and len(mo) == 2 and isinstance(mo[1], nn.PixelShuffle):
            hh, ww = L[name]
            par = model; *pth, last = name.split(".")
            for q in pth: par = getattr(par, q) if not q.isdigit() else par[int(q)]
            r = mo[1].upscale_factor
            par[int(last)] = pixelshuffle_d2s(mo[0], r, int(hh), int(ww)) if last.isdigit() \
                else setattr(par, last, pixelshuffle_d2s(mo[0], r, int(hh), int(ww)))
            n += 1
    print(f"  swapped {n} PixelShuffle -> depth-to-space ZeroStuffConvT2d")


def build():
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
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path); it.allocate_tensors()
    ops = collections.Counter(d.get("op_name", "?") for d in it._get_ops_details())
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
    over = sum(1 for d in it.get_tensor_details() if len(d.get("shape", [])) > 4)
    print(f"[{label}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    print(f"[{label}] banned:{bad or 'NONE'} >4D:{over} size {os.path.getsize(path)/1e6:.1f}MB",
          "GPU-CLEAN" if not bad and not over else "BLOCKERS")
    return it


def to_fp16(fp32, fp16):
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT), algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(fp16): os.remove(fp16)
    q = quantizer.Quantizer(float_model=fp32); q.load_quantization_recipe(rm.get_quantization_recipe())
    q.quantize().export_model(fp16); return fp16


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    m = build()
    img = torch.randn(1, 3, SIZE, SIZE)
    with torch.no_grad():
        ref = m(img)
    print(f"forward: out {tuple(ref.shape)} range [{ref.min():.3f},{ref.max():.3f}]")
    if stage == "forward":
        return
    import litert_torch
    fp32 = os.path.join(HERE, "nafnet.tflite")
    litert_torch.convert(m, (img,)).export(fp32)
    it = opcheck(fp32, "nafnet")
    d = it.get_input_details()[0]; it.set_tensor(d["index"], img.numpy().astype(d["dtype"])); it.invoke()
    o = it.get_tensor(it.get_output_details()[0]["index"])
    print(f"tflite vs torch: corr {np.corrcoef(o.ravel(), ref.numpy().ravel())[0,1]:.6f} max|d| {np.abs(o-ref.numpy()).max():.3e}")
    if stage == "all":
        to_fp16(fp32, os.path.join(HERE, "nafnet_fp16.tflite"))
        opcheck(os.path.join(HERE, "nafnet_fp16.tflite"), "nafnet_fp16")


if __name__ == "__main__":
    main()
