"""Convert VDSR (Very Deep Super-Resolution, CVPR'16) to LiteRT for this sample.

VDSR refines the luminance (Y) of an image with 20 conv layers and a global residual.
It has NO in-network upsampling, so the graph is conv + ReLU + a residual add -- no
PixelShuffle, no PReLU, no attention -> it converts with no patches and runs fully on
the GPU delegate. (Many SR models use PixelShuffle/PReLU, which lower to GPU-hostile ops
through the PyTorch path; VDSR's pre-upscaled-input design sidesteps that.)

Weights: twtygqyy/pytorch-vdsr `model_epoch_50.pth` (MIT). It is a whole-model pickle, so
we register the arch under a fake `vdsr` module to unpickle it.

    pip install litert-torch ai-edge-quantizer torch
    python convert_vdsr_litert.py [out_dir] [size]

Produces:
    vdsr_<size>.tflite        fp32
    vdsr_<size>_fp16.tflite    fp16 (used by the app)
"""
import os
import sys
import types
import urllib.request

import torch
import torch.nn as nn
import litert_torch
from ai_edge_quantizer import quantizer, recipe_manager
from ai_edge_quantizer.recipe import AlgorithmName, qtyping

WEIGHTS_URL = "https://github.com/twtygqyy/pytorch-vdsr/raw/master/model/model_epoch_50.pth"


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual_layer = nn.Sequential(*[Conv_ReLU_Block() for _ in range(18)])
        self.input = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.output = nn.Conv2d(64, 1, 3, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        return torch.add(out, residual)


# Register the classes under a fake `vdsr` module so the whole-model pickle unpickles.
_m = types.ModuleType("vdsr")
_m.Conv_ReLU_Block = Conv_ReLU_Block
_m.Net = Net
sys.modules["vdsr"] = _m


def fp16_recipe():
    rm = recipe_manager.RecipeManager()
    op_config = qtyping.OpQuantizationConfig(
        weight_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
        compute_precision=qtyping.ComputePrecision.FLOAT,
    )
    rm.add_quantization_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=op_config,
        algorithm_key=AlgorithmName.FLOAT_CASTING,
    )
    return rm.get_quantization_recipe()


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    os.makedirs(out_dir, exist_ok=True)

    weights = os.path.join(out_dir, "model_epoch_50.pth")
    if not os.path.exists(weights):
        print("Downloading VDSR weights...")
        urllib.request.urlretrieve(WEIGHTS_URL, weights)

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    loaded = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    loaded = getattr(loaded, "module", loaded)  # unwrap DataParallel
    model = Net().eval()
    model.load_state_dict(loaded.state_dict())

    # Channel-last NHWC I/O (luminance Y, 1 channel, range [0,1]).
    nchw = torch.rand(1, 1, size, size)
    clio = litert_torch.to_channel_last_io(model, args=[0], outputs=[0])
    nhwc = nchw.permute(0, 2, 3, 1).contiguous()
    edge = litert_torch.convert(clio, (nhwc,))

    fp32 = os.path.join(out_dir, f"vdsr_{size}.tflite")
    edge.export(fp32)
    print(f"fp32 -> {fp32} ({os.path.getsize(fp32) / 1e6:.2f} MB)")

    fp16 = os.path.join(out_dir, f"vdsr_{size}_fp16.tflite")
    qt = quantizer.Quantizer(float_model=fp32)
    qt.load_quantization_recipe(fp16_recipe())
    qt.quantize().export_model(fp16)
    print(f"fp16 -> {fp16} ({os.path.getsize(fp16) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
