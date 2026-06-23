"""Convert Zero-DCE (low-light enhancement) to LiteRT for this sample.

Zero-DCE's DCE-Net is a tiny 7-conv CNN that estimates per-pixel tone curves; the
curves are applied iteratively (8x) to brighten the image. Both the estimation and the
iterative application are baked into the exported graph, so the model maps a dark image
straight to the enhanced one. `x*x` is used instead of `pow(x, 2)` so the curve step
lowers to MUL (GPU-clean), not POW.

The export uses channel-last NHWC I/O (interleaved RGB, range [0,1]) on both the input
and the output, matching the litert-samples convention.

    pip install litert-torch ai-edge-quantizer torch
    python convert_zerodce_litert.py [out_dir] [size]

Produces:
    zerodce_<size>.tflite        fp32
    zerodce_<size>_fp16.tflite   fp16 (used by the app)
"""
import os
import sys
import urllib.request

import torch
import torch.nn as nn
import litert_torch
from ai_edge_quantizer import quantizer, recipe_manager
from ai_edge_quantizer.recipe import AlgorithmName, qtyping

WEIGHTS_URL = (
    "https://github.com/Li-Chongyi/Zero-DCE/raw/master/Zero-DCE_code/snapshots/Epoch99.pth"
)


class ZeroDCE(nn.Module):
    """DCE-Net (enhance_net_nopool), returning only the final enhanced image."""

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        f = 32
        self.e_conv1 = nn.Conv2d(3, f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(f, f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(f, f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(f, f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(f * 2, f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(f * 2, f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(f * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        for r in torch.split(x_r, 3, dim=1):  # 8 curve iterations (x^2 == x*x -> MUL)
            x = x + r * (x * x - x)
        return x


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
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    os.makedirs(out_dir, exist_ok=True)

    weights = os.path.join(out_dir, "Epoch99.pth")
    if not os.path.exists(weights):
        print("Downloading Zero-DCE weights...")
        urllib.request.urlretrieve(WEIGHTS_URL, weights)

    model = ZeroDCE().eval()
    sd = torch.load(weights, map_location="cpu")
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})

    # Channel-last NHWC I/O on both input and output (interleaved RGB, [0,1]).
    nchw = torch.rand(1, 3, size, size)
    clio = litert_torch.to_channel_last_io(model, args=[0], outputs=[0])
    nhwc = nchw.permute(0, 2, 3, 1).contiguous()
    edge = litert_torch.convert(clio, (nhwc,))

    fp32 = os.path.join(out_dir, f"zerodce_{size}.tflite")
    edge.export(fp32)
    print(f"fp32 -> {fp32} ({os.path.getsize(fp32) / 1e6:.2f} MB)")

    fp16 = os.path.join(out_dir, f"zerodce_{size}_fp16.tflite")
    qt = quantizer.Quantizer(float_model=fp32)
    qt.load_quantization_recipe(fp16_recipe())
    qt.quantize().export_model(fp16)
    print(f"fp16 -> {fp16} ({os.path.getsize(fp16) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
