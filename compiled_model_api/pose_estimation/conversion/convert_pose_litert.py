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

"""Convert lightweight-OpenPose to LiteRT for this sample.

The model is a MobileNet-based heatmap pose network. We export ONLY the final-stage
keypoint heatmaps and do the keypoint decode (argmax) in Kotlin, so the graph stays
pure-conv and fully GPU-resident -- unlike MoveNet's official tflite, whose baked-in
decode (GATHER_ND) the GPU delegate can't run.

Weights: Daniil-Osokin/lightweight-human-pose-estimation.pytorch (Apache-2.0).

    pip install litert-torch ai-edge-quantizer torch
    python convert_pose_litert.py [out_dir] [size]

Produces:
    pose_<size>.tflite        fp32
    pose_<size>_fp16.tflite    fp16 (used by the app)
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
    "https://download.01.org/opencv/openvino_training_extensions/models/"
    "human_pose_estimation/checkpoint_iter_370000.pth"
)


def conv(in_c, out_c, k=3, p=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    m = [nn.Conv2d(in_c, out_c, k, stride, p, dilation, bias=bias)]
    if bn:
        m.append(nn.BatchNorm2d(out_c))
    if relu:
        m.append(nn.ReLU(inplace=True))
    return nn.Sequential(*m)


def conv_dw(in_c, out_c, k=3, p=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_c, in_c, k, stride, p, dilation=dilation, groups=in_c, bias=False),
        nn.BatchNorm2d(in_c), nn.ReLU(inplace=True),
        nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )


def conv_dw_no_bn(in_c, out_c, k=3, p=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_c, in_c, k, stride, p, dilation=dilation, groups=in_c, bias=False),
        nn.ELU(inplace=True),
        nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
        nn.ELU(inplace=True),
    )


class Cpm(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.align = conv(in_c, out_c, k=1, p=0, bn=False)
        self.trunk = nn.Sequential(conv_dw_no_bn(out_c, out_c), conv_dw_no_bn(out_c, out_c),
                                    conv_dw_no_bn(out_c, out_c))
        self.conv = conv(out_c, out_c, bn=False)

    def forward(self, x):
        x = self.align(x)
        return self.conv(x + self.trunk(x))


class InitialStage(nn.Module):
    def __init__(self, nc, nh, npa):
        super().__init__()
        self.trunk = nn.Sequential(conv(nc, nc, bn=False), conv(nc, nc, bn=False), conv(nc, nc, bn=False))
        self.heatmaps = nn.Sequential(conv(nc, 512, k=1, p=0, bn=False), conv(512, nh, k=1, p=0, bn=False, relu=False))
        self.pafs = nn.Sequential(conv(nc, 512, k=1, p=0, bn=False), conv(512, npa, k=1, p=0, bn=False, relu=False))

    def forward(self, x):
        t = self.trunk(x)
        return [self.heatmaps(t), self.pafs(t)]


class RefinementStageBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.initial = conv(in_c, out_c, k=1, p=0, bn=False)
        self.trunk = nn.Sequential(conv(out_c, out_c), conv(out_c, out_c, dilation=2, p=2))

    def forward(self, x):
        i = self.initial(x)
        return i + self.trunk(i)


class RefinementStage(nn.Module):
    def __init__(self, in_c, out_c, nh, npa):
        super().__init__()
        self.trunk = nn.Sequential(*[RefinementStageBlock(in_c if i == 0 else out_c, out_c) for i in range(5)])
        self.heatmaps = nn.Sequential(conv(out_c, out_c, k=1, p=0, bn=False), conv(out_c, nh, k=1, p=0, bn=False, relu=False))
        self.pafs = nn.Sequential(conv(out_c, out_c, k=1, p=0, bn=False), conv(out_c, npa, k=1, p=0, bn=False, relu=False))

    def forward(self, x):
        t = self.trunk(x)
        return [self.heatmaps(t), self.pafs(t)]


class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stages=1, nc=128, nh=19, npa=38):
        super().__init__()
        self.model = nn.Sequential(
            conv(3, 32, stride=2, bias=False), conv_dw(32, 64), conv_dw(64, 128, stride=2),
            conv_dw(128, 128), conv_dw(128, 256, stride=2), conv_dw(256, 256), conv_dw(256, 512),
            conv_dw(512, 512, dilation=2, p=2), conv_dw(512, 512), conv_dw(512, 512),
            conv_dw(512, 512), conv_dw(512, 512))
        self.cpm = Cpm(512, nc)
        self.initial_stage = InitialStage(nc, nh, npa)
        self.refinement_stages = nn.ModuleList(
            [RefinementStage(nc + nh + npa, nc, nh, npa) for _ in range(num_refinement_stages)])

    def forward(self, x):
        backbone = self.cpm(self.model(x))
        out = self.initial_stage(backbone)
        for stage in self.refinement_stages:
            out.extend(stage(torch.cat([backbone, out[-2], out[-1]], dim=1)))
        return out


class PoseHeatmaps(nn.Module):
    """Return only the final-stage heatmaps (drop PAFs/intermediate stages)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)[-2]


def fp16_recipe():
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(
        regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT),
        algorithm_key=AlgorithmName.FLOAT_CASTING)
    return rm.get_quantization_recipe()


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    os.makedirs(out_dir, exist_ok=True)

    weights = os.path.join(out_dir, "checkpoint_iter_370000.pth")
    if not os.path.exists(weights):
        print("Downloading lightweight-OpenPose weights...")
        urllib.request.urlretrieve(WEIGHTS_URL, weights)

    net = PoseEstimationWithMobileNet(num_refinement_stages=1).eval()
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=False)  # checkpoint has extra refinement stages
    model = PoseHeatmaps(net).eval()

    nchw = torch.rand(1, 3, size, size)
    clio = litert_torch.to_channel_last_io(model, args=[0], outputs=[0])
    nhwc = nchw.permute(0, 2, 3, 1).contiguous()
    edge = litert_torch.convert(clio, (nhwc,))

    fp32 = os.path.join(out_dir, f"pose_{size}.tflite")
    edge.export(fp32)
    print(f"fp32 -> {fp32} ({os.path.getsize(fp32) / 1e6:.2f} MB)")

    fp16 = os.path.join(out_dir, f"pose_{size}_fp16.tflite")
    qt = quantizer.Quantizer(float_model=fp32)
    qt.load_quantization_recipe(fp16_recipe())
    qt.quantize().export_model(fp16)
    print(f"fp16 -> {fp16} ({os.path.getsize(fp16) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
