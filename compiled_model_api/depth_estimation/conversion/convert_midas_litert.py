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

"""Convert MiDaS_small (monocular depth) to LiteRT for the depth_estimation sample.

Self-contained: PyTorch -> litert-torch (channel-last NHWC I/O) -> fp16 weights,
then prints the op histogram and a numerical fidelity check. The CNN MiDaS
(EfficientNet-Lite3 backbone) converts with no patches and lowers entirely to
GPU-clean builtins.

    pip install litert-torch ai-edge-quantizer torch timm matplotlib pillow
    python convert_midas_litert.py [out_dir] [size]

Outputs:
    midas_small_<size>.tflite        (fp32)
    midas_small_<size>_fp16.tflite   (fp16, recommended for the app)
"""
import os
import sys

import numpy as np
import torch


def convert(out_dir: str, size: int):
    import litert_torch

    os.makedirs(out_dir, exist_ok=True)
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).eval()

    nchw = torch.randn(1, 3, size, size)
    with torch.no_grad():
        ref = model(nchw)  # native NCHW signature -> reference output

    # Channel-last I/O so the exported model takes NHWC (1,H,W,3): GPU-friendlier
    # and matches the interleaved-RGB input the Android sample feeds.
    clio = litert_torch.to_channel_last_io(model, args=[0])
    nhwc = nchw.permute(0, 2, 3, 1).contiguous()
    fp32 = os.path.join(out_dir, f"midas_small_{size}.tflite")
    litert_torch.convert(clio, (nhwc,)).export(fp32)
    print(f"[fp32] {fp32}  {os.path.getsize(fp32)/1e6:.1f} MB")
    return fp32, nhwc, ref


def quantize_fp16(fp32: str, out: str):
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping

    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT,
        ),
        algorithm_key=AlgorithmName.FLOAT_CASTING,
    )
    if os.path.exists(out):
        os.remove(out)
    qt = quantizer.Quantizer(float_model=fp32)
    qt.load_quantization_recipe(rm.get_quantization_recipe())
    qt.quantize().export_model(out)
    print(f"[fp16] {out}  {os.path.getsize(out)/1e6:.1f} MB")


def report(tflite: str, nhwc, ref):
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=tflite)
    interp.allocate_tensors()
    hist = {}
    for d in interp._get_ops_details():
        hist[d["op_name"]] = hist.get(d["op_name"], 0) + 1
    print("ops:", dict(sorted(hist.items(), key=lambda x: -x[1])))

    di, do = interp.get_input_details()[0], interp.get_output_details()[0]
    interp.set_tensor(di["index"], nhwc.numpy().astype(di["dtype"]))
    interp.invoke()
    out = interp.get_tensor(do["index"]).astype("float64").reshape(-1)
    r = ref.numpy().astype("float64").reshape(-1)
    n = min(len(out), len(r))
    corr = float(np.corrcoef(out[:n], r[:n])[0, 1])
    print(f"fidelity vs PyTorch: corr {corr:.7f}  max|diff| {np.max(np.abs(out[:n]-r[:n])):.2e}")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    fp32, nhwc, ref = convert(out_dir, size)
    print("== fp32 ==")
    report(fp32, nhwc, ref)
    fp16 = os.path.join(out_dir, f"midas_small_{size}_fp16.tflite")
    quantize_fp16(fp32, fp16)


if __name__ == "__main__":
    main()
