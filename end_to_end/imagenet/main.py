#!/usr/bin/env python3
# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs MobileNet classification with LiteRT CompiledModel API.

Also supports converting a pretrained torchvision model to LiteRT (TFLite).
"""

import argparse
import os
import sys
import numpy as np
from ai_edge_litert.compiled_model import CompiledModel
from imagenet_preprocessing import infer_input_size, load_image, pick_preprocess_config


_LITERT_TYPE_TO_NP = {
    1: np.float32,  # kLiteRtElementTypeFloat32
    9: np.int8,  # kLiteRtElementTypeInt8
    3: np.uint8,  # kLiteRtElementTypeUInt8
    2: np.int32,  # kLiteRtElementTypeInt32
}


def _default_model_path() -> str:
  return os.path.join(os.getcwd(), "mobilenet_v2.tflite")


def _default_label_path(filename: str) -> str:
  return os.path.join(os.getcwd(), filename)


def _pick_output_dtype(requirements: dict) -> np.dtype:
  supported = requirements.get("supported_types", [])
  for type_id in (1, 9, 3, 2):
    if type_id in supported:
      return _LITERT_TYPE_TO_NP[type_id]
  if supported:
    return _LITERT_TYPE_TO_NP.get(supported[0], np.float32)
  return np.float32


def _read_output(buffer, requirements: dict) -> np.ndarray:
  output_dtype = _pick_output_dtype(requirements)
  buffer_size = requirements.get("buffer_size", 0)
  itemsize = np.dtype(output_dtype).itemsize
  num_elements = buffer_size // itemsize if itemsize else buffer_size
  if num_elements == 0:
    raise ValueError("Output buffer size is zero")
  return buffer.read(num_elements, output_dtype)


def _softmax(scores: np.ndarray) -> np.ndarray:
  scores = scores.astype(np.float32, copy=False)
  max_score = np.max(scores)
  exp_scores = np.exp(scores - max_score)
  return exp_scores / np.sum(exp_scores)


def _load_synsets(synsets_path: str):
  if not synsets_path:
    return None
  with open(synsets_path, "r", encoding="utf-8") as f:
    return [line.strip() for line in f if line.strip()]


def _load_metadata(metadata_path: str):
  if not metadata_path:
    return {}
  mapping = {}
  with open(metadata_path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      synset, _, label = line.partition("\t")
      if synset and label:
        mapping[synset] = label
  return mapping


def _build_imagenet_labels(synsets_path: str, metadata_path: str, output_size: int):
  synsets = _load_synsets(synsets_path)
  if not synsets:
    return None
  if output_size == len(synsets) + 1:
    synsets = ["background"] + synsets
  if output_size != len(synsets):
    return None
  metadata = _load_metadata(metadata_path)
  labels = []
  for synset in synsets:
    label = metadata.get(synset, synset)
    labels.append(f"{synset} {label}" if label != synset else synset)
  return labels


def _pick_labels_for_output(output_size: int, args):
  labels = _build_imagenet_labels(args.labels, args.metadata, output_size)
  if labels is None:
    print(
        f"Warning: label file does not match output size {output_size}. "
        "Falling back to class indices.",
        file=sys.stderr,
    )
  return labels


def _print_topk(scores: np.ndarray, labels, top_k: int):
  flat = scores.reshape(-1)
  top_k = min(top_k, flat.size)
  if top_k == 1:
    indices = [int(np.argmax(flat))]
  else:
    indices = np.argsort(flat)[-top_k:][::-1]
  for rank, idx in enumerate(indices, start=1):
    label = labels[idx] if labels and idx < len(labels) else f"class_{idx}"
    print(f"{rank}: {label} ({flat[idx]:.6f})")


def _parse_classify_args(argv: list[str]):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", default=_default_model_path())
  parser.add_argument("--image", required=True)
  parser.add_argument("--labels", default=_default_label_path("imagenet_lsvrc_2015_synsets.txt"))
  parser.add_argument("--metadata", default=_default_label_path("imagenet_metadata.txt"))
  parser.add_argument("--top_k", type=int, default=5)
  return parser.parse_args(argv)


def _parse_convert_args(argv: list[str]):
  parser = argparse.ArgumentParser(
      description="Convert a pretrained torchvision model to LiteRT (TFLite)."
  )
  parser.add_argument(
      "--arch",
      choices=(
          "efficientnet_b0",
          "efficientnet_b1",
          "efficientnet_b2",
          "efficientnet_b3",
          "efficientnet_b4",
          "efficientnet_b5",
          "efficientnet_b6",
          "efficientnet_b7",
          "efficientnet_v2_s",
          "efficientnet_v2_m",
          "efficientnet_v2_l",
          "alexnet",
          "convnext_tiny",
          "convnext_small",
          "convnext_base",
          "convnext_large",
          "vgg11",
          "vgg11_bn",
          "vgg13",
          "vgg13_bn",
          "vgg16",
          "vgg16_bn",
          "vgg19",
          "vgg19_bn",
          "mobilenet_v2",
          "mobilenet_v3_large",
          "mobilenet_v3_small",
          "resnet18",
          "resnet34",
          "resnet50",
          "resnet101",
          "resnet152",
          "shufflenet_v2_x0_5",
          "shufflenet_v2_x1_0",
          "shufflenet_v2_x1_5",
          "shufflenet_v2_x2_0",
          "squeezenet1_0",
          "squeezenet1_1",
          "inception_v3",
      ),
      default="mobilenet_v2",
      help="Torchvision model architecture.",
  )
  parser.add_argument(
      "--output",
      default=None,
      help="Path to output .tflite file.",
  )
  parser.add_argument(
      "--quantize",
      action="store_true",
      help="Enable dynamic weight int8 quantization (activations float32).",
  )
  return parser.parse_args(argv)


# Used in the convert path
def _infer_input_size_from_weights(weights) -> tuple[int, int]:
  """Infer input HxW from torchvision weights metadata."""
  default_size = (224, 224)
  if weights is None:
    return default_size
  try:
    transforms = weights.transforms()
  except Exception:
    return default_size
  crop_size = getattr(transforms, "crop_size", None)
  if crop_size is None:
    return default_size
  if isinstance(crop_size, (list, tuple)):
    if len(crop_size) >= 2:
      return int(crop_size[0]), int(crop_size[1])
    if len(crop_size) == 1:
      size = int(crop_size[0])
      return size, size
  try:
    size = int(crop_size)
  except Exception:
    return default_size
  return size, size


def _init_torchvision_model(arch: str):
  import torch  # pylint: disable=import-outside-toplevel
  import torchvision  # pylint: disable=import-outside-toplevel
  model_specs = {
      "efficientnet_b0": (
          torchvision.models.efficientnet_b0,
          torchvision.models.EfficientNet_B0_Weights.DEFAULT,
      ),
      "efficientnet_b1": (
          torchvision.models.efficientnet_b1,
          torchvision.models.EfficientNet_B1_Weights.DEFAULT,
      ),
      "efficientnet_b2": (
          torchvision.models.efficientnet_b2,
          torchvision.models.EfficientNet_B2_Weights.DEFAULT,
      ),
      "efficientnet_b3": (
          torchvision.models.efficientnet_b3,
          torchvision.models.EfficientNet_B3_Weights.DEFAULT,
      ),
      "efficientnet_b4": (
          torchvision.models.efficientnet_b4,
          torchvision.models.EfficientNet_B4_Weights.DEFAULT,
      ),
      "efficientnet_b5": (
          torchvision.models.efficientnet_b5,
          torchvision.models.EfficientNet_B5_Weights.DEFAULT,
      ),
      "efficientnet_b6": (
          torchvision.models.efficientnet_b6,
          torchvision.models.EfficientNet_B6_Weights.DEFAULT,
      ),
      "efficientnet_b7": (
          torchvision.models.efficientnet_b7,
          torchvision.models.EfficientNet_B7_Weights.DEFAULT,
      ),
      "efficientnet_v2_s": (
          torchvision.models.efficientnet_v2_s,
          torchvision.models.EfficientNet_V2_S_Weights.DEFAULT,
      ),
      "efficientnet_v2_m": (
          torchvision.models.efficientnet_v2_m,
          torchvision.models.EfficientNet_V2_M_Weights.DEFAULT,
      ),
      "efficientnet_v2_l": (
          torchvision.models.efficientnet_v2_l,
          torchvision.models.EfficientNet_V2_L_Weights.DEFAULT,
      ),
      "alexnet": (
          torchvision.models.alexnet,
          torchvision.models.AlexNet_Weights.DEFAULT,
      ),
      "convnext_tiny": (
          torchvision.models.convnext_tiny,
          torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT,
      ),
      "convnext_small": (
          torchvision.models.convnext_small,
          torchvision.models.ConvNeXt_Small_Weights.DEFAULT,
      ),
      "convnext_base": (
          torchvision.models.convnext_base,
          torchvision.models.ConvNeXt_Base_Weights.DEFAULT,
      ),
      "convnext_large": (
          torchvision.models.convnext_large,
          torchvision.models.ConvNeXt_Large_Weights.DEFAULT,
      ),
      "vgg11": (
          torchvision.models.vgg11,
          torchvision.models.VGG11_Weights.DEFAULT,
      ),
      "vgg11_bn": (
          torchvision.models.vgg11_bn,
          torchvision.models.VGG11_BN_Weights.DEFAULT,
      ),
      "vgg13": (
          torchvision.models.vgg13,
          torchvision.models.VGG13_Weights.DEFAULT,
      ),
      "vgg13_bn": (
          torchvision.models.vgg13_bn,
          torchvision.models.VGG13_BN_Weights.DEFAULT,
      ),
      "vgg16": (
          torchvision.models.vgg16,
          torchvision.models.VGG16_Weights.DEFAULT,
      ),
      "vgg16_bn": (
          torchvision.models.vgg16_bn,
          torchvision.models.VGG16_BN_Weights.DEFAULT,
      ),
      "vgg19": (
          torchvision.models.vgg19,
          torchvision.models.VGG19_Weights.DEFAULT,
      ),
      "vgg19_bn": (
          torchvision.models.vgg19_bn,
          torchvision.models.VGG19_BN_Weights.DEFAULT,
      ),
      "mobilenet_v2": (
          torchvision.models.mobilenet_v2,
          torchvision.models.MobileNet_V2_Weights.DEFAULT,
      ),
      "mobilenet_v3_large": (
          torchvision.models.mobilenet_v3_large,
          torchvision.models.MobileNet_V3_Large_Weights.DEFAULT,
      ),
      "mobilenet_v3_small": (
          torchvision.models.mobilenet_v3_small,
          torchvision.models.MobileNet_V3_Small_Weights.DEFAULT,
      ),
      "resnet18": (
          torchvision.models.resnet18,
          torchvision.models.ResNet18_Weights.DEFAULT,
      ),
      "resnet34": (
          torchvision.models.resnet34,
          torchvision.models.ResNet34_Weights.DEFAULT,
      ),
      "resnet50": (
          torchvision.models.resnet50,
          torchvision.models.ResNet50_Weights.DEFAULT,
      ),
      "resnet101": (
          torchvision.models.resnet101,
          torchvision.models.ResNet101_Weights.DEFAULT,
      ),
      "resnet152": (
          torchvision.models.resnet152,
          torchvision.models.ResNet152_Weights.DEFAULT,
      ),
      "shufflenet_v2_x0_5": (
          torchvision.models.shufflenet_v2_x0_5,
          torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT,
      ),
      "shufflenet_v2_x1_0": (
          torchvision.models.shufflenet_v2_x1_0,
          torchvision.models.ShuffleNet_V2_X1_0_Weights.DEFAULT,
      ),
      "shufflenet_v2_x1_5": (
          torchvision.models.shufflenet_v2_x1_5,
          torchvision.models.ShuffleNet_V2_X1_5_Weights.DEFAULT,
      ),
      "shufflenet_v2_x2_0": (
          torchvision.models.shufflenet_v2_x2_0,
          torchvision.models.ShuffleNet_V2_X2_0_Weights.DEFAULT,
      ),
      "squeezenet1_0": (
          torchvision.models.squeezenet1_0,
          torchvision.models.SqueezeNet1_0_Weights.DEFAULT,
      ),
      "squeezenet1_1": (
          torchvision.models.squeezenet1_1,
          torchvision.models.SqueezeNet1_1_Weights.DEFAULT,
      ),
      "inception_v3": (
          torchvision.models.inception_v3,
          torchvision.models.Inception_V3_Weights.DEFAULT,
      ),
  }
  if arch not in model_specs:
    raise ValueError(f"Unsupported model architecture: {arch}")
  model_fn, weights = model_specs[arch]
  model = model_fn(weights).eval()
  input_height, input_width = _infer_input_size_from_weights(weights)
  return model, torch, input_height, input_width


def _convert_pretrained(argv: list[str]) -> int:
  args = _parse_convert_args(argv)
  if not args.output:
    args.output = os.path.join(os.getcwd(), f"{args.arch}.tflite")
  import litert_torch  # pylint: disable=import-outside-toplevel
  if args.quantize:
    from ai_edge_quantizer import quantizer, recipe  # pylint: disable=import-outside-toplevel

  model, torch, input_height, input_width = _init_torchvision_model(args.arch)
  model.eval()
  sample_inputs = (torch.randn(1, 3, input_height, input_width),)
  edge_model = litert_torch.convert(model, sample_inputs)
  if args.quantize:
    import tempfile  # pylint: disable=import-outside-toplevel
    tmp_path = None
    try:
      with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as tmp:
        tmp_path = tmp.name

      edge_model.export(tmp_path)
      qt = quantizer.Quantizer(tmp_path)
      qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
      qt.quantize().export_model(args.output)
      print(f"Saved quantized LiteRT model to: {args.output}")
    finally:
      if tmp_path:
        try:
          os.remove(tmp_path)
        except FileNotFoundError:
          print(f"Warning: temp file already deleted: {tmp_path}", file=sys.stderr)
  else:
    edge_model.export(args.output)
    print(f"Saved LiteRT model to: {args.output}")
  return 0


def _classify(argv: list[str]) -> int:
  args = _parse_classify_args(argv)

  if not os.path.exists(args.model):
    raise FileNotFoundError(f"Model not found: {args.model}")
  if not os.path.exists(args.image):
    raise FileNotFoundError(f"Image not found: {args.image}")

  model = CompiledModel.from_file(args.model)
  signature_index = 0

  channels = 3

  input_height, input_width, channels_first = infer_input_size(model, signature_index)
  preprocess = pick_preprocess_config(args.model, input_height, input_width)
  layout = "NCHW" if channels_first else "NHWC"
  print(
      "Model input:",
      f"{input_height}x{input_width}",
      layout,
      f"resize={preprocess['resize_size']}",
      f"crop={preprocess['crop_height']}x{preprocess['crop_width']}",
  )
  image_array = load_image(
      args.image,
      channels,
      preprocess["resize_size"],
      preprocess["crop_height"],
      preprocess["crop_width"],
      preprocess["mean"],
      preprocess["std"],
      preprocess["resample"],
      channels_first,
  )
  input_buffers = model.create_input_buffers(signature_index)
  output_buffers = model.create_output_buffers(signature_index)
  input_buffers[0].write(image_array)

  model.run_by_index(signature_index, input_buffers, output_buffers)

  output_requirements = model.get_output_buffer_requirements(0, signature_index)
  output_array = _read_output(output_buffers[0], output_requirements)
  output_probs = _softmax(output_array.reshape(-1))
  labels = _pick_labels_for_output(output_array.size, args)
  _print_topk(output_probs, labels, args.top_k)
  return 0


def main(argv: list[str]) -> int:
  if argv and argv[0] == "convert":
    return _convert_pretrained(argv[1:])
  return _classify(argv)


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
