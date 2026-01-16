#!/usr/bin/env python3
# Copyright 2025 Google LLC.
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
from PIL import Image


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


# https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html
def _load_image(
    image_path: str,
    channels: int,
    resize_size: int,
    crop_height: int,
    crop_width: int,
    mean: np.ndarray,
    std: np.ndarray,
    resample: int,
) -> np.ndarray:
  # Torchvision ImageNet models assume this preprocessing (resize/crop plus
  # mean/std normalization from torchvision docs). If your model was converted
  # from a different training pipeline, update these constants and transforms or
  # you may get poor accuracy.
  if channels != 3:
    raise ValueError(f"Expected 3 channels, got {channels}")
  if resize_size <= 0 or crop_height <= 0 or crop_width <= 0:
    raise ValueError(
        f"Invalid resize/crop size: resize={resize_size}, crop={crop_height}x{crop_width}"
    )
  image = Image.open(image_path).convert("RGB")
  width, height = image.size
  if width < height:
    new_width = resize_size
    new_height = int(round(height * resize_size / width))
  else:
    new_height = resize_size
    new_width = int(round(width * resize_size / height))
  image = image.resize((new_width, new_height), resample)
  left = int(round((new_width - crop_width) / 2.0))
  top = int(round((new_height - crop_height) / 2.0))
  image = image.crop((left, top, left + crop_width, top + crop_height))
  array = np.asarray(image, dtype=np.int32)
  array = array.astype(np.float32) / 255.0
  array = (array - mean) / std
  array = np.transpose(array, (2, 0, 1))
  return array


def _infer_input_size(model, signature_index: int) -> tuple[int, int]:
  """Infer the model's input HxW for preprocessing from compiled metadata."""
  default_size = (224, 224)
  # Try to infer HxW from the compiled model input metadata; fall back to 224x224.
  try:
    requirements = model.get_input_buffer_requirements(0, signature_index)
  except Exception:
    return default_size
  dims = (
      requirements.get("dimensions")
      or requirements.get("shape")
      or requirements.get("dims")
  )
  if not dims:
    return default_size
  # Handle common NCHW/NHWC shapes, tolerating unknown batch or dim values.
  try:
    dims = [int(dim) for dim in dims]
  except Exception:
    return default_size
  if len(dims) == 4:
    if dims[1] == 3:
      return dims[2], dims[3]
    if dims[-1] == 3:
      return dims[1], dims[2]
    if dims[0] in (1, -1):
      dims = dims[1:]
  if len(dims) == 3:
    if dims[0] == 3:
      return dims[1], dims[2]
    if dims[-1] == 3:
      return dims[0], dims[1]
  return default_size


def _pick_preprocess_config(
    model_path: str, input_height: int, input_width: int
) -> dict:
  model_name = os.path.basename(model_path).lower()
  if "efficientnet_v2_s" in model_name:
    return {
        "resize_size": 384,
        "crop_height": 384,
        "crop_width": 384,
        "mean": np.array([0.485, 0.456, 0.406], dtype=np.float32),
        "std": np.array([0.229, 0.224, 0.225], dtype=np.float32),
        "resample": Image.BILINEAR,
    }
  if "efficientnet_v2_m" in model_name:
    return {
        "resize_size": 480,
        "crop_height": 480,
        "crop_width": 480,
        "mean": np.array([0.485, 0.456, 0.406], dtype=np.float32),
        "std": np.array([0.229, 0.224, 0.225], dtype=np.float32),
        "resample": Image.BILINEAR,
    }
  if "efficientnet_v2_l" in model_name:
    return {
        "resize_size": 480,
        "crop_height": 480,
        "crop_width": 480,
        "mean": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "std": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "resample": Image.BICUBIC,
    }
  if "efficientnet_b" in model_name:
    return {
        "resize_size": 600,
        "crop_height": 600,
        "crop_width": 600,
        "mean": np.array([0.485, 0.456, 0.406], dtype=np.float32),
        "std": np.array([0.229, 0.224, 0.225], dtype=np.float32),
        "resample": Image.BICUBIC,
    }
  crop_height = input_height if input_height > 0 else 224
  crop_width = input_width if input_width > 0 else 224
  resize_size = int(round(max(crop_height, crop_width) / 0.875))
  return {
      "resize_size": resize_size,
      "crop_height": crop_height,
      "crop_width": crop_width,
      "mean": np.array([0.485, 0.456, 0.406], dtype=np.float32),
      "std": np.array([0.229, 0.224, 0.225], dtype=np.float32),
      "resample": Image.BILINEAR,
  }


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
          "mobilenet_v2",
          "resnet18",
          "resnet34",
          "resnet50",
          "resnet101",
          "resnet152",
      ),
      default="mobilenet_v2",
      help="Torchvision model architecture.",
  )
  parser.add_argument(
      "--output",
      default=None,
      help="Path to output .tflite file.",
  )
  return parser.parse_args(argv)


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
          torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1,
      ),
      "efficientnet_b1": (
          torchvision.models.efficientnet_b1,
          torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1,
      ),
      "efficientnet_b2": (
          torchvision.models.efficientnet_b2,
          torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1,
      ),
      "efficientnet_b3": (
          torchvision.models.efficientnet_b3,
          torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1,
      ),
      "efficientnet_b4": (
          torchvision.models.efficientnet_b4,
          torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1,
      ),
      "efficientnet_b5": (
          torchvision.models.efficientnet_b5,
          torchvision.models.EfficientNet_B5_Weights.IMAGENET1K_V1,
      ),
      "efficientnet_b6": (
          torchvision.models.efficientnet_b6,
          torchvision.models.EfficientNet_B6_Weights.IMAGENET1K_V1,
      ),
      "efficientnet_b7": (
          torchvision.models.efficientnet_b7,
          torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1,
      ),
      "efficientnet_v2_s": (
          torchvision.models.efficientnet_v2_s,
          torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
      ),
      "efficientnet_v2_m": (
          torchvision.models.efficientnet_v2_m,
          torchvision.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1,
      ),
      "efficientnet_v2_l": (
          torchvision.models.efficientnet_v2_l,
          torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1,
      ),
      "mobilenet_v2": (
          torchvision.models.mobilenet_v2,
          torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1,
      ),
      "resnet18": (
          torchvision.models.resnet18,
          torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
      ),
      "resnet34": (
          torchvision.models.resnet34,
          torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
      ),
      "resnet50": (
          torchvision.models.resnet50,
          torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
      ),
      "resnet101": (
          torchvision.models.resnet101,
          torchvision.models.ResNet101_Weights.IMAGENET1K_V1,
      ),
      "resnet152": (
          torchvision.models.resnet152,
          torchvision.models.ResNet152_Weights.IMAGENET1K_V1,
      ),
  }
  if arch not in model_specs:
    raise ValueError(f"Unsupported architecture: {arch}")
  model_fn, weights = model_specs[arch]
  model = model_fn(weights).eval()
  input_height, input_width = _infer_input_size_from_weights(weights)
  return model, torch, input_height, input_width


def _convert_pretrained(argv: list[str]) -> int:
  args = _parse_convert_args(argv)
  if not args.output:
    args.output = os.path.join(os.getcwd(), f"{args.arch}.tflite")
  import ai_edge_torch  # pylint: disable=import-outside-toplevel

  model, torch, input_height, input_width = _init_torchvision_model(args.arch)
  sample_inputs = (torch.randn(1, 3, input_height, input_width),)
  edge_model = ai_edge_torch.convert(model, sample_inputs)
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

  input_height, input_width = _infer_input_size(model, signature_index)
  preprocess = _pick_preprocess_config(args.model, input_height, input_width)
  image_array = _load_image(
      args.image,
      channels,
      preprocess["resize_size"],
      preprocess["crop_height"],
      preprocess["crop_width"],
      preprocess["mean"],
      preprocess["std"],
      preprocess["resample"],
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
