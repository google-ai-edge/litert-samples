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

"""Shared ImageNet preprocessing helpers for LiteRT vision eval/classify."""

from __future__ import annotations

import os

import numpy as np
from PIL import Image


def infer_input_size(model, signature_index: int) -> tuple[int, int, bool]:
    """Infer model input HxW and layout (channels_first) for preprocessing."""
    default_size = (224, 224)
    default_channels_first = True
    try:
        signature = model.get_signature_by_index(signature_index)
        signature_key = signature.get("key") if isinstance(signature, dict) else None
        if signature_key:
            details = model.get_input_tensor_details(signature_key)
            if isinstance(details, dict) and details:
                first = next(iter(details.values()))
                shape = first.get("shape") if isinstance(first, dict) else None
                if shape and len(shape) >= 3:
                    if len(shape) == 4:
                        if shape[1] == 3:
                            return int(shape[2]), int(shape[3]), True
                        if shape[-1] == 3:
                            return int(shape[1]), int(shape[2]), False
                    if len(shape) == 3:
                        if shape[0] == 3:
                            return int(shape[1]), int(shape[2]), True
                        if shape[-1] == 3:
                            return int(shape[0]), int(shape[1]), False
    except Exception:
        pass
    try:
        requirements = model.get_input_buffer_requirements(0, signature_index)
    except Exception:
        return default_size[0], default_size[1], default_channels_first
    dims = (
        requirements.get("dimensions")
        or requirements.get("shape")
        or requirements.get("dims")
    )
    if not dims:
        return default_size[0], default_size[1], default_channels_first
    try:
        dims = [int(dim) for dim in dims]
    except Exception:
        return default_size[0], default_size[1], default_channels_first

    if len(dims) == 4:
        if dims[1] == 3:
            return dims[2], dims[3], True
        if dims[-1] == 3:
            return dims[1], dims[2], False
        if dims[0] in (1, -1):
            dims = dims[1:]

    if len(dims) == 3:
        if dims[0] == 3:
            return dims[1], dims[2], True
        if dims[-1] == 3:
            return dims[0], dims[1], False

    return default_size[0], default_size[1], default_channels_first


def pick_preprocess_config(model_path: str, input_height: int, input_width: int) -> dict:
    def _fit_to_model_input(
        resize_size: int, crop_height: int, crop_width: int, resample: int
    ) -> dict:
        ch = input_height if input_height > 0 else crop_height
        cw = input_width if input_width > 0 else crop_width
        if ch != crop_height or cw != crop_width:
            resize_size = int(round(max(ch, cw) / 0.875))
            crop_height = ch
            crop_width = cw
        return {
            "resize_size": resize_size,
            "crop_height": crop_height,
            "crop_width": crop_width,
            "mean": np.array([0.485, 0.456, 0.406], dtype=np.float32),
            "std": np.array([0.229, 0.224, 0.225], dtype=np.float32),
            "resample": resample,
        }

    model_name = os.path.basename(model_path).lower()
    efficientnet_b_cfg = {
        "efficientnet_b0": (256, 224, 224, Image.BICUBIC),
        "efficientnet_b1": (255, 240, 240, Image.BILINEAR),
        "efficientnet_b2": (288, 288, 288, Image.BICUBIC),
        "efficientnet_b3": (320, 300, 300, Image.BICUBIC),
        "efficientnet_b4": (384, 380, 380, Image.BICUBIC),
        "efficientnet_b5": (456, 456, 456, Image.BICUBIC),
        "efficientnet_b6": (528, 528, 528, Image.BICUBIC),
        "efficientnet_b7": (600, 600, 600, Image.BICUBIC),
    }
    for arch, (resize, crop_h, crop_w, resample) in efficientnet_b_cfg.items():
        if arch in model_name:
            return _fit_to_model_input(resize, crop_h, crop_w, resample)

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


def preprocess_image(
    image: Image.Image,
    channels: int,
    resize_size: int,
    crop_height: int,
    crop_width: int,
    mean: np.ndarray,
    std: np.ndarray,
    resample: int,
    channels_first: bool,
) -> np.ndarray:
    if channels != 3:
        raise ValueError(f"Expected 3 channels, got {channels}")
    if resize_size <= 0 or crop_height <= 0 or crop_width <= 0:
        raise ValueError(
            f"Invalid resize/crop size: resize={resize_size}, crop={crop_height}x{crop_width}"
        )
    image = image.convert("RGB")
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
    if channels_first:
        return np.transpose(array, (2, 0, 1))
    return array


def load_image(
    image_path: str,
    channels: int,
    resize_size: int,
    crop_height: int,
    crop_width: int,
    mean: np.ndarray,
    std: np.ndarray,
    resample: int,
    channels_first: bool,
) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return preprocess_image(
        image=image,
        channels=channels,
        resize_size=resize_size,
        crop_height=crop_height,
        crop_width=crop_width,
        mean=mean,
        std=std,
        resample=resample,
        channels_first=channels_first,
    )
