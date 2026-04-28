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

"""Evaluate a LiteRT vision model on ImageNet-1K validation."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import evaluate
import numpy as np
import transformers
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator
from datasets import load_dataset
from PIL import Image
from imagenet_preprocessing import (
    infer_input_size,
    pick_preprocess_config,
    preprocess_image,
)


_LITERT_TYPE_TO_NP = {
    1: np.float32,  # kLiteRtElementTypeFloat32
    9: np.int8,  # kLiteRtElementTypeInt8
    3: np.uint8,  # kLiteRtElementTypeUInt8
    2: np.int32,  # kLiteRtElementTypeInt32
}


def _default_model_path() -> str:
    return os.path.join(os.getcwd(), "mobilenet_v2.tflite")


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


class LiteRTImageClassifier:
    def __init__(self, model_path: str, labels: list[str], cpu_only: bool = False):
        self.task = "image-classification"
        accel = (
            HardwareAccelerator.CPU
            if cpu_only
            else (HardwareAccelerator.GPU | HardwareAccelerator.CPU)
        )
        self._model = CompiledModel.from_file(model_path, hardware_accel=accel)
        self._signature_index = 0
        self._channels = 3
        self._labels = labels
        self._input_height, self._input_width, self._channels_first = infer_input_size(
            self._model, self._signature_index
        )
        self._preprocess = pick_preprocess_config(
            model_path, self._input_height, self._input_width
        )
        self._input_buffers = self._model.create_input_buffers(self._signature_index)
        self._output_buffers = self._model.create_output_buffers(self._signature_index)
        self._output_requirements = self._model.get_output_buffer_requirements(
            0, self._signature_index
        )
        layout = "NCHW" if self._channels_first else "NHWC"
        print(
            "Model input:",
            f"{self._input_height}x{self._input_width}",
            layout,
            f"resize={self._preprocess['resize_size']}",
            f"crop={self._preprocess['crop_height']}x{self._preprocess['crop_width']}",
        )

    def __call__(self, inputs: Iterable[Image.Image]):
        outputs = []
        for image in inputs:
            array = preprocess_image(
                image,
                self._channels,
                self._preprocess["resize_size"],
                self._preprocess["crop_height"],
                self._preprocess["crop_width"],
                self._preprocess["mean"],
                self._preprocess["std"],
                self._preprocess["resample"],
                self._channels_first,
            )
            self._input_buffers[0].write(array)
            self._model.run_by_index(
                self._signature_index, self._input_buffers, self._output_buffers
            )
            output_array = _read_output(
                self._output_buffers[0], self._output_requirements
            )
            probs = _softmax(output_array.reshape(-1))
            topk = np.argsort(probs)[-5:][::-1]
            outputs.append(
                [
                    {
                        "label": self._labels[idx] if self._labels else str(idx),
                        "score": float(probs[idx]),
                    }
                    for idx in topk
                ]
            )
        return outputs


def _parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=_default_model_path())
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Evaluate at most this many samples (0 means full validation set).",
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Run evaluation on CPU only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not hasattr(transformers, "TFPreTrainedModel"):
        transformers.TFPreTrainedModel = type("_TFPreTrainedModel", (), {})
    dataset = load_dataset("imagenet-1k", split="validation", token=True)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    label_names = None
    try:
        label_names = dataset.features["label"].names
    except Exception:
        label_names = None
    label_mapping = None
    if label_names:
        label_mapping = {name: idx for idx, name in enumerate(label_names)}
    pipeline = LiteRTImageClassifier(args.model, label_names or [], cpu_only=args.cpu_only)
    evaluator = evaluate.evaluator("image-classification")
    results = evaluator.compute(
        model_or_pipeline=pipeline,
        data=dataset,
        metric="accuracy",
        label_mapping=label_mapping,
    )
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
