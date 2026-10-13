// Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// YOLO26n raw-head contract: input, output, decode and the model file.
/// Every detector threshold lives here.
library;

/// Input `[1, 3, 640, 640]` NCHW float32, RGB, value / 255, letterboxed.
const kDetInput = 640;

/// Output `[1, 8400, 84]`: per anchor x1, y1, x2, y2 (640 input px), then 80
/// sigmoid class scores.
const kDetAnchors = 8400;
const kDetClasses = 80;
const kDetRawStride = 4 + kDetClasses;

/// Letterbox pad, Ultralytics grey 114.
const kDetPadByte = 114;

/// Byte sizes the compiled model must report at load: another file (Arm's
/// original, say) fails there.
const kDetInputBytes = 3 * kDetInput * kDetInput * 4;
const kDetOutputBytes = kDetAnchors * kDetRawStride * 4;

/// Decode floor (the manifest's confidence threshold).
const kDetDecodeScore = 0.25;

/// Boxes drawn on the live view.
const kDetDisplayScore = 0.35;

/// Detections kept per frame after the top-k sort; the painter draws at most
/// `kMaxPaintedBoxes` of them.
const kDetMaxDet = 100;

/// The derived file `yolo26n_fp16_rawhead.tflite`: Arm's conversion with its
/// head cut at the raw `[1, 8400, 84]` output (`tool/prune_yolo26n_head.py`).
const kDetModelName = 'yolo26n_fp16_rawhead';
const kDetModelBytes = 10361332;
const kDetModelSha256 =
    '5ddd5eebad18587d56500a30b0995c07c9e1a241640750f568e4a974f66ed80a';

/// The detector built into the app: a Flutter asset, loaded from memory with
/// `CompiledModel.fromBuffer`.
const kDetModelAsset = 'assets/models/$kDetModelName.tflite';

/// Arm's original `yolo26n_conv2d_f16_weights.tflite`: its in-graph
/// TopK/GatherND head never runs on the GPU.
const kDetArmOriginalBytes = 10363712;
