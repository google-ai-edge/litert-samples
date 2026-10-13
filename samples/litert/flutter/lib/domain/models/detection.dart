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

import 'dart:typed_data';

/// Where the detector runs. `gpu` is the default and must take the whole
/// graph; `cpu` is a chosen, labelled mode (Demo 3's Detector setting or
/// `DETECTOR_BACKEND=cpu`), never a fallback.
enum DetectorBackend {
  gpu,
  cpu;

  /// Parses `DETECTOR_BACKEND`; null for anything but `gpu` or `cpu`.
  static DetectorBackend? tryParse(String value) =>
      switch (value.trim().toLowerCase()) {
        'gpu' => gpu,
        'cpu' => cpu,
        _ => null,
      };
}

/// The plain-CPU path a loaded detector's output was checked against at
/// load.
enum VerifyReference {
  /// flutter_litert's `verifyCompiledModel`: a TFLite Interpreter with no
  /// delegate.
  interpreter('TFLite interpreter'),

  /// LiteRT's own CPU `CompiledModel`, used where the TFLite C library cannot
  /// load (Linux arm64, glibc < 2.38).
  liteRtCpu('LiteRT CPU');

  const VerifyReference(this.label);

  final String label;
}

/// What a loaded detector reports, once, after the load checks: the build on
/// exactly the requested backend, full acceleration on the GPU, the tensor
/// byte sizes, and the output against a plain-CPU reference.
final class const DetectorInfo({
  required final DetectorBackend backend,

  /// `CompiledModel.isFullyAccelerated`. Always true for [DetectorBackend.gpu]
  /// (a partial graph fails the load); usually false on the CPU.
  required final bool fullyAccelerated,

  /// Largest absolute and relative deviation from the plain-CPU reference
  /// ([verifyReference]).
  required final double verifyAbsolute,
  required final double verifyRelative,
  required final VerifyReference verifyReference,
  required final Duration createTime,
  required final Duration verifyTime,
  required final Duration firstRunTime,
}) {
  /// `GPU fp32 full` or `CPU (chosen)`; the overlay's and the setup row's
  /// backend label. The CPU is only ever a choice, never a fallback.
  String get label => switch (backend) {
    DetectorBackend.gpu => 'GPU fp32${fullyAccelerated ? ' full' : ''}',
    DetectorBackend.cpu => 'CPU (chosen)',
  };

  @override
  String toString() =>
      'DetectorInfo($label, verify=${verifyAbsolute.toStringAsExponential(2)} '
      '(${verifyRelative.toStringAsExponential(1)} of range) '
      'vs ${verifyReference.label}, '
      'create=${createTime.inMilliseconds}ms '
      'verify=${verifyTime.inMilliseconds}ms '
      'firstRun=${firstRunTime.inMilliseconds}ms)';
}

/// Values per box in [DetectionFrame.boxes].
const kBoxStride = 6;

/// One detected frame. Boxes are in *upright frame
/// pixels* (letterbox undone, rotation applied, clamped), sorted by score,
/// descending.
final class DetectionFrame({
  required final int frameId,

  /// Upright frame size: the painter's source size.
  required final int width,
  required final int height,

  /// `count × 6` floats: x1, y1, x2, y2, score, class id.
  required final Float32List boxes,

  /// Worker timings: gather into the input tensor, `CompiledModel.run`
  /// (including its I/O copies), top-k decode.
  required final int preMicros,
  required final int runMicros,
  required final int postMicros,
  required final DetectorBackend backend,
}) {
  int get count => boxes.length ~/ kBoxStride;

  double x1(int i) => boxes[i * kBoxStride];
  double y1(int i) => boxes[i * kBoxStride + 1];
  double x2(int i) => boxes[i * kBoxStride + 2];
  double y2(int i) => boxes[i * kBoxStride + 3];
  double score(int i) => boxes[i * kBoxStride + 4];
  int classId(int i) => boxes[i * kBoxStride + 5].toInt();

  /// Same boxes, bit for bit (the coexistence check).
  bool sameBoxes(DetectionFrame other) {
    if (other.boxes.length != boxes.length) return false;
    final a = boxes.buffer.asUint32List(boxes.offsetInBytes, boxes.length);
    final b = other.boxes.buffer.asUint32List(
      other.boxes.offsetInBytes,
      other.boxes.length,
    );
    for (var i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}
