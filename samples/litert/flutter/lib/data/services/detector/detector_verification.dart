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

// native.dart: CompiledModel / BackendVerification are native-only (see
// detector_engine.dart).
import 'package:flutter_litert/native.dart';

/// The `verifyCompiledModel` skip reason when its TFLite Interpreter cannot be
/// built (flutter_litert `native/backend_verification.dart`, `skip('reference
/// Interpreter failed: $e')`). On Linux that is the TFLite C library failing to
/// load: x86_64-only, and it needs glibc 2.38.
const kInterpreterUnavailablePrefix = 'reference Interpreter failed';

/// The deterministic ramp `verifyCompiledModel` feeds both backends: not
/// constant, and 251 is prime so the pattern does not align with strides.
Float32List verificationRamp(int length) {
  final input = Float32List(length);
  for (var i = 0; i < length; i++) {
    input[i] = (i % 251) / 251.0;
  }
  return input;
}

/// [compiled] against LiteRT's own CPU path on the same model and ramp.
///
/// The fallback reference where the TFLite Interpreter is unavailable. The CPU
/// `CompiledModel` runs XNNPACK kernels, so it is a CPU reference, not the
/// delegate-free one `verifyCompiledModel` builds; GPU corruption (tens of
/// percent of the range) is still far outside the 1% tolerance. Consumes one
/// inference on [compiled], like `verifyCompiledModel`.
BackendVerification verifyAgainstLiteRtCpu(
  Uint8List modelBytes,
  CompiledModel compiled, {
  required int inputFloats,
  double tolerance = kDefaultBackendTolerance,
}) {
  final input = verificationRamp(inputFloats);
  final reference = CompiledModel.fromBuffer(
    modelBytes,
    // The CPU reference at the detector's pinned settings, none of them left
    // to flutter_litert's defaults.
    accelerators: const {Accelerator.cpu},
    precision: Precision.fp32,
    tensorBufferMode: TensorBufferMode.managed,
  );
  final List<double> expected;
  try {
    expected = [
      for (final o in reference.run([input])) ...o,
    ];
  } finally {
    reference.close();
  }
  final List<double> actual;
  try {
    actual = [
      for (final o in compiled.run([input])) ...o,
    ];
  } catch (e) {
    return BackendVerification(
      agrees: false,
      absoluteDeviation: double.infinity,
      outputRange: outputRange(expected),
      relativeDeviation: double.infinity,
      error: e,
    );
  }
  return compareOutputs(expected, actual, tolerance: tolerance);
}

/// Largest deviation of [actual] from [expected], relative to the reference's
/// range; the same rules as `verifyCompiledModel` (NaN or infinity on one side
/// only, or a length mismatch, never agrees).
BackendVerification compareOutputs(
  List<double> expected,
  List<double> actual, {
  double tolerance = kDefaultBackendTolerance,
}) {
  final range = outputRange(expected);
  if (actual.length != expected.length) {
    return BackendVerification(
      agrees: false,
      absoluteDeviation: double.infinity,
      outputRange: range,
      relativeDeviation: double.infinity,
      skippedReason:
          'output length mismatch: CompiledModel produced ${actual.length} '
          'values, reference produced ${expected.length}',
    );
  }
  var deviation = 0.0;
  for (var i = 0; i < expected.length; i++) {
    final a = actual[i];
    final e = expected[i];
    if (a.isNaN != e.isNaN || a.isInfinite != e.isInfinite) {
      deviation = double.infinity;
      break;
    }
    if (a.isNaN || a.isInfinite) continue;
    final d = (a - e).abs();
    if (d > deviation) deviation = d;
  }
  // A constant reference has no range: scale by the values' magnitude instead.
  final scale = range > 0
      ? range
      : expected.fold<double>(
          0,
          (m, v) => v.isFinite && v.abs() > m ? v.abs() : m,
        );
  final relative = scale > 0
      ? deviation / scale
      : (deviation == 0 ? 0.0 : double.infinity);
  return BackendVerification(
    agrees: relative <= tolerance,
    absoluteDeviation: deviation,
    outputRange: range,
    relativeDeviation: relative,
  );
}

/// Max minus min over the finite values; 0 when there are none.
double outputRange(List<double> values) {
  var lo = double.infinity;
  var hi = double.negativeInfinity;
  for (final v in values) {
    if (!v.isFinite) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return hi >= lo ? hi - lo : 0;
}
