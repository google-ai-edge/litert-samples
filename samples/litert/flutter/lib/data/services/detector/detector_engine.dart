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

import 'package:flutter/foundation.dart';
// native.dart: verifyCompiledModel / BackendVerification are native-only, and
// the analyzer resolves flutter_litert.dart's conditional export to the web
// surface.
import 'package:flutter_litert/native.dart';

import '../../../domain/models/detection.dart';
import '../../../domain/models/detector_spec.dart';
import '../../../domain/models/scene_snapshot.dart';
import 'detector_codec.dart';
import 'detector_verification.dart';
import 'frame_message.dart';

/// Builds and verifies the `CompiledModel`. A const instance is sent to the
/// detector worker and called there, so implementations hold no state.
abstract interface class DetectorRuntime {
  /// `CompiledModel.fromBuffer` with exactly [accelerators] (never a policy
  /// with a CPU retry) and fp32.
  CompiledModel create(Uint8List modelBytes, Set<Accelerator> accelerators);

  /// One run on a ramp input against a plain-CPU reference, and which one.
  DetectorVerification verify(Uint8List modelBytes, CompiledModel model);
}

/// A [BackendVerification] and the CPU path it compared against.
typedef DetectorVerification = ({
  BackendVerification result,
  VerifyReference reference,
});

/// The real runtime: `flutter_litert` 3.9.3.
final class LiteRtDetectorRuntime implements DetectorRuntime {
  const LiteRtDetectorRuntime();

  @override
  CompiledModel create(
    Uint8List modelBytes,
    Set<Accelerator> accelerators,
  ) => CompiledModel.fromBuffer(
    modelBytes,
    accelerators: accelerators,
    // Pinned, not left to flutter_litert's defaults: fp16 fails verification.
    precision: Precision.fp32,
    tensorBufferMode: TensorBufferMode.managed,
  );

  /// `verifyCompiledModel` (TFLite Interpreter reference); where the TFLite C
  /// library cannot load, LiteRT's CPU path instead. Never "unverified".
  @override
  DetectorVerification verify(Uint8List modelBytes, CompiledModel model) {
    final viaInterpreter = verifyCompiledModel(modelBytes, model);
    final reason = viaInterpreter.skippedReason;
    if (reason == null || !reason.startsWith(kInterpreterUnavailablePrefix)) {
      return (result: viaInterpreter, reference: VerifyReference.interpreter);
    }
    debugPrint(
      '[Detector] TFLite interpreter unavailable ($reason); verifying '
      'against LiteRT CPU',
    );
    return (
      result: verifyAgainstLiteRtCpu(
        modelBytes,
        model,
        inputFloats: kDetInputBytes ~/ 4,
      ),
      reference: VerifyReference.liteRtCpu,
    );
  }
}

/// How the load errors about the model file itself begin: missing, not the
/// raw-head file, wrong I/O. No other backend can load such a file.
const kDetectorFileNotFound = 'Detector model file not found';
const kDetectorNotRawHead = 'Not the YOLO26n raw-head file';
const kDetectorWrongIo = 'Wrong model I/O';

/// Whether a detector load error ([message]) is about the file, not about
/// the backend it was loaded on.
bool isDetectorFileProblem(String message) =>
    message.startsWith(kDetectorFileNotFound) ||
    message.startsWith(kDetectorNotRawHead) ||
    message.startsWith(kDetectorWrongIo);

/// The detector could not be loaded; [message] is shown as-is.
final class DetectorLoadException implements Exception {
  const DetectorLoadException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// The loaded detector: owns the `CompiledModel` and the reused input tensor.
/// Synchronous; lives in the worker isolate (tests drive it directly).
final class DetectorEngine {
  DetectorEngine._(this._model, this.info, this._backend, this._input);

  /// Loads [modelBytes] with these checks, in order: size, a strict build on
  /// the requested accelerator, the whole graph accelerated (GPU only), the I/O
  /// sizes, agreement with a plain-CPU reference, a warm-up run. Any failure
  /// closes what was built and throws [DetectorLoadException]; there is no
  /// retry on another backend.
  static DetectorEngine load({
    required Uint8List modelBytes,
    required DetectorBackend backend,
    required DetectorRuntime runtime,
    int expectedBytes = kDetModelBytes,
    void Function(String message)? log,
  }) {
    log ??= debugPrint;
    // 0. Identity: the raw-head file has a known size.
    if (modelBytes.length != expectedBytes) {
      throw DetectorLoadException(
        '$kDetectorNotRawHead: it has ${modelBytes.length} bytes, '
        'expected $expectedBytes ($kDetModelName.tflite, derived by '
        'tool/prune_yolo26n_head.py).'
        '${modelBytes.length == kDetArmOriginalBytes ? " This is Arm's "
                  'original yolo26n_conv2d_f16_weights.tflite: its in-graph '
                  'TopK/GatherND head cannot run on the GPU.' : ''}',
      );
    }

    // 1. Build with exactly the requested accelerator.
    final accelerators = switch (backend) {
      DetectorBackend.gpu => const {Accelerator.gpu},
      DetectorBackend.cpu => const {Accelerator.cpu},
    };
    final createWatch = Stopwatch()..start();
    final CompiledModel model;
    try {
      model = runtime.create(modelBytes, accelerators);
    } catch (e) {
      throw DetectorLoadException(switch (backend) {
        DetectorBackend.gpu =>
          'GPU rejected YOLO26n: $e. Run the detector on the CPU instead '
              "(Live camera's settings, or DETECTOR_BACKEND=cpu)",
        DetectorBackend.cpu => 'The CPU could not build YOLO26n: $e',
      });
    }
    final createTime = createWatch.elapsed;

    try {
      // 2. Whole graph on the GPU, nothing narrowed or retried.
      if (backend == DetectorBackend.gpu) {
        if (!setEquals(model.accelerators, accelerators) || model.didFallback) {
          throw DetectorLoadException(
            'YOLO26n compiled with ${model.accelerators} '
            '(didFallback=${model.didFallback}) instead of strict GPU',
          );
        }
        if (!model.isFullyAccelerated) {
          throw const DetectorLoadException(
            'YOLO26n is only partly on the GPU (the rest would run on the CPU '
            'at CPU speed). No CPU fallback: check the model file and the GPU '
            "runtime, or choose the CPU (Live camera's settings, or "
            'DETECTOR_BACKEND=cpu).',
          );
        }
      }

      // 3. The I/O contract: [1,3,640,640] in, [1,8400,84] out.
      if (!listEquals(model.inputByteSizes, const [kDetInputBytes]) ||
          !listEquals(model.outputByteSizes, const [kDetOutputBytes])) {
        throw DetectorLoadException(
          '$kDetectorWrongIo: input bytes ${model.inputByteSizes}, output bytes '
          '${model.outputByteSizes}; expected [$kDetInputBytes] and '
          '[$kDetOutputBytes] (is this the raw-head file?)',
        );
      }

      // 4. Same numbers as a plain-CPU interpreter.
      final verifyWatch = Stopwatch()..start();
      final (result: verification, :reference) = runtime.verify(
        modelBytes,
        model,
      );
      final verifyTime = verifyWatch.elapsed;
      if (!verification.agrees) {
        throw DetectorLoadException(
          'YOLO26n on ${backend.name} disagrees with the CPU reference '
          '(${reference.label}): $verification',
        );
      }
      if (backend == DetectorBackend.gpu &&
          verification.absoluteDeviation == 0.0) {
        log(
          '[Detector] warning: GPU output is bit-identical to the CPU '
          'reference; the GPU may not have run',
        );
      }

      // 5. Warm-up on the all-pad frame.
      final input = Float32List(3 * kDetInput * kDetInput)
        ..fillRange(0, 3 * kDetInput * kDetInput, kDetPadUnit);
      final warmWatch = Stopwatch()..start();
      model.run([input]);
      final info = DetectorInfo(
        backend: backend,
        fullyAccelerated: model.isFullyAccelerated,
        verifyAbsolute: verification.absoluteDeviation,
        verifyRelative: verification.relativeDeviation,
        verifyReference: reference,
        createTime: createTime,
        verifyTime: verifyTime,
        firstRunTime: warmWatch.elapsed,
      );
      log('[Detector] loaded $info');
      return DetectorEngine._(model, info, backend, input);
    } catch (e) {
      _closeQuietly(model, log);
      if (e is DetectorLoadException) rethrow;
      throw DetectorLoadException('YOLO26n failed its load checks: $e');
    }
  }

  final CompiledModel _model;
  final DetectorInfo info;
  final DetectorBackend _backend;
  final FrameGatherer _gatherer = FrameGatherer();

  /// The one input tensor, reused for every frame.
  final Float32List _input;
  bool _closed = false;

  /// Gather → run → top-k decode for one frame.
  DetectionFrame detect(FrameMessage frame) =>
      _detect(frame, frame.materialize());

  /// [detect] plus the same frame as upright RGBA (a question's snapshot),
  /// converted from the bytes the detector just read, and the conversion time.
  (DetectionFrame, RgbaPixels, Duration) detectWithSnapshot(
    FrameMessage frame,
  ) {
    final data = frame.materialize();
    final detection = _detect(frame, data);
    final watch = Stopwatch()..start();
    final pixels = uprightRgba(frame, data);
    return (detection, pixels, watch.elapsed);
  }

  DetectionFrame _detect(FrameMessage frame, Uint8List data) {
    if (_closed) throw StateError('The detector is closed');
    final watch = Stopwatch()..start();
    final plan = _gatherer.gather(frame, data, _input);
    final pre = watch.elapsedMicroseconds;
    final raw = _model.run([_input]).single;
    final run = watch.elapsedMicroseconds;
    final boxes = decodeDetections(raw, plan.letterbox);
    final post = watch.elapsedMicroseconds;
    return DetectionFrame(
      frameId: frame.frameId,
      width: plan.letterbox.uprightW,
      height: plan.letterbox.uprightH,
      boxes: boxes,
      preMicros: pre,
      runMicros: run - pre,
      postMicros: post - run,
      backend: _backend,
    );
  }

  /// Releases the model. Safe to call more than once.
  void close() {
    if (_closed) return;
    _closed = true;
    _closeQuietly(_model, debugPrint);
  }

  static void _closeQuietly(CompiledModel model, void Function(String) log) {
    try {
      model.close();
    } catch (e) {
      log('[Detector] closing the model failed: $e');
    }
  }
}
