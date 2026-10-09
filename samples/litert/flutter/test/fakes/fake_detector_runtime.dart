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

import 'dart:io' show File, sleep;
import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter_litert/native.dart';
import 'package:litert_edge_demos/data/services/detector/detector_engine.dart';
import 'package:litert_edge_demos/data/services/detector/detector_service.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/detector_spec.dart';
import 'package:litert_edge_demos/utils/result.dart';

/// The box every [FakeCompiledModel] run reports, in 640 input px: a cat at
/// 0.9 covering (100, 180)–(300, 380), i.e. (100, 100)–(300, 300) in a
/// 640×480 frame (padY 80).
const kFakeBox = [100.0, 180.0, 300.0, 380.0];
const kFakeClass = 15;
const kFakeScore = 0.9;

/// A `CompiledModel` without LiteRT. Reports what [FakeDetectorRuntime] says
/// (acceleration, I/O sizes) and returns one fixed detection per run.
class FakeCompiledModel implements CompiledModel {
  FakeCompiledModel(this._runtime, this._requested);

  final FakeDetectorRuntime _runtime;
  final Set<Accelerator> _requested;
  int runs = 0;
  bool closed = false;

  @override
  bool get isFullyAccelerated => _runtime.fullyAccelerated;

  @override
  Set<Accelerator> get accelerators =>
      _runtime.effectiveAccelerators ?? _requested;

  @override
  Set<Accelerator> get requestedAccelerators => _requested;

  @override
  bool get didFallback => false;

  @override
  List<int> get inputByteSizes => [_runtime.inputBytes];

  @override
  List<int> get outputByteSizes => [_runtime.outputBytes];

  @override
  int get inputCount => 1;

  @override
  int get outputCount => 1;

  @override
  List<Float32List> run(List<Float32List> inputs) {
    if (closed) throw StateError('CompiledModel is already closed.');
    runs++;
    if (runs == _runtime.exitOnRun) {
      Isolate.exit(); // a native crash, as far as Dart can tell
    }
    if (runs >= _runtime.delayFromRun && _runtime.runDelay > Duration.zero) {
      sleep(_runtime.runDelay);
    }
    final raw = Float32List(_runtime.outputBytes ~/ 4);
    if (raw.length == kDetAnchors * kDetRawStride) {
      raw.setAll(0, kFakeBox);
      raw[4 + kFakeClass] = kFakeScore;
    }
    return [raw];
  }

  @override
  void close() => closed = true;

  @override
  Object? noSuchMethod(Invocation invocation) =>
      throw UnimplementedError('FakeCompiledModel: ${invocation.memberName}');
}

/// A [DetectorRuntime] whose models behave as configured. Const and free of
/// closures, so it can be sent to the detector worker isolate.
class FakeDetectorRuntime implements DetectorRuntime {
  const FakeDetectorRuntime({
    this.fullyAccelerated = true,
    this.effectiveAccelerators,
    this.inputBytes = kDetInputBytes,
    this.outputBytes = kDetOutputBytes,
    this.agrees = true,
    this.deviation = 1.8e-3,
    this.reference = VerifyReference.interpreter,
    this.createError,
    this.runDelay = Duration.zero,
    this.delayFromRun = 1,
    this.exitOnRun,
  });

  final bool fullyAccelerated;

  /// What the model reports as compiled-with; null = what was requested.
  final Set<Accelerator>? effectiveAccelerators;
  final int inputBytes;
  final int outputBytes;
  final bool agrees;
  final double deviation;

  /// The CPU path [verify] reports it compared against.
  final VerifyReference reference;

  /// When set, [create] throws this, like LiteRT's 504 for the Arm file.
  final String? createError;

  /// Each run from number [delayFromRun] on blocks the worker this long.
  /// Load runs the model twice (verify, warm-up), so 3 is the first frame.
  final Duration runDelay;
  final int delayFromRun;

  /// The worker isolate exits during this run (1-based); null = never.
  final int? exitOnRun;

  @override
  CompiledModel create(Uint8List modelBytes, Set<Accelerator> accelerators) {
    if (createError case final error?) throw StateError(error);
    return FakeCompiledModel(this, accelerators);
  }

  @override
  DetectorVerification verify(Uint8List modelBytes, CompiledModel model) {
    model.run([Float32List(inputBytes ~/ 4)]);
    return (
      result: BackendVerification(
        agrees: agrees,
        absoluteDeviation: deviation,
        outputRange: 864,
        relativeDeviation: deviation / 864,
      ),
      reference: reference,
    );
  }
}

/// Wraps a [FakeDetectorRuntime] and records every model it builds (for
/// tests that drive `DetectorEngine` in the test isolate).
class RecordingRuntime implements DetectorRuntime {
  RecordingRuntime([this._inner = const FakeDetectorRuntime()]);

  final FakeDetectorRuntime _inner;
  final List<Set<Accelerator>> requested = [];
  final List<FakeCompiledModel> models = [];

  @override
  CompiledModel create(Uint8List modelBytes, Set<Accelerator> accelerators) {
    requested.add(accelerators);
    final model = _inner.create(modelBytes, accelerators) as FakeCompiledModel;
    models.add(model);
    return model;
  }

  @override
  DetectorVerification verify(Uint8List modelBytes, CompiledModel model) =>
      _inner.verify(modelBytes, model);
}

/// The built-in detector in unit and widget tests: no worker isolate (a
/// widget test's fake clock never sees an isolate's reply), the size check
/// against [kFakeBundledDetectorBytes] like the real one.
DetectorService fakeDetectorService() => InProcessDetectorService();

/// [DetectorService] that loads in the test isolate: a file or bytes of
/// [expectedBytes] load as a GPU detector, anything else fails with the
/// real size message's numbers.
final class InProcessDetectorService extends DetectorService {
  InProcessDetectorService({this.expectedBytes = kFakeBundledDetectorBytes});

  final int expectedBytes;
  DetectorInfo? _loaded;
  final List<DetectorModelSource> sources = [];

  @override
  DetectorInfo? get info => _loaded;

  @override
  bool get isLoaded => _loaded != null;

  @override
  Future<Result<DetectorInfo>> load({
    required DetectorModelSource source,
    required DetectorBackend backend,
  }) async {
    sources.add(source);
    if (source case DetectorFile(:final path) when !File(path).existsSync()) {
      return Result.error(
        DetectorUnavailableException('Detector model file not found: $path'),
      );
    }
    final size = switch (source) {
      DetectorBytes(:final bytes) => bytes.length,
      DetectorFile(:final path) =>
        File(path).existsSync() ? File(path).lengthSync() : -1,
    };
    if (size != expectedBytes) {
      return Result.error(
        DetectorUnavailableException(
          'Detector model is $size bytes, expected $expectedBytes '
          '(${source.label})',
        ),
      );
    }
    return Result.ok(
      _loaded = DetectorInfo(
        backend: backend,
        fullyAccelerated: backend == DetectorBackend.gpu,
        verifyAbsolute: 1.8e-3,
        verifyRelative: 2e-6,
        verifyReference: VerifyReference.interpreter,
        createTime: const Duration(milliseconds: 50),
        verifyTime: const Duration(milliseconds: 300),
        firstRunTime: const Duration(milliseconds: 5),
      ),
    );
  }

  @override
  Future<void> close() async => _loaded = null;
}

/// What the fake asset bundle holds for `assets/models/yolo26n_…tflite`.
const kFakeBundledDetectorBytes = 64;

/// `ModelRepository.bundledDetector` in unit tests (no asset bundle read).
Future<Uint8List> fakeBundledDetector() async =>
    Uint8List(kFakeBundledDetectorBytes);
