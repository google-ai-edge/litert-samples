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

/// The detector's long-lived worker isolate: it owns the `CompiledModel` and
/// handles one request at a time, in order. `CompiledModel.run` is called here
/// directly (not `runAsync`), so the 4.9 MB input tensor never crosses an
/// isolate boundary.
library;

import 'dart:io';
import 'dart:isolate';

import 'package:flutter/foundation.dart';

import '../../../domain/models/detection.dart';
import 'detector_engine.dart';
import 'frame_message.dart';

/// The spawn message.
final class DetectorWorkerBoot {
  const DetectorWorkerBoot(this.replyTo, this.runtime);

  final SendPort replyTo;
  final DetectorRuntime runtime;
}

/// Main → worker.
sealed class DetectorRequest {
  const DetectorRequest(this.id);

  final int id;
}

/// Reads the model file, or takes the model bytes, and runs the load checks.
final class DetectorLoadRequest extends DetectorRequest {
  const DetectorLoadRequest(
    super.id, {
    this.modelPath,
    this.modelBytes,
    required this.backend,
    required this.expectedBytes,
  }) : assert((modelPath == null) != (modelBytes == null));

  /// A file the worker reads (the self-test's `--detector`, a test's copy).
  final String? modelPath;

  /// The bundled asset's bytes, moved without a copy.
  final TransferableTypedData? modelBytes;
  final DetectorBackend backend;
  final int expectedBytes;
}

final class DetectorDetectRequest extends DetectorRequest {
  const DetectorDetectRequest(
    super.id,
    this.frame, {
    this.withSnapshot = false,
  });

  final FrameMessage frame;

  /// Also reply with the frame as upright RGBA ([SnapshotReply]).
  final bool withSnapshot;
}

/// Worker → main for a [DetectorDetectRequest.withSnapshot] frame: the
/// detection plus the upright RGBA pixels, which cross back without a copy.
final class SnapshotReply {
  const SnapshotReply({
    required this.frame,
    required this.width,
    required this.height,
    required this.rgba,
    required this.convertTime,
  });

  final DetectionFrame frame;
  final int width;
  final int height;
  final TransferableTypedData rgba;
  final Duration convertTime;
}

/// Closes the model; the worker then closes its port and exits.
final class DetectorCloseRequest extends DetectorRequest {
  const DetectorCloseRequest(super.id);
}

/// Worker → main: [value] on success, otherwise [error].
final class DetectorReply {
  const DetectorReply(this.id, {this.value, this.error});

  final int id;
  final Object? value;
  final String? error;
}

/// The worker's entry point (for `Isolate.spawn`).
void detectorWorkerMain(DetectorWorkerBoot boot) {
  final requests = ReceivePort('detector worker');
  boot.replyTo.send(requests.sendPort);
  DetectorEngine? engine;

  void reply(int id, {Object? value, String? error}) =>
      boot.replyTo.send(DetectorReply(id, value: value, error: error));

  requests.listen((message) {
    switch (message) {
      case DetectorLoadRequest(
        :final id,
        :final modelPath,
        :final modelBytes,
        :final backend,
        :final expectedBytes,
      ):
        engine?.close();
        engine = null;
        try {
          final Uint8List bytes;
          if (modelBytes != null) {
            bytes = modelBytes.materialize().asUint8List();
          } else {
            final file = File(modelPath!);
            if (!file.existsSync()) {
              reply(id, error: '$kDetectorFileNotFound: $modelPath');
              return;
            }
            bytes = file.readAsBytesSync();
          }
          final loaded = DetectorEngine.load(
            modelBytes: bytes,
            backend: backend,
            runtime: boot.runtime,
            expectedBytes: expectedBytes,
          );
          engine = loaded;
          reply(id, value: loaded.info);
        } on DetectorLoadException catch (e) {
          reply(id, error: e.message);
        } catch (e, st) {
          debugPrint('[DetectorWorker] load failed: $e\n$st');
          reply(id, error: 'Detector load failed: $e');
        }
      case DetectorDetectRequest(:final id, :final frame, :final withSnapshot):
        final current = engine;
        if (current == null) {
          reply(id, error: 'The detector is not loaded');
          return;
        }
        try {
          if (withSnapshot) {
            final (detection, pixels, convertTime) = current.detectWithSnapshot(
              frame,
            );
            reply(
              id,
              value: SnapshotReply(
                frame: detection,
                width: pixels.width,
                height: pixels.height,
                rgba: TransferableTypedData.fromList([pixels.bytes]),
                convertTime: convertTime,
              ),
            );
          } else {
            reply(id, value: current.detect(frame));
          }
        } catch (e, st) {
          debugPrint('[DetectorWorker] detect failed: $e\n$st');
          reply(id, error: 'Detection failed: $e');
        }
      case DetectorCloseRequest(:final id):
        engine?.close();
        engine = null;
        reply(id);
        requests.close(); // no open ports left: the isolate exits
      default:
        debugPrint('[DetectorWorker] unknown message $message');
    }
  });
}
