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

/// What the self-test drives: the interfaces its adapters implement
/// (`self_test_adapters.dart` wraps the app's own services; tests use fakes)
/// and the values they hand back.
library;

import 'dart:typed_data';

import 'package:flutter_edge_ai/flutter_edge_ai.dart' show PreferredBackend;

import '../config/model_catalog.dart';
import '../data/services/detector/frame_message.dart';
import '../data/services/frames/frame_source.dart' show FrameView;
import '../domain/models/audio_devices.dart' show DeviceReady;
import '../domain/models/chat_model_config.dart';
import '../domain/models/detection.dart';
import '../domain/models/detector_spec.dart' show kDetModelAsset;
import '../utils/result.dart';
import 'cats_golden.dart' show kCatsGolden;

/// A model file for the self-test and where its path came from.
final class const ModelFileChoice({
  /// Null when there is none ([problem] says why). For a [bundled] model,
  /// the asset key.
  final String? path,

  /// `--gemma`, `GEMMA_MODEL_PATH`, `model store`, `bundled`.
  required final String source,

  /// Why the file cannot be used (missing, unreadable, not provisioned).
  final String? problem,

  /// Built into the app: [path] is a Flutter asset, read from the bundle.
  final bool bundled = false,
}) {
  /// The detector built into the app (without `--detector`).
  static const bundledDetector = ModelFileChoice(
    path: kDetModelAsset,
    source: 'bundled',
    bundled: true,
  );
}

/// The detector as the self-test drives it (the app's `DetectorService`).
abstract interface class SelfTestDetector {
  /// Loads [model] (a file, or the bundled asset) on exactly [backend].
  Future<Result<DetectorInfo>> load(
    ModelFileChoice model,
    DetectorBackend backend,
  );
  Future<Result<DetectionFrame>> detect(FrameMessage frame);
  Future<void> close();
}

/// The chat model the self-test loads: its file, what it is loaded with,
/// and where those settings came from.
final class const SelfTestChatModel({
  required final ModelFileChoice file,
  required final ChatModelConfig config,

  /// `the app's Gemma 4 E2B settings` (`--gemma` or `GEMMA_MODEL_PATH`,
  /// [kDefineChatModel]), `your own .litertlm, saved settings`, `your own
  /// .litertlm` (blocked) or `no chat model`.
  required final String settingsSource,

  /// The file's SHA-256 when known (computed when it was stored or hashed in
  /// place; pinned for an adopted earlier Gemma 4 E2B download).
  final String? sha256,
}) {
  /// [config] on [backend] instead (`--gemma-backend`); itself when null.
  SelfTestChatModel withBackend(PreferredBackend? backend) =>
      backend == null || backend == config.llm.backend
      ? this
      : SelfTestChatModel(
          file: file,
          settingsSource: '$settingsSource, backend from --gemma-backend',
          sha256: sha256,
          config: ChatModelConfig(
            name: config.name,
            modelType: config.modelType,
            tools: config.tools,
            llm: LlmConfig(
              maxTokens: config.llm.maxTokens,
              backend: backend,
              supportImage: config.llm.supportImage,
              maxNumImages: config.llm.maxNumImages,
            ),
          ),
        );

  /// `npu · ctx 1280 · images off · tools off · type gemmaIt`.
  String get settingsLine => [
    config.llm.backend.name,
    'ctx ${config.llm.maxTokens}',
    'images ${config.llm.supportImage ? 'on' : 'off'}',
    'tools ${config.tools ? 'on' : 'off'}',
    'type ${config.modelType.name}',
  ].join(' · ');
}

/// [GemmaLoad.config] for [config] on an engine built with [engineContext]
/// tokens: `maxTokens 8192 (engine 4096) · image on · tools on · type
/// gemma4` (the engine's context only when it differs).
String gemmaLoadConfigLine(
  ChatModelConfig config, {
  required int engineContext,
}) => [
  'maxTokens ${config.llm.maxTokens}'
      '${engineContext == config.llm.maxTokens ? '' : ' (engine $engineContext)'}',
  'image ${config.llm.supportImage ? 'on' : 'off'}',
  'tools ${config.tools ? 'on' : 'off'}',
  'type ${config.modelType.name}',
].join(' · ');

/// The chat model loaded with its configuration and warmed up.
final class const GemmaLoad({
  required final String modelId,

  /// `InferenceModel.activeBackend`.
  required final PreferredBackend backend,
  required final Duration loadTime,
  required final Duration warmUpTime,

  /// What was loaded ([gemmaLoadConfigLine]).
  required final String config,

  /// The context the engine was built with (`InferenceModel.maxTokens`).
  final int? contextTokens,
});

/// One timed generation.
final class const GemmaGeneration({
  /// Streamed chunks (about one token each).
  required final int chunks,

  /// The engine's own output token count and rate
  /// (`getSessionMetrics`); null when it reports none.
  final int? engineTokens,
  final double? engineTokensPerSecond,
  required final Duration firstChunk,
  required final Duration total,
  required final String text,
}) {
  /// Chunks after the first, per second of decoding.
  double get decodeRate {
    final decode = total - firstChunk;
    return chunks > 1 && decode > Duration.zero
        ? (chunks - 1) / (decode.inMicroseconds / 1e6)
        : 0;
  }
}

/// The chat model as the self-test drives it (the app's `LlmService`).
abstract interface class SelfTestGemma {
  /// Install with [config]'s type, load with exactly its backend, context and
  /// image setting (no fallback), warm up.
  Future<Result<GemmaLoad>> load(String path, ChatModelConfig config);
  Future<Result<GemmaGeneration>> generate(String prompt, int maxTokens);
  Future<void> close();
}

/// A capture of the default sink's monitor that is running.
abstract interface class SelfTestMonitor {
  /// Waits for the tail still in the sound server's buffers, stops, and
  /// returns 16 kHz mono PCM16.
  Future<Result<Uint8List>> stop();
}

/// One microphone recording and the input it came from.
final class const MicCapture({
  /// `Built-in Audio Analog Stereo · PulseAudio (on PipeWire 1.0.5)`.
  required final String device,

  /// 16 kHz mono PCM16.
  required final Uint8List pcm,
});

/// The audio path as the self-test drives it: the app's audio repository
/// (session, soloud, the output check) and its `record` microphone.
abstract interface class SelfTestAudio {
  /// Starts the output engine and checks the output device as the app does
  /// (`AudioRepository.prepare`). Ok: the device it plays to.
  Future<Result<DeviceReady>> prepareOutput();

  /// Null when the default sink's monitor can be recorded here (Linux:
  /// `parecord -d @DEFAULT_MONITOR@`); else why not.
  String? get monitorUnavailable;

  /// Starts recording the default sink's monitor.
  Future<Result<SelfTestMonitor>> startMonitor();

  /// Plays [pcm] (PCM16 mono at [sampleRate]) through the app's player;
  /// completes once it has played.
  Future<Result<void>> play(Uint8List pcm, int sampleRate);

  /// [duration] from the default input through the app's `record` path
  /// (16 kHz mono PCM16), with the input's name.
  Future<Result<MicCapture>> recordMic(Duration duration);

  /// Stops whatever still runs (a timed-out step) and releases the devices.
  Future<void> close();
}

/// The image the detector runs on.
final class const SelfTestImage({
  required final FrameView frame,

  /// `bundled cats (COCO val2017 39769)` or the path.
  required final String label,

  /// Whether [kCatsGolden] applies.
  required final bool golden,
});
