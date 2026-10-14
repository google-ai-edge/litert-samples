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
import '../domain/models/chat_model.dart' show CustomChatModel, LitertlmSection;
import '../domain/models/chat_model_config.dart';
import '../domain/models/detection.dart';
import '../domain/models/detector_spec.dart' show kDetModelAsset;
import '../domain/models/knowledge_query.dart';
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

  /// The user's own `.litertlm` (the app's plan): its towers follow from
  /// [header], the backend and its saved choices, as in the app. Null for
  /// the app's fixed Gemma 4 E2B settings (`--gemma`, `GEMMA_MODEL_PATH`:
  /// [kDefineChatModel]), whose towers stay as they are on any backend.
  final CustomChatModel? custom,

  /// The sections of [custom]'s file header as the app last read them (the
  /// plan's); null when it could not be read (no tower loads).
  final List<LitertlmSection>? header,
}) {
  /// [config] on [backend] instead (`--gemma-backend`); itself when null.
  /// The towers of [custom] are placed again for that backend
  /// ([CustomChatModel.towersOf]); fixed ones stay.
  SelfTestChatModel withBackend(PreferredBackend? backend) =>
      backend == null || backend == config.llm.backend
      ? this
      : SelfTestChatModel(
          file: file,
          settingsSource: '$settingsSource, backend from --gemma-backend',
          sha256: sha256,
          custom: custom,
          header: header,
          config: ChatModelConfig(
            name: config.name,
            modelType: config.modelType,
            tools: config.tools,
            llm: LlmConfig(
              maxTokens: config.llm.maxTokens,
              backend: backend,
              maxNumImages: config.llm.maxNumImages,
              towers: switch ((custom, header)) {
                (final custom?, final header?) =>
                  custom.copyWith(backend: backend).towersOf(header),
                _ => config.llm.towers,
              },
            ),
          ),
        );

  /// `npu · ctx 1280 · vision cpu · audio npu · tools off · type gemmaIt`:
  /// each loaded tower with where it runs.
  String get settingsLine => [
    config.llm.backend.name,
    'ctx ${config.llm.maxTokens}',
    ...config.llm.towers.parts,
    'tools ${config.tools ? 'on' : 'off'}',
    'type ${config.modelType.name}',
  ].join(' · ');
}

/// [GemmaLoad.config] for [config] on an engine built with [engineContext]
/// tokens: `maxTokens 8192 (engine 4096) · vision cpu · tools on · type
/// gemma4` (the engine's context only when it differs; each loaded tower
/// with where it runs).
String gemmaLoadConfigLine(
  ChatModelConfig config, {
  required int engineContext,
}) => [
  'maxTokens ${config.llm.maxTokens}'
      '${engineContext == config.llm.maxTokens ? '' : ' (engine $engineContext)'}',
  ...config.llm.towers.parts,
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

/// The embedder ready for steps 7a/7b.
final class const SelfTestEmbedderLoad({
  /// `embeddinggemma-2-740m`.
  required final String modelId,

  /// `EmbeddingModel.activeBackend`.
  required final String backend,

  /// What it accepts: `audio+text`.
  required final String media,

  /// The load and warm-up of a run that loaded it; null for the app's own
  /// embedder, loaded before the run ([borrowed]).
  final Duration? loadTime,
  final Duration? warmUpTime,

  /// The app's own embedder (the in-app run): used as it is, never closed.
  final bool borrowed = false,
});

/// The image encoder's load for step 7c ([SelfTestEmbedder.enableImages]):
/// the reload's time and the process RSS around it; [alreadyLoaded] when the
/// app's camera demo had loaded it before the run.
final class const SelfTestImagesLoad({
  required final Duration loadTime,
  final int? rssBeforeBytes,
  final int? rssAfterBytes,
  final bool alreadyLoaded = false,
});

/// The embedder as the self-test drives it: the app's `EmbedderService`.
/// In the app it is the one setup loaded (one embedder per process: a
/// second load would hand back that same model, and closing it would break
/// the knowledge base); headless, one loaded for the run from the built-in
/// file and closed after it.
abstract interface class SelfTestEmbedder {
  /// Ready to embed, as the app loads it (audio declared), or why not.
  Future<Result<SelfTestEmbedderLoad>> prepare();

  /// A question's vector, mapped exactly as the knowledge base maps it
  /// (text with the `retrievalQuery` prefix, audio as it is).
  Future<Result<List<double>>> embed(KnowledgeQuery query);

  /// Loads the image encoder as the camera demo does the first time its
  /// visual memory starts (`EmbedderService.enableImages`: one reload),
  /// unless it is loaded already; what it took.
  Future<Result<SelfTestImagesLoad>> enableImages();

  /// A photo's vector, as the visual memory embeds a camera frame (the
  /// encoded image as it is). Needs [enableImages] first.
  Future<Result<List<double>>> embedImage(Uint8List encoded);

  /// A phrase's vector, as the visual memory embeds a question (raw text,
  /// no task prefix).
  Future<Result<List<double>>> embedMemoryText(String text);

  /// Closes what [prepare] loaded; nothing for a borrowed embedder.
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
