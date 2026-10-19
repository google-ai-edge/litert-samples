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

import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart'
    show LiteRtLmBundleProblem;

import '../../utils/result.dart';
import 'model_id.dart';

/// The chat model's facts for the Models screen, the "This device" card and
/// the diagnostics report.
final class const ChatModelFacts({
  /// `Gemma 4 E2B`, or the user's name for their own model.
  required final String name,

  /// The user's own `.litertlm`; false only for `GEMMA_MODEL_PATH` (a
  /// `--dart-define`).
  required final bool custom,

  /// Where the file came from (`imported from …`, `downloaded from …`,
  /// `in place: …`, `GEMMA_MODEL_PATH=…`).
  required final String source,

  /// The file's SHA-256 when known: computed when it was imported or
  /// downloaded, or in the background for a file used in place; null for
  /// `GEMMA_MODEL_PATH`, and for a file in place loaded before its hash
  /// was done.
  final String? sha256,

  /// [sha256] matched the checksum the user entered for "Download from
  /// URL…"; false when it was only computed and recorded.
  final bool checksumMatched = false,

  /// The backend asked for; the backend it runs on is `LoadedModelInfo
  /// .backend` (`InferenceModel.activeBackend`, equal or the load failed).
  required final String requestedBackend,

  /// The context asked for, and the one the engine was built with.
  required final int requestedContext,
  required final int contextTokens,
  required final bool tools,

  /// flutter_edge_ai's `ModelType` name (`gemma4`).
  required final String modelType,

  /// Where the engine put the vision encoder (`cpu`, `gpu`, `npu`:
  /// `InferenceModel.activeVisionBackend`; the app asks for none); null when
  /// it was not loaded (the file has none) or the engine does not say.
  final String? visionBackend,

  /// The same for the audio encoder (`InferenceModel.activeAudioBackend`).
  final String? audioBackend,

  /// The process's resident memory before and after the engine was built
  /// (`ProcessInfo.currentRss`; GPU memory outside the process's pages is not
  /// in it). Null when it could not be read.
  final int? rssBeforeBytes,
  final int? rssAfterBytes,
}) {
  /// `1a2b3c4d…` for the card.
  String? get shortSha => sha256 == null ? null : '${sha256!.substring(0, 8)}…';

  /// The engine reports a vision encoder ([visionBackend]): photos and
  /// camera frames can be sent.
  bool get images => visionBackend != null;

  /// The engine reports an audio encoder ([audioBackend]).
  bool get audio => audioBackend != null;

  /// `ctx 1280 · vision cpu · audio npu · tools off · gemma4`: each loaded
  /// tower with where the engine put it (none for a text-only load).
  String get capabilityLine => [
    contextTokens == requestedContext
        ? 'ctx $contextTokens'
        : 'ctx $contextTokens (asked $requestedContext)',
    if (visionBackend case final v?) 'vision $v',
    if (audioBackend case final a?) 'audio $a',
    'tools ${tools ? 'on' : 'off'}',
    modelType,
  ].join(' · ');
}

/// What a loaded model reports, for the setup screen and the debug overlay.
final class const LoadedModelInfo({
  required final String modelId,
  required final String backend,
  required final Duration loadTime,
  required final Duration warmUpTime,

  /// A longer backend label when there is one (`GPU fp32 full`,
  /// `CPU (chosen)`); shown instead of [backend].
  final String? detail,

  /// Loaded on the CPU because the user (Demo 3's Detector setting) or a
  /// flag (`DETECTOR_BACKEND=cpu`) chose it: ready, but shown in amber
  /// everywhere it appears.
  final bool explicitCpu = false,

  /// False when the runtime cannot report where it runs (speech): [backend]
  /// is then only what was requested.
  final bool backendReported = true,

  /// The native log lines printed during this model's load (the tap in
  /// release builds on Linux and macOS; empty elsewhere).
  final List<String> nativeLog = const [],

  /// The chat model's facts; null for every other model.
  final ChatModelFacts? chat,
});

/// Lifecycle of one model on the setup screen.
sealed class const ModelState();

/// Not started yet.
final class const ModelPending() extends ModelState;

/// Downloading or registering the file. [percent] is 0–100, null when the
/// source reports no progress yet.
final class const ModelInstalling({final int? percent}) extends ModelState;

/// Building the runtime (and, on first launch, compiling the GPU programs).
final class const ModelLoading() extends ModelState;

/// Running one tiny generation so the first real turn is not the slow one.
final class const ModelWarmingUp() extends ModelState;

/// Loaded on the requested backend and warmed up.
final class const ModelReady(final LoadedModelInfo info) extends ModelState;

/// Not loadable in this build or configuration (for example a flag is not
/// set). Retry cannot change it; [reason] says what would.
final class const ModelUnavailable(final String reason) extends ModelState;

/// Setup stopped. [message] is shown as-is, with a Retry button unless
/// [retryable] is false (the same load cannot help; [message] says what
/// does). [backend] is the backend the load failed on (`gpu`, `cpu`) when
/// another backend might work (the detector: Demo 3 offers the other one);
/// null otherwise.
final class const ModelFailed(
  final String message, {
  final String? backend,
  final bool retryable = true,

  /// The chat model only: flutter_edge_ai_litertlm refused the file before
  /// any engine (`LiteRtLmBundleException.problem`: `incomplete`,
  /// `wrongBackend`, `unsupportedNpu`). The same load is refused again, so
  /// such a failure is never [retryable]; the Models screen offers what the
  /// plugin's message advises, where it can (`wrongBackend` for a build
  /// compiled for the NPU: Run on NPU). Null for every other failure.
  final LiteRtLmBundleProblem? bundleProblem,
}) extends ModelState;

/// Makes a demo's speech recognizer the active one (`ModelRepository
/// .activateStt`); the demo's view model calls it on entry.
typedef SttActivation = Future<Result<void>> Function(ModelId stt);
