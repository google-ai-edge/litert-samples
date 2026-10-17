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

import 'dart:async';

import 'package:flutter_edge_ai/flutter_edge_ai.dart' hide ModelSpec;
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart'
    show
        BackendInitAttemptFailure,
        BackendInitException,
        LiteRtLmEngineCreateException,
        LiteRtLmNpuTarget;
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/services/llm/llm_service.dart';
import 'package:litert_edge_demos/domain/models/chat_model_config.dart';
import 'package:litert_edge_demos/utils/result.dart';

/// What [LlmService.load] returns when the strict engine cannot create the
/// engine on [backend]: the plugin's [BackendInitException] with LiteRT-LM's
/// own [nativeMessage], wrapped as `LlmService` wraps it. The engine's
/// message is the plugin's (flutter_edge_ai_litertlm 1.14.0
/// `LiteRtLmFfiClient.initialize`):
/// `Failed to create engine on {backend}: {nativeMessage}. Model: {path}`,
/// with the SoC hint `LiteRtLmEngine` adds to an NPU attempt of a bundle
/// compiled for [compiledFor]. The default [nativeMessage] is a string
/// LiteRT's Android OpenCL accelerator carries.
ChatModelLoadException engineCreateFailure({
  String modelName = 'Gemma 4 E2B',
  PreferredBackend backend = PreferredBackend.gpu,
  String nativeMessage = 'Can not find OpenCL library on this device',
  String modelPath = '/models/gemma-4-E2B-it.litertlm',
  LiteRtLmNpuTarget? compiledFor,
}) => ChatModelLoadException(
  modelName: modelName,
  requested: backend,
  cause: BackendInitException(
    attempts: [
      BackendInitAttemptFailure(
        backend: backend,
        error: LiteRtLmEngineCreateException(
          'Failed to create engine on ${backend.name}: $nativeMessage. '
          'Model: $modelPath'
          '${compiledFor == null || backend != PreferredBackend.npu ? '' : '. The bundle is compiled for the $compiledFor NPU; if this device is not a $compiledFor, that is the cause.'}',
          nativeCode: 13,
          nativeMessage: nativeMessage,
        ),
        stackTrace: StackTrace.empty,
      ),
    ],
  ),
);

/// [LlmService] on the strict engine (`LiteRtLmEngine(backendFallback:
/// false)`): a load comes up on the backend it asks for, or fails with
/// [loadError] (a strict failure: [engineCreateFailure]).
class FakeLlmService implements LlmService {
  /// False: the engine reports no backend (`activeBackend` null).
  bool reportsBackend = true;

  /// Where the engine puts each encoder that loads
  /// (`activeVisionBackend` / `activeAudioBackend`): its own choice.
  PreferredBackend? visionBackend = PreferredBackend.cpu;
  PreferredBackend? audioBackend = PreferredBackend.cpu;
  final List<LlmConfig> loads = [];

  /// Every model config [load] was given.
  final List<ChatModelConfig> models = [];
  int warmUpCalls = 0;
  final List<bool> warmUpsWithImage = [];

  /// When set, the next [install] fails with it (a file that vanished:
  /// [ModelFileMissingException]).
  Exception? installError;

  /// When set, the next [load] fails with it (a wrong SoC, a context the
  /// build does not have).
  Exception? loadError;

  /// When set, the next [warmUp] fails with it (LiteRT-LM's "Failed to
  /// create conversation": an encoder the backend cannot run).
  Exception? warmUpError;

  /// When set, [load] waits for it, like a first GPU compile.
  Completer<void>? loadGate;
  int closeCalls = 0;
  int unloadCalls = 0;
  ChatModelConfig? _loaded;

  @override
  bool get isLoaded => _loaded != null;

  @override
  ChatModelConfig? get loaded => _loaded;

  @override
  InferenceModel get model => throw UnimplementedError('not used here');

  /// The paths [install] was given, and the types.
  final List<String> installs = [];
  final List<ModelType> installTypes = [];

  @override
  Future<Result<String>> install({
    required String path,
    ModelType modelType = ModelType.gemma4,
    required void Function(int percent) onProgress,
  }) async {
    installs.add(path);
    installTypes.add(modelType);
    if (installError case final error?) {
      installError = null;
      return Result.error(error);
    }
    onProgress(50);
    onProgress(100);
    return Result.ok(path.split('/').last.replaceAll('.litertlm', ''));
  }

  @override
  Future<Result<LlmInfo>> load(ChatModelConfig model) async {
    final config = model.llm;
    loads.add(config);
    models.add(model);
    _loaded = null;
    await loadGate?.future;
    if (loadError case final error?) {
      loadError = null;
      return Result.error(error);
    }
    _loaded = model;
    return Result.ok(
      LlmInfo(
        modelId: 'gemma-4-E2B-it',
        backend: reportsBackend ? config.backend : null,
        visionBackend: config.supportImage ? visionBackend : null,
        audioBackend: config.supportAudio ? audioBackend : null,
        loadTime: const Duration(milliseconds: 2400),
        contextTokens: config.maxTokens,
      ),
    );
  }

  @override
  Future<Result<Duration>> warmUp(
    SamplerConfig sampler, {
    required bool withImage,
    bool withAudio = false,
  }) async {
    warmUpCalls++;
    warmUpsWithImage.add(withImage);
    warmUpsWithAudio.add(withAudio);
    if (warmUpError case final error?) {
      warmUpError = null;
      return Result.error(error);
    }
    return const Result.ok(Duration(milliseconds: 150));
  }

  /// [warmUp]'s `withAudio`, per call.
  final List<bool> warmUpsWithAudio = [];

  @override
  Future<void> unload() async {
    unloadCalls++;
    _loaded = null;
  }

  @override
  Future<void> close() async {
    closeCalls++;
    _loaded = null;
  }
}
