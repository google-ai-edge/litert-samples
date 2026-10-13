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
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/services/llm/llm_service.dart';
import 'package:litert_edge_demos/domain/models/chat_model_config.dart';
import 'package:litert_edge_demos/utils/result.dart';

/// [LlmService] whose engine reports [activeBackend] — set it to the CPU to
/// play a silent GPU→CPU fallback. [activeBackend] null plays "the engine
/// came up on the backend requested".
class FakeLlmService implements LlmService {
  PreferredBackend? activeBackend = PreferredBackend.gpu;

  /// Follow the requested backend instead of [activeBackend].
  bool followRequested = false;
  final List<LlmConfig> loads = [];

  /// Every model config [load] was given.
  final List<ChatModelConfig> models = [];
  int warmUpCalls = 0;
  final List<bool> warmUpsWithImage = [];

  /// When set, the next [load] fails with it (a wrong SoC, a context the
  /// build does not have).
  Exception? loadError;

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
    final active = followRequested ? config.backend : activeBackend;
    if (active != config.backend) {
      return Result.error(
        BackendMismatchException(
          requested: config.backend,
          active: active,
          modelName: model.name,
        ),
      );
    }
    _loaded = model;
    return Result.ok(
      LlmInfo(
        modelId: 'gemma-4-E2B-it',
        backend: config.backend,
        loadTime: const Duration(milliseconds: 2400),
        contextTokens: config.maxTokens,
      ),
    );
  }

  @override
  Future<Result<Duration>> warmUp(
    SamplerConfig sampler, {
    required bool withImage,
  }) async {
    warmUpCalls++;
    warmUpsWithImage.add(withImage);
    return const Result.ok(Duration(milliseconds: 150));
  }

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
