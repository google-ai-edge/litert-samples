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

import '../../config/env.dart';
import '../../domain/models/chat_model.dart';
import '../../domain/models/model_id.dart';
import '../../domain/models/model_source_resolver.dart';
import '../../domain/models/provisioning.dart';
import '../../domain/ports/chat_model_planner.dart';
import '../services/model_store/bundled_model_files.dart'
    show bundledModelBytes;
import '../services/model_store/model_store.dart';

/// Which models are present for loading, for the Models screen: every model but
/// the chat model is built into the app, the chat slot holds the user's own
/// `.litertlm`, or `GEMMA_MODEL_PATH` while none is chosen:
/// [ModelSourceResolver]'s rule, the one `ModelRepository` loads by. Also
/// clears what earlier builds downloaded, once setup succeeded.
class ProvisioningRepository {
  ProvisioningRepository({
    required this._store,
    String gemmaModelPath = kGemmaModelPath,
    this._chatModels,
  }) : _sources = ModelSourceResolver(gemmaModelPath: gemmaModelPath);

  final ModelStore _store;

  /// Where the chat model's file comes from: the same rule setup loads by.
  final ModelSourceResolver _sources;

  /// What the chat slot ([ModelId.chat]) holds; null acts as none chosen
  /// (`GEMMA_MODEL_PATH`, else nothing).
  final ChatModelPlanner? _chatModels;

  /// True while the model store runs an import or download (the Chat model
  /// card's).
  ValueListenable<bool> get busy => _store.busy;

  /// Why [id] counts as present for loading, or not. The chat slot holds the
  /// user's own model when it is chosen, else `GEMMA_MODEL_PATH`, else nothing;
  /// every other model is built into the app.
  ModelPresence presenceOf(ModelId id) {
    if (id != ModelId.chat) return const PresentBundled();
    return switch (_sources.chat(
      _chatModels?.plan ?? const NoChatModelPlan(),
    )) {
      CustomChatSource(plan: CustomChatPlan(:final model)) =>
        PresentAsCustomChatModel(
          model.displayName,
          model.file?.sizeBytes ?? 0,
          where: model.source is LocalModelSource
              ? model.sourceLine
              : 'model store (custom/) · verified',
        ),
      BlockedChatSource(:final reason) => CustomChatModelBlocked(reason),
      DefineChatSource(:final path) => PresentByDefine(
        kGemmaModelPathDefine,
        path,
      ),
      NoChatSource(:final note) => ChatModelNotChosen(note: note),
    };
  }

  /// Every required model is present: loading can start.
  bool get requiredPresent => ModelId.values
      .where((id) => id.spec.required)
      .every((id) => isPresent(presenceOf(id)));

  /// [presence] lets the model load.
  static bool isPresent(ModelPresence presence) =>
      presence is! ChatModelNotChosen;

  /// The size of the files built-in model [id] ships in the app.
  static int bundledBytes(ModelId id) => bundledModelBytes(id);

  /// After a successful start, once: deletes the store folders earlier
  /// builds downloaded models into (the retired model manifest's, the
  /// retired Gemma 4 E2B download), keeping the chosen chat model's file in
  /// place. Skipped while the chat model is blocked (its file may be one of
  /// them).
  Future<List<String>> pruneOldModelFolders() async {
    if (_pruned) return const [];
    final Set<String> keep;
    switch (_chatModels?.plan) {
      case CustomChatPlan(:final path):
        keep = {path};
      case NoChatModelPlan() || null:
        keep = const {};
      case ChatPlanBlocked():
        return const [];
    }
    _pruned = true;
    return _store.pruneOldModelFolders(keepPaths: keep);
  }

  bool _pruned = false;
}
