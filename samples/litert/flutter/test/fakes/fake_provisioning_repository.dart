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
import 'package:litert_edge_demos/data/repositories/provisioning_repository.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/provisioning.dart';

/// [ProvisioningRepository] for widget tests: every model present (built
/// in, the chat model by `GEMMA_MODEL_PATH`) unless [presence] says
/// otherwise. No disk.
class FakeProvisioningRepository implements ProvisioningRepository {
  FakeProvisioningRepository({Map<ModelId, ModelPresence>? presence})
    : presence = presence ?? {};

  /// Overrides per model; the rest are present.
  final Map<ModelId, ModelPresence> presence;
  final ValueNotifier<bool> busyNotifier = ValueNotifier(false);

  @override
  ValueListenable<bool> get busy => busyNotifier;

  @override
  ModelPresence presenceOf(ModelId id) =>
      presence[id] ??
      (id == ModelId.chat
          ? const PresentByDefine('GEMMA_MODEL_PATH', '/store/chat/model')
          : const PresentBundled());

  @override
  bool get requiredPresent => ModelId.values
      .where((id) => id.spec.required)
      .every((id) => ProvisioningRepository.isPresent(presenceOf(id)));

  int prunes = 0;

  @override
  Future<List<String>> pruneOldModelFolders() async {
    prunes++;
    return const [];
  }
}
