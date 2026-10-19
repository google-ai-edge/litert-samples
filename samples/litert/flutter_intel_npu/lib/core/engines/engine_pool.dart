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

import 'package:flutter_edge_ai/flutter_edge_ai.dart';

import 'engine_slot.dart';

/// `resident`: every backend stays loaded, switching is instant. Backed by
/// one raw `LiteRtLmFfiClient` per backend ([FfiEngineSlot]); validated on
/// macOS (CPU + GPU) and Windows on Lunar Lake (CPU + GPU + NPU).
///
/// `sequential`: one engine at a time through the public flutter_edge_ai API
/// ([LiteRtLmEngineSlot]); switching costs a reload (2–10 s cold).
enum EnginePolicy { resident, sequential }

/// Identity of one engine: the same model file on the same backend with the
/// same context window. Two acts asking for different windows or models get
/// different engines even on the same backend.
typedef EngineKey = ({PreferredBackend backend, String modelPath, int maxTokens});

class EnginePool {
  EnginePool({required this.policy, required this.modelPath});

  final EnginePolicy policy;

  /// Default model file (Act 1 / Act 3). Acts may pass their own.
  final String modelPath;
  final Map<EngineKey, EngineSlot> _slots = {};

  Map<EngineKey, EngineSlot> get resident => Map.unmodifiable({
        for (final e in _slots.entries)
          if (e.value.isLoaded) e.key: e.value,
      });

  /// Returns a loaded slot for [backend] (and [modelPath] / [maxTokens]),
  /// loading it — and, under `sequential`, unloading everything else — as
  /// needed.
  Future<EngineSlot> acquire(
    PreferredBackend backend, {
    required int maxTokens,
    String? modelPath,
  }) async {
    final key = (backend: backend, modelPath: modelPath ?? this.modelPath, maxTokens: maxTokens);
    final existing = _slots[key];
    if (existing != null && existing.isLoaded) return existing;

    if (policy == EnginePolicy.sequential) {
      for (final slot in _slots.values) {
        if (slot.isLoaded) await slot.unload();
      }
    }

    final slot = _slots.putIfAbsent(key, () => _newSlot(backend));
    await slot.load(modelPath: key.modelPath, maxTokens: maxTokens);
    return slot;
  }

  /// Unloads one engine (e.g. an act's contrast engine when leaving the act).
  Future<void> release(PreferredBackend backend, {required int maxTokens, String? modelPath}) async {
    final key = (backend: backend, modelPath: modelPath ?? this.modelPath, maxTokens: maxTokens);
    final slot = _slots.remove(key);
    if (slot != null && slot.isLoaded) await slot.unload();
  }

  Future<void> releaseAll() async {
    for (final slot in _slots.values) {
      if (slot.isLoaded) await slot.unload();
    }
  }

  EngineSlot _newSlot(PreferredBackend backend) => switch (policy) {
        EnginePolicy.resident => FfiEngineSlot(backend),
        EnginePolicy.sequential => LiteRtLmEngineSlot(backend),
      };
}
