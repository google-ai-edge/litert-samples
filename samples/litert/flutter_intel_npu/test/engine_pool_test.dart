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

import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:intel_npu_demo/core/engines/engine_pool.dart';
import 'package:intel_npu_demo/core/engines/engine_slot.dart';
import 'package:flutter_test/flutter_test.dart';

/// A slot whose load and close finish only when the test says so, and which
/// behaves like LiteRtLmEngineSlot: unload waits for a load in flight.
class _Slot implements EngineSlot {
  _Slot(this.backend, this.id, this.log);

  @override
  final PreferredBackend backend;
  final int id;
  final List<String> log;
  final loadGate = Completer<void>();
  final closeGate = Completer<void>();
  Future<void>? _loading;
  bool _loaded = false;

  @override
  PreferredBackend? get activeBackend => _loaded ? backend : null;
  @override
  bool get isLoaded => _loaded;
  @override
  Duration? get loadTime => null;

  @override
  Future<void> load({required String modelPath, required int maxTokens}) => _loading ??= () async {
        log.add('load $id');
        await loadGate.future;
        _loaded = true;
        log.add('loaded $id');
      }();

  @override
  Future<void> unload() async {
    await _loading;
    if (!_loaded) return;
    await closeGate.future;
    _loaded = false;
    log.add('closed $id');
  }

  @override
  Future<EngineTurn> openTurn({double temperature = 0.8, int topK = 1, int? maxOutputTokens}) =>
      throw UnimplementedError();
  @override
  Future<InferenceChat> openChat({
    required List<Tool> tools,
    required String systemInstruction,
    required bool enableThinking,
    required double temperature,
    required int topK,
    required int randomSeed,
    int? maxOutputTokens,
  }) =>
      throw UnimplementedError();
}

void main() {
  late List<String> log;
  late List<_Slot> slots;
  EnginePool pool(EnginePolicy policy) => EnginePool(
        policy: policy,
        modelPath: '/m/12b.litertlm',
        newSlot: (b) {
          final s = _Slot(b, slots.length, log);
          slots.add(s);
          return s;
        },
      );

  setUp(() {
    log = [];
    slots = [];
  });

  test('leaving mid-load and coming back loads the engine once the old copy is closed', () async {
    final p = pool(EnginePolicy.resident);
    final first = p.acquire(PreferredBackend.gpu, maxTokens: 16384);
    await pumpEventQueue();
    final released = p.release(PreferredBackend.gpu, maxTokens: 16384);
    final second = p.acquire(PreferredBackend.gpu, maxTokens: 16384);
    await pumpEventQueue();
    expect(log, ['load 0'], reason: 'the second load must wait for the first copy to close');

    slots[0].loadGate.complete();
    await first;
    await pumpEventQueue();
    expect(log, ['load 0', 'loaded 0'], reason: 'release closes what the load opened');
    slots[0].closeGate.complete();
    await released;
    await pumpEventQueue();
    expect(log, ['load 0', 'loaded 0', 'closed 0', 'load 1']);

    slots[1].loadGate.complete();
    expect((await second).isLoaded, isTrue);
  });

  test('sequential closes an engine that is still loading before it loads the next', () async {
    final p = pool(EnginePolicy.sequential);
    unawaited(p.acquire(PreferredBackend.cpu, maxTokens: 4096));
    await pumpEventQueue();
    final gpu = p.acquire(PreferredBackend.gpu, maxTokens: 4096);
    await pumpEventQueue();
    expect(log, ['load 0']);

    slots[0].loadGate.complete();
    await pumpEventQueue();
    slots[0].closeGate.complete();
    await pumpEventQueue();
    expect(log, ['load 0', 'loaded 0', 'closed 0', 'load 1']);
    slots[1].loadGate.complete();
    await gpu;
    expect(p.resident.keys.map((k) => k.backend), [PreferredBackend.gpu]);
  });

  test('releaseAll closes a slot that is still loading', () async {
    final p = pool(EnginePolicy.resident);
    unawaited(p.acquire(PreferredBackend.npu, maxTokens: 4096));
    await pumpEventQueue();
    final all = p.releaseAll();
    slots[0].loadGate.complete();
    await pumpEventQueue();
    slots[0].closeGate.complete();
    await all;
    expect(log, ['load 0', 'loaded 0', 'closed 0']);
    expect(p.resident, isEmpty);
  });
}
