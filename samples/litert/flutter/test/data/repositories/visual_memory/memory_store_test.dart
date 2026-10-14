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

import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/live_camera_config.dart';
import 'package:litert_edge_demos/data/repositories/visual_memory/memory_store.dart';
import 'package:litert_edge_demos/domain/models/visual_memory.dart';

MemoryEntry entry(int id, {int bytes = 100, List<double>? vector}) =>
    MemoryEntry(
      id: id,
      frameId: id,
      atMicros: id * 1000000,
      png: Uint8List(bytes),
      width: 320,
      height: 180,
      counts: const {},
      vector: Float32List.fromList(vector ?? [1, 0]),
      grey: Uint8List(1024),
    );

void main() {
  test('the caps are 200 frames and 16 MB of thumbnails', () {
    expect(kMemoryMaxEntries, 200);
    expect(kMemoryMaxBytes, 16 * 1024 * 1024);
  });

  test('a frame seen again unchanged: the newest frame\'s time is refreshed '
      'in place, nothing is added and the bytes stay', () {
    final store = MemoryStore(maxEntries: 3, maxBytes: 1 << 20);
    expect(store.refreshNewest(9000000), isFalse, reason: 'nothing stored');
    store
      ..add(entry(0))
      ..add(entry(1, vector: [0, 1]));
    expect(store.refreshNewest(9000000), isTrue);
    expect(store.length, 2);
    expect(store.bytes, 200);
    final newest = store.newest!;
    expect(newest.id, 1);
    expect(newest.atMicros, 9000000);
    expect(newest.vector, [0, 1]);
    expect(store.oldest!.atMicros, 0, reason: 'only the newest');
  });

  test('over the frame cap the oldest go first', () {
    final store = MemoryStore(maxEntries: 3, maxBytes: 1 << 20);
    for (var i = 0; i < 3; i++) {
      expect(store.add(entry(i)), 0);
    }
    expect(store.add(entry(3)), 1);
    expect([for (final e in store.entries) e.id], [1, 2, 3]);
    expect(store.oldest!.id, 1);
    expect(store.newest!.id, 3);
    expect(store.bytes, 300);
  });

  test('over the byte cap (measured PNG lengths) the oldest go first, as '
      'many as it takes', () {
    final store = MemoryStore(maxEntries: 200, maxBytes: 1000);
    store
      ..add(entry(0, bytes: 400))
      ..add(entry(1, bytes: 400));
    expect(store.add(entry(2, bytes: 900)), 2);
    expect([for (final e in store.entries) e.id], [2]);
    expect(store.bytes, 900);
  });

  test('a frame larger than the byte cap on its own is not kept', () {
    final store = MemoryStore(maxEntries: 200, maxBytes: 100);
    expect(store.add(entry(0, bytes: 101)), 1);
    expect(store.isEmpty, isTrue);
    expect(store.bytes, 0);
  });

  test('cosines are per frame, oldest first; clear empties it', () {
    final store = MemoryStore()
      ..add(entry(0, vector: [1, 0]))
      ..add(entry(1, vector: [0, 1]))
      ..add(entry(2, vector: [1, 1]));
    final c = store.cosines([1, 0]);
    expect(c[0], closeTo(1, 1e-9));
    expect(c[1], closeTo(0, 1e-9));
    expect(c[2], closeTo(0.7071, 1e-4));
    store.clear();
    expect(store.length, 0);
    expect(store.bytes, 0);
    expect(store.oldest, isNull);
  });
}
