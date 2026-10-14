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

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/live_camera_config.dart';
import 'package:litert_edge_demos/data/repositories/visual_memory/memory_sampler.dart';

const _s = 1000000; // µs

void main() {
  test('the defaults are the spec\'s: 3 s, 4 × the embed time, 10 s', () {
    expect(kMemoryMinInterval, const Duration(seconds: 3));
    expect(kMemoryEmbedSpacing, 4);
    expect(kMemoryMaxInterval, const Duration(seconds: 10));
  });

  test('the first attempt is due at once and forced (nothing stored)', () {
    final sampler = MemorySampler();
    expect(sampler.due(0), isTrue);
    expect(sampler.forced(0), isTrue);
  });

  test('the next attempt waits 3 s after the previous one ended', () {
    final sampler = MemorySampler()..attemptEnded(10 * _s);
    expect(sampler.due(10 * _s + 2999999), isFalse);
    expect(sampler.due(13 * _s), isTrue);
  });

  test('a slow embedder spaces attempts at 4 × its time', () {
    final sampler = MemorySampler()
      ..attemptEnded(0, embedTime: const Duration(milliseconds: 1500));
    expect(sampler.gap, const Duration(seconds: 6));
    expect(sampler.due(5 * _s), isFalse);
    expect(sampler.due(6 * _s), isTrue);
    // A fast embedder: the 3 s floor applies.
    sampler.attemptEnded(10 * _s, embedTime: const Duration(milliseconds: 200));
    expect(sampler.gap, const Duration(seconds: 3));
    expect(sampler.lastEmbed, const Duration(milliseconds: 200));
  });

  test('an attempt without an embedding (unchanged, dropped) keeps the last '
      'embed time', () {
    final sampler = MemorySampler()
      ..attemptEnded(0, embedTime: const Duration(seconds: 1))
      ..attemptEnded(4 * _s);
    expect(sampler.gap, const Duration(seconds: 4));
  });

  test('a store is forced 10 s after the last stored frame', () {
    final sampler = MemorySampler()..stored(20 * _s);
    expect(sampler.forced(29 * _s), isFalse);
    expect(sampler.forced(30 * _s), isTrue);
  });

  test('reset starts afresh but keeps the device\'s embed time', () {
    final sampler = MemorySampler()
      ..attemptEnded(5 * _s, embedTime: const Duration(seconds: 2))
      ..stored(5 * _s)
      ..reset();
    expect(sampler.due(5 * _s), isTrue);
    expect(sampler.forced(5 * _s), isTrue);
    expect(sampler.lastEmbed, const Duration(seconds: 2));
  });
}
