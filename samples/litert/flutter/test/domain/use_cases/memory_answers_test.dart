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
import 'package:litert_edge_demos/domain/models/visual_memory.dart';
import 'package:litert_edge_demos/domain/use_cases/memory_answers.dart';
import 'package:litert_edge_demos/domain/vision/coco_vocabulary.dart';

int cls(String name) => kCocoNames.indexOf(name);

MemoryHit hit({
  Map<int, int> counts = const {},
  int seconds = 12,
  bool ambiguous = false,
}) => MemoryHit(
  entry: MemoryEntry(
    id: 1,
    frameId: 42,
    atMicros: 0,
    png: Uint8List(4),
    width: 320,
    height: 180,
    counts: counts,
    vector: Float32List(3),
    grey: Uint8List(1024),
  ),
  cosine: 0.74,
  age: Duration(seconds: seconds),
  ambiguous: ambiguous,
);

void main() {
  const answers = MemoryAnswers();

  test('nothing remembered yet', () {
    expect(
      MemoryAnswers.empty,
      "I don't remember anything yet: keep the camera running for a few "
      'seconds.',
    );
  });

  test('not seen: how far back the memory reaches, in seconds or minutes', () {
    expect(
      answers.notSeen(const Duration(seconds: 20)),
      "I haven't seen that in the last 20 seconds.",
    );
    expect(
      answers.notSeen(const Duration(seconds: 70)),
      "I haven't seen that in the last minute.",
    );
    expect(
      answers.notSeen(const Duration(minutes: 3, seconds: 10)),
      "I haven't seen that in the last 3 minutes.",
    );
    expect(
      answers.notSeen(Duration.zero),
      "I haven't seen that in the last 1 second.",
    );
  });

  test('switched off in the camera settings', () {
    expect(MemoryAnswers.disabled, 'Visual memory is off (camera settings).');
  });

  test('ambiguous: maybe in more than one place, the latest first', () {
    expect(
      answers.ambiguousLead(hit(ambiguous: true)),
      'I may have seen that in more than one place; the latest was 12 '
      'seconds ago.',
    );
    expect(
      answers.seen(hit(ambiguous: true, counts: {cls('cup'): 1})),
      'I may have seen that in more than one place; the latest was 12 '
      'seconds ago. In that frame the detector saw a cup.',
    );
    expect(
      answers.prompt('Where was the sign?', hit(ambiguous: true)),
      'This frame is from the camera 12 seconds ago: the most recent of '
      'several frames from different places that matched the question about '
      'equally well.\n\nQuestion: Where was the sign?',
    );
  });

  test('off: the reason, as a sentence', () {
    expect(
      answers.off('EmbeddingGemma is still loading.'),
      'My visual memory is off: EmbeddingGemma is still loading.',
    );
  });

  test('seen (images off): when, and what the detector saw in that frame', () {
    expect(
      answers.seen(hit(counts: {cls('cat'): 2, cls('remote'): 1})),
      'I saw it 12 seconds ago. In that frame the detector saw two cats and '
      'a remote.',
    );
    expect(
      answers.seen(hit(seconds: 1)),
      "I saw it a moment ago. The detector didn't recognize anything in that "
      'frame.',
    );
  });

  test(
    'the prompt: the frame\'s age, the detector hint, then the question',
    () {
      expect(
        answers.prompt(' Where was the mug? ', hit(counts: {cls('cup'): 1})),
        'This frame is from the camera 12 seconds ago: the best match in its '
        'visual memory for the question. Objects a detector found in it (may '
        'be incomplete): a cup.\n\nQuestion: Where was the mug?',
      );
      expect(
        answers.prompt('Where was the sign?', hit()),
        'This frame is from the camera 12 seconds ago: the best match in its '
        'visual memory for the question.\n\nQuestion: Where was the sign?',
      );
    },
  );

  test('ages and the chip label', () {
    expect(agoWords(const Duration(seconds: 2)), 'a moment ago');
    expect(agoWords(const Duration(seconds: 45)), '45 seconds ago');
    expect(agoWords(const Duration(seconds: 80)), 'about a minute ago');
    expect(agoWords(const Duration(minutes: 5)), 'about 5 minutes ago');
    expect(agoLabel(const Duration(seconds: 12)), '12 s ago');
    expect(agoLabel(const Duration(minutes: 2, seconds: 10)), '2 min ago');
    expect(
      memorySpanLabel(37, const Duration(minutes: 2)),
      '37 frames · 2 min',
    );
    expect(memorySpanLabel(1, const Duration(seconds: 8)), '1 frame · 8 s');
  });
}
