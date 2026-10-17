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
import 'package:intel_npu_demo/core/transcript/transcript.dart';

void main() {
  final t = Transcript.parse(
    '[02:10] Maya: We start with the budget.\n'
    '[02:13] Jonas: The ticket is 7413-QX.\n'
    '[02:20] Priya: Noted.\n',
    source: 'test',
  );

  test('a stamp the transcript has links to that line', () {
    final c = t.citationsIn('Jonas gave the ticket at [02:13].').single;
    expect(c.segment?.speaker, 'Jonas');
  });

  test('a stamp the model made up stays unresolved', () {
    // It used to resolve to the line before it, [02:13], and look verified.
    final c = t.citationsIn('He repeated it at 02:14.').single;
    expect(c.stamp, '02:14');
    expect(c.segment, isNull);
  });

  test('each stamp is cited once, in order', () {
    final cs = t.citationsIn('[02:20] then [02:10] and again [02:20]');
    expect(cs.map((c) => c.stamp), ['02:20', '02:10']);
  });
}
