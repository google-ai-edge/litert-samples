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
import 'package:litert_edge_demos/utils/cosine.dart';

void main() {
  test('same direction 1, opposite -1, orthogonal 0, scale ignored', () {
    expect(cosineSimilarity([1, 2, 3], [2, 4, 6]), closeTo(1, 1e-12));
    expect(cosineSimilarity([1, 0], [-3, 0]), closeTo(-1, 1e-12));
    expect(cosineSimilarity([1, 0], [0, 5]), closeTo(0, 1e-12));
    expect(
      cosineSimilarity([1, 1], [1, 0]),
      closeTo(0.7071067811865476, 1e-12),
    );
  });

  test('different lengths, empty or zero vectors are refused', () {
    expect(() => cosineSimilarity([1], [1, 2]), throwsArgumentError);
    expect(() => cosineSimilarity([], []), throwsArgumentError);
    expect(() => cosineSimilarity([0, 0], [1, 2]), throwsArgumentError);
  });

  test('a NaN or infinite value gives NaN (callers check vectorDefect)', () {
    expect(cosineSimilarity([double.nan, 1], [1, 1]), isNaN);
    expect(cosineSimilarity([double.infinity, 1], [1, 1]), isNaN);
  });

  test('vectorDefect names what no cosine can use, null otherwise', () {
    expect(vectorDefect([0.1, -2, 3]), isNull);
    expect(vectorDefect([1e-150]), isNull, reason: 'its square is 1e-300');
    expect(vectorDefect(const []), 'no values');
    expect(vectorDefect([1, double.nan]), 'a value that is not a number');
    expect(vectorDefect([double.negativeInfinity]), 'an infinite value');
    expect(vectorDefect([0, 0, 0]), 'no direction (all zeros)');
    expect(vectorDefect([1e-200, 0]), 'no direction (all zeros)');
    expect(vectorDefect([1e200, 1e200]), 'a length too large to compute');
    // Two vectors that pass give a finite cosine, never a throw.
    expect(cosineSimilarity([1e-150, 0], [0, 1e150]).isFinite, isTrue);
  });
}
