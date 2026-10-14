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
}
