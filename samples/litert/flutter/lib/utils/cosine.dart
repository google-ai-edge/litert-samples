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

import 'dart:math' as math;

/// Cosine similarity of two vectors of the same length: 1 = same direction.
/// Throws an [ArgumentError] for different lengths, an empty vector or a
/// zero vector (no direction to compare); a value that is NaN or infinite
/// gives NaN. A caller that must neither throw nor see NaN checks each
/// vector with [vectorDefect] first.
double cosineSimilarity(List<double> a, List<double> b) {
  if (a.length != b.length) {
    throw ArgumentError('vectors of ${a.length} and ${b.length} values');
  }
  if (a.isEmpty) throw ArgumentError('empty vectors');
  var dot = 0.0;
  var na = 0.0;
  var nb = 0.0;
  for (var i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na == 0 || nb == 0) throw ArgumentError('a zero vector');
  return dot / (math.sqrt(na) * math.sqrt(nb));
}

/// Why [vector] cannot take part in a cosine: "no values", "a value that is
/// not a number", "an infinite value", "no direction (all zeros)" (also when
/// its squares underflow to zero) or "a length too large to compute" (its
/// squares overflow); null when it can. Two vectors that both pass, of the
/// same length, give [cosineSimilarity] a finite result and never throw.
String? vectorDefect(List<double> vector) {
  if (vector.isEmpty) return 'no values';
  var squares = 0.0;
  for (final v in vector) {
    if (v.isNaN) return 'a value that is not a number';
    if (v.isInfinite) return 'an infinite value';
    squares += v * v;
  }
  if (squares == 0) return 'no direction (all zeros)';
  if (squares.isInfinite) return 'a length too large to compute';
  return null;
}
