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
import 'package:litert_edge_demos/data/services/detector/detector_verification.dart';

void main() {
  test('the ramp is the one verifyCompiledModel feeds: (i % 251) / 251', () {
    final ramp = verificationRamp(600);
    expect(ramp[0], 0);
    expect(ramp[250], closeTo(250 / 251, 1e-7));
    expect(ramp[251], 0);
    expect(ramp[599], closeTo((599 % 251) / 251, 1e-7));
  });

  test('float noise far below 1% of the range agrees', () {
    final v = compareOutputs([0, 432, 864], [0.001, 432.002, 863.999]);
    expect(v.agrees, isTrue);
    expect(v.outputRange, 864);
    expect(v.absoluteDeviation, closeTo(0.002, 1e-9));
    expect(v.relativeDeviation, closeTo(0.002 / 864, 1e-12));
    expect(v.skipped, isFalse);
  });

  test('a deviation above 1% of the range disagrees', () {
    final v = compareOutputs([0, 100], [0, 98.9]);
    expect(v.agrees, isFalse);
    expect(v.relativeDeviation, closeTo(0.011, 1e-9));
  });

  test('NaN or infinity on one side only never agrees', () {
    expect(compareOutputs([0, 1], [0, double.nan]).agrees, isFalse);
    expect(compareOutputs([0, 1], [double.infinity, 1]).agrees, isFalse);
    expect(
      compareOutputs([0, 1], [0, double.nan]).absoluteDeviation,
      double.infinity,
    );
  });

  test('a length mismatch never agrees and says why', () {
    final v = compareOutputs([0, 1, 2], [0, 1]);
    expect(v.agrees, isFalse);
    expect(v.skippedReason, contains('length mismatch'));
  });

  test('a constant reference is scaled by its magnitude, as upstream does', () {
    expect(outputRange([5, 5]), 0);
    expect(compareOutputs([5, 5], [5, 5]).agrees, isTrue);
    expect(
      compareOutputs([5, 5], [5, 5.1]).relativeDeviation,
      closeTo(0.02, 1e-9),
    );
    expect(compareOutputs([0, 0], [0, 0.1]).agrees, isFalse);
  });

  test('the range ignores non-finite values', () {
    expect(outputRange([double.nan, -2, 3, double.infinity]), 5);
    expect(outputRange(const []), 0);
  });
}
