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
import 'package:litert_edge_demos/data/services/llm/npu_availability.dart';
import 'package:litert_edge_demos/domain/models/npu_availability.dart';

/// The app's copy of flutter_edge_ai_litertlm 1.9.0's NPU gate
/// (`backend_preference.dart:54` and `:108`).
void main() {
  ({bool opened, String? error}) opens() => (opened: true, error: null);
  ({bool opened, String? error}) fails() => (
    opened: false,
    error: 'dlopen failed: library "libcdsprpc.so" not found',
  );

  test('Android: available when FastRPC opens, with the SoC', () {
    final npu = npuAvailabilityFor(
      'android',
      openFastRpc: opens,
      soc: 'QTI SM8750',
    );
    expect(npu, isA<NpuAvailable>());
    expect(describeNpu(npu), contains('SM8750'));
  });

  test("Android without FastRPC: flutter_edge_ai's reason, with the dlopen "
      'error', () {
    final npu = npuAvailabilityFor('android', openFastRpc: fails);
    expect(npu, isA<NpuUnavailable>());
    final reason = (npu as NpuUnavailable).reason;
    expect(reason, startsWith('this device has no Qualcomm FastRPC'));
    expect(reason, contains('library "libcdsprpc.so" not found'));
  });

  test('macOS, iOS and Linux ship no NPU stack; the probe never runs', () {
    for (final os in ['macos', 'ios', 'linux']) {
      final npu = npuAvailabilityFor(
        os,
        openFastRpc: () => fail('no dlopen on $os'),
      );
      expect(
        (npu as NpuUnavailable).reason,
        'no NPU dispatch stack ships for $os',
      );
    }
  });

  test('Windows is gated per OS, as flutter_edge_ai does', () {
    expect(
      npuAvailabilityFor('windows', openFastRpc: () => fail('no probe')),
      isA<NpuAvailable>(),
    );
  });

  test('this test host (macOS/Linux CI) reports the NPU unavailable', () {
    expect(probeNpu(), isA<NpuUnavailable>());
  });
}
