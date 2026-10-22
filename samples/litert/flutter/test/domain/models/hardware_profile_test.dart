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
import 'package:litert_edge_demos/domain/hardware/native_log_parser.dart';
import 'package:litert_edge_demos/domain/models/hardware_profile.dart';

void main() {
  group('isSoftwareGpuName', () {
    test("Mesa's and Google's CPU rasterizers", () {
      for (final name in [
        'llvmpipe (LLVM 15.0.7, 256 bits)',
        'lavapipe',
        'SwiftShader Device (Subzero)',
        'softpipe',
      ]) {
        expect(isSoftwareGpuName(name), isTrue, reason: name);
      }
    });

    test("Windows' WARP: the Basic Render Driver and the Basic Display "
        'Adapter, in any case', () {
      for (final name in [
        'Microsoft Basic Render Driver',
        'Microsoft Basic Display Adapter',
        'microsoft basic render driver',
      ]) {
        expect(isSoftwareGpuName(name), isTrue, reason: name);
      }
    });

    test('real GPUs and the Remote Desktop adapter are not software '
        'rasterizers', () {
      for (final name in [
        'Intel(R) Arc(TM) 140V GPU (16GB)',
        'Intel(R) Arc(TM) B390 GPU',
        'NVIDIA GeForce RTX 4060 Laptop GPU',
        'Microsoft Remote Display Adapter',
        'Tesla T4',
      ]) {
        expect(isSoftwareGpuName(name), isFalse, reason: name);
      }
    });
  });

  test('a WebGPU adapter line naming WARP counts as a software GPU', () {
    final log = parseNativeLog([
      'Selected adapter: Microsoft Basic Render Driver, arch=, '
          'vendor=microsoft, backend=D3D12, adapterType=Integrated GPU',
    ]);
    expect(log.softwareGpu, isTrue);
  });
}
