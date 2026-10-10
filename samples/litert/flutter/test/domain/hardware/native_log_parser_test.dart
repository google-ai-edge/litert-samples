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
import 'package:litert_edge_demos/domain/hardware/accelerator_inference.dart';
import 'package:litert_edge_demos/domain/hardware/native_log_parser.dart';
import 'package:litert_edge_demos/domain/models/accelerator_evidence.dart';
import 'package:litert_edge_demos/domain/models/hardware_profile.dart';

// Captured by the macOS release self-test (Gemma 4 E2B on an M4 Pro).
const _macGemma = [
  'INFO: [accelerator_registry.cc:54] RegisterAccelerator: ptr=0xb4dc50300, name=GPU Metal',
  'INFO: [gpu_registry.cc:144] Dynamically loaded GPU accelerator(@executable_path/../Frameworks/LiteRtMetalAccelerator.framework/LiteRtMetalAccelerator) registered.',
  'I0000 00:00:1791240441.593498 19514714 delegate_metal.mm:89] Created a Metal device.',
  'INFO: [gpu_environment.cc:367] Failed to create OpenCL context.',
  'INFO: [gpu_environment.cc:374] Created Metal device from provided device id',
  'INFO: Created TensorFlow Lite XNNPACK delegate for CPU.',
  'W0000 00:00:1791240447.020853 19514920 sampler_factory.cc:771] GPU sampler unavailable. Falling back to CPU sampling.',
  'INFO: [litert_lm_loader.cc:120] unrelated line',
];

// From runs on a Tesla T4 VM.
const _t4 =
    'I0000 00:00:1759700000.000000 12345 environment.cc:526] Selected adapter: '
    'Tesla T4, arch=turing, vendor=nvidia, backend=Vulkan, '
    'adapterType=Discrete GPU';

const _profile = HardwareProfile(
  platform: HostPlatform.linux,
  os: 'Ubuntu 22.04.5 LTS',
  cpu: CpuInfo(model: 'x', cores: 4),
  gpus: [GpuInfo(name: 'Tesla T4', source: '/proc/driver/nvidia')],
);

void main() {
  test('macOS: Metal confirmed, CPU delegate and CPU sampler noticed, the '
      'OpenCL failure and other lines ignored', () {
    final log = parseNativeLog(_macGemma);
    expect(log.api, 'Metal');
    expect(log.adapter, isNull);
    expect(log.cpuDelegate, isTrue);
    expect(log.samplerOnCpu, isTrue);
    expect(log.softwareGpu, isFalse);
    expect(log.lines, hasLength(6));
    expect(log.lines.any((l) => l.contains('OpenCL')), isFalse);
  });

  test('WebGPU adapter line: name, type and the API from the backend; with a '
      'CRLF ending and a logcat prefix', () {
    for (final line in [
      _t4,
      '$_t4\r',
      '10-06 12:00:00.000  1234  1234 I litert: $_t4',
    ]) {
      final log = parseNativeLog([line]);
      expect(log.adapter!.name, 'Tesla T4');
      expect(log.adapter!.vendor, 'nvidia');
      expect(log.adapter!.backend, 'Vulkan');
      expect(log.adapter!.adapterType, 'Discrete GPU');
      expect(log.api, 'WebGPU/Vulkan');
      expect(log.softwareGpu, isFalse);
    }
  });

  test('llvmpipe or adapterType=CPU is a software GPU', () {
    final llvmpipe = parseNativeLog([
      'Selected adapter: llvmpipe (LLVM 19.1.1, 256 bits), arch=, vendor=mesa, '
          'backend=Vulkan, adapterType=CPU',
    ]);
    expect(llvmpipe.softwareGpu, isTrue);
    expect(llvmpipe.adapter!.arch, isNull);
    final swiftShader = parseNativeLog([
      'Selected adapter: SwiftShader Device (Subzero), arch=, vendor=google, '
          'backend=Vulkan, adapterType=Integrated GPU',
    ]);
    expect(swiftShader.softwareGpu, isTrue);
  });

  test('no GPU accelerator', () {
    final log = parseNativeLog([
      'WARNING: GPU accelerator could not be loaded',
    ]);
    expect(log.noGpu, isTrue);
    expect(log.api, isNull);
  });

  test('problem lines after a failed load: glog E/W/F, ERROR, failed; the '
      'last 20', () {
    final lines = [
      'INFO: [gpu_registry.cc:136] Attempting to load GPU accelerator',
      'E0000 00:00:1 webgpu.cc:12] Failed to create a Vulkan instance',
      'W0000 00:00:1 sampler_factory.cc:771] GPU sampler unavailable.',
      'ERROR: [litert_lm.cc:9] engine_create failed',
      'ordinary line',
      for (var i = 0; i < 30; i++) 'E0000 repeated $i',
    ];
    final problems = nativeProblemLines(lines);
    expect(problems, hasLength(20));
    expect(problems.last, 'E0000 repeated 29');
    expect(nativeProblemLines(lines.take(5)), hasLength(3));
  });

  group('evidence', () {
    test('a log line beats the inference and is labelled as such', () {
      final e = inferEvidence(
        requested: 'gpu',
        reported: 'gpu',
        hardware: _profile,
        log: parseNativeLog([_t4]),
      );
      expect(e.backendSource, EvidenceSource.api);
      expect(e.api, 'WebGPU/Vulkan');
      expect(e.apiSource, EvidenceSource.log);
      expect(e.adapter, 'Tesla T4 (Discrete GPU)');
      expect(e.adapterSource, EvidenceSource.log);
      expect(e.isError, isFalse);
    });

    test('a confirmed software adapter is an error even when the probe saw '
        'a real GPU', () {
      final e = inferEvidence(
        requested: 'gpu',
        reported: 'gpu',
        hardware: _profile,
        log: parseNativeLog([
          'Selected adapter: llvmpipe, arch=, vendor=mesa, backend=Vulkan, '
              'adapterType=CPU',
        ]),
      );
      expect(e.softwareGpu, isTrue);
      expect(e.isError, isTrue);
    });

    test('not reportable → requested; a reported other backend → mismatch', () {
      final speech = inferEvidence(
        requested: 'cpu',
        reported: null,
        hardware: _profile,
      );
      expect(speech.backendSource, EvidenceSource.requested);
      expect(speech.mismatch, isFalse);
      final fallback = inferEvidence(
        requested: 'gpu',
        reported: 'cpu',
        hardware: _profile,
      );
      expect(fallback.mismatch, isTrue);
      expect(fallback.isError, isTrue);
    });
  });
}
