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

import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show Skill, SkillResult, SkillType, TextResult;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/hardware/diagnostics_report.dart';
import 'package:litert_edge_demos/domain/models/accelerator_evidence.dart';
import 'package:litert_edge_demos/domain/models/hardware_profile.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/skills/app_intent_executor.dart';
import 'package:litert_edge_demos/domain/skills/app_intent_handlers.dart';
import 'package:litert_edge_demos/domain/skills/app_intents.dart';

Skill intent(String name) => Skill(
  name: name,
  description: '',
  instructions: '',
  type: SkillType.intent,
);

void main() {
  late DateTime now;
  late AppIntentExecutor executor;

  setUp(() {
    now = DateTime(2026, 10, 2, 14, 3, 7);
    executor = AppIntentExecutor(
      buildAppIntents(deviceFacts: () => 'facts', now: () => now),
    );
  });

  Future<SkillResult> run(String name, [String data = '{}']) =>
      executor.execute(intent(name), data);

  Future<String> text(String name, [String data = '{}']) async {
    final result = await run(name, data);
    expect(result, isA<TextResult>(), reason: '$result');
    return (result as TextResult).text;
  }

  test('the app has exactly these intents', () {
    expect(AppIntent.all, {'device_info', 'current_time'});
  });

  test('current_time: the app\'s clock, in the spec\'s format', () async {
    expect(
      await text(AppIntent.currentTime),
      'It is 2:03 PM on Friday, October 2, 2026.',
    );
    now = DateTime(2026, 1, 5, 0, 7);
    expect(
      await text(AppIntent.currentTime),
      'It is 12:07 AM on Monday, January 5, 2026.',
    );
  });

  test('device_info reports the facts source', () async {
    expect(await text(AppIntent.deviceInfo), 'facts');
  });

  group('describeHardware', () {
    ModelDiagnostics model(
      String name,
      String backend,
      EvidenceSource source, {
      ChatModelFacts? chat,
    }) => ModelDiagnostics(
      name: name,
      state: 'ready',
      evidence: AcceleratorEvidence(
        requested: backend,
        actual: backend,
        backendSource: source,
      ),
      chat: chat,
    );

    ChatModelFacts chat(String name, {bool custom = false}) => ChatModelFacts(
      name: name,
      custom: custom,
      source: custom ? 'imported' : 'bundled',
      requestedBackend: 'gpu',
      requestedContext: 4096,
      contextTokens: 4096,
      images: !custom,
      tools: !custom,
      modelType: 'gemma4',
    );

    const mac = HardwareProfile(
      platform: HostPlatform.macos,
      os: 'macOS 26.5.1',
      cpu: CpuInfo(model: 'Apple M4 Pro', cores: 14),
      memory: MemoryInfo(totalBytes: 24 << 30),
      gpus: [
        GpuInfo(
          name: 'Apple M4 Pro',
          source: 'chip name (sysctl)',
          inferred: true,
          kind: GpuKind.integrated,
          api: 'Metal',
        ),
      ],
    );

    test('the chip, memory and GPU, then where each model runs and how that '
        'is known, the chat model first; numbers in words', () {
      final facts = describeHardware(
        hardware: mac,
        models: [
          model('Whisper base', 'cpu', EvidenceSource.requested),
          model('Inflect nano', 'cpu', EvidenceSource.requested),
          model('YOLO26n', 'gpu', EvidenceSource.api),
          model('EmbeddingGemma 300M', 'cpu', EvidenceSource.requested),
          model(
            'Gemma 4 E2B',
            'gpu',
            EvidenceSource.api,
            chat: chat('Gemma 4 E2B'),
          ),
          const ModelDiagnostics(name: 'moonshine tiny', state: 'pending'),
        ],
      );

      expect(
        facts,
        'This Mac has an Apple M4 Pro with twenty-four gigabytes of memory '
        'and an Apple M4 Pro GPU (Metal). Gemma 4 E2B and the detector run '
        'on the GPU (confirmed); speech recognition, speech synthesis and the '
        'embedder on the CPU (requested).',
      );
      expect(facts, isNot(contains('24')));
      expect(facts, isNot(contains('moonshine')), reason: 'not loaded');
    });

    test('a phone: the SoC by its marketing name, your own chat model on the '
        'Qualcomm NPU', () {
      final facts = describeHardware(
        hardware: const HardwareProfile(
          platform: HostPlatform.android,
          os: 'Android 15',
          soc: SocInfo(
            manufacturer: 'QTI',
            model: 'SM8650',
            source: 'ro.soc.model',
            name: 'Snapdragon 8 Gen 3',
          ),
          cpu: CpuInfo(model: 'Snapdragon 8 Gen 3 (SM8650)', cores: 8),
          memory: MemoryInfo(totalBytes: 12 << 30),
          gpus: [GpuInfo(name: 'Adreno 750', source: 'GL', api: 'OpenCL')],
        ),
        models: [
          model('YOLO26n', 'gpu', EvidenceSource.api),
          model(
            'Gemma 4 E2B',
            'npu',
            EvidenceSource.api,
            chat: chat('Gemma 3 1B NPU', custom: true),
          ),
          model('Whisper base', 'cpu', EvidenceSource.requested),
        ],
      );

      expect(
        facts,
        'This phone has a Snapdragon 8 Gen 3 (SM8650) with twelve gigabytes '
        'of memory and an Adreno 750 GPU (OpenCL). Gemma 3 1B NPU runs on the '
        'Qualcomm NPU (confirmed); the detector on the GPU (confirmed); '
        'speech recognition on the CPU (requested).',
      );
    });

    test('an inferred backend is said as inferred; a VM display adapter is '
        'not a GPU', () {
      final facts = describeHardware(
        hardware: const HardwareProfile(
          platform: HostPlatform.linux,
          os: 'Ubuntu 22.04',
          cpu: CpuInfo(model: 'Intel Xeon', cores: 4),
          gpus: [
            GpuInfo(
              name: 'QEMU VGA',
              source: 'PCI',
              kind: GpuKind.virtualDisplay,
            ),
          ],
        ),
        models: [model('YOLO26n', 'cpu', EvidenceSource.inferred)],
      );

      expect(
        facts,
        'This computer has an Intel Xeon. The detector runs on the CPU '
        '(inferred).',
      );
    });

    test('no probe yet and nothing loaded says so', () {
      expect(
        describeHardware(hardware: null, models: const []),
        'No model is loaded yet.',
      );
    });

    test('an NPU on a phone the table does not know is just "the NPU"', () {
      final facts = describeHardware(
        hardware: const HardwareProfile(
          platform: HostPlatform.android,
          os: 'Android 14',
          soc: SocInfo(model: 'MT6989', source: 'ro.soc.model'),
          cpu: CpuInfo(model: 'MT6989', cores: 8),
        ),
        models: [model('Gemma 4 E2B', 'npu', EvidenceSource.api)],
      );
      expect(
        facts,
        'This phone has an MT6989. Gemma 4 E2B runs on the NPU (confirmed).',
        reason: 'an initialism takes the article of its first letter\'s name',
      );
    });

    test('Jetson: "an NVIDIA", and a GPU named "…iGPU…" is not called a GPU '
        'twice', () {
      final facts = describeHardware(
        hardware: const HardwareProfile(
          platform: HostPlatform.linux,
          os: 'Ubuntu 22.04',
          cpu: CpuInfo(model: '6× Cortex-A78AE', cores: 6),
          memory: MemoryInfo(totalBytes: 8 << 30),
          gpus: [
            GpuInfo(
              name: 'NVIDIA Orin iGPU (Ampere)',
              source: 'device tree (tegra234)',
              inferred: true,
              kind: GpuKind.integrated,
            ),
          ],
        ),
        models: [model('YOLO26n', 'gpu', EvidenceSource.log)],
      );

      expect(
        facts,
        'This computer has a 6× Cortex-A78AE with eight gigabytes of memory '
        'and an NVIDIA Orin iGPU (Ampere). The detector runs on the GPU '
        '(confirmed).',
      );
    });

    test('a core count takes the article of the number said', () {
      String said(String cpu) => describeHardware(
        hardware: HardwareProfile(
          platform: HostPlatform.linux,
          os: 'Ubuntu 22.04',
          cpu: CpuInfo(model: cpu, cores: 8),
        ),
        models: const [],
      ).split('.').first;

      expect(said('8× Cortex-A78AE'), 'This computer has an 8× Cortex-A78AE');
      expect(said('6× Cortex-A78AE'), 'This computer has a 6× Cortex-A78AE');
      expect(said('11× Neoverse'), 'This computer has an 11× Neoverse');
      expect(said('12× Cortex-A78AE'), 'This computer has a 12× Cortex-A78AE');
    });

    test('a GPU known only by its PCI id is called a GPU once', () {
      final facts = describeHardware(
        hardware: const HardwareProfile(
          platform: HostPlatform.linux,
          os: 'Ubuntu 24.04',
          cpu: CpuInfo(model: 'AMD EPYC 7B13', cores: 8),
          gpus: [
            GpuInfo(
              name: 'NVIDIA GPU 0x2b85',
              source: 'PCI 10de:2b85',
              kind: GpuKind.discrete,
            ),
          ],
        ),
        models: const [],
      );

      expect(
        facts,
        'This computer has an AMD EPYC 7B13 and an NVIDIA GPU 0x2b85. No '
        'model is loaded yet.',
      );
    });
  });
}
