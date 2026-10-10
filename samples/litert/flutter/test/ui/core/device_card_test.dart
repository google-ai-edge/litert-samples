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

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/build_info.dart';
import 'package:litert_edge_demos/data/repositories/hardware_repository.dart';
import 'package:litert_edge_demos/data/repositories/model_repository.dart';
import 'package:litert_edge_demos/data/services/hardware/android_hardware_info_service.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/domain/hardware/accelerator_inference.dart';
import 'package:litert_edge_demos/domain/hardware/device_summary.dart';
import 'package:litert_edge_demos/domain/hardware/diagnostics_report.dart';
import 'package:litert_edge_demos/domain/hardware/native_log_parser.dart';
import 'package:litert_edge_demos/domain/models/hardware_profile.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/npu_availability.dart';
import 'package:litert_edge_demos/ui/core/device_card.dart';
import 'package:litert_edge_demos/ui/core/warning_color.dart';
import 'package:litert_edge_demos/ui/features/home/view_models/home_view_model.dart';
import 'package:litert_edge_demos/ui/features/home/views/home_screen.dart';
import 'package:provider/provider.dart';

import '../../fakes/fake_bundled_files.dart';
import '../../fakes/fake_detector_runtime.dart';
import '../../fakes/fake_hardware.dart';
import '../../fakes/fake_llm_service.dart';
import '../../fakes/fake_model_files.dart';
import '../../fakes/fake_speech.dart';
import '../../support/android_devices.dart';

ModelDiagnostics _gemma(
  HardwareProfile hw, {
  String reported = 'gpu',
  List<String> log = const [],
}) => ModelDiagnostics(
  name: 'Gemma 4 E2B',
  state: 'ready',
  evidence: inferEvidence(
    requested: 'gpu',
    reported: reported,
    hardware: hw,
    log: parseNativeLog(log),
  ),
);

Future<void> _pump(
  WidgetTester tester,
  DeviceSummary summary, {
  String Function()? report,
}) => tester.pumpWidget(
  MaterialApp(
    home: Scaffold(
      body: SingleChildScrollView(
        child: DeviceCard(summary: summary, report: report ?? () => 'report'),
      ),
    ),
  ),
);

Color? _colorOf(WidgetTester tester, String text) =>
    tester.widget<Text>(find.text(text)).style?.color;

const _t4 = HardwareProfile(
  platform: HostPlatform.linux,
  os: 'Ubuntu 22.04.5 LTS',
  libc: 'glibc 2.35',
  machine: 'Google Compute Engine',
  cpu: CpuInfo(model: 'Intel(R) Xeon(R) CPU @ 2.00GHz', cores: 4),
  gpus: [GpuInfo(name: 'Tesla T4', source: '/proc/driver/nvidia')],
  vulkan: [
    VulkanDevice(name: 'Tesla T4', type: 'DISCRETE_GPU', apiVersion: '1.4.312'),
  ],
);

void main() {
  testWidgets('inferred (macOS): chip line, GPU inferred in amber, Gemma '
      'amber with the "inferred" tag', (tester) async {
    await _pump(
      tester,
      buildDeviceSummary(
        hardware: kFakeMacProfile,
        hardwareNote: null,
        models: [_gemma(kFakeMacProfile)],
      ),
    );
    expect(find.text('THIS DEVICE'), findsOneWidget);
    expect(find.text('Mac16,8 · macOS 26.5.1 (25F80)'), findsOneWidget);
    expect(
      find.text('Apple M4 Pro · 14 cores (10P+4E) · RAM 24.00 GB'),
      findsOneWidget,
    );
    expect(_colorOf(tester, 'Apple M4 Pro · Metal'), kWarningColor);
    expect(find.text('GPU → GPU · Metal · Apple M4 Pro'), findsOneWidget);
    expect(_colorOf(tester, 'GPU → GPU · Metal · Apple M4 Pro'), kWarningColor);
    expect(find.text('inferred'), findsNWidgets(2));
    expect(find.text('Apple Neural Engine (not used)'), findsOneWidget);
  });

  testWidgets('confirmed (T4 adapter line in the log): green, "confirmed"', (
    tester,
  ) async {
    await _pump(
      tester,
      buildDeviceSummary(
        hardware: _t4,
        hardwareNote: null,
        models: [
          _gemma(
            _t4,
            log: const [
              'Selected adapter: Tesla T4, arch=turing, vendor=nvidia, '
                  'backend=Vulkan, adapterType=Discrete GPU',
            ],
          ),
        ],
      ),
    );
    const line = 'GPU → GPU · WebGPU/Vulkan · Tesla T4 (Discrete GPU)';
    expect(find.text(line), findsOneWidget);
    expect(_colorOf(tester, line), const Color(0xFF2E7D32));
    expect(find.text('confirmed'), findsOneWidget);
    expect(find.text('Tesla T4 · 1.4.312'), findsOneWidget, reason: 'Vulkan');
  });

  testWidgets('software GPU and mismatch are red', (tester) async {
    const llvmpipe = HardwareProfile(
      platform: HostPlatform.linux,
      os: 'Ubuntu 24.04.3 LTS',
      cpu: CpuInfo(model: 'AMD EPYC 7B12', cores: 2),
      vulkan: [VulkanDevice(name: 'llvmpipe', type: 'CPU')],
    );
    await _pump(
      tester,
      buildDeviceSummary(
        hardware: llvmpipe,
        hardwareNote: null,
        models: [
          _gemma(llvmpipe),
          ModelDiagnostics(
            name: 'YOLO26n detector',
            state: 'ready',
            evidence: inferEvidence(
              requested: 'gpu',
              reported: 'cpu',
              hardware: llvmpipe,
            ),
          ),
        ],
      ),
    );
    final error = ThemeData().colorScheme.error;
    expect(_colorOf(tester, 'llvmpipe: software only'), error);
    expect(
      _colorOf(tester, 'GPU → GPU · WebGPU/Vulkan · llvmpipe · SOFTWARE GPU'),
      error,
    );
    expect(_colorOf(tester, 'GPU → CPU · MISMATCH'), error);
  });

  testWidgets('Android (Galaxy S24): phone and Android version as the '
      'title, the SoC as the chip, the Adreno amber and "inferred", the '
      'Hexagon hint', (tester) async {
    final s24 = androidProfileFrom(
      propsOf(kGalaxyS24Snapdragon),
      files: phoneFiles(),
      npu: const NpuAvailable(soc: 'QTI SM8650'),
      cores: 8,
    );
    await _pump(
      tester,
      buildDeviceSummary(
        hardware: s24,
        hardwareNote: null,
        models: [_gemma(s24)],
      ),
    );
    expect(find.text('Samsung SM-S921U · Android 14 (API 34)'), findsOneWidget);
    expect(
      find.text('Snapdragon 8 Gen 3 (SM8650) · 8 cores · RAM 7.27 GB'),
      findsOneWidget,
    );
    expect(find.text('Chip: unknown'), findsNothing);
    expect(find.text('none found'), findsNothing);
    expect(_colorOf(tester, 'Adreno 750'), kWarningColor);
    expect(find.text('GPU → GPU · OpenCL · Adreno 750'), findsOneWidget);
    expect(find.text('inferred'), findsNWidgets(2));
    expect(
      find.textContaining('Qualcomm Hexagon V75 · libcdsprpc.so opens'),
      findsOneWidget,
    );
  });

  testWidgets('Android with a SoC the table does not know: the GPU is "not '
      'identified", not "none found"', (tester) async {
    final phone = androidProfileFrom(
      propsOf(kMediatekPhone),
      files: phoneFiles(),
      npu: const NpuUnavailable('no FastRPC'),
      cores: 8,
    );
    await _pump(
      tester,
      buildDeviceSummary(hardware: phone, hardwareNote: null, models: const []),
    );
    expect(
      find.text('Mediatek MT6989 · 8 cores · RAM 7.27 GB'),
      findsOneWidget,
    );
    expect(find.text('not identified (see Copy diagnostics)'), findsOneWidget);
    expect(
      _colorOf(tester, 'not identified (see Copy diagnostics)'),
      kWarningColor,
    );
  });

  testWidgets('an explicit-CPU detector is amber although confirmed', (
    tester,
  ) async {
    await _pump(
      tester,
      buildDeviceSummary(
        hardware: kFakeMacProfile,
        hardwareNote: null,
        models: [
          ModelDiagnostics(
            name: 'YOLO26n detector',
            state: 'ready',
            explicitCpu: true,
            evidence: inferEvidence(
              requested: 'cpu',
              reported: 'cpu',
              hardware: kFakeMacProfile,
            ),
          ),
        ],
      ),
    );
    expect(_colorOf(tester, 'CPU → CPU (chosen)'), kWarningColor);
    expect(find.text('confirmed'), findsOneWidget);
  });

  testWidgets('Copy diagnostics writes the report to the clipboard', (
    tester,
  ) async {
    final calls = <MethodCall>[];
    tester.binding.defaultBinaryMessenger.setMockMethodCallHandler(
      SystemChannels.platform,
      (call) async {
        calls.add(call);
        return null;
      },
    );
    addTearDown(
      () => tester.binding.defaultBinaryMessenger.setMockMethodCallHandler(
        SystemChannels.platform,
        null,
      ),
    );
    var built = 0;
    await _pump(
      tester,
      buildDeviceSummary(
        hardware: kFakeMacProfile,
        hardwareNote: null,
        models: const [],
      ),
      report: () {
        built++;
        return 'LiteRT Demos diagnostics\n…';
      },
    );
    expect(built, 0, reason: 'the report is built on Copy, not on build');
    await tester.tap(find.byKey(DeviceCardKeys.copy));
    await tester.pump();
    final set = calls.singleWhere((c) => c.method == 'Clipboard.setData');
    expect((set.arguments as Map)['text'], 'LiteRT Demos diagnostics\n…');
    expect(find.text('Diagnostics copied to the clipboard'), findsOneWidget);
  });

  testWidgets('home shows the card from the hardware repository; Copy gets '
      'the full report', (tester) async {
    final models = ModelRepository(
      gemmaModelPath: kTestChatModelPath,
      bundled: FakeBundledFiles(),
      detector: fakeDetectorService(),
      bundledDetector: fakeBundledDetector,
      llm: FakeLlmService(),
      stt: fakeSttService(),
      tts: fakeTtsService(),
      embedder: EmbedderService(),
    );
    await tester.runAsync(models.prepareAll);
    final logs = <String>[];
    final hardware = HardwareRepository(
      service: FakeHardwareInfoService(),
      models: models.states,
      memory: FakeMemoryProbe(),
      log: logs.add,
      build: currentBuildInfo(),
    );
    await tester.runAsync(hardware.probe);
    expect(logs.first, startsWith('[Hardware] macOS 26.5.1 (25F80) · Mac16,8'));
    expect(
      logs,
      contains(
        '[Hardware] chat req=gpu act=gpu(confirmed: API) api=Metal(inferred) '
        'adapter="Apple M4 Pro"(inferred)',
      ),
    );

    await tester.pumpWidget(
      MaterialApp(
        home: ChangeNotifierProvider(
          create: (_) =>
              HomeViewModel(models: models.states, hardware: hardware),
          child: HomeScreen(onOpen: (_) async {}),
        ),
      ),
    );
    await tester.scrollUntilVisible(find.byKey(DeviceCardKeys.card), 200);
    // The chat model and the built-in detector.
    expect(find.text('GPU → GPU · Metal · Apple M4 Pro'), findsNWidgets(2));
    final report = hardware.report();
    expect(report, contains('DEVICE\nos         macOS 26.5.1 (25F80)'));
    expect(report, contains(ModelId.chat.spec.displayName));
    expect(report, contains('MEMORY\nnow        available'));

    await tester.pumpWidget(const SizedBox.shrink());
    hardware.dispose();
    await tester.runAsync(models.close);
  });

  testWidgets('a failed probe is shown, not thrown', (tester) async {
    final states = ValueNotifier<Map<ModelId, ModelState>>(const {});
    final hardware = HardwareRepository(
      service: FakeHardwareInfoService(kFakeMacProfile, StateError('boom')),
      models: states,
      memory: FakeMemoryProbe(),
      log: (_) {},
      build: currentBuildInfo(),
    );
    await tester.runAsync(hardware.probe);
    await _pump(tester, hardware.summary(), report: hardware.report);
    expect(find.text('probe failed: Bad state: boom'), findsOneWidget);
    expect(
      hardware.report(),
      contains('probe      probe failed: Bad state: boom'),
    );
    hardware.dispose();
    states.dispose();
  });
}
