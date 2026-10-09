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

// Golden text for the diagnostics report. Regenerate after an intended
// change with:
//   UPDATE_GOLDENS=1 fvm flutter test test/domain/hardware/diagnostics_report_test.dart

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/hardware_repository.dart';
import 'package:litert_edge_demos/data/services/hardware/linux_hardware_info_service.dart';
import 'package:litert_edge_demos/domain/hardware/device_summary.dart';
import 'package:litert_edge_demos/domain/hardware/diagnostics_report.dart';
import 'package:litert_edge_demos/domain/models/hardware_profile.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';

import '../../support/hardware_trees.dart';

const _build = BuildInfo(
  appVersion: '0.1.0+1',
  buildMode: 'release',
  flutterVersion: '3.47.3',
  dartVersion: '3.13.3',
  packages: {
    'flutter_edge_ai': '2.1.0',
    'flutter_edge_ai_litertlm': '1.9.0',
    'flutter_litert': '3.9.3',
  },
);

const _m4Pro = HardwareProfile(
  platform: HostPlatform.macos,
  os: 'macOS 26.5.1 (25F80)',
  machine: 'Mac16,8',
  cpu: CpuInfo(
    model: 'Apple M4 Pro',
    cores: 14,
    performanceCores: 10,
    efficiencyCores: 4,
    architecture: 'arm64',
  ),
  memory: MemoryInfo(totalBytes: 24 << 30, availableBytes: 9 << 30),
  gpus: [
    GpuInfo(
      name: 'Apple M4 Pro',
      source: 'chip name (sysctl); Metal device not queried',
      inferred: true,
      kind: GpuKind.integrated,
      api: 'Metal',
    ),
  ],
  npuHints: [NpuHint('Apple Neural Engine (not used)')],
);

const _memory = MemorySnapshot(
  availableBytes: 8 << 30,
  rssBytes: 1600 << 20,
  peakRssBytes: 1700 << 20,
);

Map<ModelId, ModelState> _states({
  required List<String> gemmaLog,
  required List<String> detectorLog,
}) => {
  ModelId.chat: ModelReady(
    LoadedModelInfo(
      modelId: 'gemma-4-E2B-it',
      backend: 'gpu',
      loadTime: const Duration(milliseconds: 3606),
      warmUpTime: const Duration(milliseconds: 2253),
      nativeLog: gemmaLog,
      chat: const ChatModelFacts(
        name: 'Gemma 4 E2B',
        custom: false,
        source: 'GEMMA_MODEL_PATH=/models/gemma-4-E2B-it.litertlm',
        requestedBackend: 'gpu',
        requestedContext: 8192,
        contextTokens: 8192,
        images: true,
        tools: true,
        modelType: 'gemma4',
      ),
    ),
  ),
  ModelId.whisperBase: const ModelReady(
    LoadedModelInfo(
      modelId: 'whisper-base',
      backend: 'cpu',
      detail: 'CPU (requested)',
      loadTime: Duration(milliseconds: 410),
      warmUpTime: Duration(milliseconds: 1800),
      backendReported: false,
    ),
  ),
  ModelId.inflectNano: const ModelFailed('TTS download failed: offline'),
  ModelId.yolo26n: ModelReady(
    LoadedModelInfo(
      modelId: 'yolo26n_fp16_rawhead',
      backend: 'gpu',
      detail: 'GPU fp32 full',
      loadTime: const Duration(milliseconds: 240),
      warmUpTime: const Duration(milliseconds: 5),
      nativeLog: detectorLog,
    ),
  ),
  ModelId.moonshineTiny: const ModelLoading(),
  ModelId.embeddingGemma: const ModelUnavailable(
    'EmbeddingGemma is not in the model store yet',
  ),
};

String _report(HardwareProfile hw, Map<ModelId, ModelState> states) =>
    formatDiagnostics(
      DiagnosticsInput(
        generatedAt: DateTime.utc(2026, 10, 6, 12, 34, 56, 789),
        build: _build,
        hardware: hw,
        models: [
          for (final id in ModelId.values)
            diagnosticsFor(id, states[id]!, hardware: hw),
        ],
        memory: _memory,
        nativeLog: 'native stderr → /tmp/app/logs/native.log',
      ),
    );

void _expectGolden(String actual, String name) {
  final file = File('test/goldens/$name');
  if (Platform.environment['UPDATE_GOLDENS'] == '1') {
    file
      ..createSync(recursive: true)
      ..writeAsStringSync(actual);
  }
  expect(actual, file.readAsStringSync(), reason: 'golden $name');
}

void main() {
  test('macOS M4 Pro report', () {
    _expectGolden(
      _report(
        _m4Pro,
        _states(
          gemmaLog: const [
            'INFO: [accelerator_registry.cc:54] RegisterAccelerator: ptr=0xb4dc50300, name=GPU Metal',
            'I0000 00:00:1791240441.593498 19514714 delegate_metal.mm:89] Created a Metal device.',
            'W0000 00:00:1791240447.020853 19514920 sampler_factory.cc:771] GPU sampler unavailable. Falling back to CPU sampling.',
          ],
          detectorLog: const [],
        ),
      ),
      'diagnostics_macos.txt',
    );
  });

  test('Linux T4 report: the adapter confirmed by the log for Gemma, '
      'inferred for the detector', () async {
    final hw = await LinuxHardwareInfoService(
      files: t4Vm.systemFiles,
      processes: t4Vm.processes,
      libcVersion: () => t4Vm.glibc,
      architecture: t4Vm.architecture,
    ).probe();
    _expectGolden(
      _report(
        hw,
        _states(
          gemmaLog: const [
            'I0000 00:00:1759700000.000000 12345 environment.cc:526] Selected adapter: Tesla T4, arch=turing, vendor=nvidia, backend=Vulkan, adapterType=Discrete GPU',
          ],
          detectorLog: const [],
        ),
      ),
      'diagnostics_linux_t4.txt',
    );
  });

  test('the chat model rows: your own NPU model, its source, checksum, '
      'context, images and tools; the card names it', () {
    const facts = ChatModelFacts(
      name: 'Gemma 3 1B NPU',
      custom: true,
      source: 'downloaded from https://example.com/g3_ekv1280.litertlm',
      sha256:
          '1a2b3c4d00000000000000000000000000000000000000000000000000000000',
      requestedBackend: 'npu',
      requestedContext: 1280,
      contextTokens: 1280,
      images: false,
      tools: false,
      modelType: 'gemmaIt',
    );
    final row = diagnosticsFor(
      ModelId.chat,
      const ModelReady(
        LoadedModelInfo(
          modelId: 'g3_ekv1280',
          backend: 'npu',
          loadTime: Duration(seconds: 4),
          warmUpTime: Duration(milliseconds: 300),
          chat: facts,
        ),
      ),
      hardware: _m4Pro,
    );
    final text = formatDiagnostics(
      DiagnosticsInput(
        generatedAt: DateTime.utc(2026, 10, 6),
        build: _build,
        models: [row],
        nativeLog: 'off',
      ),
    );
    expect(text, contains('Chat model\n'));
    expect(text, contains('  backend  npu → npu (confirmed: API)'));
    expect(
      text,
      contains('  model    Gemma 3 1B NPU (your own .litertlm, type gemmaIt)'),
    );
    expect(text, contains('  source   downloaded from https://example.com/'));
    expect(text, contains('  sha256   1a2b3c4d000'));
    expect(text, contains('computed at import'));
    expect(text, contains('  context  1280 tokens'));
    expect(text, contains('  images   off'));
    expect(text, contains('  tools    off'));
    expect(text, isNot(contains('vision encoder')), reason: 'images off');

    final card = buildDeviceSummary(
      hardware: _m4Pro,
      hardwareNote: null,
      models: [row],
    );
    expect(card.models.first.label, 'Chat model');
    expect(card.models.first.value, startsWith('NPU → NPU'));
    expect(
      card.models.last.value,
      'Gemma 3 1B NPU (your own) · ctx 1280 · images off · tools off · '
      'gemmaIt · sha256 1a2b3c4d…',
    );

    // Downloaded with the checksum the user entered, and it matched.
    const matched = ChatModelFacts(
      name: 'Gemma 3 1B NPU',
      custom: true,
      source: 'downloaded from https://example.com/g3_ekv1280.litertlm',
      sha256:
          '1a2b3c4d00000000000000000000000000000000000000000000000000000000',
      checksumMatched: true,
      requestedBackend: 'npu',
      requestedContext: 1280,
      contextTokens: 1280,
      images: false,
      tools: false,
      modelType: 'gemmaIt',
    );
    final sha = chatModelRows(matched).singleWhere((r) => r.$1 == '  sha256');
    expect(sha.$2, endsWith('(matches the published one)'));
    expect(sha.$2, isNot(contains('computed at import')));
  });

  test('one-line logs', () {
    expect(
      hardwareStartupLine(_m4Pro, _build),
      '[Hardware] macOS 26.5.1 (25F80) · Mac16,8 · Apple M4 Pro 10P+4E · GPU Apple '
      'M4 Pro Metal (inferred) · 24.00 GB · app 0.1.0+1 release · '
      'flutter_edge_ai 2.1.0 · flutter_edge_ai_litertlm 1.9.0 · flutter_litert '
      '3.9.3',
    );
    final gemma = diagnosticsFor(
      ModelId.chat,
      _states(gemmaLog: const [], detectorLog: const [])[ModelId.chat]!,
      hardware: _m4Pro,
    );
    expect(
      evidenceLogLine('chat', gemma.evidence!),
      '[Hardware] chat req=gpu act=gpu(confirmed: API) api=Metal(inferred) '
      'adapter="Apple M4 Pro"(inferred)',
    );
  });

  test('bytes', () {
    expect(formatBytes(24 << 30), '24.00 GB');
    expect(formatBytes(182 << 20), '182 MB');
    expect(formatBytes(-(51 << 20)), '-51 MB');
    expect(formatBytes(512), '512 B');
  });
}
