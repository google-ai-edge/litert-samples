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

// Your own chat model end to end on macOS: the real app's dependencies and
// the Models screen's ChatModelViewModel, with only the file picker replaced
// (the OS panel cannot be driven from a test): it returns GEMMA_MODEL_PATH,
// imported as "your own .litertlm" (APFS clone + SHA-256 computed and
// recorded), applied on the GPU
// (the chat is released, the model reloaded with exactly these settings),
// then NPU is refused with flutter_edge_ai's reason on this host, and the
// in-app self-test runs on the custom model.
//
// The choice stays saved (the same container as the release build), so a
// release build started afterwards loads the custom model; choose another
// .litertlm on the Models screen's Chat model card to change it.
//
//   flutter test integration_test/custom_chat_model_test.dart -d macos \
//     --dart-define=GEMMA_MODEL_PATH=<path/to/gemma-4-E2B-it.litertlm>
//
// GEMMA_MODEL_PATH must be an absolute path to the Gemma 4 E2B file whose
// size and SHA-256 the test pins (litert-community/gemma-4-E2B-it-litert-lm);
// the sandboxed debug build reads it from ~/Downloads. The baseline setup
// starts from the saved choice as a user's app does; with no choice saved
// yet it loads GEMMA_MODEL_PATH (any setup failure is only logged). Prints
// `CUSTOM_MODEL import_ms=… sha256=… reload_ms=… backend=… npu=… selftest=…`.

import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart' show PreferredBackend;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/dependencies.dart';
import 'package:litert_edge_demos/config/env.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/self_test.dart';
import 'package:litert_edge_demos/domain/ports/model_file_picker.dart';
import 'package:litert_edge_demos/ui/features/setup/view_models/chat_model_view_model.dart';
import 'package:litert_edge_demos/utils/result.dart';

import 'support/app_window.dart';

/// Returns one fixed file, like a user picking it in the OS panel.
final class _PickedFile implements ModelFilePicker {
  _PickedFile(this.path);

  final String path;

  @override
  ImportSupport get support => const ImportFromFolder();

  @override
  Future<String?> pickModelFile() async => path;

  @override
  Future<String?> temporaryCopiesDirectory() async => null;
}

Future<void> until(
  WidgetTester tester,
  bool Function() condition, {
  required Duration timeout,
  required String reason,
}) async {
  final watch = Stopwatch()..start();
  while (!condition()) {
    if (watch.elapsed > timeout) fail('Timed out after $timeout: $reason');
    await tester.pump(const Duration(milliseconds: 100));
  }
}

void main() {
  initIntegrationTest();

  testWidgets('import your own .litertlm, run it on the GPU, NPU refused '
      'with the reason, the in-app self-test on it', (tester) async {
    const source = kGemmaModelPath;
    if (source.isEmpty) fail('Pass --dart-define=GEMMA_MODEL_PATH=…');
    if (!File(source).existsSync()) fail('No $source');
    final deps = await AppDependencies.create(picker: _PickedFile(source));
    addTearDown(deps.dispose);
    await tester.pumpWidget(const MaterialApp(home: SizedBox.shrink()));

    // Setup as the app does it (whatever the saved choice is now).
    final setup = await tester.runAsync(deps.prepareModels);
    debugPrint('CUSTOM_MODEL baseline setup: $setup');

    final vm = ChatModelViewModel(
      chatModels: deps.chatModels,
      models: deps.models,
      switcher: deps.chatModelSwitcher,
      picker: deps.picker,
    );
    addTearDown(vm.dispose);

    // 1. Import file… (the picker returns GEMMA_MODEL_PATH).
    final importWatch = Stopwatch()..start();
    unawaited(vm.importFile.execute());
    await until(
      tester,
      () => !vm.importFile.running,
      timeout: const Duration(minutes: 3),
      reason: 'the import',
    );
    final importMs = importWatch.elapsedMilliseconds;
    expect(vm.importFile.completed, isTrue, reason: '${vm.transferError}');
    final file = vm.saved!.file!;
    expect(file.name, 'gemma-4-E2B-it.litertlm');
    expect(file.sizeBytes, 2538799104);
    expect(
      file.sha256,
      '2c902a8c1c7675ec57f51020f01571d765af2ca84859e8c8fcf663a56e6e587a',
      reason: 'the pinned Gemma 4 E2B bytes',
    );
    expect(file.checksumMatched, isFalse, reason: 'computed, not entered');
    debugPrint('CUSTOM_MODEL file ${vm.fileLine}');

    // 2. NPU is not offered on macOS: flutter_edge_ai's own reason.
    expect(vm.npuOffered, isFalse);
    expect(
      vm.npuUnavailableReason,
      contains('no NPU dispatch stack ships for macos'),
    );
    vm
      ..setBackend(PreferredBackend.npu)
      ..setContext('4096');
    expect(vm.draftProblem, contains('no NPU dispatch stack ships for macos'));
    expect(vm.canApply, isFalse);
    final npuReason = vm.draftProblem;

    // 3. Use this model on the GPU, images on (Gemma 4 E2B has a vision
    //    encoder), tools on, as gemma4.
    vm
      ..setBackend(PreferredBackend.gpu)
      ..setName('My Gemma (custom)')
      ..setImages(enabled: true)
      ..setTools(enabled: true);
    expect(vm.draftProblem, isNull);
    final reloadWatch = Stopwatch()..start();
    unawaited(vm.apply.execute());
    await until(
      tester,
      () => !vm.apply.running,
      timeout: const Duration(minutes: 3),
      reason: 'the reload',
    );
    final reloadMs = reloadWatch.elapsedMilliseconds;
    expect(vm.apply.result, isA<Ok<void>>(), reason: '${vm.loadError}');
    final ready = deps.models.states.value[ModelId.chat];
    expect(ready, isA<ModelReady>(), reason: '$ready');
    final info = (ready! as ModelReady).info;
    expect(info.backend, 'gpu', reason: 'activeBackend');
    final chat = info.chat!;
    expect(chat.custom, isTrue);
    expect(chat.name, 'My Gemma (custom)');
    expect(chat.contextTokens, 4096);
    expect(chat.sha256, file.sha256);
    expect(chat.source, startsWith('imported from $source'));
    expect(deps.conversation.capabilities.modelName, 'My Gemma (custom)');
    expect(deps.conversation.capabilities.images, isTrue);
    debugPrint('CUSTOM_MODEL ${vm.activeLine}');

    // The diagnostics report names it.
    final report = deps.hardware.report();
    expect(report, contains('My Gemma (custom) (your own .litertlm'));
    expect(report, contains('sha256   ${file.sha256} (computed'));
    debugPrint(
      report
          .split('\n')
          .where((l) => l.contains('Chat model') || l.startsWith('  '))
          .take(12)
          .join('\n'),
    );

    // 4. Run self-test (the Models screen's button) on the custom model.
    final lines = <String>[];
    final outcome = await tester.runAsync(
      () => deps.selfTest.run(progress: lines.add),
    );
    final SelfTestOutcome result;
    switch (outcome) {
      case Ok(:final value):
        result = value;
      case final other:
        fail('The self-test could not run: $other');
    }
    debugPrint(result.text);
    expect(result.text, contains('chat model My Gemma (custom) (your own'));
    expect(result.text, contains('PASS  4a  chat model load + warm-up (gpu)'));
    expect(result.text, contains('PASS  4b  chat model generate'));
    expect(result.text, contains('detector (bundled)'));
    expect(
      deps.models.states.value[ModelId.chat],
      isA<ModelReady>(),
      reason: 'the app has its chat model back after the run',
    );

    debugPrint(
      'CUSTOM_MODEL import_ms=$importMs sha256=${file.sha256!.substring(0, 12)} '
      'reload_ms=$reloadMs backend=${info.backend} npu="$npuReason" '
      'selftest=${result.passed ? 'PASS' : 'FAIL'} report=${result.reportPath}',
    );
    expect(deps.chatModels.state.value.kind, ChatModelKind.custom);
    expect(kDefineChatModel.name, 'Gemma 4 E2B');
  });
}
