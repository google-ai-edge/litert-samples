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

import 'dart:io';

import 'package:flutter_edge_ai/flutter_edge_ai.dart' show PreferredBackend;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';
import 'package:litert_edge_demos/selftest/self_test_adapters.dart';
import 'package:litert_edge_demos/selftest/self_test_runner.dart'
    show SelfTestChatModel, gemmaLoadConfigLine;

/// The self-test loads exactly what the app would (GEMMA_MODEL_PATH must
/// not override the user's own model), except for an explicit
/// `--gemma`.
void main() {
  late Directory tmp;
  late String defineFile;
  late String customFile;
  late String argumentFile;

  setUp(() {
    tmp = Directory.systemTemp.createTempSync('self_test_adapters');
    String file(String name) =>
        (File('${tmp.path}/$name')..writeAsStringSync('x')).path;
    defineFile = file('define.litertlm');
    customFile = file('custom.litertlm');
    argumentFile = file('argument.litertlm');
  });
  tearDown(() => tmp.deleteSync(recursive: true));

  CustomChatPlan custom() => CustomChatPlan(
    path: customFile,
    model: CustomChatModel(
      displayName: 'My Gemma',
      source: ImportedModelSource(customFile),
      file: const CustomModelFile(
        name: 'custom.litertlm',
        sizeBytes: 1,
        sha256: 'abc',
        checksumMatched: false,
      ),
      backend: PreferredBackend.gpu,
      maxTokens: 4096,
    ),
  );

  Future<SelfTestChatModel> resolve(
    ChatModelPlan plan, {
    String? argument,
    String define = '',
  }) =>
      resolveSelfTestChatModel(argument: argument, define: define, plan: plan);

  test('your own model wins over GEMMA_MODEL_PATH, like the app', () async {
    final chosen = await resolve(custom(), define: defineFile);

    expect(chosen.file.path, customFile);
    expect(chosen.settingsSource, 'your own .litertlm, saved settings');
    expect(chosen.config.name, 'My Gemma');
    expect(chosen.sha256, 'abc');
  });

  test('a file used in place says so, not "model store"', () async {
    final local = custom();
    final chosen = await resolve(
      CustomChatPlan(
        path: customFile,
        model: local.model.copyWith(source: LocalModelSource(customFile)),
      ),
    );

    expect(chosen.file.path, customFile);
    expect(chosen.file.source, 'in place');
    expect((await resolve(local)).file.source, 'model store, custom/');
  });

  test('no chat model chosen: GEMMA_MODEL_PATH when set, else a problem '
      '(the app ships none)', () async {
    final fromDefine = await resolve(
      const NoChatModelPlan(),
      define: defineFile,
    );
    expect(fromDefine.file.path, defineFile);
    expect(fromDefine.file.source, 'GEMMA_MODEL_PATH');
    expect(fromDefine.sha256, isNull);

    final none = await resolve(const NoChatModelPlan(note: 'retired'));
    expect(none.file.path, isNull);
    expect(none.file.problem, startsWith('No chat model yet'));
    expect(none.file.problem, endsWith('retired'));
  });

  test('an explicit --gemma wins over everything (the CLI override)', () async {
    final chosen = await resolve(
      custom(),
      argument: argumentFile,
      define: defineFile,
    );

    expect(chosen.file.path, argumentFile);
    expect(chosen.file.source, '--gemma');
  });

  test('an explicit --detector wins over the built-in one; without it, the '
      'built-in one', () async {
    final chosen = await resolveSelfTestDetector(argument: argumentFile);

    expect(chosen.path, argumentFile);
    expect(chosen.source, '--detector');
    expect(chosen.bundled, isFalse);

    final missing = await resolveSelfTestDetector(
      argument: '${tmp.path}/nowhere.tflite',
    );
    expect(missing.path, isNull);
    expect(missing.problem, startsWith('not readable'));

    final builtIn = await resolveSelfTestDetector(argument: null);
    expect(builtIn.bundled, isTrue);
    expect(builtIn.problem, isNull);
  });

  test('the load line says what was asked for, and the engine\'s context '
      'when it differs', () {
    expect(
      gemmaLoadConfigLine(kDefineChatModel, engineContext: 8192),
      'maxTokens 8192 · image on · tools on · type gemma4',
    );
    expect(
      gemmaLoadConfigLine(kDefineChatModel, engineContext: 4096),
      'maxTokens 8192 (engine 4096) · image on · tools on · type gemma4',
    );
  });

  test('a blocked choice stays blocked, never Gemma 4 E2B', () async {
    final chosen = await resolve(
      const ChatPlanBlocked('the file changed'),
      define: defineFile,
    );

    expect(chosen.file.path, isNull);
    expect(chosen.file.problem, 'the file changed');
  });
}
