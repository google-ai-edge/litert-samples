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

// Platform-setup smoke test: Gemma 4 E2B loads from a local .litertlm on the
// requested backend and answers a factual prompt. Proves the per-platform
// config (entitlements, sandbox, LiteRT-LM staging, minSdk) on a real target.
//
// GEMMA_MODEL_PATH: absolute path, or a path relative to the app's documents
// directory (copy the file in first: `xcrun devicectl device copy to` on iOS).
// On macOS the sandboxed debug build reads model files from ~/Downloads.
// GEMMA_BACKEND: gpu (default) or cpu. A silent GPU→CPU fallback fails the test.
//
//   flutter test integration_test/model_smoke_test.dart -d macos \
//     --dart-define=GEMMA_MODEL_PATH=<path/to/gemma-4-E2B-it.litertlm>

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:path_provider/path_provider.dart';

const _modelPath = String.fromEnvironment('GEMMA_MODEL_PATH');
const _backendName = String.fromEnvironment(
  'GEMMA_BACKEND',
  defaultValue: 'gpu',
);

Future<String> _resolveModelPath() async {
  if (_modelPath.isEmpty) {
    throw StateError(
      'Pass --dart-define=GEMMA_MODEL_PATH=<absolute or documents-relative path>',
    );
  }
  if (_modelPath.startsWith('/')) return _modelPath;
  final docs = await getApplicationDocumentsDirectory();
  return '${docs.path}/$_modelPath';
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('Gemma 4 E2B loads on the requested backend and answers', (
    tester,
  ) async {
    final backend = PreferredBackend.values.byName(_backendName);
    final path = await _resolveModelPath();

    await FlutterEdgeAi.initialize(inferenceEngines: [const LiteRtLmEngine()]);
    await FlutterEdgeAi.installModel(
      modelType: ModelType.gemma4,
      fileType: ModelFileType.litertlm,
    ).fromFile(path).install();

    final loadWatch = Stopwatch()..start();
    final model = await FlutterEdgeAi.getActiveModel(
      maxTokens: 1024,
      preferredBackend: backend,
    );
    final loadMs = loadWatch.elapsedMilliseconds;
    try {
      expect(
        model.activeBackend,
        backend,
        reason: 'requested $backend, engine loaded ${model.activeBackend}',
      );

      final session = await model.createSession(maxOutputTokens: 32);
      final reply = StringBuffer();
      final genWatch = Stopwatch()..start();
      int? firstTokenMs;
      try {
        await session.addQueryChunk(
          const Message(
            text: 'What is the capital of France? Answer with only the city name.',
            isUser: true,
          ),
        );
        await for (final token in session.getResponseAsync().timeout(
          const Duration(seconds: 90),
        )) {
          firstTokenMs ??= genWatch.elapsedMilliseconds;
          reply.write(token);
        }
      } finally {
        await session.close();
      }

      final answer = reply.toString().trim();
      debugPrint(
        'SMOKE backend=${model.activeBackend} load=${loadMs}ms '
        'firstToken=${firstTokenMs}ms total=${genWatch.elapsedMilliseconds}ms '
        'answer="$answer"',
      );
      expect(answer.toLowerCase(), contains('paris'));
    } finally {
      await model.close();
    }
  }, timeout: const Timeout(Duration(minutes: 8)));
}
