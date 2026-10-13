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

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/demos.dart';
import 'package:litert_edge_demos/data/repositories/model_repository.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/ui/features/home/view_models/home_view_model.dart';
import 'package:litert_edge_demos/ui/features/home/views/home_screen.dart';
import 'package:provider/provider.dart';

import '../../../../fakes/fake_bundled_files.dart';
import '../../../../fakes/fake_detector_runtime.dart';
import '../../../../fakes/fake_llm_service.dart';
import '../../../../fakes/fake_model_files.dart';
import '../../../../fakes/fake_speech.dart';

void main() {
  testWidgets('home: Demo 1 opens, Demo 3 is disabled with the reason (a '
      'built-in detector that cannot be read), and the history caption '
      'shows', (tester) async {
    final models = ModelRepository(
      gemmaModelPath: kTestChatModelPath,
      bundled: FakeBundledFiles(),
      detector: fakeDetectorService(),
      bundledDetector: () async => throw StateError('no asset'),
      llm: FakeLlmService(),
      stt: fakeSttService(),
      tts: fakeTtsService(),
      embedder: EmbedderService(),
    );
    await tester.runAsync(models.prepareAll);
    final opened = <Demo>[];

    await tester.pumpWidget(
      MaterialApp(
        home: ChangeNotifierProvider(
          create: (_) => HomeViewModel(models: models.states),
          child: HomeScreen(onOpen: (demo) async => opened.add(demo)),
        ),
      ),
    );

    final camera = find.byKey(HomeKeys.tile(Demo.liveCamera));
    expect(
      find.descendant(
        of: camera,
        matching: find.textContaining('The built-in detector'),
      ),
      findsOneWidget,
    );
    expect(tester.widget<ListTile>(camera).enabled, isFalse);
    await tester.tap(camera);
    await tester.pump();
    expect(opened, isEmpty, reason: 'a disabled tile does not open');

    final chat = find.byKey(HomeKeys.tile(Demo.voiceChat));
    expect(
      find.descendant(of: chat, matching: find.text('Ready')),
      findsOneWidget,
    );
    await tester.tap(chat);
    await tester.pump();
    expect(opened, [Demo.voiceChat]);

    expect(find.text(HomeViewModel.historyNote), findsOneWidget);

    await tester.pumpWidget(const SizedBox.shrink());
    // Real async: the models' operation queues ran in runAsync.
    await tester.runAsync(models.close);
  });

  testWidgets('a fast double tap opens a demo once; after it closes, the '
      'tile opens again', (tester) async {
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
    final opened = <Demo>[];
    var route = Completer<void>();

    await tester.pumpWidget(
      MaterialApp(
        home: ChangeNotifierProvider(
          create: (_) => HomeViewModel(models: models.states),
          child: HomeScreen(
            onOpen: (demo) {
              opened.add(demo);
              return route.future; // completes when the demo is popped
            },
          ),
        ),
      ),
    );
    final chat = find.byKey(HomeKeys.tile(Demo.voiceChat));

    await tester.tap(chat);
    await tester.tap(chat);
    await tester.pump();
    expect(opened, [Demo.voiceChat], reason: 'one push, one view model');

    route.complete();
    route = Completer<void>();
    await tester.pump();
    await tester.tap(chat);
    await tester.pump();
    expect(opened, [Demo.voiceChat, Demo.voiceChat]);

    route.complete();
    await tester.pumpWidget(const SizedBox.shrink());
    // Real async: the models' operation queues ran in runAsync.
    await tester.runAsync(models.close);
  });

  testWidgets('the menu opens the Models screen', (tester) async {
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
    addTearDown(models.close);
    var opened = 0;

    await tester.pumpWidget(
      MaterialApp(
        home: ChangeNotifierProvider(
          create: (_) => HomeViewModel(models: models.states),
          child: HomeScreen(
            onOpen: (_) async {},
            onOpenModels: () async => opened++,
          ),
        ),
      ),
    );

    await tester.tap(find.byKey(HomeKeys.menu));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(HomeKeys.models));
    await tester.pumpAndSettle();

    expect(opened, 1);
  });

  testWidgets('without a Models route there is no menu', (tester) async {
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
    addTearDown(models.close);

    await tester.pumpWidget(
      MaterialApp(
        home: ChangeNotifierProvider(
          create: (_) => HomeViewModel(models: models.states),
          child: HomeScreen(onOpen: (_) async {}),
        ),
      ),
    );

    expect(find.byKey(HomeKeys.menu), findsNothing);
  });
}
