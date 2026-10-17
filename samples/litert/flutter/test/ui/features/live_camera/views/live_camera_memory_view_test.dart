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

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/data/repositories/live_camera_settings_repository.dart';
import 'package:litert_edge_demos/data/repositories/live_detection_repository.dart';
import 'package:litert_edge_demos/data/repositories/speech_repository.dart';
import 'package:litert_edge_demos/data/repositories/visual_memory_repository.dart';
import 'package:litert_edge_demos/data/services/images/png_encoder.dart';
import 'package:litert_edge_demos/data/services/settings/typed_settings.dart';
import 'package:litert_edge_demos/domain/models/camera_side_event.dart';
import 'package:litert_edge_demos/domain/models/chat_capabilities.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/visual_memory.dart';
import 'package:litert_edge_demos/domain/use_cases/camera_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/ui/features/live_camera/view_models/live_camera_view_model.dart';
import 'package:litert_edge_demos/ui/features/live_camera/views/camera_settings_sheet.dart';
import 'package:litert_edge_demos/ui/features/live_camera/views/live_camera_screen.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:provider/provider.dart';

import '../../../../fakes/fake_audio_repository.dart';
import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_detector.dart';
import '../../../../fakes/fake_frame_source.dart';
import '../../../../fakes/fake_settings_store.dart';
import '../../../../fakes/fake_speech.dart';
import '../../../../fakes/fake_visual_memory.dart';

/// 4×2 mid grey, RGB.
final _grey = Uint8List(24)..fillRange(0, 24, 128);

void main() {
  late FakeConversationRepository conversation;
  late LiveDetectionRepository live;
  late ValueNotifier<Map<ModelId, ModelState>> models;
  late DiagnosticsRepository diagnostics;
  late FakeMemoryCapture capture;
  late FakeMemoryEmbedder embedder;
  late VisualMemoryRepository memory;
  late InMemorySettingsStore store;
  late int now;

  setUp(() {
    store = InMemorySettingsStore();
    conversation = FakeConversationRepository()
      ..capabilities = const ChatCapabilities(
        modelName: 'Gemma 3 NPU',
        images: false,
        tools: false,
      );
    live = LiveDetectionRepository(
      detector: FakeDetector(autoComplete: true),
      createSource: (_) => Result.ok(FakeFrameSource()),
    );
    models = ValueNotifier(const {});
    diagnostics = DiagnosticsRepository(
      models: models,
      minInterval: Duration.zero,
      sampleRss: false,
    );
    capture = FakeMemoryCapture();
    embedder = FakeMemoryEmbedder(
      imageVectors: {
        0x89: [1, 0, 0], // a real PNG's first byte
      },
      textVectors: {
        'yellow mug': [0.95, 0.1, 0.1],
      },
    );
    now = 100000000;
    memory = VisualMemoryRepository(
      capture: capture.call,
      embedder: embedder,
      enableImages: embedder.enableImages,
      sourceStarts: live.sourceStarts,
      blackFrames: live.blackFrames,
      liveState: live.state,
      capturePending: () => live.capturePending,
      tickInterval: null,
      wallClockMicros: () => now,
      monotonicMicros: () => now,
    );
  });

  Future<LiveCameraViewModel> pumpScreen(
    WidgetTester tester, {
    String recognized = 'Where was the yellow mug?',
    int? physicalMemoryBytes = 12 << 30,
  }) async {
    final SpeechRepository speech = (await tester.runAsync(
      () => loadedSpeech(
        recognizer: FakeRecognizer(recognized),
        synthesizer: RecordingSynth(),
      ),
    ))!;
    late LiveCameraViewModel viewModel;
    await tester.pumpWidget(
      MaterialApp(
        home: ChangeNotifierProvider(
          create: (_) => viewModel = LiveCameraViewModel(
            conversation: conversation,
            live: live,
            memory: memory,
            settings: LiveCameraSettingsRepository(
              settings: TypedSettings(store: store),
              environment: const Result.ok(FixtureSourceSpec(['/fixtures'])),
              physicalMemoryBytes: physicalMemoryBytes,
            ),
            diagnostics: diagnostics,
            activateStt: (_) async => const Result.ok(null),
            assistant: VoiceAssistant<CameraSideEvent>(
              speech: speech,
              audio: FakeAudioRepository()..autoDrain = true,
              responders: CameraTurnResponder(
                capture: live.capture,
                conversation: conversation,
                encode: live.encodeForLlm,
                recall: memory.recall,
              ),
              diagnostics: diagnostics,
            ),
          ),
          child: const LiveCameraScreen(),
        ),
      ),
    );
    await tester.runAsync(() => Future<void>.delayed(Duration.zero));
    await tester.pump();
    return viewModel;
  }

  Future<void> tearDownScreen(WidgetTester tester) async {
    await tester.pumpWidget(const SizedBox.shrink());
    await tester.pump(Duration.zero);
    memory.close();
    await tester.runAsync(live.close);
    await conversation.close();
    diagnostics.dispose();
    models.dispose();
  }

  Future<void> remember(WidgetTester tester, {bool changed = true}) async {
    capture.results.add(
      Result.ok(fakeShot(png: changed, encoded: encodePngRgb(4, 2, _grey))),
    );
    await tester.runAsync(memory.tick);
    await tester.pump();
  }

  testWidgets('the chip: nothing yet, then the frames and how far back they '
      'reach; the memory off is an amber banner with the reason', (
    tester,
  ) async {
    await pumpScreen(tester);
    expect(find.text('Memory: nothing yet'), findsOneWidget);

    await remember(tester);
    now += 20000000;
    await remember(tester, changed: false); // the same scene: nothing new
    // Refreshed in place 20 s later: the memory still reaches back to the
    // frame's first sighting.
    expect(find.text('Memory: 1 frame · 20 s'), findsOneWidget);

    for (var i = 0; i < 3; i++) {
      capture.results.add(
        const Result.error(MemoryFrameException('PNG failed')),
      );
      now += 10000000;
      await tester.runAsync(memory.tick);
    }
    await tester.pump();
    expect(find.byKey(LiveCameraKeys.memoryOff), findsOneWidget);
    expect(
      find.textContaining('Visual memory off: 3 frames in a row failed'),
      findsOneWidget,
    );
    await tearDownScreen(tester);
  });

  testWidgets('"Memory: preparing…" while the image encoder loads, then the '
      'frames; a failed load is the banner with its reason', (tester) async {
    embedder.unavailable = 'EmbeddingGemma was loaded without images';
    embedder.enableGate = Completer();
    await pumpScreen(tester);
    expect(find.text('Memory: preparing…'), findsOneWidget);
    expect(embedder.enables, 1);

    embedder.enableGate!.complete();
    await tester.runAsync(pumpEventQueue);
    await tester.pump();
    expect(find.text('Memory: nothing yet'), findsOneWidget);
    await tearDownScreen(tester);
  });

  testWidgets('a failed image encoder load: the banner says why', (
    tester,
  ) async {
    embedder.unavailable = 'EmbeddingGemma was loaded without images';
    embedder.enableError = Exception('no vision tower');
    await pumpScreen(tester);
    await tester.runAsync(pumpEventQueue);
    await tester.pump();
    expect(
      find.textContaining(
        'Visual memory off: loading its image encoder failed',
      ),
      findsOneWidget,
    );
    await tearDownScreen(tester);
  });

  testWidgets('the "Visual memory" switch: off hides the chip, forgets the '
      'frames and is saved; a memory question then says it is off', (
    tester,
  ) async {
    final viewModel = await pumpScreen(tester);
    await remember(tester);
    expect(find.byKey(LiveCameraKeys.memoryChip), findsOneWidget);
    expect(viewModel.memoryEnabled, isTrue);

    await tester.tap(find.byKey(LiveCameraKeys.settings));
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.byKey(CameraSettingsKeys.memorySwitch));
    await tester.tap(find.byKey(CameraSettingsKeys.memorySwitch));
    await tester.runAsync(pumpEventQueue);
    await tester.pump();

    expect(viewModel.memoryEnabled, isFalse);
    expect(memory.entries, isEmpty);
    expect(memory.status.value.enabled, isFalse);
    expect(store.values['camera.visualMemory'], false);
    expect(
      find.byKey(CameraSettingsKeys.clearMemory),
      findsNothing,
      reason: 'nothing to clear while off',
    );
    Navigator.of(tester.element(find.byKey(CameraSettingsKeys.sheet))).pop();
    await tester.pumpAndSettle();
    expect(find.byKey(LiveCameraKeys.memoryChip), findsNothing);
    expect(find.byKey(LiveCameraKeys.memoryOff), findsNothing);
    final recall = await tester.runAsync(() => memory.recall('yellow mug'));
    expect((recall! as Ok<MemoryRecall>).value, isA<MemoryDisabled>());
    await tearDownScreen(tester);
  });

  testWidgets('a device with less than 8 GB of RAM: the switch starts off, '
      'the reason under it, no image encoder loaded; turning it on works', (
    tester,
  ) async {
    final viewModel = await pumpScreen(tester, physicalMemoryBytes: 4 << 30);
    expect(viewModel.memoryEnabled, isFalse);
    expect(embedder.enables, 0);
    expect(memory.status.value.enabled, isFalse);

    await tester.tap(find.byKey(LiveCameraKeys.settings));
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.byKey(CameraSettingsKeys.memorySwitch));
    expect(
      find.text(
        'Off by default on this device (4.0 GB RAM): the visual memory '
        'loads about 300–450 MB more.',
      ),
      findsOneWidget,
    );
    await tester.tap(find.byKey(CameraSettingsKeys.memorySwitch));
    await tester.runAsync(pumpEventQueue);
    await tester.pump();
    expect(viewModel.memoryEnabled, isTrue);
    expect(store.values['camera.visualMemory'], true);
    expect(memory.status.value.enabled, isTrue);
    Navigator.of(tester.element(find.byKey(CameraSettingsKeys.sheet))).pop();
    await tester.pumpAndSettle();
    await tearDownScreen(tester);
  });

  testWidgets('saved off: the memory starts off and never loads the image '
      'encoder', (tester) async {
    store.values['camera.visualMemory'] = false;
    embedder.unavailable = 'EmbeddingGemma was loaded without images';
    final viewModel = await pumpScreen(tester);
    expect(viewModel.memoryEnabled, isFalse);
    expect(embedder.enables, 0);
    expect(memory.status.value.enabled, isFalse);
    expect(find.byKey(LiveCameraKeys.memoryChip), findsNothing);
    await tearDownScreen(tester);
  });

  testWidgets('a memory question shows the remembered frame on a card; a tap '
      'closes it', (tester) async {
    await pumpScreen(tester);
    await remember(tester);
    now += 12000000;

    final gesture = await tester.startGesture(
      tester.getCenter(find.byKey(LiveCameraKeys.mic)),
    );
    await tester.pump();
    await gesture.up();
    for (
      var i = 0;
      i < 100 && find.byKey(LiveCameraKeys.memoryCard).evaluate().isEmpty;
      i++
    ) {
      await tester.pump(const Duration(milliseconds: 5));
      await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 5)),
      );
    }
    await tester.pump();
    expect(find.byKey(LiveCameraKeys.memoryCard), findsOneWidget);
    expect(find.text('Seen 12 s ago · “yellow mug”'), findsOneWidget);

    await tester.tap(find.byKey(LiveCameraKeys.memoryCard));
    await tester.pump();
    expect(find.byKey(LiveCameraKeys.memoryCard), findsNothing);
    await tester.runAsync(
      () => Future<void>.delayed(const Duration(milliseconds: 50)),
    );
    await tearDownScreen(tester);
  });

  testWidgets('"Clear memory" in the settings sheet forgets every frame', (
    tester,
  ) async {
    final viewModel = await pumpScreen(tester);
    await remember(tester);
    expect(memory.entries, hasLength(1));

    await tester.tap(find.byKey(LiveCameraKeys.settings));
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.byKey(CameraSettingsKeys.clearMemory));
    await tester.tap(find.byKey(CameraSettingsKeys.clearMemory));
    await tester.pump();

    expect(memory.entries, isEmpty);
    expect(viewModel.memoryStatus?.value.frames, 0);
    expect(find.text('Memory cleared'), findsOneWidget);
    await tearDownScreen(tester);
  });
}
