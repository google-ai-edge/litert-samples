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

import 'dart:ui' show AppLifecycleState;

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/data/repositories/live_camera_settings_repository.dart';
import 'package:litert_edge_demos/data/repositories/live_detection_repository.dart';
import 'package:litert_edge_demos/data/repositories/speech_repository.dart';
import 'package:litert_edge_demos/data/repositories/visual_memory_repository.dart';
import 'package:litert_edge_demos/data/services/settings/typed_settings.dart';
import 'package:litert_edge_demos/domain/models/camera_side_event.dart';
import 'package:litert_edge_demos/domain/models/chat_capabilities.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/use_cases/camera_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/ui/core/app_foreground.dart';
import 'package:litert_edge_demos/ui/features/live_camera/view_models/live_camera_view_model.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../../fakes/fake_audio_repository.dart';
import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_detector.dart';
import '../../../../fakes/fake_frame_source.dart';
import '../../../../fakes/fake_settings_store.dart';
import '../../../../fakes/fake_speech.dart';
import '../../../../fakes/fake_visual_memory.dart';

Future<void> until(bool Function() condition, {String? reason}) async {
  final watch = Stopwatch()..start();
  while (!condition() && watch.elapsed < const Duration(seconds: 15)) {
    await Future<void>.delayed(const Duration(milliseconds: 2));
  }
  expect(condition(), isTrue, reason: reason);
}

const _fixture = Result<FrameSourceSpec>.ok(FixtureSourceSpec(['/fixtures']));

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  late FakeConversationRepository conversation;
  late LiveDetectionRepository live;
  late List<FakeFrameSource> sources;
  late SpeechRepository speech;
  late ValueNotifier<Map<ModelId, ModelState>> models;
  late DiagnosticsRepository diagnostics;
  late FakeRecognizer recognizer;
  late FakeAudioRepository audio;
  late FakeMemoryCapture capture;
  late FakeMemoryEmbedder embedder;
  late VisualMemoryRepository memory;
  late int nowMicros;

  setUp(() async {
    audio = FakeAudioRepository(log: [])..autoDrain = true;
    recognizer = FakeRecognizer('Where was the yellow mug?');
    speech = await loadedSpeech(
      recognizer: recognizer,
      synthesizer: RecordingSynth(),
    );
    models = ValueNotifier(const {});
    diagnostics = DiagnosticsRepository(
      models: models,
      minInterval: Duration.zero,
      sampleRss: false,
    );
    conversation = FakeConversationRepository()
      ..capabilities = const ChatCapabilities(
        modelName: 'Gemma 3 NPU',
        images: false,
        tools: false,
      );
    sources = [];
    live = LiveDetectionRepository(
      detector: FakeDetector(autoComplete: true),
      createSource: (_) {
        final source = FakeFrameSource();
        sources.add(source);
        return Result.ok(source);
      },
    );
    capture = FakeMemoryCapture();
    embedder = FakeMemoryEmbedder(
      imageVectors: {
        1: [1, 0, 0],
      },
      textVectors: {
        'yellow mug': [0.95, 0.1, 0.1],
      },
    );
    nowMicros = 100000000;
    memory = VisualMemoryRepository(
      capture: capture.call,
      embedder: embedder,
      enableImages: embedder.enableImages,
      sourceStarts: live.sourceStarts,
      blackFrames: live.blackFrames,
      liveState: live.state,
      capturePending: () => live.capturePending,
      tickInterval: null,
      clockMicros: () => nowMicros,
    );
  });

  tearDown(() async {
    memory.close();
    await live.close();
    await conversation.close();
    diagnostics.dispose();
    models.dispose();
  });

  LiveCameraViewModel create({
    LiveCameraSettingsRepository? settings,
    AppForeground? foreground,
  }) => LiveCameraViewModel(
    conversation: conversation,
    live: live,
    memory: memory,
    settings: settings,
    foreground: foreground,
    assistant: VoiceAssistant<CameraSideEvent>(
      speech: speech,
      audio: audio,
      responders: CameraTurnResponder(
        capture: live.capture,
        conversation: conversation,
        encode: live.encodeForLlm,
        recall: memory.recall,
      ),
      diagnostics: diagnostics,
    ),
    diagnostics: diagnostics,
    activateStt: (_) async => const Result.ok(null),
    source: _fixture,
  );

  /// Remembers one frame (scene 1) through the sampler.
  Future<void> remember() async {
    capture.results.add(Result.ok(fakeShot(frameId: 5)));
    await memory.tick();
    expect(memory.entries, hasLength(1));
  }

  test(
    'entering samples (the memory is active); leaving forgets everything',
    () async {
      final viewModel = create();
      await until(() => live.state.value is LiveRunning);
      expect(viewModel.memoryStatus, same(memory.status));
      expect(memory.status.value.active, isTrue);
      await remember();
      expect(memory.status.value.frames, 1);

      viewModel.dispose();
      expect(memory.entries, isEmpty);
      expect(memory.status.value.active, isFalse);
    },
  );

  test('a trip to the background (inactive, then resumed) keeps the '
      'frames: the restart after the release is the same detection session; '
      'a Retry after a failure is a new one', () async {
    final foreground = AppForeground(platform: TargetPlatform.android);
    addTearDown(foreground.dispose);
    final viewModel = create(foreground: foreground);
    await until(() => live.state.value is LiveRunning);
    await until(() => viewModel.micAccess.result != null);
    await remember();

    foreground.onStateChange(AppLifecycleState.inactive);
    await until(() => live.state.value is LiveStopped);
    foreground.onStateChange(AppLifecycleState.resumed);
    await until(() => live.state.value is LiveRunning);
    await pumpEventQueue();
    expect(memory.entries, hasLength(1), reason: 'kept across the release');
    expect(memory.status.value.frames, 1);

    expect(sources, hasLength(2));
    // A Retry asked for while the app is away is still a new session when
    // it runs on return.
    foreground.onStateChange(AppLifecycleState.inactive);
    await until(() => live.state.value is LiveStopped);
    await viewModel.retryStart();
    foreground.onStateChange(AppLifecycleState.resumed);
    await until(() => live.state.value is LiveRunning);
    await pumpEventQueue();
    expect(memory.entries, isEmpty, reason: 'the deferred Retry cleared');
    await remember();
    expect(sources, hasLength(3));
    sources.last.failAtRuntime(Exception('unplugged'));
    await until(() => live.state.value is LiveFailed);
    expect(memory.entries, hasLength(1), reason: 'a failure clears nothing');
    await viewModel.retryStart();
    await until(() => live.state.value is LiveRunning);
    expect(memory.entries, isEmpty, reason: 'Retry starts a new session');
    viewModel.dispose();
  });

  test('a press makes the demo busy: the sampler takes nothing while the '
      'mic is held (and does take a frame before it)', () async {
    final viewModel = create();
    await until(() => live.state.value is LiveRunning);
    // The control: idle, the first tick asks for a frame.
    await remember();
    expect(capture.calls, hasLength(1));

    await viewModel.pressMic();
    nowMicros += 30000000; // well past any gap: only the press stops it
    capture.results.add(Result.ok(fakeShot(frameId: 6, scene: 2)));
    await memory.tick();
    expect(capture.calls, hasLength(1), reason: 'nothing while held');
    expect(memory.entries, hasLength(1));
    viewModel.dispose();
  });

  test('a memory question puts the remembered frame on the card; the next '
      'press closes it; "Clear memory" forgets it all', () async {
    final viewModel = create();
    await until(() => live.state.value is LiveRunning && viewModel.chatReady);
    await remember();
    nowMicros += 12000000;

    await viewModel.pressMic();
    await viewModel.releaseMic();
    await until(() => viewModel.memoryCard.value != null);
    final card = viewModel.memoryCard.value!;
    expect(card.png, memory.entries.single.png);
    expect(card.age, const Duration(seconds: 12));
    expect(card.query, 'yellow mug');
    expect(card.cosine, greaterThan(0.9));
    expect(viewModel.exchange.route, startsWith('memory · seen 12 s ago'));
    await until(() => viewModel.exchange.answer != null);
    expect(viewModel.exchange.answer, startsWith('I saw it 12 seconds ago.'));

    await viewModel.pressMic();
    expect(viewModel.memoryCard.value, isNull);
    await viewModel.releaseMic();

    viewModel.closeMemoryCard();
    viewModel.clearMemory();
    expect(memory.entries, isEmpty);
    viewModel.dispose();
  });

  test('a switch setting that cannot be read keeps the memory off (no image '
      'encoder load on a guess) and says why; switching it on works and is '
      'saved', () async {
    final store = InMemorySettingsStore()..failWith = Exception('prefs broken');
    embedder.unavailable = 'EmbeddingGemma was loaded without images';
    final viewModel = create(
      settings: LiveCameraSettingsRepository(
        settings: TypedSettings(store: store),
        environment: _fixture,
      ),
    );
    await until(() => viewModel.memoryEnabled != null);
    expect(viewModel.memoryEnabled, isFalse);
    expect(viewModel.memorySettingError, contains('prefs broken'));
    expect(memory.status.value.enabled, isFalse);
    expect(embedder.enables, 0);

    store.failWith = null;
    await viewModel.setMemoryEnabled.execute(true);
    expect(viewModel.memoryEnabled, isTrue);
    expect(viewModel.memorySettingError, isNull);
    expect(store.values['camera.visualMemory'], true);
    await until(() => embedder.enables == 1);
    viewModel.dispose();
  });

  test(
    'a failed save keeps the switch as set for this visit and says so',
    () async {
      final store = InMemorySettingsStore()
        ..failWritesOf.add('camera.visualMemory');
      final viewModel = create(
        settings: LiveCameraSettingsRepository(
          settings: TypedSettings(store: store),
          environment: _fixture,
        ),
      );
      await until(() => viewModel.memoryEnabled != null);
      expect(viewModel.memoryEnabled, isTrue);
      await viewModel.setMemoryEnabled.execute(false);
      expect(viewModel.memoryEnabled, isFalse);
      expect(memory.status.value.enabled, isFalse);
      expect(
        viewModel.memorySettingError,
        contains('Could not save the visual memory setting'),
      );
      viewModel.dispose();
    },
  );
}
