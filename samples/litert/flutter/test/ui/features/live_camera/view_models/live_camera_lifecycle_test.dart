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

// Demo 3 and the app's lifecycle: on Android and iOS leaving the app
// releases the camera (and ends a held push-to-talk press without a turn),
// coming back starts it again; desktop keeps both.
import 'dart:async';
import 'dart:ui' show AppLifecycleState;

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/data/repositories/live_detection_repository.dart';
import 'package:litert_edge_demos/domain/models/camera_side_event.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/domain/use_cases/camera_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/ui/core/app_foreground.dart';
import 'package:litert_edge_demos/ui/features/live_camera/view_models/live_camera_view_model.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../../fakes/fake_audio_repository.dart';
import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_detector.dart';
import '../../../../fakes/fake_frame_source.dart';
import '../../../../fakes/fake_speech.dart';

/// Polls [condition], bounded by real time.
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
  late List<FakeFrameSource> sources;
  late LiveDetectionRepository live;
  late ValueNotifier<Map<ModelId, ModelState>> models;
  late DiagnosticsRepository diagnostics;
  late FakeRecognizer recognizer;
  late FakeAudioRepository audio;
  late AppForeground foreground;

  /// Holds the next source's start (a camera still opening, a network
  /// camera still connecting).
  Completer<void>? nextStartGate;

  setUp(() {
    audio = FakeAudioRepository()..autoDrain = true;
    recognizer = FakeRecognizer('How many cats do you see?');
    models = ValueNotifier(const {});
    diagnostics = DiagnosticsRepository(
      models: models,
      minInterval: Duration.zero,
      sampleRss: false,
    );
    conversation = FakeConversationRepository();
    sources = [];
    nextStartGate = null;
    live = LiveDetectionRepository(
      detector: FakeDetector(autoComplete: true),
      createSource: (_) {
        final s = FakeFrameSource()..startGate = nextStartGate;
        nextStartGate = null;
        sources.add(s);
        return Result.ok(s);
      },
    );
    foreground = AppForeground(platform: TargetPlatform.android);
  });

  tearDown(() async {
    await live.close();
    await conversation.close();
    diagnostics.dispose();
    models.dispose();
    foreground.dispose();
  });

  Future<LiveCameraViewModel> create() async => LiveCameraViewModel(
    conversation: conversation,
    live: live,
    assistant: VoiceAssistant<CameraSideEvent>(
      speech: await loadedSpeech(
        recognizer: recognizer,
        synthesizer: RecordingSynth(),
      ),
      audio: audio,
      responders: CameraTurnResponder(
        capture: live.capture,
        conversation: conversation,
        encode: live.encodeForLlm,
      ),
      diagnostics: diagnostics,
    ),
    diagnostics: diagnostics,
    activateStt: (_) async => const Result.ok(null),
    source: _fixture,
    foreground: foreground,
  );

  /// Live detection runs and the entry's microphone request has ended (its
  /// OS dialog would make the app inactive on its own).
  Future<void> running(LiveCameraViewModel viewModel) => until(
    () => live.state.value is LiveRunning && viewModel.micAccess.result != null,
  );

  void leave() => foreground
    ..onStateChange(AppLifecycleState.inactive)
    ..onStateChange(AppLifecycleState.hidden)
    ..onStateChange(AppLifecycleState.paused);

  void comeBack() => foreground
    ..onStateChange(AppLifecycleState.hidden)
    ..onStateChange(AppLifecycleState.inactive)
    ..onStateChange(AppLifecycleState.resumed);

  test('leaving the app releases the running camera at inactive; coming '
      'back starts a new source', () async {
    final viewModel = await create();
    await running(viewModel);

    foreground.onStateChange(AppLifecycleState.inactive);
    await until(() => live.state.value is LiveStopped);
    expect(sources.single.stopped, isTrue);
    expect(viewModel.preview.value, isNull);

    foreground
      ..onStateChange(AppLifecycleState.hidden)
      ..onStateChange(AppLifecycleState.paused);
    comeBack();
    await until(() => live.state.value is LiveRunning);
    expect(sources, hasLength(2));
    expect(sources.last.running, isTrue);
    expect(viewModel.startLive.result, isA<Ok<FrameSourceInfo>>());
    viewModel.dispose();
  });

  test('a held push-to-talk press ends without a turn when the app is '
      'left; the release that follows is ignored', () async {
    final viewModel = await create();
    await running(viewModel);
    await viewModel.pressMic();
    expect(viewModel.isListening, isTrue);

    leave();
    await until(() => viewModel.phase == TurnPhase.idle);
    expect(audio.captures.single.cancelled, isTrue);

    await viewModel.releaseMic();
    expect(recognizer.calls, 0);
    expect(viewModel.exchange.question, isNull);
    viewModel.dispose();
  });

  test('a source still starting when the app is left (a permission dialog, '
      'a network camera connecting) finishes its start, is released, and '
      'starts again on return', () async {
    final gate = nextStartGate = Completer<void>();
    final viewModel = await create();
    await until(() => live.state.value is LiveStarting);

    leave();
    await pumpEventQueue();
    expect(sources.single.stopped, isFalse, reason: 'the start goes on');

    gate.complete();
    await until(() => sources.single.stopped);
    await until(() => live.state.value is LiveStopped);

    comeBack();
    await until(() => live.state.value is LiveRunning);
    expect(sources, hasLength(2));
    viewModel.dispose();
  });

  test("the screen's own microphone dialog (inactive) keeps the camera; "
      'putting the app away with the dialog up releases it', () async {
    audio.micAccessGate = Completer<void>();
    final viewModel = await create();
    await until(() => viewModel.micAccess.running);
    expect(live.state.value, isA<LiveRunning>());

    foreground.onStateChange(AppLifecycleState.inactive);
    await pumpEventQueue();
    expect(live.state.value, isA<LiveRunning>());
    expect(sources.single.running, isTrue);

    foreground.onStateChange(AppLifecycleState.hidden);
    await until(() => live.state.value is LiveStopped);
    expect(sources.single.stopped, isTrue);

    audio.micAccessGate!.complete();
    comeBack();
    await until(() => live.state.value is LiveRunning);
    expect(sources, hasLength(2));
    viewModel.dispose();
  });

  test('a camera permission dialog during the start (inactive only) does not '
      'release the camera that starts behind it', () async {
    final gate = nextStartGate = Completer<void>();
    final viewModel = await create();
    await until(() => live.state.value is LiveStarting);

    foreground.onStateChange(AppLifecycleState.inactive);
    gate.complete();
    await until(() => live.state.value is LiveRunning);
    foreground.onStateChange(AppLifecycleState.resumed);
    await until(() => viewModel.micAccess.result != null);
    await pumpEventQueue();

    expect(sources, hasLength(1));
    expect(sources.single.running, isTrue);
    viewModel.dispose();
  });

  test('a start asked for out of the foreground (Retry, an Apply) opens '
      'nothing until the app comes back, then once', () async {
    final viewModel = await create();
    await running(viewModel);
    leave();
    await until(() => live.state.value is LiveStopped);

    await viewModel.retryStart();
    await pumpEventQueue();
    expect(sources, hasLength(1));
    expect(viewModel.startLive.running, isFalse);

    comeBack();
    await until(() => live.state.value is LiveRunning);
    await pumpEventQueue();
    expect(sources, hasLength(2));
    viewModel.dispose();
  });

  test('a failed source stays failed: coming back does not retry '
      'it', () async {
    final viewModel = await create();
    await running(viewModel);
    sources.single.failAtRuntime(Exception('unplugged'));
    await until(() => live.state.value is LiveFailed);

    leave();
    comeBack();
    await pumpEventQueue();
    expect(live.state.value, isA<LiveFailed>());
    expect(sources, hasLength(1));
    viewModel.dispose();
  });

  test('a screen left while out of the foreground starts nothing on '
      'return', () async {
    final viewModel = await create();
    await running(viewModel);
    leave();
    await until(() => live.state.value is LiveStopped);

    viewModel.dispose();
    comeBack();
    await pumpEventQueue();
    expect(sources, hasLength(1));
    expect(live.state.value, isA<LiveStopped>());
  });

  test('on desktop the camera and a held press stay when the window loses '
      'focus or is hidden', () async {
    foreground.dispose();
    foreground = AppForeground(platform: TargetPlatform.macOS);
    final viewModel = await create();
    await running(viewModel);
    await viewModel.pressMic();

    leave();
    await pumpEventQueue();
    expect(live.state.value, isA<LiveRunning>());
    expect(sources.single.running, isTrue);
    expect(viewModel.isListening, isTrue);
    expect(audio.captures.single.isOpen, isTrue);
    viewModel.dispose();
  });
}
