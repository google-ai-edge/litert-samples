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
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/data/repositories/live_detection_repository.dart';
import 'package:litert_edge_demos/data/repositories/speech_repository.dart';
import 'package:litert_edge_demos/domain/models/camera_side_event.dart';
import 'package:litert_edge_demos/domain/models/chat_capabilities.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/use_cases/camera_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/ui/core/detection_painter.dart';
import 'package:litert_edge_demos/ui/core/level_meter.dart';
import 'package:litert_edge_demos/ui/core/warning_color.dart';
import 'package:litert_edge_demos/ui/features/live_camera/view_models/live_camera_view_model.dart';
import 'package:litert_edge_demos/ui/features/live_camera/views/live_camera_screen.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:provider/provider.dart';

import '../../../../fakes/fake_audio_repository.dart';
import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_detector.dart';
import '../../../../fakes/fake_frame_source.dart';
import '../../../../fakes/fake_speech.dart';
import '../../../../support/frames.dart';

void main() {
  late FakeConversationRepository conversation;
  late FakeFrameSource source;
  late LiveDetectionRepository live;
  late int clock; // µs, driven by the test
  late ValueNotifier<Map<ModelId, ModelState>> models;
  late DiagnosticsRepository diagnostics;

  setUp(() {
    conversation = FakeConversationRepository();
    source = FakeFrameSource(label: 'fixture (2 images)');
    clock = 0;
    live = LiveDetectionRepository(
      detector: FakeDetector(autoComplete: true),
      createSource: (_) => Result.ok(source),
      clockMicros: () => clock,
    );
    models = ValueNotifier(const {});
    diagnostics = DiagnosticsRepository(
      models: models,
      minInterval: Duration.zero,
      sampleRss: false,
    );
  });

  Future<void> pumpScreen(
    WidgetTester tester, {
    FakeAudioRepository? audio,
  }) async {
    final SpeechRepository speech = (await tester.runAsync(
      () => loadedSpeech(
        recognizer: FakeRecognizer(),
        synthesizer: RecordingSynth(),
      ),
    ))!;
    await tester.pumpWidget(
      MaterialApp(
        home: ChangeNotifierProvider(
          create: (_) => LiveCameraViewModel(
            conversation: conversation,
            live: live,
            diagnostics: diagnostics,
            activateStt: (_) async => const Result.ok(null),
            assistant: VoiceAssistant<CameraSideEvent>(
              speech: speech,
              audio: audio ?? (FakeAudioRepository()..autoDrain = true),
              responders: CameraTurnResponder(
                capture: live.capture,
                conversation: conversation,
                encode: live.encodeForLlm,
              ),
              diagnostics: diagnostics,
            ),
            source: const Result.ok(FixtureSourceSpec(['/fixtures'])),
          ),
          child: const LiveCameraScreen(),
        ),
      ),
    );
    await tester.runAsync(() => Future<void>.delayed(Duration.zero));
    await tester.pump();
  }

  Future<void> tearDownScreen(WidgetTester tester) async {
    await tester.pumpWidget(const SizedBox.shrink());
    await tester.pump(Duration.zero);
    await tester.runAsync(live.close);
    await conversation.close();
    diagnostics.dispose();
    models.dispose();
  }

  testWidgets('a failed chat open shows the error and a Retry that reopens '
      'the chat', (tester) async {
    conversation.openResult = Result.error(Exception('no model'));
    await pumpScreen(tester);
    expect(find.textContaining('no model'), findsOneWidget);

    conversation.openResult = const Result.ok(null);
    await tester.tap(find.byKey(LiveCameraKeys.retryChat));
    await tester.pump();

    expect(conversation.openedProfiles, [kCameraProfile, kCameraProfile]);
    expect(find.textContaining('no model'), findsNothing);
    expect(find.textContaining('Chat ready'), findsOneWidget);
    await tearDownScreen(tester);
  });

  testWidgets('a chat model without images: the voice bar says detailed '
      'answers are off', (tester) async {
    conversation.capabilities = const ChatCapabilities(
      modelName: 'Gemma 3 NPU',
      images: false,
      tools: false,
    );
    await pumpScreen(tester);

    final label = tester.widget<Text>(find.byKey(LiveCameraKeys.detailedOff));
    expect(label.data, contains('Detailed answers are off'));
    expect(label.data, contains('Gemma 3 NPU'));
    expect(find.textContaining('send the frame to'), findsNothing);
    await tearDownScreen(tester);
  });

  testWidgets('running: the boxes layer paints the live frames; the chip '
      'names the source and the detector; paused shows the reason', (
    tester,
  ) async {
    await pumpScreen(tester);
    expect(live.state.value, isA<LiveRunning>());
    expect(find.text('fixture (2 images) · GPU fp32 full'), findsOneWidget);
    final paint = tester.widget<CustomPaint>(find.byKey(LiveCameraKeys.boxes));
    expect((paint.painter! as DetectionPainter).frames, same(live.frames));

    live.setDuty(DetectorDuty.paused, reason: 'Gemma');
    await tester.pump();
    expect(find.text('Detector paused (Gemma)'), findsOneWidget);
    await tearDownScreen(tester);
  });

  testWidgets('black frames show an amber warning chip above the status, '
      'which clears when the luma recovers', (tester) async {
    await pumpScreen(tester);
    Future<void> emit(int count, int fill) async {
      for (var i = 0; i < count; i++) {
        clock += 66667; // 15 fps
        source.emit(TestFrame.rgba(fill: fill));
        await tester.pump();
      }
    }

    await emit(10, 0);
    expect(find.byKey(LiveCameraKeys.blackFrames), findsNothing);

    await emit(30, 0); // 2.7 s of black
    expect(find.byKey(LiveCameraKeys.blackFrames), findsOneWidget);
    expect(find.textContaining('Camera delivers black frames'), findsOneWidget);
    final chip = tester.widget<DecoratedBox>(
      find.byKey(LiveCameraKeys.blackFrames),
    );
    expect((chip.decoration as BoxDecoration).color, kWarningColor);
    expect(live.state.value, isA<LiveRunning>(), reason: 'not a failure');
    expect(find.text('fixture (2 images) · GPU fp32 full'), findsOneWidget);

    await emit(16, 128);
    expect(find.byKey(LiveCameraKeys.blackFrames), findsNothing);
    await tearDownScreen(tester);
  });

  testWidgets('a live failure shows its message and a Retry', (tester) async {
    source.startResult = const Result.error(
      FrameSourceUnavailableException('No images in /fixtures'),
    );
    await pumpScreen(tester);
    expect(find.textContaining('No images'), findsWidgets);
    expect(find.byKey(LiveCameraKeys.retryLive), findsOneWidget);
    await tearDownScreen(tester);
  });

  testWidgets('a press during a slow mic start shows "Opening the mic…" on '
      'the voice bar and the button, and Listening only once the capture '
      'runs', (tester) async {
    final audio = FakeAudioRepository()
      ..autoDrain = true
      ..startGate = Completer<void>();
    await pumpScreen(tester, audio: audio);
    final viewModel = Provider.of<LiveCameraViewModel>(
      tester.element(find.byType(LiveCameraScreen)),
      listen: false,
    );
    String? phaseLine() =>
        tester.widget<Text>(find.byKey(LiveCameraKeys.phase)).data;
    Finder inMic(Finder matching) =>
        find.descendant(of: find.byKey(LiveCameraKeys.mic), matching: matching);

    final gesture = await tester.startGesture(
      tester.getCenter(find.byKey(LiveCameraKeys.mic)),
    );
    await tester.pump();
    await tester.pump();
    expect(viewModel.isOpeningMic, isTrue);
    expect(viewModel.isListening, isFalse);
    expect(phaseLine(), kOpeningMicLabel);
    expect(find.text('Listening… release to ask'), findsNothing);
    expect(inMic(find.bySemanticsLabel(kOpeningMicLabel)), findsOneWidget);
    expect(inMic(find.byType(CircularProgressIndicator)), findsOneWidget);

    audio.startGate!.complete();
    await tester.pump();
    await tester.pump();
    expect(viewModel.isListening, isTrue);
    expect(phaseLine(), 'Listening… release to ask');
    expect(inMic(find.byType(CircularProgressIndicator)), findsNothing);
    expect(inMic(find.byIcon(Icons.mic)), findsOneWidget);

    unawaited(viewModel.stop.execute());
    await gesture.up();
    await tester.pump();
    await tearDownScreen(tester);
  });

  testWidgets('a release before the mic opens says to wait for Listening in '
      'the caption; nothing is asked', (tester) async {
    final audio = FakeAudioRepository()
      ..autoDrain = true
      ..startGate = Completer<void>();
    await pumpScreen(tester, audio: audio);

    final gesture = await tester.startGesture(
      tester.getCenter(find.byKey(LiveCameraKeys.mic)),
    );
    await tester.pump();
    await gesture.up();
    await tester.pump();
    expect(
      tester.widget<Text>(find.byKey(LiveCameraKeys.phase)).data,
      kOpeningMicLabel,
    );

    audio.startGate!.complete();
    await tester.pump();
    await tester.pump();
    expect(
      find.descendant(
        of: find.byKey(LiveCameraKeys.caption),
        matching: find.text(
          'The mic was still opening — wait for Listening… before you speak.',
        ),
      ),
      findsOneWidget,
    );
    expect(
      tester.widget<Text>(find.byKey(LiveCameraKeys.phase)).data,
      'Hold the mic and ask about the scene',
    );
    expect(audio.captures.single.cancelled, isTrue);
    expect(conversation.prompts, isEmpty);

    await tearDownScreen(tester);
  });

  testWidgets('holding and releasing the mic asks about the frame: the '
      'caption shows the question, the answer and the route chip', (
    tester,
  ) async {
    await pumpScreen(tester);
    expect(find.byKey(LiveCameraKeys.mic), findsOneWidget);
    expect(find.byKey(LiveCameraKeys.caption), findsNothing);

    final gesture = await tester.startGesture(
      tester.getCenter(find.byKey(LiveCameraKeys.mic)),
    );
    await tester.pump();
    expect(find.text('Listening… release to ask'), findsOneWidget);

    await gesture.up();
    // Real async for the turn (VoiceSession completes on the root zone);
    // the frame after the release is the snapshot. The wait
    // for what it leads to is settleTurn's below.
    await tester.runAsync(() async {
      await pumpEventQueue();
      clock += 100000;
      source.emit(TestFrame.rgba());
      await pumpEventQueue();
    });
    await tester.pump();

    // A capital-of question is not a detector question: the frame goes to
    // Gemma, and the view freezes on it while Gemma answers. The turn runs
    // in two zones (it started from the gesture; VoiceSession completes on
    // the root zone): alternate frames and real time.
    Future<void> settleTurn(bool Function() done) async {
      for (var i = 0; i < 100 && !done(); i++) {
        await tester.pump(const Duration(milliseconds: 5));
        await tester.runAsync(
          () => Future<void>.delayed(const Duration(milliseconds: 5)),
        );
      }
      await tester.pump();
    }

    await settleTurn(
      () =>
          conversation.prompts.isNotEmpty &&
          find.byKey(LiveCameraKeys.frozen).evaluate().isNotEmpty,
    );
    expect(conversation.prompts.single, contains('Question: What is the'));
    expect(find.text('“What is the capital of France?”'), findsOneWidget);
    expect(find.byKey(LiveCameraKeys.routeChip), findsOneWidget);
    expect(find.textContaining('→ Gemma'), findsOneWidget);
    expect(find.byKey(LiveCameraKeys.frozen), findsOneWidget);
    expect(find.byKey(LiveCameraKeys.frozenLabel), findsOneWidget);

    conversation.emit('It is Paris.');
    // Completes once the responder took the done event (in the fake zone).
    unawaited(conversation.finish());
    // Until the turn is over (committed after playback drained), not just
    // until the streamed text shows.
    await settleTurn(
      () => find
          .text('Hold the mic and ask about the scene')
          .evaluate()
          .isNotEmpty,
    );
    expect(find.text('It is Paris.'), findsOneWidget);
    expect(
      find.byKey(LiveCameraKeys.frozen),
      findsNothing,
      reason: 'spoken (playback drained): back to the live view',
    );
    await tearDownScreen(tester);
  });
}
