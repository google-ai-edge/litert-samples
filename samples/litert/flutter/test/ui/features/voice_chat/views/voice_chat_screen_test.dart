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
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/data/repositories/skill_repository.dart';
import 'package:litert_edge_demos/domain/models/chat_capabilities.dart';
import 'package:litert_edge_demos/domain/models/llm_image.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/ui/core/level_meter.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/view_models/voice_chat_view_model.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/chat_keys.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/voice_chat_screen.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:provider/provider.dart';

import '../../../../fakes/fake_audio_repository.dart';
import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_image_input.dart';
import '../../../../fakes/fake_skill_store.dart';
import '../../../../fakes/fake_speech.dart';
import '../../../../support/pcm.dart';

void main() {
  late FakeConversationRepository conversation;
  late FakeAudioRepository audio;
  late ValueNotifier<Map<ModelId, ModelState>> models;
  late DiagnosticsRepository diagnostics;
  late FakeImageInputService imageInput;
  late SkillRepository skills;
  Result<void> sttResult = const Result.ok(null);

  Future<VoiceChatViewModel> pumpScreen(WidgetTester tester) async {
    final speech = (await tester.runAsync(
      () => loadedSpeech(
        recognizer: FakeRecognizer(),
        synthesizer: RecordingSynth(),
      ),
    ))!;
    final viewModel = VoiceChatViewModel(
      activateStt: (_) async => sttResult,
      conversation: conversation,
      diagnostics: diagnostics,
      images: fakeImageRepository(imageInput),
      skills: skills,
      assistant: VoiceAssistant(
        speech: speech,
        audio: audio,
        responders: ChatTurnResponder(conversation: conversation),
        diagnostics: diagnostics,
      ),
    );
    await tester.pumpWidget(
      ChangeNotifierProvider.value(
        value: viewModel,
        child: const MaterialApp(home: VoiceChatScreen()),
      ),
    );
    await tester.pump(); // open() completes
    return viewModel;
  }

  Future<void> tearDownScreen(
    WidgetTester tester,
    VoiceChatViewModel viewModel,
  ) async {
    await tester.pumpWidget(const SizedBox.shrink());
    viewModel.dispose();
    await tester.pump(Duration.zero);
    diagnostics.dispose();
    models.dispose();
    skills.dispose();
    await conversation.close();
  }

  setUp(() {
    skills = SkillRepository(store: FakeSkillStore());
    sttResult = const Result.ok(null);
    conversation = FakeConversationRepository();
    imageInput = FakeImageInputService();
    audio = FakeAudioRepository()..autoDrain = true;
    models = ValueNotifier(const {});
    diagnostics = DiagnosticsRepository(
      models: models,
      minInterval: Duration.zero,
      sampleRss: false,
    );
  });

  testWidgets('a failed open shows the error and New conversation recovers '
      'the chat without a restart', (tester) async {
    conversation.openResult = Result.error(Exception('no model'));
    final viewModel = await pumpScreen(tester);

    IconButton button(Key key) => tester.widget<IconButton>(find.byKey(key));
    expect(find.textContaining('no model'), findsOneWidget);
    expect(button(ChatKeys.send).onPressed, isNull);
    // In the app bar's More menu.
    await tester.tap(find.byKey(ChatKeys.more));
    await tester.pumpAndSettle();
    expect(
      tester
          .widget<PopupMenuItem<void>>(find.byKey(ChatKeys.newConversation))
          .enabled,
      isTrue,
    );

    conversation.openResult = const Result.ok(null);
    await tester.tap(find.byKey(ChatKeys.newConversation));
    await tester.pumpAndSettle();

    expect(find.textContaining('no model'), findsNothing);
    await tester.enterText(find.byKey(ChatKeys.input), 'Hi');
    await tester.pump();
    expect(button(ChatKeys.send).onPressed, isNotNull);

    await tearDownScreen(tester, viewModel);
  });

  testWidgets('holding the mic shows Listening; a silent release shows '
      '"Didn\'t catch that"', (tester) async {
    audio.nextUtterance = Utterance(
      pcm: quiet(const Duration(seconds: 1)),
      held: const Duration(seconds: 1),
    );
    final viewModel = await pumpScreen(tester);
    expect(
      find.text('Hold the mic to talk, or type a question'),
      findsOneWidget,
    );

    final gesture = await tester.startGesture(
      tester.getCenter(find.byKey(ChatKeys.mic)),
    );
    await tester.pump();
    await tester.pump();
    expect(viewModel.phase, TurnPhase.listening);
    expect(find.text('Listening… release to send'), findsOneWidget);
    expect(find.byIcon(Icons.mic), findsOneWidget);

    await gesture.up();
    await tester.pump();
    await tester.pump();
    expect(find.textContaining("Didn't catch that"), findsOneWidget);
    expect(conversation.prompts, isEmpty);

    await tearDownScreen(tester, viewModel);
  });

  group('a slow mic start (the audio warm-up still running)', () {
    String? phaseLine(WidgetTester tester) =>
        tester.widget<Text>(find.byKey(ChatKeys.phase)).data;

    Finder inMic(Finder matching) =>
        find.descendant(of: find.byKey(ChatKeys.mic), matching: matching);

    testWidgets('a press shows "Opening the mic…" on the phase line and the '
        'button, and Listening only once the capture runs', (tester) async {
      audio.nextUtterance = Utterance(
        pcm: quiet(const Duration(seconds: 1)),
        held: const Duration(seconds: 1),
      );
      final viewModel = await pumpScreen(tester);
      audio.startGate = Completer<void>();

      final gesture = await tester.startGesture(
        tester.getCenter(find.byKey(ChatKeys.mic)),
      );
      await tester.pump();
      await tester.pump();
      expect(viewModel.isOpeningMic, isTrue);
      expect(viewModel.isListening, isFalse);
      expect(phaseLine(tester), kOpeningMicLabel);
      expect(find.text('Listening… release to send'), findsNothing);
      expect(inMic(find.bySemanticsLabel(kOpeningMicLabel)), findsOneWidget);
      expect(inMic(find.byType(CircularProgressIndicator)), findsOneWidget);
      expect(inMic(find.byIcon(Icons.mic)), findsNothing);

      audio.startGate!.complete();
      await tester.pump();
      await tester.pump();
      expect(viewModel.isListening, isTrue);
      expect(phaseLine(tester), 'Listening… release to send');
      expect(inMic(find.bySemanticsLabel('Release to send')), findsOneWidget);
      expect(inMic(find.byType(CircularProgressIndicator)), findsNothing);
      expect(inMic(find.byIcon(Icons.mic)), findsOneWidget);

      await gesture.up();
      await tester.pump();
      await tester.pump();
      expect(find.textContaining("Didn't catch that"), findsOneWidget);

      await tearDownScreen(tester, viewModel);
    });

    testWidgets('a release before the capture starts says to wait for '
        'Listening; nothing is transcribed', (tester) async {
      final viewModel = await pumpScreen(tester);
      audio.startGate = Completer<void>();

      final gesture = await tester.startGesture(
        tester.getCenter(find.byKey(ChatKeys.mic)),
      );
      await tester.pump();
      await gesture.up();
      await tester.pump();
      expect(phaseLine(tester), kOpeningMicLabel, reason: 'still starting');

      audio.startGate!.complete();
      await tester.pump();
      await tester.pump();
      expect(
        find.text(
          'The mic was still opening — wait for Listening… before you speak.',
        ),
        findsOneWidget,
      );
      expect(find.textContaining("Didn't catch that"), findsNothing);
      expect(phaseLine(tester), 'Hold the mic to talk, or type a question');
      expect(audio.captures.single.cancelled, isTrue);
      expect(conversation.prompts, isEmpty);
      expect(viewModel.phase, TurnPhase.idle);

      await tearDownScreen(tester, viewModel);
    });
  });

  testWidgets('the speak-replies toggle flips the setting', (tester) async {
    final viewModel = await pumpScreen(tester);
    expect(find.byIcon(Icons.volume_up), findsOneWidget);

    await tester.tap(find.byKey(ChatKeys.speakReplies));
    await tester.pump();

    expect(viewModel.speakReplies, isFalse);
    expect(find.byIcon(Icons.volume_off_outlined), findsOneWidget);

    await tearDownScreen(tester, viewModel);
  });

  testWidgets('attachment bar: gallery attaches a thumbnail, the next '
      "question's entry shows it, and remove detaches it", (tester) async {
    final viewModel = await pumpScreen(tester);
    expect(find.byKey(ChatKeys.attachGallery), findsOneWidget);
    expect(
      find.byKey(ChatKeys.attachCamera),
      findsNothing,
      reason: 'no camera in the picker on this platform',
    );
    expect(find.byKey(ChatKeys.attachmentThumbnail), findsNothing);

    await tester.tap(find.byKey(ChatKeys.attachGallery));
    await tester.pump();
    await tester.pump();
    expect(find.byKey(ChatKeys.attachmentThumbnail), findsOneWidget);
    expect(find.textContaining('1024×768'), findsOneWidget);

    await tester.enterText(find.byKey(ChatKeys.input), 'What is this?');
    await tester.pump();
    await tester.tap(find.byKey(ChatKeys.send));
    await tester.pump();
    await tester.pump();
    expect(find.byKey(ChatKeys.entryImage), findsOneWidget);
    expect(conversation.images.single, same(viewModel.attachment!.png));
    conversation.emit('A cat.');
    await conversation.finish();
    await tester.pump();

    await tester.tap(find.byKey(ChatKeys.removeAttachment));
    await tester.pump();
    expect(find.byKey(ChatKeys.attachmentThumbnail), findsNothing);
    expect(viewModel.attachment, isNull);
    expect(
      find.byKey(ChatKeys.entryImage),
      findsOneWidget,
      reason: 'the sent entry keeps its thumbnail',
    );

    await tearDownScreen(tester, viewModel);
  });

  testWidgets('a chat model without images or tools: the photo buttons are '
      'disabled with the reason, and the skills line says why', (tester) async {
    conversation.capabilities = const ChatCapabilities(
      modelName: 'Gemma 3 NPU',
      images: false,
      tools: false,
    );
    final viewModel = await pumpScreen(tester);

    final gallery = tester.widget<IconButton>(
      find.byKey(ChatKeys.attachGallery),
    );
    expect(gallery.onPressed, isNull);
    expect(gallery.tooltip, contains('Photos are off'));
    expect(
      tester.widget<Text>(find.byKey(ChatKeys.photosOff)).data,
      contains('Gemma 3 NPU was set up without images'),
    );
    expect(find.byKey(ChatKeys.skillsOff), findsNothing, reason: 'no skills');
    await tearDownScreen(tester, viewModel);
  });

  testWidgets('the camera button shows where the picker has a camera', (
    tester,
  ) async {
    imageInput = FakeImageInputService(
      supported: {ImageSourceKind.gallery, ImageSourceKind.camera},
    );
    final viewModel = await pumpScreen(tester);

    await tester.tap(find.byKey(ChatKeys.attachCamera));
    await tester.pump();
    await tester.pump();

    expect(imageInput.picks, [ImageSourceKind.camera]);
    expect(find.byKey(ChatKeys.attachmentThumbnail), findsOneWidget);

    await tearDownScreen(tester, viewModel);
  });

  testWidgets('a failed recognizer switch shows a lasting bar '
      'with Retry; a successful retry removes it', (tester) async {
    sttResult = Result.error(Exception('whisper missing'));
    final viewModel = await pumpScreen(tester);
    await tester.pump();

    expect(find.byKey(ChatKeys.sttError), findsOneWidget);
    expect(find.textContaining('whisper missing'), findsOneWidget);

    sttResult = const Result.ok(null);
    await tester.tap(find.byKey(ChatKeys.retryStt));
    await tester.pump();
    await tester.pump();

    expect(find.byKey(ChatKeys.sttError), findsNothing);
    await tearDownScreen(tester, viewModel);
  });
}
