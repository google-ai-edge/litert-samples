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
import 'package:flutter/services.dart' show PlatformException;
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show Skill, SkillType;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/data/repositories/image_repository.dart';
import 'package:litert_edge_demos/data/repositories/skill_repository.dart';
import 'package:litert_edge_demos/data/repositories/speech_repository.dart';
import 'package:litert_edge_demos/data/services/images/image_normalizer.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/chat_capabilities.dart';
import 'package:litert_edge_demos/domain/models/chat_entry.dart';
import 'package:litert_edge_demos/domain/models/llm_image.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/view_models/voice_chat_view_model.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../../fakes/fake_audio_repository.dart';
import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_image_input.dart';
import '../../../../fakes/fake_skill_store.dart';
import '../../../../fakes/fake_speech.dart';
import '../../../../support/pcm.dart';

/// Lets queued microtasks and zero-length timers run (VoiceSession awaits
/// several times per turn).
Future<void> settle() => pumpEventQueue();

const _skill = Skill(
  name: 'timer',
  description: 'Timers.',
  instructions: 'Call run_intent.',
  type: SkillType.intent,
);

void main() {
  late FakeConversationRepository conversation;
  late FakeAudioRepository audio;
  late FakeRecognizer recognizer;
  late RecordingSynth synth;
  late SpeechRepository speech;
  late ValueNotifier<Map<ModelId, ModelState>> models;
  late DiagnosticsRepository diagnostics;
  late FakeImageInputService imageInput;
  late SkillRepository skills;
  late VoiceChatViewModel viewModel;

  final sttActivations = <ModelId>[];
  Result<void> sttResult = const Result.ok(null);

  VoiceChatViewModel createViewModel() => VoiceChatViewModel(
    activateStt: (id) async {
      sttActivations.add(id);
      return sttResult;
    },
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

  setUp(() async {
    conversation = FakeConversationRepository();
    skills = SkillRepository(store: FakeSkillStore());
    imageInput = FakeImageInputService();
    audio = FakeAudioRepository()..autoDrain = true;
    recognizer = FakeRecognizer();
    synth = RecordingSynth();
    speech = await loadedSpeech(recognizer: recognizer, synthesizer: synth);
    models = ValueNotifier(const {});
    diagnostics = DiagnosticsRepository(
      models: models,
      minInterval: Duration.zero,
      sampleRss: false,
    );
    sttActivations.clear();
    sttResult = const Result.ok(null);
    viewModel = createViewModel();
    await settle();
  });

  test('a retry by New conversation that fails again stays '
      'visible (it is awaited, not wiped with the old entries)', () async {
    viewModel.dispose();
    sttResult = Result.error(Exception('whisper missing'));
    viewModel = createViewModel();
    await settle();

    final before = sttActivations.length;
    await viewModel.newConversation.execute();

    expect(sttActivations.length, before + 1, reason: 'the retry ran');
    expect(viewModel.sttError, contains('whisper missing'));
    expect(viewModel.entries, isEmpty, reason: 'a fresh conversation');

    sttResult = const Result.ok(null);
    await viewModel.selectStt.execute(); // the Retry button
    expect(viewModel.sttError, isNull);
  });

  test('entering makes Whisper the active recognizer; a failed switch is '
      'shown and New conversation retries it', () async {
    expect(sttActivations, [ModelId.whisperBase]);
    expect(viewModel.error, isNull);
    viewModel.dispose();

    sttResult = Result.error(Exception('whisper missing'));
    viewModel = createViewModel();
    await settle();
    expect(viewModel.sttError, contains('whisper missing'));

    sttResult = const Result.ok(null);
    await viewModel.newConversation.execute();
    await settle();
    expect(sttActivations, [
      ModelId.whisperBase,
      ModelId.whisperBase,
      ModelId.whisperBase,
    ]);
    expect(viewModel.selectStt.result, isA<Ok<void>>());
  });

  tearDown(() async {
    viewModel.dispose();
    await settle();
    diagnostics.dispose();
    models.dispose();
    skills.dispose();
    await conversation.close();
  });

  test('opens the shared chat with the voice-chat profile on creation', () {
    expect(conversation.openedProfiles, [kVoiceChatProfile]);
    expect(viewModel.isReady, isTrue);
    expect(viewModel.canSend, isTrue);
    expect(viewModel.canTalk, isTrue);
    expect(viewModel.entries, isEmpty);
    expect(viewModel.phase, TurnPhase.idle);
    expect(viewModel.speakReplies, isTrue);
    expect(audio.prepareCalls, 1, reason: 'audio warmed up on entry');
  });

  test('while open() waits for the previous demo\'s turn to drain, the chat '
      'is not ready and Send and the mic are disabled', () async {
    viewModel.dispose();
    await conversation.close();
    // Demo 3 left the shared chat open; its turn is still draining.
    conversation = FakeConversationRepository()
      ..leaveOpen(kCameraProfile)
      ..openGate = Completer<void>();
    viewModel = createViewModel();
    await settle();

    expect(viewModel.open.running, isTrue);
    expect(viewModel.isReady, isFalse, reason: 'the open chat is not ours');
    expect(viewModel.canSend, isFalse);
    expect(viewModel.canTalk, isFalse);

    conversation.openGate!.complete();
    await settle();

    expect(viewModel.open.running, isFalse);
    expect(viewModel.isReady, isTrue);
    expect(viewModel.canSend, isTrue);
  });

  test('New conversation during a reply, then leaving before the turn '
      'drains: no late voice-chat open after dispose', () async {
    final sending = viewModel.send.execute('Tell me a story');
    await settle();
    conversation.stopGate = Completer<void>();
    final renewing = viewModel.newConversation.execute();
    await settle();

    viewModel.dispose(); // back to home while the stopped turn drains
    conversation.stopGate!.complete();
    await sending;
    await renewing;
    await settle();

    expect(
      conversation.openedProfiles,
      [kVoiceChatProfile],
      reason:
          'only the initial open; a late one would replace the next '
          "demo's chat",
    );
    viewModel = createViewModel(); // for tearDown
    await settle();
  });

  test('a failed open is visible and blocks sending', () async {
    viewModel.dispose();
    conversation.openResult = Result.error(Exception('no model'));
    viewModel = createViewModel();
    await settle();

    expect(viewModel.isReady, isFalse);
    expect(viewModel.canSend, isFalse);
    expect(viewModel.error, contains('no model'));
    expect(viewModel.entries.single.role, ChatRole.error);
  });

  test('after a failed open, New conversation retries the open and '
      'recovers the chat', () async {
    viewModel.dispose();
    conversation.openResult = Result.error(Exception('no model'));
    viewModel = createViewModel();
    await settle();
    expect(viewModel.canSend, isFalse);

    conversation.openResult = const Result.ok(null);
    final opensBefore = conversation.openCalls;
    await viewModel.newConversation.execute();

    expect(conversation.openCalls, opensBefore + 1);
    expect(conversation.openedProfiles.last, kVoiceChatProfile);
    expect(viewModel.newConversation.completed, isTrue);
    expect(viewModel.isReady, isTrue);
    expect(viewModel.canSend, isTrue);
    expect(viewModel.error, isNull);
    expect(viewModel.entries, isEmpty);
  });

  test('typed turn: tokens update only partialReply, then the reply is '
      'committed after its audio', () async {
    var screenNotifications = 0;
    viewModel.addListener(() => screenNotifications++);
    var bubbleUpdates = 0;
    viewModel.partialReply.addListener(() => bubbleUpdates++);

    final sending = viewModel.send.execute('  Hello?  ');
    await settle();
    expect(conversation.prompts, ['Hello?']);
    expect(recognizer.calls, 0, reason: 'typed turns skip STT');
    expect(viewModel.isGenerating, isTrue);
    expect(viewModel.isStreaming, isTrue);
    expect(viewModel.canSend, isFalse);
    expect(viewModel.entries.single.text, 'Hello?');

    final beforeTokens = screenNotifications;
    conversation.emit('Hi');
    await settle();
    conversation.emit(' there');
    await settle();
    expect(viewModel.partialReply.value, 'Hi there');
    expect(bubbleUpdates, 2);
    expect(screenNotifications, beforeTokens, reason: 'no rebuild per token');

    await conversation.finish();
    await sending;

    expect(viewModel.isStreaming, isFalse);
    expect(viewModel.isGenerating, isFalse);
    expect(viewModel.send.completed, isTrue);
    expect(viewModel.partialReply.value, isEmpty);
    expect(viewModel.entries.map((e) => (e.role, e.text)), [
      (ChatRole.user, 'Hello?'),
      (ChatRole.assistant, 'Hi there'),
    ]);
    expect(viewModel.entries.last.interrupted, isFalse);
    expect(synth.synthesized, ['Hi there']);
    expect(diagnostics.latest.lastGeneration?.chunks, 2);
  });

  test('stop forwards to the repository, keeps the partial reply, and the '
      'next turn works', () async {
    final sending = viewModel.send.execute('Tell me a story');
    await settle();
    conversation.emit('Once upon');
    await settle();

    await viewModel.stop.execute();
    await sending;

    expect(conversation.stopCalls, 1);
    expect(viewModel.isGenerating, isFalse);
    expect(viewModel.entries.last.role, ChatRole.assistant);
    expect(viewModel.entries.last.text, 'Once upon');
    expect(viewModel.entries.last.interrupted, isTrue);
    expect(diagnostics.latest.lastGeneration?.stopped, isTrue);

    final next = viewModel.send.execute('Again');
    await settle();
    conversation.emit('Sure.');
    await conversation.finish();
    await next;
    expect(viewModel.entries.last.text, 'Sure.');
    expect(viewModel.entries.last.interrupted, isFalse);
  });

  test(
    'a failed turn shows a visible error and the next turn clears it',
    () async {
      final sending = viewModel.send.execute('Hi');
      await settle();
      conversation.emit('Par');
      await settle();
      await conversation.fail(Exception('GPU lost'));
      await sending;

      expect(viewModel.send.error, isTrue);
      expect(viewModel.isGenerating, isFalse);
      expect(viewModel.isStreaming, isFalse);
      expect(viewModel.phase, TurnPhase.error);
      expect(viewModel.error, contains('GPU lost'));
      expect(viewModel.entries.map((e) => e.role), [
        ChatRole.user,
        ChatRole.assistant,
        ChatRole.error,
      ]);

      final next = viewModel.send.execute('Retry');
      await settle();
      expect(viewModel.error, isNull);
      conversation.emit('Fine.');
      await conversation.finish();
      await next;
      expect(viewModel.phase, TurnPhase.idle);
    },
  );

  test('an empty reply is a visible error, not an empty bubble', () async {
    final sending = viewModel.send.execute('Hi');
    await settle();
    await conversation.finish();
    await sending;

    expect(viewModel.send.error, isTrue);
    expect(viewModel.error, contains('empty reply'));
    expect(viewModel.entries.map((e) => e.role), [
      ChatRole.user,
      ChatRole.error,
    ]);
  });

  test('new conversation stops a running turn and clears the screen', () async {
    final sending = viewModel.send.execute('Hi');
    await settle();
    conversation.emit('Hel');
    await settle();

    await viewModel.newConversation.execute();
    await sending;

    expect(conversation.stopCalls, greaterThanOrEqualTo(1));
    expect(conversation.openedProfiles, [kVoiceChatProfile, kVoiceChatProfile]);
    expect(viewModel.entries, isEmpty);
    expect(viewModel.error, isNull);
    expect(viewModel.canSend, isTrue);
  });

  test('a failed reopen is visible', () async {
    conversation.openResult = Result.error(Exception('reopen broke'));
    await viewModel.newConversation.execute();

    expect(viewModel.newConversation.error, isTrue);
    expect(viewModel.error, contains('reopen broke'));
    expect(viewModel.canSend, isFalse);
    expect(viewModel.canStartNewConversation, isTrue);
  });

  group('push-to-talk', () {
    test(
      'press, release: the transcript and the reply are committed',
      () async {
        await viewModel.pressMic();
        expect(viewModel.isListening, isTrue);
        expect(viewModel.canSend, isFalse);

        final releasing = viewModel.releaseMic();
        await settle();
        expect(viewModel.entries.single.text, 'What is the capital of France?');
        conversation.emit('Paris.');
        await conversation.finish();
        await releasing;

        expect(viewModel.entries.map((e) => (e.role, e.text)), [
          (ChatRole.user, 'What is the capital of France?'),
          (ChatRole.assistant, 'Paris.'),
        ]);
        expect(viewModel.phase, TurnPhase.idle);
      },
    );

    test(
      'a silent press says "Didn\'t catch that" and makes no LLM call',
      () async {
        audio.nextUtterance = Utterance(
          pcm: quiet(const Duration(seconds: 1)),
          held: const Duration(seconds: 1),
        );
        await viewModel.pressMic();
        await viewModel.releaseMic();

        expect(viewModel.entries.single.role, ChatRole.notice);
        expect(viewModel.entries.single.text, contains("Didn't catch that"));
        expect(viewModel.error, isNull);
        expect(conversation.prompts, isEmpty);
      },
    );

    test('an all-zero capture shows the mic-access error, not "Didn\'t catch '
        'that"', () async {
      audio.nextUtterance = Utterance(
        pcm: zeros(const Duration(seconds: 1)),
        held: const Duration(seconds: 1),
      );
      await viewModel.pressMic();
      await viewModel.releaseMic();

      expect(viewModel.entries.single.role, ChatRole.error);
      expect(viewModel.error, contains('Microphone access is off'));
      expect(conversation.prompts, isEmpty);
    });

    test('pressing the mic during a reply is the barge-in: the reply is '
        'committed as interrupted and the mic is open', () async {
      audio.autoDrain = false;
      final sending = viewModel.send.execute('Tell me a story');
      await settle();
      conversation.emit('The first sentence is here. ');
      await settle();
      expect(viewModel.phase, TurnPhase.speaking);

      await viewModel.pressMic();
      await sending;

      expect(audio.lastPlayback!.stopped, isTrue);
      expect(viewModel.isListening, isTrue);
      expect(viewModel.entries.last.interrupted, isTrue);
      expect(viewModel.entries.last.text, 'The first sentence is here.');
      await viewModel.stop.execute();
      expect(viewModel.phase, TurnPhase.idle);
    });
  });

  test('a second release while the first waits on a draining '
      'interrupt is not dropped (no capture left open at listening)', () async {
    audio.autoDrain = false;
    final sending = viewModel.send.execute('Tell me a story');
    await settle();
    conversation.emit('The first sentence is here. ');
    await settle();
    expect(viewModel.phase, TurnPhase.speaking);

    // The stopped turn drains slowly, so the first release waits on it.
    conversation.stopGate = Completer<void>();
    unawaited(viewModel.pressMic());
    await settle();
    unawaited(viewModel.releaseMic());
    await settle();
    expect(viewModel.phase, TurnPhase.transcribing);

    unawaited(viewModel.pressMic()); // capture B
    await settle();
    expect(viewModel.isListening, isTrue);
    unawaited(viewModel.releaseMic()); // must close capture B
    await settle();

    expect(
      audio.log.where((e) => e == 'capture.stop'),
      hasLength(2),
      reason: 'both captures closed',
    );
    expect(viewModel.isListening, isFalse);
    conversation.stopGate!.complete();
    await sending;
    await settle();
    conversation.emit('Paris.');
    await conversation.finish();
    await settle();
    expect(viewModel.phase, TurnPhase.speaking);
    audio.lastPlayback!.completeDrain();
    await settle();
    expect(viewModel.phase, TurnPhase.idle);
  });

  test('a double press opens one capture and runs one turn', () async {
    unawaited(viewModel.pressMic());
    unawaited(viewModel.pressMic());
    await settle();
    expect(audio.log.where((e) => e == 'startCapture'), hasLength(1));
    unawaited(viewModel.releaseMic());
    await settle();
    conversation.emit('Paris.');
    await conversation.finish();
    await settle();
    expect(conversation.prompts, hasLength(1));
    expect(viewModel.phase, TurnPhase.idle);
  });

  test(
    'speak replies off: the next reply is not synthesized or played',
    () async {
      viewModel.setSpeakReplies(enabled: false);
      expect(viewModel.speakReplies, isFalse);

      final sending = viewModel.send.execute('Hi');
      await settle();
      conversation.emit('Hello there. How are you? ');
      await conversation.finish();
      await sending;

      expect(synth.synthesized, isEmpty);
      expect(audio.playbacks, isEmpty);
      expect(viewModel.entries.last.text, 'Hello there. How are you?');
    },
  );

  group('a chat model without images or tools', () {
    test(
      'photos are off with a reason, and a stale attach opens no picker',
      () async {
        conversation.capabilities = const ChatCapabilities(
          modelName: 'Gemma 3 NPU',
          images: false,
          tools: true,
        );

        expect(viewModel.photosEnabled, isFalse);
        expect(viewModel.photosOffReason, contains('Gemma 3 NPU'));
        expect(viewModel.photosOffReason, contains('without images'));

        await viewModel.attach.execute(ImageSourceKind.gallery);

        expect(imageInput.picks, isEmpty);
        expect(viewModel.attachment, isNull);
        expect(viewModel.attach.error, isTrue);
      },
    );

    test('Gemma 4 E2B: photos on, no label', () {
      expect(viewModel.photosEnabled, isTrue);
      expect(viewModel.photosOffReason, isNull);
      expect(viewModel.skillsOffReason, isNull);
    });

    test('tools off: the skills label says why and what still works', () async {
      conversation.capabilities = const ChatCapabilities(
        modelName: 'Gemma 3 NPU',
        images: true,
        tools: false,
      );
      await conversation.open(kVoiceChatProfile, skills: [_skill]);

      expect(viewModel.skillsOffReason, contains('Gemma 3 NPU has tools off'));
      expect(viewModel.skillsOffReason, contains('device facts still work'));
      expect(viewModel.hasSkills, isFalse);
    });
  });

  group('attachment', () {
    Future<void> attachFromGallery() async {
      await viewModel.attach.execute(ImageSourceKind.gallery);
      expect(viewModel.attachment, isNotNull, reason: '${viewModel.error}');
    }

    Future<void> typedTurn(String text, {String reply = 'Ok.'}) async {
      final sending = viewModel.send.execute(text);
      await settle();
      conversation.emit(reply);
      await conversation.finish();
      await sending;
    }

    test('a gallery pick is normalized and attached, and the overlay gets '
        'its size', () async {
      var notified = 0;
      viewModel.addListener(() => notified++);

      await attachFromGallery();

      final image = viewModel.attachment!;
      expect(imageInput.picks, [ImageSourceKind.gallery]);
      expect((image.width, image.height), (1024, 768));
      expect(image.sourceBytes, 4);
      expect(diagnostics.latest.attachment, same(image));
      expect(notified, greaterThan(0));
      expect(viewModel.canUseGallery, isTrue);
      expect(viewModel.canUseCamera, isFalse, reason: 'macOS-like platform');
    });

    test('the picture is sticky: every turn carries the same object, and the '
        "user's entries show it", () async {
      await attachFromGallery();
      final png = viewModel.attachment!.png;

      await typedTurn('What animal is this?', reply: 'A cat.');
      await typedTurn('How many are there?', reply: 'Two.');

      expect(conversation.images, [same(png), same(png)]);
      final users = viewModel.entries.where((e) => e.role == ChatRole.user);
      expect(users.map((e) => e.image), [same(png), same(png)]);
    });

    test('a voice turn carries the picture too', () async {
      await attachFromGallery();

      await viewModel.pressMic();
      final releasing = viewModel.releaseMic();
      await settle();
      conversation.emit('Paris.');
      await conversation.finish();
      await releasing;

      expect(conversation.images.single, same(viewModel.attachment!.png));
      expect(viewModel.entries.first.image, same(viewModel.attachment!.png));
    });

    test('removing it: the next turn goes without a picture', () async {
      await attachFromGallery();
      await typedTurn('What animal is this?');

      viewModel.removeAttachment();
      await typedTurn('Tell me a joke.');

      expect(viewModel.attachment, isNull);
      expect(diagnostics.latest.attachment, isNull);
      expect(conversation.images.last, isNull);
      expect(viewModel.entries.last.role, ChatRole.assistant);
      expect(viewModel.entries[viewModel.entries.length - 2].image, isNull);
    });

    test('a picture changed during a turn applies from the next turn; the '
        'running turn keeps its own', () async {
      await attachFromGallery();
      final first = viewModel.attachment!.png;
      final sending = viewModel.send.execute('What is this?');
      await settle();

      imageInput.nextPick = Uint8List.fromList([1, 2, 3]);
      await viewModel.attach.execute(ImageSourceKind.gallery);
      final second = viewModel.attachment!.png;
      conversation.emit('A cat.');
      await conversation.finish();
      await sending;
      await typedTurn('And this?');

      expect(identical(first, second), isFalse);
      expect(conversation.images, [same(first), same(second)]);
      expect(viewModel.entries.first.image, same(first));
    });

    test('a cancelled pick keeps the current picture', () async {
      await attachFromGallery();
      final before = viewModel.attachment;

      imageInput.nextPick = null;
      await viewModel.attach.execute(ImageSourceKind.gallery);

      expect(viewModel.attachment, same(before));
      expect(viewModel.error, isNull);
    });

    test('leaving the screen clears the picture from the overlay', () async {
      await attachFromGallery();

      viewModel.dispose();
      await settle();

      expect(diagnostics.latest.attachment, isNull);
      viewModel = createViewModel(); // for tearDown
      await settle();
    });

    test('New conversation drops the picture', () async {
      await attachFromGallery();

      await viewModel.newConversation.execute();

      expect(viewModel.attachment, isNull);
      expect(diagnostics.latest.attachment, isNull);
    });

    test('the camera on a platform without one is an explicit error, never '
        'a silent no-op', () async {
      await viewModel.attach.execute(ImageSourceKind.camera);

      expect(viewModel.attachment, isNull);
      expect(viewModel.attach.error, isTrue);
      expect(viewModel.entries.single.role, ChatRole.error);
      expect(viewModel.error, contains('Could not attach the picture'));
      expect(viewModel.error, contains('pick an image from the gallery'));
      expect(imageInput.picks, isEmpty, reason: 'refused before the picker');
    });

    test('a denied picker is shown as an error', () async {
      imageInput.nextError = PlatformException(
        code: 'photo_access_denied',
        message: 'The user did not allow photo access.',
      );

      await viewModel.attach.execute(ImageSourceKind.gallery);

      expect(viewModel.attachment, isNull);
      expect(viewModel.error, contains('did not allow photo access'));
    });

    test('an unreadable file is shown as an error', () async {
      final undecodable = ImageRepository(
        input: imageInput,
        normalize: (_) async =>
            throw const UndecodableImageException('Invalid image data'),
      );
      viewModel.dispose();
      viewModel = VoiceChatViewModel(
        activateStt: (_) async => const Result.ok(null),
        conversation: conversation,
        diagnostics: diagnostics,
        images: undecodable,
        skills: skills,
        assistant: VoiceAssistant(
          speech: speech,
          audio: audio,
          responders: ChatTurnResponder(conversation: conversation),
          diagnostics: diagnostics,
        ),
      );
      await settle();

      await viewModel.attach.execute(ImageSourceKind.gallery);

      expect(viewModel.attachment, isNull);
      expect(viewModel.error, contains('could not be read'));
    });

    test('a context reset is one notice, from its own event, '
        'not from the end-of-turn metrics', () async {
      await attachFromGallery();
      final sending = viewModel.send.execute('One more?');
      await settle();
      conversation.emitContextReset();
      await settle();
      conversation.emit('Sure.');
      await conversation.finish(
        metricsOverride: FakeConversationRepository.metrics(
          imageAttached: true,
          imageSent: true,
          imageResent: ImageLoss.budget,
          contextReset: true,
        ),
      );
      await sending;

      expect(viewModel.entries.map((e) => e.role), [
        ChatRole.user,
        ChatRole.notice,
        ChatRole.assistant,
      ]);
      expect(viewModel.entries[1].text, contextResetText);
      expect(diagnostics.latest.lastGeneration?.imageResent, ImageLoss.budget);
    });

    test('the notice shows when a barge-in already detached the '
        'turn', () async {
      final sending = viewModel.send.execute('Tell me a story');
      await settle();
      conversation.emit('Once');
      await settle();
      conversation.stopGate = Completer<void>(); // the stopped turn drains
      await viewModel.pressMic(); // barge-in: the turn is detached
      await settle();

      conversation.emitContextReset();
      await settle();
      conversation.stopGate!.complete();
      await sending;
      await settle();

      expect(
        viewModel.entries.where((e) => e.text == contextResetText),
        hasLength(1),
      );
      await viewModel.stop.execute();
    });

    test('the notice shows when the turn fails after the reset', () async {
      final sending = viewModel.send.execute('One more?');
      await settle();
      conversation.emitContextReset();
      await settle();
      await conversation.fail(Exception('GPU lost'));
      await sending;

      expect(viewModel.entries.map((e) => e.role), [
        ChatRole.user,
        ChatRole.notice,
        ChatRole.error,
      ]);
      expect(viewModel.entries[1].text, contextResetText);
    });

    test('a picture picked while New conversation drains stays '
        'attached', () async {
      await attachFromGallery();
      final sending = viewModel.send.execute('Tell me a story');
      await settle();
      conversation.stopGate = Completer<void>();
      final renewing = viewModel.newConversation.execute();
      await settle();
      expect(viewModel.attachment, isNull, reason: 'cleared at once');

      imageInput.nextPick = Uint8List.fromList([7, 7, 7]);
      await viewModel.attach.execute(ImageSourceKind.gallery);
      final picked = viewModel.attachment;
      expect(picked, isNotNull);
      conversation.stopGate!.complete();
      await sending;
      await renewing;

      expect(viewModel.attachment, same(picked));
      expect(diagnostics.latest.attachment, same(picked));
    });

    test("a view model left while New conversation opens does "
        "not clear the next screen's overlay attachment", () async {
      await attachFromGallery();
      conversation.openGate = Completer<void>();
      final renewing = viewModel.newConversation.execute();
      await settle();

      viewModel.dispose(); // back to home while the open waits
      final next = LlmImage(
        png: Uint8List(4),
        width: 2,
        height: 2,
        sourceWidth: 2,
        sourceHeight: 2,
        sourceBytes: 4,
        normalizeTime: Duration.zero,
      );
      diagnostics.recordAttachment(next); // the next screen's picture
      conversation.openGate!.complete();
      await renewing;
      await settle();

      expect(diagnostics.latest.attachment, same(next));
      viewModel = createViewModel(); // for tearDown
      await settle();
    });
  });
}
