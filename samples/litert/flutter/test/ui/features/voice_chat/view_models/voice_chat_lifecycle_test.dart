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

// Demo 1 and the app's lifecycle: on Android and iOS leaving the app with
// the push-to-talk button held ends the press without a turn; on iOS hiding
// the app also stops a running turn, on Android it goes on; desktop keeps
// the microphone.
import 'dart:async';
import 'dart:ui' show AppLifecycleState;

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/data/repositories/skill_repository.dart';
import 'package:litert_edge_demos/domain/models/chat_entry.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/ui/core/app_foreground.dart';
import 'package:litert_edge_demos/ui/core/voice_notices.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/view_models/voice_chat_view_model.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../../fakes/fake_audio_repository.dart';
import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_image_input.dart';
import '../../../../fakes/fake_skill_store.dart';
import '../../../../fakes/fake_speech.dart';
import '../../../../support/until.dart';

void main() {
  late FakeConversationRepository conversation;
  late FakeAudioRepository audio;
  late FakeRecognizer recognizer;
  late ValueNotifier<Map<ModelId, ModelState>> models;
  late DiagnosticsRepository diagnostics;
  late SkillRepository skills;
  late VoiceChatViewModel viewModel;

  Future<VoiceChatViewModel> build(AppForeground foreground) async =>
      VoiceChatViewModel(
        activateStt: (_) async => const Result.ok(null),
        conversation: conversation,
        diagnostics: diagnostics,
        images: fakeImageRepository(FakeImageInputService()),
        skills: skills,
        foreground: foreground,
        assistant: VoiceAssistant(
          speech: await loadedSpeech(
            recognizer: recognizer,
            synthesizer: RecordingSynth(),
          ),
          audio: audio,
          responders: ChatTurnResponder(conversation: conversation),
          diagnostics: diagnostics,
        ),
      );

  setUp(() {
    conversation = FakeConversationRepository();
    audio = FakeAudioRepository()..autoDrain = true;
    recognizer = FakeRecognizer();
    models = ValueNotifier(const {});
    skills = SkillRepository(store: FakeSkillStore());
    diagnostics = DiagnosticsRepository(
      models: models,
      minInterval: Duration.zero,
      sampleRss: false,
    );
  });

  tearDown(() async {
    viewModel.dispose();
    await pumpEventQueue();
    diagnostics.dispose();
    models.dispose();
    skills.dispose();
    await conversation.close();
  });

  Future<void> holdMic(AppForeground foreground) async {
    viewModel = await build(foreground);
    await untilNotified(viewModel, () => viewModel.isReady, what: 'Demo 1');
    await viewModel.pressMic();
    expect(viewModel.phase, TurnPhase.listening);
  }

  test('on Android, leaving the app with the button held cancels the '
      'capture: idle, no STT, the release that follows is ignored', () async {
    final foreground = AppForeground(platform: TargetPlatform.android);
    addTearDown(foreground.dispose);
    await holdMic(foreground);

    foreground.onStateChange(AppLifecycleState.inactive);
    await pumpEventQueue();

    expect(audio.captures.single.cancelled, isTrue);
    expect(viewModel.phase, TurnPhase.idle);
    await viewModel.releaseMic();
    expect(recognizer.calls, 0);
    expect(conversation.prompts, isEmpty);
  });

  /// A typed turn whose reply is being generated.
  Future<void> generating(AppForeground foreground) async {
    viewModel = await build(foreground);
    await untilNotified(viewModel, () => viewModel.isReady, what: 'Demo 1');
    unawaited(viewModel.send.execute('Tell me a story'));
    await pumpEventQueue();
    conversation.emit('Once upon a time');
    await pumpEventQueue();
    expect(viewModel.phase, TurnPhase.thinking);
  }

  test('on iOS, hiding the app stops a reply being generated (no GPU work in '
      'the background): the model is stopped, the partial reply stays as '
      'interrupted, a notice says why', () async {
    final foreground = AppForeground(platform: TargetPlatform.iOS);
    addTearDown(foreground.dispose);
    await generating(foreground);

    // Control Centre or a banner (inactive) leaves the reply alone.
    foreground.onStateChange(AppLifecycleState.inactive);
    await pumpEventQueue();
    expect(viewModel.phase, TurnPhase.thinking);
    expect(conversation.stopCalls, 0);

    foreground.onStateChange(AppLifecycleState.hidden);
    await untilNotified(
      viewModel,
      () => viewModel.phase == TurnPhase.idle,
      what: 'the turn stopped',
    );

    expect(conversation.stopCalls, 1);
    final [..., reply, notice] = viewModel.entries;
    expect(reply.role, ChatRole.assistant);
    expect(reply.text, 'Once upon a time');
    expect(reply.interrupted, isTrue);
    expect(notice.role, ChatRole.notice);
    expect(notice.text, interruptedText(InterruptionCause.background));

    // Paused next: nothing more to stop.
    foreground.onStateChange(AppLifecycleState.paused);
    await pumpEventQueue();
    expect(conversation.stopCalls, 1);
  });

  test('on Android the reply goes on in the background', () async {
    final foreground = AppForeground(platform: TargetPlatform.android);
    addTearDown(foreground.dispose);
    await generating(foreground);

    foreground
      ..onStateChange(AppLifecycleState.inactive)
      ..onStateChange(AppLifecycleState.hidden)
      ..onStateChange(AppLifecycleState.paused);
    await pumpEventQueue();
    expect(viewModel.phase, TurnPhase.thinking);
    expect(conversation.stopCalls, 0);

    conversation.emit(' there was a cat.');
    await conversation.finish();
    await untilNotified(
      viewModel,
      () => viewModel.phase == TurnPhase.idle,
      what: 'the turn completed',
    );
    final reply = viewModel.entries.last;
    expect(reply.text, 'Once upon a time there was a cat.');
    expect(reply.interrupted, isFalse);
  });

  test('on Linux the held press goes on when the window loses focus or is '
      'hidden', () async {
    final foreground = AppForeground(platform: TargetPlatform.linux);
    addTearDown(foreground.dispose);
    await holdMic(foreground);

    foreground
      ..onStateChange(AppLifecycleState.inactive)
      ..onStateChange(AppLifecycleState.hidden);
    await pumpEventQueue();

    expect(audio.captures.single.isOpen, isTrue);
    expect(viewModel.phase, TurnPhase.listening);
  });
}
