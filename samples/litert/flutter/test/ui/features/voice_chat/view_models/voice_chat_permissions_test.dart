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

import 'package:fake_async/fake_async.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/voice_config.dart';
import 'package:litert_edge_demos/data/repositories/audio_repository.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/data/repositories/skill_repository.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/view_models/voice_chat_view_model.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../../fakes/fake_audio_repository.dart';
import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_image_input.dart';
import '../../../../fakes/fake_skill_store.dart';
import '../../../../fakes/fake_speech.dart';
import '../../../../support/until.dart';

/// Demo 1 entered: the entry access request settled and the chat is open
/// (the view model notifies on both).
Future<void> entered(VoiceChatViewModel vm) =>
    untilNotified(vm, () => vm.isReady, what: 'Demo 1 to be ready');

/// Entering Demo 1 asks for microphone access, so the OS dialog never lands
/// inside a turn. A denial stays visible with the settings path until a
/// Retry finds access on. Demo 1 never asks for the camera on entry.
void main() {
  late List<String> log;
  late FakeConversationRepository conversation;
  late FakeAudioRepository audio;
  late ValueNotifier<Map<ModelId, ModelState>> models;
  late DiagnosticsRepository diagnostics;
  late SkillRepository skills;
  late VoiceChatViewModel viewModel;

  Future<VoiceChatViewModel> build({
    Duration accessSettleCap = const Duration(seconds: 3),
  }) async => VoiceChatViewModel(
    activateStt: (_) async => const Result.ok(null),
    conversation: conversation,
    diagnostics: diagnostics,
    images: fakeImageRepository(FakeImageInputService()),
    accessSettleCap: accessSettleCap,
    skills: skills,
    assistant: VoiceAssistant(
      speech: await loadedSpeech(
        recognizer: FakeRecognizer(),
        synthesizer: RecordingSynth(),
      ),
      audio: audio,
      responders: ChatTurnResponder(conversation: conversation),
      diagnostics: diagnostics,
    ),
  );

  setUp(() {
    log = [];
    conversation = FakeConversationRepository();
    audio = FakeAudioRepository(log: log)..autoDrain = true;
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
    // Its unawaited entry work (open, recognizer, audio warm-up) runs on
    // fakes: microtasks and zero timers only.
    await pumpEventQueue();
    diagnostics.dispose();
    models.dispose();
    skills.dispose();
    await conversation.close();
  });

  test('on entry: the microphone only; granted, no error', () async {
    viewModel = await build();
    await entered(viewModel);
    // Whatever else entering started has run too (fakes: no real timers).
    await pumpEventQueue();

    expect(log.where((e) => e.contains('Access') || e.contains('camera')), [
      'requestMicAccess',
    ]);
    expect(viewModel.micAccessError, isNull);
  });

  test('a denial is a lasting, visible error with the settings path; Retry '
      'asks again and clears it once access is on', () async {
    audio.micAccess = Result.error(MicAccessException(kMicAccessMessage));
    viewModel = await build();
    await entered(viewModel);

    expect(viewModel.micAccessError, contains(kMicAccessMessage));

    // A turn does not clear it.
    final sending = viewModel.send.execute('Hi');
    await untilNotified(
      conversation.isGenerating,
      () => conversation.isGenerating.value,
      what: 'the turn to reach the chat',
    );
    conversation.emit('Hello.');
    await conversation.finish();
    await sending;
    expect(viewModel.micAccessError, isNotNull);

    audio.micAccess = const Result.ok(null);
    await viewModel.requestAccess.execute();
    expect(viewModel.micAccessError, isNull);
    expect(audio.micAccessRequests, 2);
  });

  test('no turn starts until the entry access request has finished', () async {
    audio.micAccessGate = Completer<void>();
    viewModel = await build();
    await untilNotified(
      viewModel.open,
      () => viewModel.open.result != null,
      what: 'the chat to open',
    );
    expect(conversation.isOpen, isTrue);
    expect(viewModel.isReady, isFalse, reason: 'the dialog is still open');
    expect(viewModel.canSend, isFalse);
    expect(viewModel.canTalk, isFalse);

    audio.micAccessGate!.complete();
    await entered(viewModel);
    expect(viewModel.isReady, isTrue);
  });

  test('a request that hangs (an unanswered OS dialog) holds the chat for '
      'exactly the cap', () {
    // Fake time: the cap is a Timer, and a real 50 ms cap checked at 80 ms
    // left 30 ms of margin.
    fakeAsync((async) {
      const cap = Duration(milliseconds: 50);
      audio.micAccessGate = Completer<void>();
      unawaited(build(accessSettleCap: cap).then((vm) => viewModel = vm));
      async.flushMicrotasks();
      expect(viewModel.isReady, isFalse);

      async.elapse(cap - const Duration(milliseconds: 1));
      expect(viewModel.isReady, isFalse, reason: 'the dialog is still open');
      async.elapse(const Duration(milliseconds: 1));
      expect(viewModel.isReady, isTrue, reason: 'the cap ran out');

      audio.micAccessGate!.complete();
      async.flushMicrotasks();
      expect(viewModel.isReady, isTrue);
    });
  });

  test('a failure that is not about access (audio session) shows no access '
      'bar; the turn reports it', () async {
    audio.micAccess = const Result.error(PlaybackException('session'));
    viewModel = await build();
    await untilNotified(
      viewModel.requestAccess,
      () => viewModel.requestAccess.result != null,
      what: 'the entry access request',
    );
    expect(viewModel.micAccessError, isNull);
  });
}
