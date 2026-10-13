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
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/data/repositories/skill_repository.dart';
import 'package:litert_edge_demos/data/services/skills/skill_store_service.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/chat_entry.dart';
import 'package:litert_edge_demos/domain/models/chat_side_event.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/skill_step.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/view_models/voice_chat_view_model.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../../../integration_test/support/skill_fixtures.dart';
import '../../../../fakes/fake_audio_repository.dart';
import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_image_input.dart';
import '../../../../fakes/fake_speech.dart';
import '../../../../support/until.dart';

final class FixedBundle implements BundledSkillSource {
  @override
  Future<Map<String, String>> load() async => {
    'current-time': File('assets/skills/current-time/SKILL.md')
        .readAsStringSync(),
  };
}

/// A short real-time window for the checks that nothing happens (real file
/// I/O, the skills scan, needs event-loop time, not just microtasks).
Future<void> settle() async {
  for (var i = 0; i < 10; i++) {
    await Future<void>.delayed(const Duration(milliseconds: 5));
  }
}

/// Demo 1 with skills — opened with the scanned skills, re-opened
/// only when they changed and only between turns, steps under replies.
void main() {
  late Directory root;
  late SkillRepository skills;
  late FakeConversationRepository conversation;
  late FakeAudioRepository audio;
  late ValueNotifier<Map<ModelId, ModelState>> models;
  late DiagnosticsRepository diagnostics;
  late VoiceAssistant<ChatSideEvent> assistant;
  late VoiceChatViewModel viewModel;

  /// Waits on the view model's notifications until [condition] holds.
  Future<void> untilVm(bool Function() condition, String what) =>
      untilNotified(viewModel, condition, what: what);

  /// The turn reached the chat (the fake generates until finished).
  Future<void> turnStarted() => untilNotified(
    conversation.isGenerating,
    () => conversation.isGenerating.value,
    what: 'the turn to reach the chat',
  );

  /// The turn is over: committed and back to idle.
  Future<void> turnEnded() => untilVm(
    () => viewModel.phase == TurnPhase.idle && !viewModel.send.running,
    'the turn to end',
  );

  /// The latest skills apply (a re-open with the scanned set) finished.
  Future<void> applied() => untilVm(
    () =>
        viewModel.applySkills.result != null && !viewModel.applySkills.running,
    'the skills apply to finish',
  );

  void writeSkill(String relative, String text) {
    File('${root.path}/skills/$relative')
      ..parent.createSync(recursive: true)
      ..writeAsStringSync(text);
  }

  setUp(() async {
    root = Directory.systemTemp.createTempSync('voice_chat_skills');
    skills = SkillRepository(
      store: SkillStoreService(
        directory: () async => Directory('${root.path}/skills'),
        bundled: FixedBundle(),
      ),
    );
    conversation = FakeConversationRepository();
    audio = FakeAudioRepository()..autoDrain = true;
    models = ValueNotifier(const {});
    diagnostics = DiagnosticsRepository(
      models: models,
      minInterval: Duration.zero,
      sampleRss: false,
    );
    final speech = await loadedSpeech(
      recognizer: FakeRecognizer(),
      synthesizer: RecordingSynth(),
    );
    assistant = VoiceAssistant(
      speech: speech,
      audio: audio,
      responders: ChatTurnResponder(conversation: conversation),
      diagnostics: diagnostics,
    );
    viewModel = VoiceChatViewModel(
      activateStt: (_) async => const Result.ok(null),
      conversation: conversation,
      diagnostics: diagnostics,
      images: fakeImageRepository(FakeImageInputService()),
      assistant: assistant,
      skills: skills,
    );
    await untilVm(
      () => conversation.openCalls == 1 && !viewModel.open.running,
      'the first open',
    );
  });

  tearDown(() async {
    viewModel.dispose();
    await settle();
    diagnostics.dispose();
    models.dispose();
    skills.dispose();
    await conversation.close();
    root.deleteSync(recursive: true);
  });

  List<String> openedNames() =>
      conversation.openedSkills.last.map((s) => s.name).toList();

  test('entering Demo 1 seeds, scans and opens with the skills', () {
    expect(conversation.openedProfiles, [kVoiceChatProfile]);
    expect(openedNames(), ['current-time']);
    expect(viewModel.skillCatalog?.skills, hasLength(1));
    expect(viewModel.skillsPending, isFalse);
    expect(viewModel.isReady, isTrue);
  });

  test('Reload with unchanged files leaves the conversation alone', () async {
    await viewModel.reloadSkills.execute();
    await settle();

    expect(conversation.openCalls, 1);
    expect(viewModel.lastReload, SkillsReload.unchanged);
  });

  test('Reload when idle: a dropped-in skill re-opens the chat with it and '
      'the chat says the conversation started over', () async {
    writeSkill('kid-clock/SKILL.md', kidClockSkillMd);
    writeSkill('broken/SKILL.md', brokenSkillMd);

    await viewModel.reloadSkills.execute();
    await applied();

    expect(conversation.openCalls, 2);
    expect(openedNames(), ['current-time', 'kid-clock']);
    expect(viewModel.lastReload, SkillsReload.applied);
    expect(viewModel.skillsPending, isFalse);
    expect(viewModel.skillCatalog!.errors.single.path, 'broken/SKILL.md');
    expect(
      viewModel.entries.last,
      isA<ChatEntry>()
          .having((e) => e.role, 'role', ChatRole.notice)
          .having((e) => e.text, 'text', contains('started over'))
          .having((e) => e.text, 'text', contains('1 with errors')),
    );
  });

  test(
    'Reload during a reply waits for the turn to end, then applies',
    () async {
      final sending = viewModel.send.execute('Tell me a story');
      await turnStarted();
      expect(viewModel.phase, TurnPhase.thinking);
      writeSkill('kid-clock/SKILL.md', kidClockSkillMd);

      await viewModel.reloadSkills.execute();
      await settle();
      expect(viewModel.lastReload, SkillsReload.pending);
      expect(viewModel.skillsPending, isTrue);
      expect(conversation.openCalls, 1, reason: 'never mid-turn');

      conversation.emit('Once upon a time.');
      await conversation.finish();
      await sending;
      await applied();

      expect(conversation.openCalls, 2);
      expect(openedNames(), ['current-time', 'kid-clock']);
      expect(viewModel.lastReload, SkillsReload.applied);
      expect(viewModel.entries.last.text, contains('started over'));
    },
  );

  test(
    'a rescan on resume (repository refresh) applies by itself when idle',
    () async {
      writeSkill('kid-clock/SKILL.md', kidClockSkillMd);

      await skills.refresh(); // what AppDependencies.onResumed does
      await applied();

      expect(conversation.openCalls, 2);
      expect(openedNames(), contains('kid-clock'));
    },
  );

  test(
    'a failed apply is shown and not retried in a loop; Reload retries',
    () async {
      conversation.openResult = Result.error(Exception('engine busy'));
      writeSkill('kid-clock/SKILL.md', kidClockSkillMd);

      await skills.refresh();
      await applied();
      expect(conversation.openCalls, 2);
      expect(viewModel.error, contains('Could not reload the skills'));
      await settle();
      expect(conversation.openCalls, 2, reason: 'no retry loop');

      conversation.openResult = const Result.ok(null);
      await viewModel.reloadSkills.execute();
      await applied();
      expect(conversation.openCalls, 3);
      expect(openedNames(), contains('kid-clock'));
    },
  );

  test('a rebuild after an interrupted skill call is a notice that says so, '
      'not "too long"', () async {
    final sending = viewModel.send.execute('Hi');
    await turnStarted();
    conversation
      ..emitContextReset(reason: ContextResetReason.interruptedSkill)
      ..emit('Hello.');
    await conversation.finish();
    await sending;
    await turnEnded();

    final notices = [
      for (final e in viewModel.entries)
        if (e.role == ChatRole.notice) e.text,
    ];
    expect(notices, [interruptedSkillResetText]);
    expect(interruptedSkillResetText, isNot(contextResetText));
    expect(interruptedSkillResetText, contains('started over'));
  });

  test('steps are live while the turn runs (one update per step, no '
      'screen-wide notify), then move to the reply', () async {
    final sending = viewModel.send.execute('How late is it?');
    await turnStarted();
    var stepUpdates = 0;
    var screenUpdates = 0;
    viewModel.liveSteps.addListener(() => stepUpdates++);
    viewModel.addListener(() => screenUpdates++);
    const loaded = SkillLoaded(
      'current-time',
      found: true,
      at: Duration(seconds: 1),
    );
    const called = IntentCalled('current_time', '{}', at: Duration(seconds: 2));

    conversation.emitStep(loaded);
    expect(viewModel.liveSteps.value, [loaded]);
    conversation.emitStep(called);
    await settle();

    expect(viewModel.liveSteps.value, [loaded, called]);
    expect(stepUpdates, 2);
    expect(screenUpdates, 0);

    conversation.emit('It is just after two.');
    await conversation.finish();
    await sending;
    await turnEnded();
    expect(viewModel.liveSteps.value, isEmpty);
    expect(viewModel.entries.last.steps, [loaded, called]);
  });

  test('skill steps go under the reply they belong to', () async {
    final sending = viewModel.send.execute('How late is it?');
    await turnStarted();
    const loaded = SkillLoaded(
      'current-time',
      found: true,
      at: Duration(seconds: 1),
    );
    const succeeded = IntentSucceeded(
      'current_time',
      'It is 2:03 PM on Friday, October 2, 2026.',
      elapsed: Duration(milliseconds: 2),
      at: Duration(seconds: 2),
    );
    conversation
      ..emitStep(loaded)
      ..emitStep(succeeded)
      ..emit('It is just after two.');
    await conversation.finish();
    await sending;
    await turnEnded();

    final reply = viewModel.entries.last;
    expect(reply.role, ChatRole.assistant);
    expect(reply.steps, [loaded, succeeded]);
  });

  test(
    'an intent that ran after a barge-in is still reported, as a notice',
    () async {
      unawaited(viewModel.send.execute('How late is it?'));
      await turnStarted();
      // Barge-in: the running turn is detached at once.
      await viewModel.pressMic();
      await settle();

      conversation.emitStep(
        const IntentSucceeded(
          'current_time',
          'It is 2:03 PM on Friday, October 2, 2026.',
          elapsed: Duration(milliseconds: 2),
          at: Duration(seconds: 2),
        ),
      );
      await settle();

      expect(
        viewModel.entries.last.text,
        '$skillAfterInterruptionText It is 2:03 PM on Friday, October 2, 2026.',
      );
      expect(viewModel.entries.last.role, ChatRole.notice);
    },
  );

  test(
    'New conversation clears the chat and reopens with the skills',
    () async {
      final sending = viewModel.send.execute('Hi');
      await turnStarted();
      conversation.emit('Hello.');
      await conversation.finish();
      await sending;
      await turnEnded();
      expect(viewModel.entries, isNotEmpty);

      // It awaits the re-open.
      await viewModel.newConversation.execute();

      expect(viewModel.entries, isEmpty);
      expect(openedNames(), [
        'current-time',
      ], reason: 'reopened with the skills');
    },
  );

  test('a skill turn whose final reply is empty shows and '
      'speaks the intent result instead of failing', () async {
    final sending = viewModel.send.execute('What time is it?');
    await turnStarted();
    conversation.emitStep(
      const IntentSucceeded(
        'current_time',
        'It is 2:03 PM on Friday, October 2, 2026.',
        elapsed: Duration(milliseconds: 300),
        at: Duration(milliseconds: 1800),
      ),
    );
    await conversation.finish();
    await sending;
    await turnEnded();

    expect(viewModel.send.error, isFalse, reason: '${viewModel.error}');
    expect(viewModel.error, isNull);
    final reply = viewModel.entries.last;
    expect(reply.role, ChatRole.assistant);
    expect(reply.text, 'It is 2:03 PM on Friday, October 2, 2026.');
    expect(reply.steps.single, isA<IntentSucceeded>());
    expect(audio.playbacks, isNotEmpty, reason: 'spoken');
  });
}
