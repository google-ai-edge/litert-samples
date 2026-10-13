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

// When Demo 1 re-opens its chat with a changed skill set, and what the
// Skills sheet says about it: the paths voice_chat_skills_test.dart (real
// files) does not reach — a change while the chat is being opened, re-opened
// or the recognizer switched, another demo's chat, an apply that fails after
// a Reload, and the Reload right after a failed open.
import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show Skill, SkillType;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/data/repositories/skill_repository.dart';
import 'package:litert_edge_demos/domain/models/chat_entry.dart';
import 'package:litert_edge_demos/domain/models/chat_side_event.dart';
import 'package:litert_edge_demos/domain/models/llm_image.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/skill_catalog.dart';
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

Future<void> settle() => pumpEventQueue();

SkillCatalog _catalog(String fingerprint, List<String> names) => SkillCatalog(
  directory: '/fake/skills',
  fingerprint: fingerprint,
  skills: [
    for (final name in names)
      LoadedSkill(
        skill: Skill(
          name: name,
          description: 'The $name skill.',
          instructions: 'Call run_intent.',
          type: SkillType.intent,
        ),
        path: '$name/SKILL.md',
      ),
  ],
);

void main() {
  late FakeSkillStore store;
  late SkillRepository skills;
  late FakeConversationRepository conversation;
  late FakeAudioRepository audio;
  late ValueNotifier<Map<ModelId, ModelState>> models;
  late DiagnosticsRepository diagnostics;
  late VoiceAssistant<ChatSideEvent> assistant;
  late VoiceChatViewModel viewModel;
  Completer<void>? sttGate;

  VoiceChatViewModel createViewModel() => VoiceChatViewModel(
    activateStt: (_) async {
      await sttGate?.future;
      return const Result.ok(null);
    },
    conversation: conversation,
    diagnostics: diagnostics,
    images: fakeImageRepository(FakeImageInputService()),
    assistant: assistant,
    skills: skills,
  );

  setUp(() async {
    store = FakeSkillStore(catalog: _catalog('v1', ['current-time']));
    skills = SkillRepository(store: store);
    conversation = FakeConversationRepository();
    audio = FakeAudioRepository()..autoDrain = true;
    models = ValueNotifier(const {});
    diagnostics = DiagnosticsRepository(
      models: models,
      minInterval: Duration.zero,
      sampleRss: false,
    );
    assistant = VoiceAssistant(
      speech: await loadedSpeech(
        recognizer: FakeRecognizer(),
        synthesizer: RecordingSynth(),
      ),
      audio: audio,
      responders: ChatTurnResponder(conversation: conversation),
      diagnostics: diagnostics,
    );
    sttGate = null;
  });

  tearDown(() async {
    viewModel.dispose();
    await settle();
    diagnostics.dispose();
    models.dispose();
    skills.dispose();
    await conversation.close();
  });

  List<String> openedNames() =>
      conversation.openedSkills.last.map((s) => s.name).toList();

  Iterable<String> notices() => [
    for (final e in viewModel.entries)
      if (e.role == ChatRole.notice) e.text,
  ];

  /// Changes the skills on disk and rescans (what Reload and resume do).
  Future<void> changeSkills(String fingerprint, List<String> names) async {
    store.catalog = _catalog(fingerprint, names);
    await skills.refresh();
  }

  group('opened', () {
    setUp(() async {
      viewModel = createViewModel();
      await settle();
    });

    test('with the scanned skills: nothing pending, no Reload yet', () {
      expect(conversation.openCalls, 1);
      expect(openedNames(), ['current-time']);
      expect(viewModel.skillCatalog?.fingerprint, 'v1');
      expect(viewModel.skillsPending, isFalse);
      expect(viewModel.lastReload, isNull);
    });

    test('a Reload that finds a change is "applying" while the chat '
        're-opens, then "applied", with the notice', () async {
      store.catalog = _catalog('v2', ['current-time', 'kid-clock']);
      conversation.openGate = Completer<void>();
      var notified = 0;
      viewModel.addListener(() => notified++);

      await viewModel.reloadSkills.execute();

      expect(viewModel.lastReload, SkillsReload.applying);
      expect(viewModel.applySkills.running, isTrue);
      expect(viewModel.isReady, isFalse, reason: 'the chat is re-opening');
      expect(notified, greaterThan(0));

      conversation.openGate!.complete();
      await settle();

      expect(viewModel.lastReload, SkillsReload.applied);
      expect(openedNames(), ['current-time', 'kid-clock']);
      expect(viewModel.skillsPending, isFalse);
      expect(notices(), [
        'Skills reloaded: 2 skills. The conversation started over.',
      ]);
      expect(viewModel.isReady, isTrue);
    });

    test('a Reload whose apply fails says so; the set is not retried by '
        'itself; the next Reload retries', () async {
      conversation.openResult = Result.error(Exception('engine busy'));
      store.catalog = _catalog('v2', ['kid-clock']);

      await viewModel.reloadSkills.execute();
      await settle();

      expect(viewModel.lastReload, SkillsReload.failed);
      expect(
        viewModel.error,
        'Could not reload the skills: Exception: '
        'engine busy',
      );
      expect(viewModel.entries.last.role, ChatRole.error);
      expect(viewModel.skillsPending, isFalse, reason: 'failed: not retried');
      expect(conversation.openCalls, 2);

      await changeSkills('v2', ['kid-clock']); // an unchanged rescan
      await settle();
      expect(conversation.openCalls, 2, reason: 'no retry loop');

      conversation.openResult = const Result.ok(null);
      await viewModel.reloadSkills.execute();
      await settle();
      expect(conversation.openCalls, 3);
      expect(viewModel.lastReload, SkillsReload.applied);
      expect(openedNames(), ['kid-clock']);
    });

    test('a change found by a rescan (not a Reload) applies without '
        'touching the Reload status', () async {
      await changeSkills('v2', ['kid-clock']);
      await settle();

      expect(conversation.openCalls, 2);
      expect(viewModel.lastReload, isNull);
      expect(notices(), [
        'Skills reloaded: 1 skill. The conversation started over.',
      ]);
    });

    test('a failed apply after a rescan leaves the Reload status '
        'alone', () async {
      conversation.openResult = Result.error(Exception('engine busy'));
      await changeSkills('v2', ['kid-clock']);
      await settle();

      expect(viewModel.lastReload, isNull);
      expect(viewModel.error, contains('Could not reload the skills'));
    });

    test('the notice counts the files with errors', () async {
      store.catalog = SkillCatalog(
        directory: '/fake/skills',
        fingerprint: 'v2',
        skills: _catalog('v2', ['kid-clock']).skills,
        errors: const [
          SkillLoadError(path: 'a/SKILL.md', message: 'bad'),
          SkillLoadError(path: 'b/SKILL.md', message: 'bad'),
        ],
      );
      await skills.refresh();
      await settle();

      expect(notices(), [
        'Skills reloaded: 1 skill, 2 with errors. The conversation started '
            'over.',
      ]);
    });

    test("another demo's chat is never replaced: the change waits", () async {
      conversation.leaveOpen(kCameraProfile);

      await changeSkills('v2', ['kid-clock']);
      await settle();

      expect(conversation.openCalls, 1);
      expect(viewModel.skillsPending, isTrue);
      expect(viewModel.isReady, isFalse);
    });

    test('a change during New conversation applies once it has '
        'finished', () async {
      conversation.openGate = Completer<void>();
      final renewing = viewModel.newConversation.execute();
      await settle();
      expect(openedNames(), ['current-time']);

      await changeSkills('v2', ['kid-clock']);
      await settle();
      expect(conversation.openCalls, 2, reason: 'waits for New conversation');
      expect(viewModel.skillsPending, isTrue);

      conversation.openGate!.complete();
      await renewing;
      await settle();

      expect(conversation.openCalls, 3);
      expect(openedNames(), ['kid-clock']);
      expect(viewModel.skillsPending, isFalse);
      expect(notices(), [
        'Skills reloaded: 1 skill. The conversation started over.',
      ]);
    });

    test('New conversation opens with the latest skills: nothing left '
        'pending', () async {
      conversation.leaveOpen(kCameraProfile);
      await changeSkills('v2', ['kid-clock']);
      await settle();
      expect(viewModel.skillsPending, isTrue);

      await viewModel.newConversation.execute();
      await settle();

      expect(conversation.openCalls, 2);
      expect(openedNames(), ['kid-clock']);
      expect(viewModel.skillsPending, isFalse);
      expect(notices(), isEmpty, reason: 'a fresh conversation, no notice');
    });

    test('a New conversation that fails marks its set failed: not retried '
        'by itself; a Reload retries', () async {
      conversation.leaveOpen(kCameraProfile);
      await changeSkills('v2', ['kid-clock']);
      await settle();
      expect(viewModel.skillsPending, isTrue);

      conversation.openResult = Result.error(Exception('engine busy'));
      await viewModel.newConversation.execute();
      await settle();

      expect(conversation.openCalls, 2, reason: 'no retry by itself');
      expect(viewModel.skillsPending, isFalse);
      expect(viewModel.error, contains('Could not start a new conversation'));

      conversation.openResult = const Result.ok(null);
      await viewModel.reloadSkills.execute();
      await settle();
      expect(conversation.openCalls, 3);
      expect(openedNames(), ['kid-clock']);
      expect(viewModel.isReady, isTrue);
    });

    test('a change during a turn applies when the turn ends, clearing '
        "that turn's steps", () async {
      final sending = viewModel.send.execute('Hi');
      await settle();
      await changeSkills('v2', ['kid-clock']);
      await settle();
      expect(conversation.openCalls, 1, reason: 'never mid-turn');
      expect(viewModel.lastReload, isNull);

      conversation.emit('Hello.');
      await conversation.finish();
      await sending;
      await settle();

      expect(conversation.openCalls, 2);
      expect(viewModel.entries.map((e) => e.role), [
        ChatRole.user,
        ChatRole.assistant,
        ChatRole.notice,
      ]);
      expect(viewModel.liveSteps.value, isEmpty);
    });

    test('a change while the recognizer switches applies once it has '
        'switched', () async {
      sttGate = Completer<void>();
      final switching = viewModel.selectStt.execute();
      await settle();

      await changeSkills('v2', ['kid-clock']);
      await settle();
      expect(conversation.openCalls, 1);

      sttGate!.complete();
      await switching;
      await settle();
      expect(conversation.openCalls, 2);
      expect(openedNames(), ['kid-clock']);
    });
  });

  test('a change while the chat is being opened on entry applies once the '
      'open has finished', () async {
    conversation.openGate = Completer<void>();
    viewModel = createViewModel();
    await settle();
    expect(viewModel.open.running, isTrue);

    await changeSkills('v2', ['kid-clock']);
    await settle();
    expect(conversation.openCalls, 1);
    expect(viewModel.skillsPending, isTrue);

    conversation.openGate!.complete();
    await settle();

    expect(conversation.openCalls, 2);
    expect(openedNames(), ['kid-clock']);
    expect(viewModel.skillsPending, isFalse);
  });

  test('after a failed open: nothing pending, and a Reload is the retry '
      'that opens the chat', () async {
    conversation.openResult = Result.error(Exception('no model'));
    viewModel = createViewModel();
    await settle();
    expect(viewModel.skillsPending, isFalse);
    expect(viewModel.isReady, isFalse);

    conversation.openResult = const Result.ok(null);
    await viewModel.reloadSkills.execute();
    await settle();

    expect(conversation.openCalls, 2);
    expect(conversation.openedProfiles.last, kVoiceChatProfile);
    expect(viewModel.isReady, isTrue);
    expect(viewModel.lastReload, SkillsReload.applied);
    expect(notices(), [
      'Skills reloaded: 1 skill. The conversation started over.',
    ]);
  });

  test('a fed utterance runs a voice turn with the attachment and clears '
      'the last error', () async {
    viewModel = createViewModel();
    await settle();
    await viewModel.attach.execute(ImageSourceKind.gallery);
    final png = viewModel.attachment!.png;
    // No camera on this (macOS-like) platform: an error to clear.
    await viewModel.attach.execute(ImageSourceKind.camera);
    expect(viewModel.error, isNotNull);

    final fed = viewModel.submitUtterance(speechUtterance());
    expect(viewModel.error, isNull);
    await settle();
    conversation.emit('Paris.');
    await conversation.finish();

    expect((await fed).outcome, TurnOutcome.completed);
    expect(conversation.images.single, same(png));
    expect(
      viewModel.entries
          .where((e) => e.role != ChatRole.error)
          .map((e) => (e.role, e.text)),
      [
        (ChatRole.user, 'What is the capital of France?'),
        (ChatRole.assistant, 'Paris.'),
      ],
    );
  });
}
