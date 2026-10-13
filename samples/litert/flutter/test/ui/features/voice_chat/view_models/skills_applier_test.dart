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

import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show Skill, SkillType;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/skill_repository.dart';
import 'package:litert_edge_demos/domain/models/skill_catalog.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/view_models/skills_applier.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_skill_store.dart';

Future<void> settle() => pumpEventQueue();

SkillCatalog _catalog(String fingerprint, [List<String> names = const []]) =>
    SkillCatalog(
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

/// The screen's side, played by the test: what is busy, and the apply
/// command, which runs the applier's apply like the real command does.
final class _Commands implements SkillsApplyCommands {
  late SkillsApplier applier;

  /// An open command or the recognizer switch runs.
  bool opening = false;

  /// Like the screen's: an apply that runs is opening the chat too.
  @override
  bool get openingChat => opening || applying;

  @override
  bool turnRunning = false;

  @override
  bool applying = false;

  int applies = 0;
  final List<Result<SkillCatalog?>> results = [];

  @override
  Future<void> apply() async {
    applies++;
    applying = true;
    try {
      results.add(await applier.apply());
    } finally {
      applying = false;
    }
  }
}

void main() {
  late FakeSkillStore store;
  late SkillRepository skills;
  late FakeConversationRepository conversation;
  late _Commands commands;
  late SkillsApplier applier;
  var changes = 0;

  setUp(() {
    store = FakeSkillStore(catalog: _catalog('v1', ['current-time']));
    skills = SkillRepository(store: store);
    conversation = FakeConversationRepository()..leaveOpen(kVoiceChatProfile);
    commands = _Commands();
    changes = 0;
    applier = commands.applier = SkillsApplier(
      skills: skills,
      conversation: conversation,
      profile: kVoiceChatProfile,
      commands: commands,
      onChanged: () => changes++,
    );
  });

  tearDown(() async {
    applier.close();
    skills.dispose();
    await conversation.close();
  });

  /// The screen's first open, with the scan it made.
  Future<SkillCatalog> openWithScan() async {
    commands.opening = true; // the screen's open command runs
    expect(await applier.open(rescan: true), isA<Ok<void>>());
    commands.opening = false;
    changes = 0;
    return applier.catalog!;
  }

  Future<void> changeSkills(String fingerprint, List<String> names) async {
    store.catalog = _catalog(fingerprint, names);
    await skills.refresh();
  }

  List<String> openedNames() =>
      conversation.openedSkills.last.map((s) => s.name).toList();

  group('pending', () {
    test('nothing before the first scan', () {
      expect(applier.catalog, isNull);
      expect(applier.pending, isFalse);
      expect(applier.lastReload, isNull);
    });

    test('a scan the chat was opened with is not pending; a newer one '
        'is', () async {
      final first = await openWithScan();
      expect(applier.catalog, same(first));
      expect(applier.pending, isFalse);

      commands.turnRunning = true; // keep it from applying
      await changeSkills('v2', ['kid-clock']);
      expect(applier.pending, isTrue);
    });

    test('a set whose open failed is not pending (no retry by '
        'itself)', () async {
      conversation.openResult = Result.error(Exception('no model'));
      commands.opening = true;
      await applier.open(rescan: true);
      commands.opening = false;

      expect(applier.pending, isFalse);
    });

    test('a chat opened without a scan counts as no set applied', () async {
      await applier.open(rescan: false);
      commands.turnRunning = true;
      await skills.refresh();

      expect(applier.pending, isTrue);
    });
  });

  group('a rescan that changed the skills', () {
    setUp(openWithScan);

    test('applies at once when idle, and tells the screen', () async {
      await changeSkills('v2', ['kid-clock']);
      await settle();

      expect(commands.applies, 1);
      expect(openedNames(), ['kid-clock']);
      expect(applier.pending, isFalse);
      expect(changes, greaterThan(0));
      expect(applier.lastReload, isNull, reason: 'not a Reload');
    });

    for (final (name, makeBusy) in <(String, void Function(_Commands))>[
      ('the chat is being opened', (c) => c.opening = true),
      ('a turn runs', (c) => c.turnRunning = true),
    ]) {
      test('waits while $name, shown as pending', () async {
        makeBusy(commands);

        await changeSkills('v2', ['kid-clock']);
        await settle();

        expect(commands.applies, 0);
        expect(applier.pending, isTrue);
        expect(changes, 2, reason: 'busy, and the catalog changed');
      });
    }

    test("waits while another demo's chat is open", () async {
      conversation.leaveOpen(kCameraProfile);

      await changeSkills('v2', ['kid-clock']);
      await settle();

      expect(commands.applies, 0);
      expect(applier.pending, isTrue);
    });

    test('with no chat at all (a failed open) applies: the retry', () async {
      await conversation.release();

      await changeSkills('v2', ['kid-clock']);
      await settle();

      expect(commands.applies, 1);
      expect(conversation.profile, kVoiceChatProfile);
    });

    test('applies once the open commands have finished', () async {
      commands.opening = true;
      await changeSkills('v2', ['kid-clock']);
      applier.commandsChanged();
      expect(commands.applies, 0, reason: 'still opening');

      commands.opening = false;
      applier.commandsChanged();
      await settle();

      expect(commands.applies, 1);
    });

    test('applies once the turn has ended', () async {
      commands.turnRunning = true;
      await changeSkills('v2', ['kid-clock']);
      applier.turnChanged();
      expect(commands.applies, 0, reason: 'the turn still runs');

      commands.turnRunning = false;
      applier.turnChanged();
      await settle();

      expect(commands.applies, 1);
    });

    test('a turn ending with nothing pending applies nothing', () async {
      applier
        ..turnChanged()
        ..commandsChanged()
        ..maybeApply();
      await settle();

      expect(commands.applies, 0);
      expect(changes, 0);
    });
  });

  group('Reload', () {
    setUp(openWithScan);

    test('unchanged files: "unchanged", the chat is left alone', () async {
      await applier.reload();

      expect(applier.lastReload, SkillsReload.unchanged);
      expect(commands.applies, 0);
      expect(conversation.openCalls, 1, reason: 'the first open only');
      expect(changes, 1);
    });

    test('a change applied at once: "applying", then "applied"', () async {
      store.catalog = _catalog('v2', ['kid-clock']);
      conversation.openGate = Completer<void>();

      await applier.reload();

      expect(applier.lastReload, SkillsReload.applying);
      conversation.openGate!.complete();
      await settle();
      expect(applier.lastReload, SkillsReload.applied);
      expect(commands.results.single, isA<Ok<SkillCatalog?>>());
    });

    test('a change during a turn: "pending", then "applied" after '
        'it', () async {
      commands.turnRunning = true;
      store.catalog = _catalog('v2', ['kid-clock']);

      await applier.reload();
      expect(applier.lastReload, SkillsReload.pending);

      commands.turnRunning = false;
      applier.turnChanged();
      await settle();
      expect(applier.lastReload, SkillsReload.applied);
    });

    test('an apply that fails: "failed", the set is not retried by itself; '
        'the next Reload retries it', () async {
      conversation.openResult = Result.error(Exception('engine busy'));
      store.catalog = _catalog('v2', ['kid-clock']);

      await applier.reload();
      await settle();

      expect(applier.lastReload, SkillsReload.failed);
      expect(commands.results.single, isA<Error<SkillCatalog?>>());
      expect(applier.pending, isFalse);
      applier.maybeApply();
      expect(commands.applies, 1, reason: 'no retry loop');

      conversation.openResult = const Result.ok(null);
      await applier.reload();
      await settle();
      expect(commands.applies, 2);
      expect(applier.lastReload, SkillsReload.applied);
    });
  });

  group('open', () {
    test('with a rescan: the fresh scan\'s skills, nothing pending', () async {
      store.catalog = _catalog('v2', ['kid-clock']);
      commands.opening = true; // the screen's open command runs

      expect(await applier.open(rescan: true), isA<Ok<void>>());

      expect(conversation.openedProfiles.single, kVoiceChatProfile);
      expect(openedNames(), ['kid-clock']);
      expect(applier.catalog?.fingerprint, 'v2');
      expect(applier.pending, isFalse);
    });

    test('without a rescan: the latest scan\'s skills, none before the '
        'first scan', () async {
      await applier.open(rescan: false);
      expect(conversation.openedSkills.single, isEmpty);
      expect(applier.catalog, isNull, reason: 'no rescan');

      commands.turnRunning = true; // keep the scan from applying
      await skills.refresh();
      await applier.open(rescan: false);
      expect(openedNames(), ['current-time']);
      expect(applier.pending, isFalse);
    });

    test('a failure marks the set failed: the end of the open command '
        'does not retry it; a Reload does', () async {
      await openWithScan();
      conversation.leaveOpen(kCameraProfile);
      await changeSkills('v2', ['kid-clock']);
      expect(applier.pending, isTrue);

      commands.opening = true; // New conversation runs
      conversation.openResult = Result.error(Exception('engine busy'));
      expect(await applier.open(rescan: false), isA<Error<void>>());
      commands.opening = false;
      applier.commandsChanged();
      await settle();

      expect(applier.pending, isFalse);
      expect(commands.applies, 0, reason: 'no retry by itself');
      expect(conversation.openCalls, 2);

      conversation.openResult = const Result.ok(null);
      await applier.reload();
      await settle();
      expect(commands.applies, 1);
      expect(conversation.profile, kVoiceChatProfile);
    });

    test('closed while scanning: opens nothing', () async {
      final opening = applier.open(rescan: true);
      applier.close();

      expect(await opening, isA<Ok<void>>());
      expect(conversation.openCalls, 0);
    });

    test('closed while opening: records nothing, returns the '
        'outcome', () async {
      commands.turnRunning = true;
      await skills.refresh();
      conversation
        ..openGate = Completer<void>()
        ..openResult = Result.error(Exception('engine busy'));
      final opening = applier.open(rescan: false);

      applier.close();
      conversation.openGate!.complete();

      expect(await opening, isA<Error<void>>());
      expect(applier.pending, isTrue, reason: 'nothing recorded');
    });
  });

  group('apply', () {
    test('without a scan opens nothing', () async {
      final result = await applier.apply();

      expect(result, isA<Ok<SkillCatalog?>>());
      expect((result as Ok<SkillCatalog?>).value, isNull);
      expect(conversation.openCalls, 0);
    });

    test('opens the screen\'s chat with the scan\'s skills and returns the '
        'set', () async {
      commands.turnRunning = true;
      final catalog = await skills.refresh();

      final result = await applier.apply();

      expect((result as Ok<SkillCatalog?>).value, same(catalog));
      expect(conversation.openedProfiles.single, kVoiceChatProfile);
      expect(openedNames(), ['current-time']);
      expect(applier.pending, isFalse);
    });

    test('a failure is returned and the set marked failed; the Reload '
        'status is untouched outside a Reload', () async {
      commands.turnRunning = true;
      await skills.refresh();
      conversation.openResult = Result.error(Exception('engine busy'));

      final result = await applier.apply();

      expect('${(result as Error<SkillCatalog?>).error}', contains('engine'));
      expect(applier.pending, isFalse);
      expect(applier.lastReload, isNull);
    });
  });

  group('close', () {
    test('a later rescan applies nothing', () async {
      await openWithScan();
      applier.close();

      await changeSkills('v2', ['kid-clock']);
      applier.maybeApply();
      await settle();

      expect(commands.applies, 0);
      expect(changes, 0);
    });

    test('an apply in flight records nothing; its result is still '
        'returned', () async {
      commands.turnRunning = true;
      await skills.refresh();
      conversation.openGate = Completer<void>();
      final applying = applier.apply();

      applier.close();
      conversation.openGate!.complete();

      expect((await applying), isA<Ok<SkillCatalog?>>());
      expect((await applying as Ok<SkillCatalog?>).value, isNull);
      expect(applier.pending, isTrue, reason: 'nothing recorded');
    });

    test('a failed apply in flight still returns its error', () async {
      commands.turnRunning = true;
      await skills.refresh();
      conversation
        ..openGate = Completer<void>()
        ..openResult = Result.error(Exception('engine busy'));
      final applying = applier.apply();

      applier.close();
      conversation.openGate!.complete();

      expect(await applying, isA<Error<SkillCatalog?>>());
    });

    test('a Reload after it changes nothing', () async {
      await openWithScan();
      applier.close();
      store.catalog = _catalog('v2', ['kid-clock']);

      expect(await applier.reload(), isA<Ok<void>>());
      expect(applier.lastReload, isNull);
      expect(commands.applies, 0);
    });
  });
}
