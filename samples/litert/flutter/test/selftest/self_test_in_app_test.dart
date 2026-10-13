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
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository.dart';
import 'package:litert_edge_demos/data/services/hardware/native_log_tap.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/self_test.dart';
import 'package:litert_edge_demos/domain/ports/chat_model_planner.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_model_switcher.dart';
import 'package:litert_edge_demos/selftest/self_test_in_app.dart';
import 'package:litert_edge_demos/ui/features/setup/view_models/self_test_view_model.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../fakes/fake_audio_repository.dart';
import '../fakes/fake_conversation_repository.dart';

final class _Planner implements ChatModelPlanner {
  @override
  ChatModelPlan get plan => const NoChatModelPlan();
}

const _ready = ModelReady(
  LoadedModelInfo(
    modelId: 'g',
    backend: 'gpu',
    loadTime: Duration.zero,
    warmUpTime: Duration.zero,
  ),
);

void main() {
  late List<String> log;
  late FakeConversationRepository conversation;
  late ValueNotifier<Map<ModelId, ModelState>> models;

  setUp(() {
    log = [];
    conversation = FakeConversationRepository();
    models = ValueNotifier({ModelId.chat: _ready});
  });

  tearDown(() async {
    await conversation.close();
    models.dispose();
  });

  ChatModelSwitcher newSwitcher() => ChatModelSwitcher(
    conversation: conversation,
    reloadChatModel: () async {
      log.add('reload');
      return const Result.ok(null);
    },
    unloadChatModel: () async => log.add('unload'),
    refuseChatModelLoads: (reason) => log.add('refuse: $reason'),
  );

  InAppSelfTest launcher({
    bool fail = false,
    bool stillRunning = false,
    bool chatEngineMayBeLoaded = true,
    ChatModelSwitcher? switcher,
    Future<void>? stepsGate,
    bool untilStopped = false,
  }) => InAppSelfTest(
    switcher: switcher ?? newSwitcher(),
    chatModels: _Planner(),
    models: models,
    logTap: const NoNativeLogTap(),
    audio: FakeAudioRepository(),
    steps: (progress, stopped) async {
      log.add('steps');
      if (untilStopped) {
        // A runner whose step in flight ends once it is asked to stop.
        await stopped;
        log.add('stopped');
      }
      await stepsGate;
      progress('step 4a chat model load + warm-up (npu) …');
      if (fail) throw StateError('runner crashed');
      return SelfTestOutcome(
        text: 'SELFTEST',
        passed: !stillRunning,
        stillRunning: stillRunning,
        chatEngineMayBeLoaded: chatEngineMayBeLoaded,
      );
    },
  );

  test('the chat is released and the chat model unloaded before the steps, '
      'and loaded again after them', () async {
    final lines = <String>[];

    final result = await launcher().run(progress: lines.add);

    expect((result as Ok<SelfTestOutcome>).value.passed, isTrue);
    expect(conversation.releaseCalls, 2, reason: 'before unload and reload');
    expect(log, ['unload', 'steps', 'reload']);
    expect(lines, contains('step 4a chat model load + warm-up (npu) …'));
  });

  test(
    'a crash in the steps is an error, and the app gets its model back',
    () async {
      final result = await launcher(fail: true).run(progress: (_) {});

      expect(
        (result as Error<SelfTestOutcome>).error.toString(),
        contains('crashed'),
      );
      expect(log, ['unload', 'steps', 'reload']);
    },
  );

  test('the previous reply is still stopping: the run is refused with that '
      'reason, no step runs and the chat model, never unloaded, is not '
      'reloaded; Run says why', () async {
    conversation.releaseError = const ConversationNotReadyException(
      'The previous reply did not finish within 5s of being stopped',
    );
    final selfTest = launcher();
    final vm = SelfTestViewModel(launcher: selfTest);
    addTearDown(vm.dispose);

    await vm.run.execute();

    expect(
      (vm.run.result as Error<SelfTestOutcome>).error,
      isA<ReplyStillStoppingException>(),
    );
    expect(log, isEmpty, reason: 'no unload, no steps, no reload');
    expect(
      vm.error,
      'The self-test could not run: The previous reply is still stopping; '
      'try again in a moment',
    );
    expect(selfTest.stuck, isFalse);

    conversation.releaseError = null;
    expect(await selfTest.run(progress: (_) {}), isA<Ok<SelfTestOutcome>>());
    expect(log, ['unload', 'steps', 'reload']);
  });

  test(
    'a chat model that never loaded is not loaded by the self-test',
    () async {
      models.value = {ModelId.chat: const ModelPending()};

      await launcher().run(progress: (_) {});

      expect(log, ['unload', 'steps']);
    },
  );

  group('the app closing', () {
    test('stops the run in flight and waits for it; the app\'s chat model is '
        'not loaded again', () async {
      final selfTest = launcher(untilStopped: true);
      final running = selfTest.run(progress: (_) {});
      await pumpEventQueue();
      expect(log, ['unload', 'steps']);

      var closed = false;
      final closing = selfTest.close().then((_) => closed = true);
      expect(await running, isA<Ok<SelfTestOutcome>>());
      await closing;

      expect(closed, isTrue);
      expect(log, ['unload', 'steps', 'stopped'], reason: 'no reload');
    });

    test(
      'a run that does not end holds the close for the grace only',
      () async {
        final selfTest = launcher(stepsGate: Completer<void>().future);
        unawaited(selfTest.run(progress: (_) {}));
        await pumpEventQueue();
        expect(log, ['unload', 'steps']);

        await selfTest.close(grace: const Duration(milliseconds: 20));

        expect(log, ['unload', 'steps']);
      },
    );

    test('with no run in flight it returns at once; no run starts '
        'afterwards', () async {
      final selfTest = launcher();

      await selfTest.close();
      final refused = await selfTest.run(progress: (_) {});

      expect(
        '${(refused as Error<SelfTestOutcome>).error}',
        contains('The app is closing'),
      );
      expect(log, isEmpty, reason: 'nothing unloaded, no steps');
    });

    test('a run still waiting for the chat model when the app closes runs '
        'nothing', () async {
      final switcher = newSwitcher();
      final hold = Completer<void>();
      final holding = switcher.exclusive((_) => hold.future);
      final selfTest = launcher(switcher: switcher);
      final running = selfTest.run(progress: (_) {});
      await pumpEventQueue();

      final closing = selfTest.close();
      hold.complete();
      await holding;

      expect(await running, isA<Error<SelfTestOutcome>>());
      await closing;
      expect(log, isEmpty, reason: 'nothing unloaded, no steps');
    });
  });

  group('a run past its time limit', () {
    test('a runner that never completes: the partial report, the runner '
        'asked to stop, and "still running" after the grace', () async {
      var stops = 0;
      final run = await runSelfTestWithin(
        Completer<String>().future, // never completes
        limit: const Duration(milliseconds: 20),
        grace: const Duration(milliseconds: 20),
        partial: () => 'partial',
        stop: () => stops++,
      );

      expect(run.report, 'partial');
      expect(run.stillRunning, isTrue);
      expect(stops, 1);
    });

    test('a runner that ends within the grace once stopped: the partial '
        'report, not still running', () async {
      final runner = Completer<String>();
      final run = await runSelfTestWithin(
        runner.future,
        limit: const Duration(milliseconds: 20),
        grace: const Duration(seconds: 5),
        partial: () => 'partial',
        stop: () => runner.complete('finished late'),
      );

      expect(run.report, 'partial');
      expect(run.stillRunning, isFalse);
    });

    test('a runner that fails after the limit has ended too', () async {
      final runner = Completer<String>();
      final run = await runSelfTestWithin(
        runner.future,
        limit: const Duration(milliseconds: 20),
        grace: const Duration(seconds: 5),
        partial: () => 'partial',
        stop: () => runner.completeError(StateError('closed under it')),
      );

      expect(run.stillRunning, isFalse);
    });

    test('a runner that finishes in time: its report, never stopped', () async {
      final run = await runSelfTestWithin(
        Future.value('full'),
        limit: const Duration(seconds: 5),
        grace: const Duration(seconds: 5),
        partial: () => fail('no partial report'),
        stop: () => fail('not stopped'),
      );

      expect(run.report, 'full');
      expect(run.stillRunning, isFalse);
    });

    test('still running: the app\'s chat model is not loaded again (its '
        'engine would be the second), the report says to restart, and no '
        'other run starts', () async {
      final lines = <String>[];
      final selfTest = launcher(stillRunning: true);

      final result = await selfTest.run(progress: lines.add);

      expect((result as Ok<SelfTestOutcome>).value.stillRunning, isTrue);
      expect(log, [
        'unload',
        'steps',
        'refuse: $kSelfTestStillRunning',
      ], reason: 'no reload, and none from anywhere else until a restart');
      expect(lines.last, kSelfTestStillRunning);
      expect(kSelfTestStillRunning, contains('restart the app'));
      expect(selfTest.stuck, isTrue);

      final again = await selfTest.run(progress: lines.add);
      expect(
        '${(again as Error<SelfTestOutcome>).error}',
        contains('still running'),
      );
      expect(log, hasLength(3), reason: 'nothing ran');
    });

    test('still running, but holding no chat model engine (hung before its '
        'load, e.g. in the detector load, or after closing it): the app\'s '
        'chat model is loaded again; no other run starts', () async {
      final lines = <String>[];
      final selfTest = launcher(
        stillRunning: true,
        chatEngineMayBeLoaded: false,
      );

      final result = await selfTest.run(progress: lines.add);

      expect((result as Ok<SelfTestOutcome>).value.stillRunning, isTrue);
      expect(log, ['unload', 'steps', 'reload'], reason: 'no refusal');
      expect(lines, contains(kSelfTestStillRunningNoChatModel));
      // Also true for a run stuck after closing its chat model (a hung
      // audio close outlives the grace), so it never says "never loaded".
      expect(
        kSelfTestStillRunningNoChatModel,
        contains('no longer holds a chat model engine'),
      );
      expect(lines, isNot(contains(kSelfTestStillRunning)));
      expect(selfTest.stuck, isTrue);
      expect(
        '${((await selfTest.run(progress: lines.add)) as Error<SelfTestOutcome>).error}',
        contains('still running'),
      );
      expect(log, hasLength(3), reason: 'nothing ran');
    });

    test(
      'still running: Run is disabled and says why; the report stays',
      () async {
        final selfTest = launcher(stillRunning: true);
        final vm = SelfTestViewModel(launcher: selfTest);
        addTearDown(vm.dispose);
        expect(vm.canRun, isTrue);

        await vm.run.execute();

        expect(vm.outcome?.stillRunning, isTrue);
        expect(vm.canRun, isFalse);
        expect(vm.blockedReason, contains('restart the app'));
      },
    );
  });

  group('exclusivity', () {
    test('the whole run holds the chat model: a reload asked for meanwhile '
        'waits until the run has given the model back', () async {
      final switcher = newSwitcher();
      final gate = Completer<void>();
      final running = launcher(
        switcher: switcher,
        stepsGate: gate.future,
      ).run(progress: (_) {});
      expect(switcher.busy.value, isTrue, reason: 'from the tap on');
      await pumpEventQueue();

      final reload = switcher.reload();
      await pumpEventQueue();
      expect(log, ['unload', 'steps'], reason: 'the reload waits');

      gate.complete();
      await running;
      await reload;

      expect(log, ['unload', 'steps', 'reload', 'reload']);
      expect(switcher.busy.value, isFalse);
    });

    test('Run is disabled while the chat model is switched, and a tapped '
        'Run disables what would switch it at once', () async {
      final switcher = newSwitcher();
      final gate = Completer<void>();
      final vm = SelfTestViewModel(
        launcher: launcher(switcher: switcher, stepsGate: gate.future),
        blockers: [switcher.busy],
      );
      addTearDown(vm.dispose);

      // An Apply holds the switcher: Run is off, with the reason.
      final applyGate = Completer<void>();
      final applying = switcher.exclusive((_) => applyGate.future);
      expect(vm.canRun, isFalse);
      expect(vm.blockedReason, contains('chat model is not being switched'));
      applyGate.complete();
      await applying;
      expect(vm.canRun, isTrue);

      // A tapped Run holds it from the same tick: an Apply (which checks
      // the switcher) is off before any await.
      final run = vm.run.execute();
      expect(switcher.busy.value, isTrue);
      gate.complete();
      await run;
      expect(switcher.busy.value, isFalse);
      expect(vm.outcome?.passed, isTrue);
    });
  });

  group('SelfTestViewModel', () {
    test('runs, collects the progress, shows the report; blocked while models '
        'load', () async {
      final preparing = ValueNotifier(false);
      addTearDown(preparing.dispose);
      final vm = SelfTestViewModel(launcher: launcher(), blockers: [preparing]);
      addTearDown(vm.dispose);

      expect(vm.canRun, isTrue);
      preparing.value = true;
      expect(vm.canRun, isFalse);
      expect(vm.blockedReason, contains('Wait until the models'));
      preparing.value = false;

      await vm.run.execute();

      expect(vm.outcome?.text, 'SELFTEST');
      expect(vm.outcome?.passed, isTrue);
      expect(vm.progress.value, isNotEmpty);
      expect(vm.error, isNull);
    });
  });
}
