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

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository_edge_ai.dart';
import 'package:litert_edge_demos/data/services/llm/llm_service.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_inference_model.dart';

/// Lets queued microtasks and zero-length timers run.
Future<void> settle() => Future<void>.delayed(Duration.zero);

/// One `ask` stream, collected.
final class Turn {
  Turn(Stream<AssistantEvent> stream) {
    done = stream.listen(events.add).asFuture<void>();
  }

  final List<AssistantEvent> events = [];
  late final Future<void> done;

  List<String> get deltas => [
    for (final e in events)
      if (e case AssistantTextDelta(:final text)) text,
  ];

  GenerationMetrics get metrics => switch (events.last) {
    AssistantDone(:final metrics) => metrics,
    final other => throw StateError('last event is $other'),
  };
}

void main() {
  late FakeInferenceModel model;
  late EdgeAiConversationRepository repo;

  FakeInferenceSession session() => model.lastSession;

  setUp(() async {
    model = FakeInferenceModel();
    final llm = LlmService(loadModel: (_) async => model);
    expect(await llm.load(kDefineChatModel), isA<Ok<LlmInfo>>());
    repo = EdgeAiConversationRepository(
      llm: llm,
      stopTimeout: const Duration(seconds: 2),
    );
    expect(await repo.open(kVoiceChatProfile), isA<Ok<void>>());
  });

  tearDown(() => repo.close());

  test('opens the chat with the shared sampler and the profile', () {
    expect(model.chatsCreated, 1);
    expect(repo.profile, same(kVoiceChatProfile));
    expect(model.sessionSettings.last, (
      temperature: kSampler.temperature,
      topK: kSampler.topK,
      maxOutputTokens: kVoiceChatProfile.maxOutputTokens,
    ));
  });

  test(
    'streams deltas, ends with AssistantDone and records the reply',
    () async {
      session().nativeTokensPerSecond = 42;
      final turn = Turn(repo.ask('Capital of France?'));
      await settle();
      expect(repo.isGenerating.value, isTrue);

      session()
        ..emit('Par')
        ..emit('is')
        ..finish();
      await turn.done;

      expect(turn.deltas, ['Par', 'is']);
      expect(turn.metrics.stopped, isFalse);
      expect(turn.metrics.chunks, 2);
      expect(turn.metrics.tokensPerSecond, 42);
      expect(turn.metrics.tokensPerSecondSource, TokenRateSource.native);
      expect(repo.isGenerating.value, isFalse);
      expect(model.chat!.fullHistory.last.text, 'Paris');
      expect(model.chat!.fullHistory.last.isUser, isFalse);
    },
  );

  // A busy or unopened chat refusing an ask: the turn paths test
  // (preconditions) and live_chat_test.dart.

  /// Lands a stop after the turn began but before native generation started,
  /// then delivers the first chunk. Returns the finished turn.
  Future<Turn> stopBeforeFirstChunk() async {
    session().queryGate = Completer<void>();
    final turn = Turn(repo.ask('Tell me a story'));
    await settle();
    expect(repo.isGenerating.value, isTrue);

    final stopping = repo.stop(); // native has nothing to cancel yet
    await settle();
    expect(session().nativeCancels, 0);
    session().queryGate!.complete();
    await settle();
    expect(session().responseRequests, 1, reason: 'the turn still starts');

    session().emit('Once'); // first chunk after the stop
    await settle();
    if (session().streaming) session().finish(); // a cancel that never landed
    await stopping;
    await turn.done;
    return turn;
  }

  /// Streams three chunks, stops, then lets one already-queued chunk through
  /// before native closes the stream. Returns the finished turn.
  Future<Turn> stopMidReplyWithLateChunk() async {
    session().closeOnCancel = false;
    final turn = Turn(repo.ask('Tell me a story'));
    await settle();
    session().emit('Once');
    await Future<void>.delayed(const Duration(milliseconds: 5));
    session().emit(' upon');
    await Future<void>.delayed(const Duration(milliseconds: 5));
    session().emit(' a');
    await settle();

    final stopping = repo.stop();
    await settle();
    session().emit(' time'); // queued before native cancelled
    await settle();
    if (session().streaming) session().finish(); // native CANCELLED
    await stopping;
    await turn.done;
    return turn;
  }

  test('stop before the first chunk: the turn still starts, the stop is '
      're-sent natively, and the turn ends stopped', () async {
    final turn = await stopBeforeFirstChunk();

    expect(session().nativeCancels, 1, reason: 'stop re-sent once generating');
    expect(turn.deltas, isEmpty, reason: 'nothing reaches the UI after stop');
    expect(turn.metrics.stopped, isTrue);
    expect(turn.metrics.stopLatency, isNotNull);
    expect(turn.metrics.chunks, 0);
    expect(repo.isGenerating.value, isFalse);
  });

  test('stop mid-reply drains late chunks without showing them and keeps the '
      'partial reply in the chat history', () async {
    final turn = await stopMidReplyWithLateChunk();

    expect(turn.deltas, ['Once', ' upon', ' a']);
    expect(turn.metrics.stopped, isTrue);
    expect(turn.metrics.chunks, 3);
    // InferenceChat's end-of-turn bookkeeping ran: its history has the same
    // partial reply LiteRT-LM's conversation recorded.
    final history = model.chat!.fullHistory;
    expect(history.last.isUser, isFalse);
    expect(history.last.text, 'Once upon a time');
    expect(repo.isGenerating.value, isFalse);
  });

  // Which rate a stopped or chunkless turn reports (never the native one):
  // generation_metrics_builder_test.dart.

  test('a native stream error ends the turn with AssistantFailed', () async {
    final turn = Turn(repo.ask('hi'));
    await settle();
    session()
      ..emit('Hel')
      ..failWith(Exception('Stream error: boom'));
    await turn.done;

    expect(turn.deltas, ['Hel']);
    expect(turn.events.last, isA<AssistantFailed>());
    expect(repo.isGenerating.value, isFalse);
  });

  test('close during a turn stops it and closes the chat', () async {
    final turn = Turn(repo.ask('Tell me a story'));
    await settle();
    session().emit('Once');
    await settle();
    final chatSession = session();

    await repo.close();
    await turn.done;

    expect(turn.metrics.stopped, isTrue);
    expect(chatSession.closed, isTrue);
    expect(repo.isOpen, isFalse);
    final after = await repo.ask('again').toList();
    expect(after.single, isA<AssistantFailed>());
  });

  group('while close waits for the running turn to end', () {
    test('an ask made the moment the reply ends is refused: no turn starts '
        'on the chat being closed', () async {
      session().closeOnCancel = false; // the stopped turn drains a while
      final turn = Turn(repo.ask('Tell me a story'));
      await settle();
      session().emit('Once');
      await settle();
      final draining = session();
      // A caller that asks again as soon as the chat is idle.
      Turn? next;
      repo.isGenerating.addListener(() {
        if (!repo.isGenerating.value) next ??= Turn(repo.ask('again'));
      });

      final closing = repo.close();
      await settle();
      draining.finish(); // native CANCELLED: the turn ends
      await turn.done;
      await closing;
      await next!.done.timeout(const Duration(seconds: 2));

      expect(
        next!.events.single,
        isA<AssistantFailed>().having(
          (e) => e.error,
          'error',
          isA<ConversationNotReadyException>(),
        ),
      );
      expect(draining.queries.map((m) => m.text), ['Tell me a story']);
    });

    test('an open is refused and builds no chat', () async {
      session().closeOnCancel = false;
      final turn = Turn(repo.ask('Tell me a story'));
      await settle();
      final draining = session();

      final closing = repo.close();
      await settle();
      final opening = repo.open(kCameraProfile);
      await settle();
      draining.finish();
      await turn.done;
      await closing;

      expect(await opening, isA<Error<void>>());
      expect(model.chatsCreated, 1);
    });
  });

  test(
    'open() during a draining turn waits for the drain, then rebuilds',
    () async {
      session().closeOnCancel = false; // native keeps delivering after cancel
      final turn = Turn(repo.ask('Tell me a story'));
      await settle();
      session().emit('Once');
      await settle();
      final draining = session();

      // Back to home mid-reply: the leaving view model stops without awaiting.
      unawaited(repo.stop());
      await settle();
      var opened = false;
      final opening = repo.open(kCameraProfile).then((r) {
        opened = true;
        return r;
      });
      await settle();
      expect(opened, isFalse, reason: 'must wait for the drain, not fail');
      expect(model.chatsCreated, 1);

      draining
        ..emit(' upon') // late chunk, drained
        ..finish(); // native CANCELLED
      final result = await opening;
      await turn.done;

      expect(result, isA<Ok<void>>());
      expect(model.chatsCreated, 2);
      expect(repo.profile, same(kCameraProfile));
      expect(turn.metrics.stopped, isTrue);
      expect(repo.isGenerating.value, isFalse);
    },
  );

  test('open() during a live turn stops it and waits', () async {
    final turn = Turn(repo.ask('Tell me a story'));
    await settle();
    session().emit('Once');
    await settle();

    final result = await repo.open(kCameraProfile);
    await turn.done;

    expect(result, isA<Ok<void>>());
    expect(turn.metrics.stopped, isTrue);
    expect(model.chatsCreated, 2);
  });

  test('switching profiles rebuilds only the chat, never the model', () async {
    var loads = 0;
    final llm = LlmService(
      loadModel: (_) async {
        loads++;
        return model;
      },
    );
    await llm.load(kDefineChatModel);
    final switching = EdgeAiConversationRepository(llm: llm);
    addTearDown(switching.close);

    expect(await switching.open(kVoiceChatProfile), isA<Ok<void>>());
    expect(await switching.open(kCameraProfile), isA<Ok<void>>());

    expect(loads, 1, reason: 'kLlmConfig stays the only model argument set');
    expect(switching.profile, same(kCameraProfile));
    expect(model.sessionSettings.last, (
      temperature: kSampler.temperature,
      topK: kSampler.topK,
      maxOutputTokens: kCameraProfile.maxOutputTokens,
    ));
  });

  test('reset keeps the profile and drops the history', () async {
    await repo.open(kCameraProfile);
    final turn = Turn(repo.ask('What is this?'));
    await settle();
    session()
      ..emit('A cat.')
      ..finish();
    await turn.done;
    expect(model.chat!.fullHistory, isNotEmpty);

    expect(await repo.reset(ifCurrent: kCameraProfile), isA<Ok<void>>());

    expect(repo.profile, same(kCameraProfile));
    expect(model.chat!.fullHistory, isEmpty);
  });

  test('concurrent opens run one at a time and the last one wins', () async {
    final gate = model.createGate = Completer<void>();
    final first = repo.open(kCameraProfile);
    final second = repo.open(kVoiceChatProfile);
    await settle();
    gate.complete();

    expect(await first, isA<Ok<void>>());
    expect(await second, isA<Ok<void>>());
    expect(model.maxConcurrentCreates, 1);
    expect(model.chatsCreated, 3);
    expect(repo.profile, same(kVoiceChatProfile));
  });

  // Which queued opens are superseded and never built: open_queue_test.dart.

  test('a post-turn reset that lands after a switch to another demo does not '
      'reopen the old profile', () async {
    expect(await repo.open(kCameraProfile), isA<Ok<void>>());
    session().closeOnCancel = false; // native keeps delivering after cancel
    final turn = Turn(repo.ask('What is this?'));
    await settle();
    session().emit('A cat');
    await settle();
    final draining = session();

    // Back to home mid-reply, straight into Demo 1: its open waits for the
    // drain while the camera chat is still the open one.
    unawaited(repo.stop());
    final opening = repo.open(kVoiceChatProfile);
    await settle();
    // Demo 3's unawaited post-turn reset runs now.
    final resetting = repo.reset(ifCurrent: kCameraProfile);
    await settle();
    draining.finish();

    expect(await opening, isA<Ok<void>>());
    expect(await resetting, isA<Ok<void>>(), reason: 'moot, not an error');
    await turn.done;
    expect(repo.profile, same(kVoiceChatProfile));
    expect(model.chatsCreated, 3, reason: 'setUp voice + camera + voice');
    expect(
      model.sessionSettings.last.maxOutputTokens,
      kVoiceChatProfile.maxOutputTokens,
    );
  });

  test('reset for the current profile rebuilds once even when an open for '
      'the same profile is still queued', () async {
    final gate = model.createGate = Completer<void>();
    final running = repo.open(kCameraProfile);
    await settle();
    final queued = repo.open(kVoiceChatProfile);
    final resetting = repo.reset(ifCurrent: kVoiceChatProfile);
    await settle();
    gate.complete();

    expect(await running, isA<Ok<void>>());
    expect(await queued, isA<Ok<void>>());
    expect(await resetting, isA<Ok<void>>());
    expect(model.chatsCreated, 3, reason: 'setUp + camera + one voice');
    expect(repo.profile, same(kVoiceChatProfile));
  });

  test('open() gives up after the stop timeout and leaves the previous chat '
      'and profile in place; a retry after the drain works', () async {
    final llm = LlmService(loadModel: (_) async => model);
    await llm.load(kDefineChatModel);
    final impatient = EdgeAiConversationRepository(
      llm: llm,
      stopTimeout: const Duration(milliseconds: 50),
    );
    addTearDown(impatient.close);
    expect(await impatient.open(kVoiceChatProfile), isA<Ok<void>>());
    final stuck = model.lastSession..closeOnCancel = false;
    final turn = Turn(impatient.ask('Tell me a story'));
    await settle();
    stuck.emit('Once'); // native never ends this turn on its own

    final result = await impatient.open(kCameraProfile);

    expect(result, isA<Error<void>>());
    expect(
      (result as Error<void>).error.toString(),
      contains('did not finish'),
    );
    expect(impatient.profile, same(kVoiceChatProfile));
    expect(impatient.isOpen, isTrue);
    expect(impatient.isGenerating.value, isTrue);
    final creates = model.chatsCreated;

    stuck.finish();
    await turn.done;
    expect(await impatient.open(kCameraProfile), isA<Ok<void>>());
    expect(impatient.profile, same(kCameraProfile));
    expect(model.chatsCreated, creates + 1);
  });

  test('close() during a rebuild waits for the chat being built, then closes '
      'it: the model is never closed under an in-flight createChat', () async {
    final gate = model.createGate = Completer<void>();
    final opening = repo.open(kCameraProfile);
    await settle();
    expect(model.chatsCreated, 2, reason: 'the camera chat is being built');

    var closed = false;
    final closing = repo.close().then((_) => closed = true);
    final late = repo.open(kVoiceChatProfile);
    await settle();
    expect(closed, isFalse, reason: 'createChat is still in flight');
    expect(
      await late,
      isA<Error<void>>().having(
        (e) => e.error,
        'error',
        isA<ConversationNotReadyException>(),
      ),
      reason: 'an open while close waits is refused at once',
    );

    gate.complete();
    await closing;

    expect(await opening, isA<Error<void>>());
    expect(model.chatsCreated, 2, reason: 'nothing built after the close');
    expect(model.lastSession.closed, isTrue, reason: 'no orphaned chat');
    expect(repo.isOpen, isFalse);
    expect(repo.profile, isNull);
  });

  test('a second close() while the first waits for a rebuild gets the same '
      'close', () async {
    final gate = model.createGate = Completer<void>();
    final opening = repo.open(kCameraProfile);
    await settle();

    final first = repo.close();
    final second = repo.close();
    expect(identical(first, second), isTrue);
    gate.complete();
    await Future.wait([first, second, opening]);

    expect(model.lastSession.closed, isTrue);
  });

  test('reset during a turn stops it and opens a fresh chat', () async {
    final turn = Turn(repo.ask('Tell me a story'));
    await settle();
    session().emit('Once');
    await settle();

    expect(await repo.reset(ifCurrent: kVoiceChatProfile), isA<Ok<void>>());
    await turn.done;

    expect(turn.metrics.stopped, isTrue);
    expect(model.chatsCreated, 2);
    expect(model.chat!.fullHistory, isEmpty);

    final next = Turn(repo.ask('Hi'));
    await settle();
    session()
      ..emit('Hello')
      ..finish();
    await next.done;
    expect(next.deltas, ['Hello']);
    expect(next.metrics.stopped, isFalse);
  });
}
