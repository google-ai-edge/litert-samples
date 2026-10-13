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

// EdgeAiConversationRepository.release() before a chat model switch: it is
// drain (queued opens), stop (the running turn), forget the requested
// profile, then close the chat, refusing opens that arrive meanwhile. Each
// step is pinned here so none can be dropped or reordered unnoticed. Also
// close() while a running turn rebuilds its chat (the model is closed right
// after it).
import 'dart:async';

import 'package:flutter_edge_ai/flutter_edge_ai.dart' hide ModelSpec;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository_edge_ai.dart';
import 'package:litert_edge_demos/data/services/llm/llm_service.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_tool_model.dart';

/// [FakeToolModel] whose session creation can be held ([createGate]).
class _GatedToolModel extends FakeToolModel {
  Completer<void>? createGate;

  @override
  Future<InferenceModelSession> createSession({
    double temperature = .8,
    int randomSeed = 1,
    int topK = 1,
    double? topP,
    String? loraPath,
    bool? enableVisionModality,
    bool? enableAudioModality,
    String? systemInstruction,
    bool enableThinking = false,
    List<Tool> tools = const [],
    int? maxOutputTokens,
  }) async {
    await createGate?.future;
    return super.createSession(
      temperature: temperature,
      randomSeed: randomSeed,
      topK: topK,
      topP: topP,
      loraPath: loraPath,
      enableVisionModality: enableVisionModality,
      enableAudioModality: enableAudioModality,
      systemInstruction: systemInstruction,
      enableThinking: enableThinking,
      tools: tools,
      maxOutputTokens: maxOutputTokens,
    );
  }
}

Future<void> settle([int rounds = 20]) async {
  for (var i = 0; i < rounds; i++) {
    await Future<void>.delayed(Duration.zero);
  }
}

void main() {
  late _GatedToolModel model;
  late EdgeAiConversationRepository repo;

  setUp(() async {
    model = _GatedToolModel();
    final llm = LlmService(loadModel: (_) async => model);
    expect(await llm.load(kDefineChatModel), isA<Ok<LlmInfo>>());
    repo = EdgeAiConversationRepository(
      llm: llm,
      stopTimeout: const Duration(seconds: 2),
    );
    expect(await repo.open(kVoiceChatProfile), isA<Ok<void>>());
  });

  tearDown(() => repo.close());

  test('waits for an open queued before it, then closes the chat that open '
      'built', () async {
    final gate = model.createGate = Completer<void>();
    final opening = repo.open(kCameraProfile);
    await settle();
    var released = false;
    final releasing = repo.release().then((_) => released = true);
    await settle();

    expect(released, isFalse, reason: 'the camera chat is still building');
    gate.complete();
    expect(await opening, isA<Ok<void>>());
    await releasing;

    expect(
      model.chats.last.systemInstruction,
      kCameraProfile.systemInstruction,
    );
    expect(model.lastSession.closed, isTrue, reason: 'no chat outlives it');
    expect(repo.isOpen, isFalse);
  });

  test('stops a running turn and lets it end before the chat is '
      'closed', () async {
    model.script.add(TextTurn(const ['One.'], gate: Completer<void>()));
    final session = model.lastSession;
    bool? closedAtDone;
    final events = <AssistantEvent>[];
    final turn = repo.ask('Tell me a story').listen((event) {
      events.add(event);
      if (event is AssistantDone) closedAtDone = session.closed;
    }).asFuture<void>();
    await settle();
    expect(repo.isGenerating.value, isTrue);

    await repo.release();
    await turn;

    expect(session.stopCalls, greaterThanOrEqualTo(1));
    final done = events.last as AssistantDone;
    expect(done.metrics.stopped, isTrue);
    expect(closedAtDone, isFalse, reason: 'the turn ended first');
    expect(session.closed, isTrue);
    expect(repo.isGenerating.value, isFalse);
  });

  test('an open that arrives while it waits for the stopped turn is refused, '
      'and a reset is moot: no chat is built after it returns', () async {
    model.script.add(TextTurn(const ['One.'], gate: Completer<void>()));
    // The stop is held: the turn ends only when the test finishes it.
    final session = model.lastSession..closeOnCancel = false;
    final turn = repo.ask('Tell me a story').drain<void>();
    await settle();
    expect(session.streaming, isTrue);

    var released = false;
    final releasing = repo.release().then((_) => released = true);
    await settle();
    expect(session.stopCalls, greaterThanOrEqualTo(1));
    expect(released, isFalse, reason: 'the stopped turn has not ended yet');

    final chats = model.chats.length;
    final opening = repo.open(kCameraProfile);
    final resetting = repo.reset(ifCurrent: kVoiceChatProfile);
    await settle();
    session.finish(); // the stopped turn ends
    await releasing;
    final opened = await opening;
    await turn;
    await settle();

    expect(
      opened,
      isA<Error<void>>().having(
        (e) => e.error,
        'error',
        isA<ConversationNotReadyException>(),
      ),
    );
    expect(await resetting, isA<Ok<void>>());
    expect(model.chats.length, chats, reason: 'no chat built after release');
    expect(repo.isOpen, isFalse);
    expect(session.closed, isTrue);

    // Once it has returned, opens are taken again.
    expect(await repo.open(kCameraProfile), isA<Ok<void>>());
    expect(repo.isOpen, isTrue);
  });

  test('an open right after it starts (nothing to drain or stop) is refused '
      'too', () async {
    final session = model.lastSession;
    final chats = model.chats.length;

    final releasing = repo.release();
    final opening = repo.open(kCameraProfile);
    await releasing;

    expect(await opening, isA<Error<void>>());
    expect(model.chats.length, chats);
    expect(repo.isOpen, isFalse);
    expect(session.closed, isTrue);
  });

  test('a stopped turn still generating after the stop timeout (the native '
      'turn ignores the stop): release refuses and closes nothing under the '
      'running generation; once the turn ends, a release works', () async {
    final llm = LlmService(loadModel: (_) async => model);
    expect(await llm.load(kDefineChatModel), isA<Ok<LlmInfo>>());
    final impatient = EdgeAiConversationRepository(
      llm: llm,
      stopTimeout: const Duration(milliseconds: 50),
    );
    addTearDown(impatient.close);
    expect(await impatient.open(kVoiceChatProfile), isA<Ok<void>>());
    model.script.add(TextTurn(const ['One.'], gate: Completer<void>()));
    final session = model.lastSession..closeOnCancel = false;
    final turn = impatient.ask('Tell me a story').drain<void>();
    await settle();
    expect(session.streaming, isTrue);

    final refused = await impatient.release();

    expect(
      refused,
      isA<Error<void>>().having(
        (e) => e.error,
        'error',
        isA<ConversationNotReadyException>().having(
          (e) => '$e',
          'message',
          contains('did not finish'),
        ),
      ),
    );
    expect(session.stopCalls, greaterThanOrEqualTo(1));
    expect(session.closed, isFalse, reason: 'it still generates');
    expect(impatient.isOpen, isTrue);
    expect(impatient.isGenerating.value, isTrue);
    // The requested profile is kept: a reset for it is not moot, it is
    // refused only because the stopped turn has not ended yet.
    expect(
      await impatient.reset(ifCurrent: kVoiceChatProfile),
      isA<Error<void>>(),
    );

    session.finish(); // the stopped turn finally ends
    await turn;
    expect(await impatient.release(), isA<Ok<void>>());
    expect(session.closed, isTrue);
    expect(impatient.isOpen, isFalse);
  });

  test('close() after a stop that timed out (the native turn ignores the '
      'stop) leaves the chat to the running generation: not closed under '
      'it', () async {
    final llm = LlmService(loadModel: (_) async => model);
    expect(await llm.load(kDefineChatModel), isA<Ok<LlmInfo>>());
    final impatient = EdgeAiConversationRepository(
      llm: llm,
      stopTimeout: const Duration(milliseconds: 50),
    );
    expect(await impatient.open(kVoiceChatProfile), isA<Ok<void>>());
    model.script.add(TextTurn(const ['One.'], gate: Completer<void>()));
    final session = model.lastSession..closeOnCancel = false;
    final turn = impatient.ask('Tell me a story').drain<void>();
    await settle();
    expect(session.streaming, isTrue);

    await impatient.close();

    expect(session.stopCalls, greaterThanOrEqualTo(1));
    expect(session.closed, isFalse, reason: 'it still generates');
    expect(impatient.isOpen, isFalse, reason: 'the repository is closed');

    session.finish(); // the stopped turn finally ends
    await turn;
  });

  test('forgets the requested profile: a late reset for it is moot and '
      'builds nothing', () async {
    await repo.release();
    final chats = model.chats.length;

    expect(await repo.reset(ifCurrent: kVoiceChatProfile), isA<Ok<void>>());

    expect(model.chats.length, chats);
    expect(repo.isOpen, isFalse);
  });

  test(
    'close() while a turn\'s budget guard rebuilds its chat for longer '
    'than the stop timeout: it waits for the chat being built and closes '
    'it, so the model is never closed under an in-flight createChat',
    () async {
      final llm = LlmService(loadModel: (_) async => model);
      expect(await llm.load(kDefineChatModel), isA<Ok<LlmInfo>>());
      final impatient = EdgeAiConversationRepository(
        llm: llm,
        stopTimeout: const Duration(milliseconds: 50),
      );
      expect(await impatient.open(kVoiceChatProfile), isA<Ok<void>>());
      model.lastSession.nativeInputTokens = 4096 - kContextHeadroomTokens - 100;
      final gate = model.createGate = Completer<void>();
      final turn = impatient.ask('abcd' * 5).toList();
      await settle();
      expect(
        impatient.isOpen,
        isFalse,
        reason: 'the budget reset is under way',
      );
      final sessions = model.created.length;

      var closed = false;
      final closing = impatient.close().then((_) => closed = true);
      await Future<void>.delayed(const Duration(milliseconds: 200));
      expect(closed, isFalse, reason: 'createChat is still in flight');

      gate.complete();
      await closing;
      final events = await turn;

      expect(model.created, hasLength(sessions + 1));
      expect(model.lastSession.closed, isTrue, reason: 'no orphaned chat');
      expect(model.lastSession.queries, isEmpty, reason: 'nothing sent on it');
      expect(events.last, isA<AssistantFailed>());
      expect(impatient.isOpen, isFalse);
    },
  );
}
