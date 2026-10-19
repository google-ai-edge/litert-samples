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
import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show ModelType, PreferredBackend;
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show TextSkillExecutor, parseSkillMd;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/conversation/history_mirror.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository_edge_ai.dart';
import 'package:litert_edge_demos/data/services/llm/llm_service.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/chat_model_config.dart';
import 'package:litert_edge_demos/domain/models/conversation_history.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:litert_edge_demos/utils/wav.dart';

import '../../fakes/fake_inference_model.dart';
import '../../fakes/fake_tool_model.dart';

/// Lets queued microtasks and zero-length timers run.
Future<void> settle() => Future<void>.delayed(Duration.zero);

/// One `ask` stream, collected.
final class _Turn {
  _Turn(Stream<AssistantEvent> stream) {
    done = stream.listen(events.add).asFuture<void>();
  }

  final List<AssistantEvent> events = [];
  late final Future<void> done;
}

/// Gemma 4 E2B with its audio encoder (and no tools: the plain path).
const _audioGemma = ChatModelConfig(
  name: 'Gemma 4 E2B',
  modelType: ModelType.gemma4,
  llm: LlmConfig(
    maxTokens: 8192,
    backend: PreferredBackend.gpu,
    maxNumImages: 1,
    supportImage: true,
    supportAudio: true,
  ),
  tools: false,
);

/// The conversation's text mirror on the real repository over the fake
/// engine session: what each turn records, the transcript that arrives
/// later, and the rebuilds that replay it instead of the engine's own text
/// history.
void main() {
  _agentPath();

  late FakeInferenceModel model;
  late EdgeAiConversationRepository repo;
  late List<String> log;
  late DebugPrintCallback originalDebugPrint;

  // 1 s of a 16 kHz mono PCM16 recording.
  final wav = wavFromPcm16(
    Uint8List.fromList(List.generate(32000, (i) => i % 7)),
    sampleRate: 16000,
  );

  FakeInferenceSession session() => model.lastSession;

  Future<_Turn> answer(
    String prompt, {
    Uint8List? audioWav,
    String? historyText,
    int? turn,
    String reply = 'Ok.',
  }) async {
    final t = _Turn(
      repo.ask(
        prompt,
        audioWav: audioWav,
        historyText: historyText,
        turn: turn,
      ),
    );
    await settle();
    session()
      ..emit(reply)
      ..finish();
    await t.done;
    return t;
  }

  /// Asks, streams one chunk, stops, and lets native end the stream.
  Future<_Turn> stopped(
    String prompt, {
    Uint8List? audioWav,
    String? historyText,
    int? turn,
  }) async {
    final t = _Turn(
      repo.ask(
        prompt,
        audioWav: audioWav,
        historyText: historyText,
        turn: turn,
      ),
    );
    await settle();
    session().emit('Once upon');
    await settle();
    await repo.stop();
    await t.done;
    return t;
  }

  setUp(() async {
    log = [];
    originalDebugPrint = debugPrint;
    debugPrint = (message, {wrapWidth}) => log.add(message ?? '');
    model = FakeInferenceModel();
    final llm = LlmService(loadModel: (_) async => model);
    expect(await llm.load(_audioGemma), isA<Ok<LlmInfo>>());
    repo = EdgeAiConversationRepository(
      llm: llm,
      stopTimeout: const Duration(seconds: 2),
    );
    expect(await repo.open(kVoiceChatProfile), isA<Ok<void>>());
  });

  tearDown(() async {
    await repo.close();
    debugPrint = originalDebugPrint;
  });

  test('a text turn records the question it is given, not the prompt with '
      'its excerpts, and the reply as it streamed', () async {
    await answer(
      'Excerpts…\n\nQuestion: What is LiteRT?',
      historyText: 'What is LiteRT?',
      turn: 1,
      reply: 'A runtime.',
    );

    expect(repo.history, hasLength(1));
    final t = repo.history.single;
    expect(
      (t.user, t.assistant, t.turn, t.audio),
      ('What is LiteRT?', 'A runtime.', 1, false),
    );
  });

  test('without a history text the prompt is what the history keeps (the '
      'camera\'s turns)', () async {
    await answer('Describe the frame.');
    expect(repo.history.single.user, 'Describe the frame.');
    expect(repo.history.single.turn, isNull);
  });

  test('an audio turn records the placeholder; its transcript replaces it '
      'later by the turn\'s number', () async {
    await answer(
      PromptBuilderLike.spoken,
      audioWav: wav,
      historyText: kSpokenQuestionPlaceholder,
      turn: 3,
      reply: 'Paris.',
    );
    expect(repo.history.single.user, kSpokenQuestionPlaceholder);
    expect(repo.history.single.audio, isTrue);

    expect(repo.setHistoryText(3, 'What is the capital of France?'), isTrue);
    expect(repo.history.single.user, 'What is the capital of France?');
    expect(repo.history.single.assistant, 'Paris.');
    expect(repo.setHistoryText(4, 'never asked'), isFalse);
  });

  test('a turn refused before it reaches the model is not recorded', () async {
    final busy = _Turn(repo.ask('first'));
    await settle();
    final refused = await repo.ask('second', historyText: 'second').toList();
    expect(refused.single, isA<AssistantFailed>());
    session()
      ..emit('Ok.')
      ..finish();
    await busy.done;
    expect(repo.history.map((t) => t.user), ['first']);
  });

  test('open and reset start an empty history', () async {
    await answer('Hi', historyText: 'Hi');
    expect(repo.history, isNotEmpty);
    expect(await repo.reset(ifCurrent: kVoiceChatProfile), isA<Ok<void>>());
    expect(repo.history, isEmpty);
    await answer('Hello', historyText: 'Hello');
    expect(await repo.open(kCameraProfile), isA<Ok<void>>());
    expect(repo.history, isEmpty);
  });

  group('a stop with an audio turn in the live chat', () {
    test(
      'the next turn rebuilds the chat from the text history '
      '(clearHistory): a new native conversation on the same chat whose '
      'first message replays the turns as text — the transcript, or the '
      'placeholder — and no recording survives in the chat\'s history',
      () async {
        await answer(
          PromptBuilderLike.spoken,
          audioWav: wav,
          historyText: kSpokenQuestionPlaceholder,
          turn: 1,
          reply: 'Paris.',
        );
        repo.setHistoryText(1, 'What is the capital of France?');
        await stopped(
          PromptBuilderLike.spoken,
          audioWav: wav,
          historyText: kSpokenQuestionPlaceholder,
          turn: 2,
        );
        expect(
          model.chat!.fullHistory.where((m) => m.hasAudio),
          hasLength(2),
          reason: 'until the rebuild the chat keeps both recordings',
        );
        final chats = model.chatsCreated;
        final stoppedSession = session();

        final next = await answer('And Germany?', historyText: 'And Germany?');

        expect(model.chatsCreated, chats, reason: 'the same chat, cleared');
        expect(stoppedSession.closed, isTrue);
        expect(session(), isNot(same(stoppedSession)));
        final queries = session().queries;
        expect(queries, hasLength(2));
        expect(
          queries.first.text,
          HistoryMirror.replayText(const [
            HistoryTurn(
              user: 'What is the capital of France?',
              assistant: 'Paris.',
            ),
            HistoryTurn(
              user: kSpokenQuestionPlaceholder,
              assistant: 'Once upon',
            ),
          ]),
        );
        expect(queries.last.text, 'And Germany?');
        expect(model.chat!.fullHistory.where((m) => m.hasAudio), isEmpty);
        expect(
          log.where(
            (l) => l.startsWith(
              '[Conversation] rebuilt the chat from the text history: 2 turns',
            ),
          ),
          hasLength(1),
        );
        expect(next.events.whereType<AssistantContextReset>(), isEmpty);
        expect(next.events.last, isA<AssistantDone>());

        // The rebuilt chat holds no audio: a further stop leaves the rebuild to
        // flutter_edge_ai_litertlm again.
        await stopped('Tell me more', historyText: 'Tell me more');
        final created = model.created.length;
        await answer('Thanks', historyText: 'Thanks');
        expect(model.created.length, created, reason: 'no second clear');
      },
    );

    test('a transcript that arrives after the rebuild replayed its '
        'placeholder reaches the model: the next turn rebuilds the chat '
        'again with the words; one for a question still heard as audio '
        'rebuilds nothing', () async {
      await answer(
        PromptBuilderLike.spoken,
        audioWav: wav,
        historyText: kSpokenQuestionPlaceholder,
        turn: 1,
        reply: 'Paris.',
      );
      await stopped(
        PromptBuilderLike.spoken,
        audioWav: wav,
        historyText: kSpokenQuestionPlaceholder,
        turn: 2,
      );
      await answer('And Spain?', historyText: 'And Spain?', reply: 'Madrid.');
      expect(
        session().queries.first.text,
        contains('User: $kSpokenQuestionPlaceholder\nAssistant: Paris.'),
        reason: 'replayed before its words arrived',
      );
      bool rebuilt(String l) =>
          l.contains('rebuilt the chat from the text history');
      expect(log.where(rebuilt), hasLength(1));

      // Turn 1's words arrive late: the live chat holds only the placeholder.
      expect(repo.setHistoryText(1, 'What is the capital of France?'), isTrue);
      final rebuiltSession = session();
      final next = await answer('Thanks', historyText: 'Thanks');

      expect(log.where(rebuilt), hasLength(2));
      expect(rebuiltSession.closed, isTrue, reason: 'cleared and rebuilt');
      expect(
        session().queries.first.text,
        HistoryMirror.replayText(const [
          HistoryTurn(
            user: 'What is the capital of France?',
            assistant: 'Paris.',
          ),
          HistoryTurn(user: kSpokenQuestionPlaceholder, assistant: 'Once upon'),
          HistoryTurn(user: 'And Spain?', assistant: 'Madrid.'),
        ]),
      );
      expect(session().queries.last.text, 'Thanks');
      expect(next.events.whereType<AssistantContextReset>(), isEmpty);

      // A question the live chat holds as audio: its words change nothing.
      await answer(
        PromptBuilderLike.spoken,
        audioWav: wav,
        historyText: kSpokenQuestionPlaceholder,
        turn: 5,
      );
      expect(repo.setHistoryText(5, 'Is that all?'), isTrue);
      await answer('Bye', historyText: 'Bye');
      expect(log.where(rebuilt), hasLength(2), reason: 'no third rebuild');
    });

    test('a turn stopped before it sends leaves the rebuild due: the next '
        'turn that sends does it', () async {
      await stopped(
        PromptBuilderLike.spoken,
        audioWav: wav,
        historyText: kSpokenQuestionPlaceholder,
        turn: 1,
      );
      bool rebuilt(String l) =>
          l.contains('rebuilt the chat from the text history');
      // Stopped while the budget guard counts: nothing reaches the model.
      final gate = session().sizeGate = Completer<void>();
      final cut = _Turn(repo.ask('Next', historyText: 'Next'));
      await settle();
      final stopping = repo.stop();
      await settle();
      gate.complete();
      session().sizeGate = null;
      await stopping;
      await cut.done;
      expect((cut.events.single as AssistantDone).metrics.stopped, isTrue);
      expect(log.where(rebuilt), isEmpty);

      await answer('Again', historyText: 'Again');
      expect(log.where(rebuilt), hasLength(1));
      expect(session().queries.last.text, 'Again');
    });
  });

  test('a stopped text turn leaves the rebuild to flutter_edge_ai_litertlm '
      '(its text history is complete): no clearHistory', () async {
    await answer('What is LiteRT?', historyText: 'What is LiteRT?');
    await stopped('Tell me more', historyText: 'Tell me more');
    final created = model.created.length;
    await answer('Thanks', historyText: 'Thanks');
    expect(model.created.length, created);
    expect(session().queries.last.text, 'Thanks');
    expect(
      log.where((l) => l.contains('the text history')),
      isEmpty,
      reason: 'no replay of the mirror',
    );
  });

  test(
    'a budget reset gives the fresh chat the newest turns that fit the '
    'replay share, oldest dropped, and the reset says how many it kept',
    () async {
      // Each reply ~250 tokens (1000 characters at the fake's 4 per token):
      // 3 of them do not fit the replay share of the 3840-token limit (960).
      for (var i = 1; i <= 5; i++) {
        await answer('Q$i', historyText: 'Q$i', reply: 'x' * 1000);
      }
      session().nativeInputTokens = 3700;

      final turn = await answer('Q6', historyText: 'Q6', reply: 'Six.');

      final reset = turn.events.whereType<AssistantContextReset>().single;
      expect(reset.reason, ContextResetReason.budget);
      expect(reset.keptTurns, 3);
      expect(reset.droppedTurns, 2, reason: 'Q1 and Q2 did not fit');
      final replay = session().queries.first.text;
      expect(replay, startsWith(HistoryMirror.replayHeader));
      expect(replay, isNot(contains('User: Q2')));
      expect(replay, contains('User: Q3'));
      expect(replay, contains('User: Q5'));
      expect(session().queries.last.text, 'Q6');
      expect(
        log.any((l) => l.contains('replayed the text history: 3 turns')),
        isTrue,
      );
      expect(log.any((l) => l.contains('2 older turns dropped')), isTrue);
      expect(repo.history, hasLength(6), reason: 'the mirror keeps every turn');
    },
  );

  test('a budget reset that replays a spoken question still waiting for its '
      'words says so: the reset counts the kept turns that are only the '
      'placeholder', () async {
    for (var i = 1; i <= 5; i++) {
      await answer(
        i == 4 ? PromptBuilderLike.spoken : 'Q$i',
        audioWav: i == 4 ? wav : null,
        historyText: i == 4 ? kSpokenQuestionPlaceholder : 'Q$i',
        turn: i,
        reply: 'x' * 1000,
      );
    }
    session().nativeInputTokens = 3700;

    final turn = await answer('Q6', historyText: 'Q6', reply: 'Six.');

    final reset = turn.events.whereType<AssistantContextReset>().single;
    expect(reset.reason, ContextResetReason.budget);
    expect(reset.keptTurns, 3);
    expect(reset.wordlessTurns, 1, reason: 'turn 4 had no transcript yet');
  });

  group('a rebuild after a stop is not capped like a budget reset', () {
    test('six long exchanges and a barge-in: the rebuilt chat gets all seven '
        'back, and no turn says the conversation got too long', () async {
      // Each reply ~250 tokens (1000 characters at the fake's 4 per token):
      // six are well over a budget reset's share (960 of the 3840-token
      // limit) and well within what a rebuild has (the limit less the turn).
      await answer(
        PromptBuilderLike.spoken,
        audioWav: wav,
        historyText: kSpokenQuestionPlaceholder,
        turn: 1,
        reply: 'x' * 1000,
      );
      repo.setHistoryText(1, 'Q1');
      final turns = <_Turn>[];
      for (var i = 2; i <= 6; i++) {
        turns.add(await answer('Q$i', historyText: 'Q$i', reply: 'x' * 1000));
      }
      turns.add(await stopped('Q7', historyText: 'Q7')); // the barge-in

      final next = await answer('Q8', historyText: 'Q8');

      for (final t in [...turns, next]) {
        expect(t.events.whereType<AssistantContextReset>(), isEmpty);
      }
      expect((next.events.last as AssistantDone).metrics.contextReset, isFalse);
      final replay = session().queries.first.text;
      expect(replay, startsWith(HistoryMirror.replayHeader));
      for (var i = 1; i <= 7; i++) {
        expect(replay, contains('User: Q$i\n'), reason: 'Q$i kept');
      }
      expect(session().queries.last.text, 'Q8');
      expect(
        log.where(
          (l) => l.startsWith(
            '[Conversation] rebuilt the chat from the text history: 7 turns',
          ),
        ),
        hasLength(1),
      );
      expect(log.any((l) => l.contains('older turns dropped')), isFalse);
    });

    test('a turn that does not fit beside the next question is dropped, '
        'announced as a rebuild after a stop (not as a context overflow), '
        'with the turns it kept', () async {
      // Two replies of ~1400 tokens; the question after the stop carries
      // ~1000 tokens of excerpts, so the rebuilt chat has ~2400 tokens for the
      // text history: the stopped turn and the second exchange fit, the first
      // does not.
      await answer(
        PromptBuilderLike.spoken,
        audioWav: wav,
        historyText: kSpokenQuestionPlaceholder,
        turn: 1,
        reply: 'x' * 5600,
      );
      repo.setHistoryText(1, 'Q1');
      await answer('Q2', historyText: 'Q2', reply: 'x' * 5600);
      await stopped(
        PromptBuilderLike.spoken,
        audioWav: wav,
        historyText: kSpokenQuestionPlaceholder,
        turn: 3,
      );

      final next = await answer(
        'Excerpts: ${'y' * 4000}\n\nQuestion: Q4',
        historyText: 'Q4',
      );

      final reset = next.events.whereType<AssistantContextReset>().single;
      expect(reset.reason, ContextResetReason.rebuildAfterStop);
      expect(reset.keptTurns, 2);
      expect(reset.droppedTurns, 1, reason: 'the first exchange');
      final metrics = (next.events.last as AssistantDone).metrics;
      expect(metrics.contextReset, isTrue);
      expect(metrics.contextResetReason, ContextResetReason.rebuildAfterStop);
      final replay = session().queries.first.text;
      expect(replay, isNot(contains('User: Q1\n')));
      expect(replay, contains('User: Q2\n'));
      expect(replay, contains('User: $kSpokenQuestionPlaceholder\n'));
      expect(
        log.where(
          (l) =>
              l.startsWith(
                '[Conversation] rebuilt the chat from the text history: 2 '
                'turns',
              ) &&
              l.endsWith('1 older turns dropped'),
        ),
        hasLength(1),
      );
    });
  });
}

/// The agent path (an agent chat over Gemma 4 with audio): an audio turn
/// is recorded the same way, a turn after a stopped audio turn rebuilds
/// the same chat from the text history, and the fresh chat after an
/// interrupted skill call gets every turn that fits back.
void _agentPath() {
  group('agent chat', () {
    late FakeToolModel model;
    late EdgeAiConversationRepository repo;
    late List<String> log;
    late DebugPrintCallback originalDebugPrint;
    final wav = wavFromPcm16(Uint8List(16000), sampleRate: 16000);

    setUp(() async {
      log = [];
      originalDebugPrint = debugPrint;
      debugPrint = (message, {wrapWidth}) => log.add(message ?? '');
      model = FakeToolModel();
      final llm = LlmService(loadModel: (_) async => model);
      expect(
        await llm.load(
          const ChatModelConfig(
            name: 'Gemma 4 E2B',
            modelType: ModelType.gemma4,
            llm: LlmConfig(
              maxTokens: 8192,
              backend: PreferredBackend.gpu,
              maxNumImages: 1,
              supportImage: true,
              supportAudio: true,
            ),
            tools: true,
          ),
        ),
        isA<Ok<LlmInfo>>(),
      );
      repo = EdgeAiConversationRepository(
        llm: llm,
        stopTimeout: const Duration(seconds: 2),
        executors: [TextSkillExecutor()],
      );
      final skill = parseSkillMd(
        File('assets/skills/current-time/SKILL.md').readAsStringSync(),
      );
      expect(
        await repo.open(kVoiceChatProfile, skills: [skill]),
        isA<Ok<void>>(),
      );
      expect(repo.hasSkills, isTrue);
    });

    tearDown(() async {
      await repo.close();
      debugPrint = originalDebugPrint;
    });

    test('an audio turn records the placeholder and the reply; the '
        'transcript replaces it', () async {
      model.script.add(const TextTurn(['Paris.']));
      final events = await repo
          .ask(
            PromptBuilderLike.spoken,
            audioWav: wav,
            historyText: kSpokenQuestionPlaceholder,
            turn: 1,
          )
          .toList();
      expect(events.last, isA<AssistantDone>());
      expect(model.lastSession.queries.single.audioBytes, same(wav));
      expect(repo.history.single.user, kSpokenQuestionPlaceholder);
      expect(repo.history.single.assistant, 'Paris.');
      expect(repo.history.single.audio, isTrue);
      expect(repo.setHistoryText(1, 'Capital of France?'), isTrue);
      expect(repo.history.single.user, 'Capital of France?');
    });

    test('an audio turn, a stop mid-reply, the next turn: the same agent chat '
        'is rebuilt once from the text history (the replay line first), no '
        'recording is left in its history, and the tools are still '
        'declared', () async {
      model.script.add(const TextTurn(['Paris.']));
      await repo
          .ask(
            PromptBuilderLike.spoken,
            audioWav: wav,
            historyText: kSpokenQuestionPlaceholder,
            turn: 1,
          )
          .drain<void>();
      repo.setHistoryText(1, 'Capital of France?');

      // The second audio question is stopped at its first words.
      model.script.add(
        const TextTurn(['Once ', 'upon ', 'a ', 'time ', 'there ', 'was.']),
      );
      final cut = <AssistantEvent>[];
      var stopping = false;
      await repo
          .ask(
            PromptBuilderLike.spoken,
            audioWav: wav,
            historyText: kSpokenQuestionPlaceholder,
            turn: 2,
          )
          .map((e) {
            cut.add(e);
            if (e is AssistantTextDelta && !stopping) {
              stopping = true;
              unawaited(repo.stop());
            }
            return e;
          })
          .drain<void>();
      expect((cut.last as AssistantDone).metrics.stopped, isTrue);
      expect(
        model.chat!.fullHistory.where((m) => m.hasAudio),
        hasLength(2),
        reason: 'until the rebuild the chat keeps both recordings',
      );
      final stoppedReply = repo.history[1].assistant;
      expect(stoppedReply, isNotEmpty);
      expect(stoppedReply, isNot('Once upon a time there was.'));
      final chats = model.chats.length;
      final sessions = model.created.length;
      final declared = [for (final t in model.lastSession.tools) t.name];
      expect(declared, hasLength(2));

      model.script.add(const TextTurn(['Berlin.']));
      final next = await repo
          .ask('And Germany?', historyText: 'And Germany?')
          .toList();

      expect(next.last, isA<AssistantDone>());
      expect(next.whereType<AssistantContextReset>(), isEmpty);
      expect(model.chats.length, chats, reason: 'the same chat, cleared');
      expect(model.created.length, sessions + 1, reason: 'one new session');
      final queries = model.lastSession.queries;
      expect(
        queries.first.text,
        HistoryMirror.replayText([
          const HistoryTurn(user: 'Capital of France?', assistant: 'Paris.'),
          HistoryTurn(
            user: kSpokenQuestionPlaceholder,
            assistant: stoppedReply,
          ),
        ]),
      );
      expect(queries.last.text, 'And Germany?');
      expect(model.chat!.fullHistory.where((m) => m.hasAudio), isEmpty);
      expect(
        log.where(
          (l) => l.startsWith('[Conversation] rebuilt the chat from the text'),
        ),
        hasLength(1),
      );
      expect([for (final t in model.lastSession.tools) t.name], declared);
      expect(model.chat!.tools, hasLength(2));
    });

    test('an interrupted skill call\'s fresh chat gets every turn that fits '
        'beside the next one back, not only a budget reset\'s share', () async {
      // Six replies of ~250 tokens: over a budget reset's share (960 of the
      // 3840-token limit), well within the room the fresh chat has.
      for (var i = 1; i <= 6; i++) {
        model.script.add(TextTurn(['x' * 1000]));
        final events = await repo.ask('Q$i', historyText: 'Q$i').toList();
        expect(events.whereType<AssistantContextReset>(), isEmpty);
        expect(events.last, isA<AssistantDone>());
      }
      // A skill call that fails mid-turn leaves input staged: the next turn
      // starts on a fresh chat.
      model.script.addAll([
        const ToolCallTurn('loadSkill', {'skillName': 'current-time'}),
        FailTurn(StateError('decode failed')),
      ]);
      final failed = await repo
          .ask('What time is it?', historyText: 'What time is it?')
          .toList();
      expect(failed.last, isA<AssistantFailed>());
      final chats = model.chats.length;
      model.script.add(const TextTurn(['Hello.']));

      final next = await repo.ask('Hi', historyText: 'Hi').toList();

      expect(model.chats.length, chats + 1, reason: 'a fresh chat');
      final reset = next.whereType<AssistantContextReset>().single;
      expect(reset.reason, ContextResetReason.interruptedSkill);
      expect(reset.keptTurns, 6);
      expect(reset.droppedTurns, 0, reason: 'every turn fit');
      final replay = model.lastSession.queries.first.text;
      for (var i = 1; i <= 6; i++) {
        expect(replay, contains('User: Q$i\n'), reason: 'Q$i kept');
      }
      expect(
        log.any((l) => l.contains('replayed the text history: 6 turns')),
        isTrue,
      );
      expect(log.any((l) => l.contains('older turns dropped')), isFalse);
    });
  });
}

/// The spoken question's prompt as the responder builds it without excerpts.
abstract final class PromptBuilderLike {
  static const spoken = 'Answer the question asked in this audio.';
}
