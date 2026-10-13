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

import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/chat_entry.dart';
import 'package:litert_edge_demos/domain/models/chat_side_event.dart';
import 'package:litert_edge_demos/domain/models/knowledge.dart';
import 'package:litert_edge_demos/domain/models/skill_step.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/ui/core/voice_notices.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/view_models/voice_chat_reducer.dart';

import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_knowledge.dart';

const _reducer = VoiceChatReducer();

typedef _Event = VoiceAssistantEvent<ChatSideEvent>;

Retrieval _used(int count) => Retrieval(
  outcome: RetrievalOutcome.used,
  passages: [for (var i = 1; i <= count; i++) passage(i)],
  gate: 0.4,
  latency: const Duration(milliseconds: 130),
);

const _loaded = SkillLoaded('current-time', found: true, at: Duration.zero);
const _called = IntentCalled(
  'current_time',
  '{}',
  at: Duration(milliseconds: 900),
);
const _succeeded = IntentSucceeded(
  'current_time',
  'It is 2:03 PM.',
  elapsed: Duration(milliseconds: 2),
  at: Duration(seconds: 1),
);

/// A turn in progress: its question, a retrieval and two steps, after an
/// earlier error.
VoiceChatState _midTurn() => const VoiceChatState(
  entries: [ChatEntry(role: ChatRole.user, text: 'What time is it?')],
  error: 'The reply failed: earlier',
  turnSteps: [_loaded, _called],
).copyWith(turnRetrieval: _used(3));

_Event _side(ChatSideEvent event, {bool detached = false}) =>
    SideEvent(event, detached: detached);

void main() {
  group('the user\'s words', () {
    test('start the turn: the last error goes, the steps start empty, the '
        'entry carries the picture', () {
      final image = Uint8List.fromList([1, 2, 3]);
      final before = _midTurn();

      final update = _reducer.reduce(
        before,
        UserSaid('And the date?', typed: false, image: image),
      );

      final state = update.state;
      expect(state.error, isNull);
      expect(state.turnSteps, isEmpty);
      expect(state.turnRetrieval, same(before.turnRetrieval));
      expect(state.entries, hasLength(2));
      expect(state.entries.last.role, ChatRole.user);
      expect(state.entries.last.text, 'And the date?');
      expect(state.entries.last.image, same(image));
      expect(update.notify, isTrue);
      expect(update.generation, isNull);
      expect(update.retrieval, isNull);
    });
  });

  group('a committed reply', () {
    test('takes the retrieval with the excerpts it cites, and the steps; '
        'the next reply starts without them', () {
      final before = _midTurn();

      final update = _reducer.reduce(
        before,
        const AssistantSaid('It is two [1], see also [3].', interrupted: false),
      );

      final reply = update.state.entries.last;
      expect(reply.role, ChatRole.assistant);
      expect(reply.text, 'It is two [1], see also [3].');
      expect(reply.interrupted, isFalse);
      expect(reply.knowledge!.retrieval, same(before.turnRetrieval));
      expect(reply.knowledge!.cited, {1, 3});
      expect(reply.steps, [_loaded, _called]);
      expect(update.state.turnRetrieval, isNull);
      expect(update.state.turnSteps, isEmpty);
      expect(update.state.error, before.error, reason: 'kept until a turn');
      expect(update.notify, isTrue);
    });

    test('without a retrieval has no knowledge', () {
      final update = _reducer.reduce(
        VoiceChatState.empty,
        const AssistantSaid('Hello.', interrupted: false),
      );

      expect(update.state.entries.single.knowledge, isNull);
      expect(update.state.entries.single.steps, isEmpty);
    });

    test('cut short before any text is still an entry, marked '
        'interrupted', () {
      final update = _reducer.reduce(
        VoiceChatState.empty,
        const AssistantSaid('', interrupted: true),
      );

      expect(update.state.entries.single.text, isEmpty);
      expect(update.state.entries.single.interrupted, isTrue);
      expect(update.notify, isTrue);
    });

    test('without text but with steps is an entry with them', () {
      final update = _reducer.reduce(
        const VoiceChatState(turnSteps: [_succeeded]),
        const AssistantSaid('', interrupted: false),
      );

      expect(update.state.entries.single.steps, [_succeeded]);
    });

    test('without text, steps or an interruption adds nothing, and still '
        'drops the retrieval', () {
      final before = VoiceChatState.empty.copyWith(turnRetrieval: _used(1));

      final update = _reducer.reduce(
        before,
        const AssistantSaid('', interrupted: false),
      );

      expect(update.state.entries, isEmpty);
      expect(update.state.turnRetrieval, isNull);
      expect(update.notify, isFalse);
    });
  });

  group('a capture with no LLM call', () {
    for (final reason in NotHeardReason.values) {
      test('${reason.name}: a notice; the retrieval goes, the steps and the '
          'error stay', () {
        final before = _midTurn();

        final update = _reducer.reduce(before, NotHeard(reason));

        final notice = update.state.entries.last;
        expect(notice.role, ChatRole.notice);
        expect(notice.text, notHeardText(reason));
        expect(update.state.turnRetrieval, isNull);
        expect(update.state.turnSteps, same(before.turnSteps));
        expect(update.state.error, before.error);
        expect(update.notify, isTrue);
      });
    }
  });

  test('a mic that gave no audio is the error, as an entry; the turn facts '
      'stay', () {
    final before = _midTurn();

    final update = _reducer.reduce(
      before,
      const MicUnavailable('Microphone access is off'),
    );

    expect(update.state.error, 'Microphone access is off');
    final entry = update.state.entries.last;
    expect(entry.role, ChatRole.error);
    expect(entry.text, 'Microphone access is off');
    expect(entry.steps, isEmpty);
    expect(update.state.turnRetrieval, same(before.turnRetrieval));
    expect(update.state.turnSteps, same(before.turnSteps));
    expect(update.notify, isTrue);
  });

  test('a failed turn is the error with the steps it got to; the turn '
      'facts go', () {
    final update = _reducer.reduce(
      _midTurn(),
      TurnFailed(Exception('GPU lost')),
    );

    expect(update.state.error, 'The reply failed: Exception: GPU lost');
    final entry = update.state.entries.last;
    expect(entry.role, ChatRole.error);
    expect(entry.text, 'The reply failed: Exception: GPU lost');
    expect(entry.steps, [_loaded, _called]);
    expect(update.state.turnRetrieval, isNull);
    expect(update.state.turnSteps, isEmpty);
    expect(update.notify, isTrue);
  });

  group('side events', () {
    for (final detached in [false, true]) {
      final which = detached ? 'a replaced turn' : 'the running turn';

      test("$which's figures go to the overlay; nothing on screen "
          'changes', () {
        final before = _midTurn();
        final metrics = FakeConversationRepository.metrics(stopped: detached);

        final update = _reducer.reduce(
          before,
          _side(ChatGenerationDone(metrics), detached: detached),
        );

        expect(update.state, same(before));
        expect(update.generation, same(metrics));
        expect(update.notify, isFalse);
      });

      for (final reason in ContextResetReason.values) {
        test("$which's context reset (${reason.name}) is a notice", () {
          final update = _reducer.reduce(
            _midTurn(),
            _side(ChatContextReset(reason: reason), detached: detached),
          );

          final notice = update.state.entries.last;
          expect(notice.role, ChatRole.notice);
          expect(notice.text, contextResetTextFor(reason));
          expect(update.notify, isTrue);
        });
      }
    }

    test("the running turn's retrieval waits for its reply; the overlay "
        'gets it now', () {
      final retrieval = _used(2);

      final update = _reducer.reduce(
        VoiceChatState.empty,
        _side(ChatRetrieval(retrieval)),
      );

      expect(update.state.turnRetrieval, same(retrieval));
      expect(update.retrieval, same(retrieval));
      expect(update.state.entries, isEmpty);
      expect(update.notify, isFalse);
    });

    test("a replaced turn's retrieval belongs to no entry; the overlay "
        'still gets it', () {
      final before = _midTurn();
      final late = _used(1);

      final update = _reducer.reduce(
        before,
        _side(ChatRetrieval(late), detached: true),
      );

      expect(update.state, same(before));
      expect(update.retrieval, same(late));
      expect(update.notify, isFalse);
    });

    test("the running turn's steps go to the bubble, one new list per "
        'step, without a rebuild', () {
      final first = _reducer.reduce(
        VoiceChatState.empty,
        _side(const ChatSkillStep(_loaded)),
      );
      final second = _reducer.reduce(
        first.state,
        _side(const ChatSkillStep(_called)),
      );

      expect(first.state.turnSteps, [_loaded]);
      expect(second.state.turnSteps, [_loaded, _called]);
      expect(first.state.turnSteps, hasLength(1), reason: 'not shared');
      expect(
        () => second.state.turnSteps.add(_succeeded),
        throwsUnsupportedError,
      );
      expect(first.notify, isFalse);
      expect(second.notify, isFalse);
      expect(second.state.entries, isEmpty);
    });

    test('an intent that ran after a barge-in is a notice with its '
        'result', () {
      final before = _midTurn();

      final update = _reducer.reduce(
        before,
        _side(const ChatSkillStep(_succeeded), detached: true),
      );

      final notice = update.state.entries.last;
      expect(notice.role, ChatRole.notice);
      expect(notice.text, '$skillAfterInterruptionText It is 2:03 PM.');
      expect(notice.steps, [_succeeded]);
      expect(update.state.turnSteps, same(before.turnSteps));
      expect(update.notify, isTrue);
    });

    for (final step in const <SkillStep>[
      _loaded,
      _called,
      IntentFailed('current_time', 'boom', at: Duration(seconds: 1)),
    ]) {
      test('any other step after a barge-in is dropped: $step', () {
        final before = _midTurn();

        final update = _reducer.reduce(
          before,
          _side(ChatSkillStep(step), detached: true),
        );

        expect(update.state, same(before));
        expect(update.notify, isFalse);
      });
    }
  });

  group('VoiceChatState', () {
    test('is immutable: a change makes a new state, the old one and its '
        'entries stay', () {
      final before = _midTurn();

      final after = before.adding(
        const ChatEntry(role: ChatRole.assistant, text: 'Two.'),
      );

      expect(before.entries, hasLength(1));
      expect(after.entries, hasLength(2));
      expect(
        () => after.entries.add(
          const ChatEntry(role: ChatRole.notice, text: 'x'),
        ),
        throwsUnsupportedError,
      );
    });

    test('copyWith keeps what it is not given and drops only what it is '
        'told to', () {
      final before = _midTurn();

      final kept = before.copyWith();
      final cleared = before.copyWith(
        clearError: true,
        clearTurnRetrieval: true,
      );

      expect(kept.entries, same(before.entries));
      expect(kept.error, before.error);
      expect(kept.turnRetrieval, same(before.turnRetrieval));
      expect(kept.turnSteps, same(before.turnSteps));
      expect(cleared.error, isNull);
      expect(cleared.turnRetrieval, isNull);
      expect(cleared.entries, same(before.entries));
      expect(cleared.turnSteps, same(before.turnSteps));
    });

    test('showingError sets the error and adds it as an entry', () {
      final state = VoiceChatState.empty.showingError(
        'Could not open the chat: no model',
      );

      expect(state.error, 'Could not open the chat: no model');
      expect(state.entries.single.role, ChatRole.error);
      expect(state.entries.single.text, 'Could not open the chat: no model');
    });

    test('empty has nothing on screen and no turn facts', () {
      const empty = VoiceChatState.empty;

      expect(empty.entries, isEmpty);
      expect(empty.error, isNull);
      expect(empty.turnRetrieval, isNull);
      expect(empty.turnSteps, isEmpty);
    });
  });

  test('each context reset reason has its own notice', () {
    expect(contextResetTextFor(ContextResetReason.budget), contextResetText);
    expect(
      contextResetTextFor(ContextResetReason.interruptedSkill),
      interruptedSkillResetText,
    );
    expect(contextResetText, isNot(interruptedSkillResetText));
  });
}
