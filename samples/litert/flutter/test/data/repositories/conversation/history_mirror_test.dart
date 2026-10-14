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

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/conversation/history_mirror.dart';
import 'package:litert_edge_demos/domain/models/conversation_history.dart';

/// Tokens as the fake session counts them: one per 4 characters.
Future<int> _tokens(String text) async => (text.length / 4).ceil();

void main() {
  late HistoryMirror mirror;

  setUp(() => mirror = HistoryMirror());

  test('a turn: begin with the user\'s words, reply as far as it got', () {
    final slot = mirror.begin(user: 'What is LiteRT?', turn: 1);
    expect(mirror.turns.single.assistant, '');
    expect(mirror.turns.single.replayable, isFalse, reason: 'nothing answered');
    mirror.reply(slot, 'A runtime.');
    expect(mirror.turns.single, isA<HistoryTurn>());
    expect(mirror.turns.single.user, 'What is LiteRT?');
    expect(mirror.turns.single.assistant, 'A runtime.');
    expect(mirror.turns.single.turn, 1);
    expect(mirror.turns.single.replayable, isTrue);
    expect(mirror.audioInLive, isFalse, reason: 'a text turn');
  });

  test('an audio turn: the placeholder until its transcript arrives; the live '
      'conversation now holds audio', () {
    final slot = mirror.begin(
      user: kSpokenQuestionPlaceholder,
      turn: 7,
      audio: true,
    );
    mirror.reply(slot, 'Paris.');
    expect(mirror.audioInLive, isTrue);
    expect(mirror.turns.single.audio, isTrue);
    expect(mirror.setUser(7, 'What is the capital of France?'), isTrue);
    expect(mirror.turns.single.user, 'What is the capital of France?');
    expect(mirror.turns.single.assistant, 'Paris.');
    expect(mirror.turns.single.audio, isTrue);
    expect(mirror.setUser(8, 'unknown'), isFalse, reason: 'no such turn');
  });

  test(
    'a spoken question replayed into the live chat before its words '
    'arrived: its transcript makes the replay outdated (the live chat holds '
    'only the placeholder); one still in the live chat as audio does not',
    () {
      final asked = mirror.begin(
        user: kSpokenQuestionPlaceholder,
        turn: 1,
        audio: true,
      );
      mirror.reply(asked, 'Paris.');
      expect(mirror.turns.single.wordless, isTrue);
      // Still audio in the live chat: the model heard it, nothing to redo.
      expect(mirror.setUser(1, 'What is the capital of France?'), isTrue);
      expect(mirror.replayOutdated, isFalse);
      expect(mirror.turns.single.wordless, isFalse);

      final second = mirror.begin(
        user: kSpokenQuestionPlaceholder,
        turn: 2,
        audio: true,
      );
      mirror.reply(second, 'Berlin.');
      final typed = mirror.begin(user: 'And Spain?', turn: 3);
      mirror.reply(typed, 'Madrid.');
      mirror.replayed(mirror.turns);
      expect(mirror.audioInLive, isFalse, reason: 'replayed as text');
      expect(mirror.wordlessInLive, [2]);
      expect(mirror.setUser(3, 'And Spain?'), isTrue);
      expect(mirror.replayOutdated, isFalse, reason: 'not a placeholder');
      expect(mirror.setUser(2, 'And of Germany?'), isTrue);
      expect(mirror.replayOutdated, isTrue);
      expect(mirror.wordlessInLive, isEmpty);

      // The rebuild that follows replays the words: up to date again.
      mirror.replayed(mirror.turns);
      expect(mirror.replayOutdated, isFalse);
      // A fresh chat, or a cleared mirror, holds no replay.
      mirror.replayed([mirror.turns[1]]);
      mirror.forgetLive();
      expect(mirror.wordlessInLive, isEmpty);
      expect(mirror.replayOutdated, isFalse);
    },
  );

  test('a transcript that lands between the replay\'s plan and its end (the '
      'plan replays the placeholder): the replay is outdated at once', () {
    final asked = mirror.begin(
      user: kSpokenQuestionPlaceholder,
      turn: 1,
      audio: true,
    );
    mirror.reply(asked, 'Paris.');
    final plan = mirror.turns; // what the turn planned to replay
    expect(mirror.setUser(1, 'What is the capital of France?'), isTrue);
    mirror.replayed(plan);
    expect(
      mirror.replayOutdated,
      isTrue,
      reason: 'the chat got the placeholder',
    );
    expect(mirror.wordlessInLive, isEmpty);
  });

  test('clear: nothing left, no audio in the live chat, and a turn that began '
      'before it cannot write into the new history', () {
    final old = mirror.begin(user: 'Hi', turn: 1, audio: true);
    mirror.clear();
    expect(mirror.isEmpty, isTrue);
    expect(mirror.audioInLive, isFalse);
    mirror.begin(user: 'Hello', turn: 2);
    mirror.reply(old, 'late reply of the old chat');
    expect(mirror.turns.single.user, 'Hello');
    expect(mirror.turns.single.assistant, '');
  });

  test('the replay: one message with a header, one line per side, newlines '
      'folded, and a blank line before the turn\'s own text', () {
    final text = HistoryMirror.replayText(const [
      HistoryTurn(user: 'What is LiteRT?', assistant: 'A runtime.\nOn device.'),
      HistoryTurn(user: kSpokenQuestionPlaceholder, assistant: 'Paris.'),
    ]);
    expect(
      text,
      '${HistoryMirror.replayHeader}'
      'User: What is LiteRT?\n'
      'Assistant: A runtime. On device.\n'
      'User: $kSpokenQuestionPlaceholder\n'
      'Assistant: Paris.\n'
      '\n',
    );
  });

  group('newestThatFit', () {
    void answered(String user, String reply, {int? turn}) {
      final slot = mirror.begin(user: user, turn: turn);
      mirror.reply(slot, reply);
    }

    test('the newest turns that fit, oldest first; the older ones counted as '
        'dropped; unanswered turns are never replayed', () async {
      answered('one', 'a' * 40);
      answered('two', 'b' * 40);
      mirror.begin(user: 'stopped before any text');
      answered('three', 'c' * 40);

      final all = await mirror.newestThatFit(1000, _tokens);
      expect(all.turns.map((t) => t.user), ['one', 'two', 'three']);
      expect(all.dropped, 0);
      // The header (34 characters: 9 tokens) and each turn (62: 16).
      expect(all.tokens, 9 + 3 * 16);

      // Header (9) + one turn (~16 each).
      final some = await mirror.newestThatFit(9 + 17 + 17, _tokens);
      expect(some.turns.map((t) => t.user), ['two', 'three']);
      expect(some.dropped, 1);
    });

    test(
      'nothing fits, nothing to replay, or no room: an empty plan',
      () async {
        expect((await mirror.newestThatFit(1000, _tokens)).turns, isEmpty);
        answered('one', 'a' * 400);
        final none = await mirror.newestThatFit(20, _tokens);
        expect(none.turns, isEmpty);
        expect(none.tokens, 0);
        expect(none.dropped, 1);
        expect((await mirror.newestThatFit(0, _tokens)).dropped, 1);
      },
    );
  });
}
