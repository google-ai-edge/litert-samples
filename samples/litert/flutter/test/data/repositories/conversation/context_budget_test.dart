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
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/conversation/context_budget.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository.dart';

const _window = 4096;
const _limit = _window - kContextHeadroomTokens;
const _reply = 384;
const _prompt = 20;

/// The plain turn without an image: prompt + reply + template.
const _turn = _prompt + _reply + kTurnOverheadTokens;
const _reserve = (system: 400, rounds: 600);

ContextBudget _budget({
  int native = 0,
  int chat = 0,
  int prompt = _prompt,
  AgentReserve? agent,
  int audio = 0,
  bool rebuilding = false,
}) => ContextBudget(
  nativeTokens: native,
  chatTokens: chat,
  promptTokens: prompt,
  replyTokens: _reply,
  maxTokens: _window,
  agentReserve: agent,
  audioTokens: audio,
  rebuilding: rebuilding,
);

void main() {
  test('used is the larger of the two counts; the limit keeps the '
      'headroom free', () {
    expect(_budget(native: 900, chat: 700).used, 900);
    expect(_budget(native: 700, chat: 900).used, 900);
    expect(_budget().limit, _limit);
  });

  group('a plain chat', () {
    test('needs the prompt, the reply and the template; an image only when '
        'it is sent (now) or attached (fresh)', () {
      final budget = _budget(native: 1000);

      expect(budget.needNow(sendsImage: false), _turn);
      expect(budget.needNow(sendsImage: true), _turn + kImageTokenAllowance);
      expect(budget.needFresh(image: false), _turn);
      expect(budget.needFresh(image: true), _turn + kImageTokenAllowance);
      expect(budget.systemNow, 0);
    });

    test('fits while used + need stays within the limit', () {
      expect(
        _budget(native: _limit - _turn).check(image: false, sendsImage: false),
        isA<BudgetFits>().having((c) => c.needNow, 'needNow', _turn),
      );
      expect(
        _budget(native: _limit - _turn + 1)
            .check(image: false, sendsImage: false),
        isA<BudgetReset>().having((c) => c.needNow, 'needNow', _turn),
      );
    });

    test('an image already in context costs nothing now: the same chat fits '
        'without the allowance and resets with it', () {
      final budget = _budget(native: _limit - _turn - 10);

      expect(budget.check(image: true, sendsImage: false), isA<BudgetFits>());
      expect(budget.check(image: true, sendsImage: true), isA<BudgetReset>());
    });

    test('too long when even a fresh chat could not hold it, whatever the '
        'chat holds: the image counts when attached', () {
      final prompt = _limit - _reply - kTurnOverheadTokens;
      final budget = _budget(prompt: prompt);

      expect(budget.check(image: false, sendsImage: false), isA<BudgetFits>());
      expect(
        budget.check(image: true, sendsImage: false),
        isA<BudgetTooLong>().having(
          (c) => c.needFresh,
          'needFresh',
          _limit + kImageTokenAllowance,
        ),
      );
    });

    test('says why in the failure and the log line', () {
      final budget = _budget(native: 300, chat: 200);

      expect(
        '${budget.tooLong(5000)}',
        'The question needs about 5000 tokens with its reply, more than the '
            '$_limit the model can hold',
      );
      expect(budget.summary(450), 'budget used=300 need=450 limit=$_limit');
      expect(budget.tooLong(5000), isA<ConversationTooLongException>());
    });
  });

  group('an agent chat', () {
    test('adds the tool rounds always, and the system prompt only before the '
        'first prefill (LiteRT-LM count 0)', () {
      final empty = _budget(chat: 50, agent: _reserve);
      final started = _budget(native: 1, agent: _reserve);

      expect(empty.systemNow, _reserve.system);
      expect(empty.needNow(sendsImage: false), _turn + 600 + 400);
      expect(started.systemNow, 0);
      expect(started.needNow(sendsImage: false), _turn + 600);
      expect(
        started.needNow(sendsImage: true),
        _turn + 600 + kImageTokenAllowance,
      );
    });

    test('a fresh chat always needs the system prompt', () {
      final started = _budget(native: 2000, agent: _reserve);

      expect(started.needFresh(image: false), _turn + 600 + 400);
      expect(
        started.needFresh(image: true),
        _turn + 600 + 400 + kImageTokenAllowance,
      );
    });

    test('a chat that would fit a plain turn is reset for the tool '
        'rounds', () {
      final native = _limit - _turn - 100;

      expect(
        _budget(native: native).check(image: false, sendsImage: false),
        isA<BudgetFits>(),
      );
      expect(
        _budget(
          native: native,
          agent: _reserve,
        ).check(image: false, sendsImage: false),
        isA<BudgetReset>().having((c) => c.needNow, 'needNow', _turn + 600),
      );
    });

    test('too long counts the skills', () {
      final prompt = _limit - _reply - kTurnOverheadTokens - 1000 + 1;
      final budget = _budget(prompt: prompt, agent: _reserve);

      expect(
        budget.check(image: false, sendsImage: false),
        isA<BudgetTooLong>().having(
          (c) => c.needFresh,
          'needFresh',
          _limit + 1,
        ),
      );
    });

    test('says why in the failure and the log line', () {
      final empty = _budget(native: 0, chat: 120, agent: _reserve);
      final started = _budget(native: 300, agent: _reserve);

      expect(
        '${empty.tooLong(5000)}',
        'The question needs about 5000 tokens with its skills and reply, '
            'more than the $_limit the model can hold',
      );
      expect(
        empty.summary(1450),
        'agent budget used=120 need=1450 (rounds 600, system 400) '
        'limit=$_limit',
      );
      expect(
        started.summary(1050),
        'agent budget used=300 need=1050 (rounds 600, system 0) '
        'limit=$_limit',
      );
    });
  });

  group('a recording (an audio turn)', () {
    test('the budget\'s arithmetic: audioTokensFor is kAudioTokensPerSecond '
        '(25, the spec\'s estimate, not measured here) per second, rounded '
        'up', () {
      expect(kAudioTokensPerSecond, 25);
      expect(audioTokensFor(Duration.zero), 0);
      expect(audioTokensFor(const Duration(seconds: 2)), 50);
      expect(audioTokensFor(const Duration(milliseconds: 3200)), 80);
      expect(audioTokensFor(const Duration(milliseconds: 1001)), 26);
      expect(audioTokensFor(const Duration(seconds: 30)), 750);
    });

    test('counts now and in a fresh chat alike (it is sent with its turn '
        'only), on top of an image', () {
      final budget = _budget(native: 1000, audio: 80);

      expect(budget.needNow(sendsImage: false), _turn + 80);
      expect(budget.needFresh(image: false), _turn + 80);
      expect(
        budget.needNow(sendsImage: true),
        _turn + 80 + kImageTokenAllowance,
      );
    });

    test('a chat that fits the text turn is reset for its recording; too '
        'long when even a fresh chat cannot hold it', () {
      expect(
        _budget(native: _limit - _turn).check(image: false, sendsImage: false),
        isA<BudgetFits>(),
      );
      expect(
        _budget(
          native: _limit - _turn,
          audio: 50,
        ).check(image: false, sendsImage: false),
        isA<BudgetReset>().having((c) => c.needNow, 'needNow', _turn + 50),
      );
      expect(
        _budget(
          prompt: _limit - _reply - kTurnOverheadTokens,
          audio: 1,
        ).check(image: false, sendsImage: false),
        isA<BudgetTooLong>(),
      );
    });

    test('says so in the failure and the log line', () {
      final plain = _budget(native: 300, audio: 75);
      final agent = _budget(native: 300, agent: _reserve, audio: 75);

      expect(
        '${plain.tooLong(5000)}',
        'The question needs about 5000 tokens with its recording and reply, '
            'more than the $_limit the model can hold',
      );
      expect(
        '${agent.tooLong(5000)}',
        contains('with its recording, skills and reply'),
      );
      expect(
        plain.summary(500),
        'budget used=300 need=500 audio=75 limit=$_limit',
      );
      expect(
        agent.summary(1100),
        'agent budget used=300 need=1100 (rounds 600, system 0 audio=75) '
        'limit=$_limit',
      );
    });
  });

  group('a rebuild from the text history (a stop left an audio turn in the '
      'live chat)', () {
    test('what the live chat holds does not count: never a reset, and an '
        'agent chat\'s system prompt is prefilled again', () {
      final full = _budget(native: _limit, rebuilding: true);
      expect(full.used, 0);
      expect(full.check(image: false, sendsImage: false), isA<BudgetFits>());
      final agent = _budget(native: 2000, agent: _reserve, rebuilding: true);
      expect(agent.systemNow, _reserve.system);
      expect(
        _budget(native: 2000, agent: _reserve).systemNow,
        0,
        reason: 'a live agent chat has its system prompt already',
      );
    });

    test('a turn too long even for a fresh chat still fails', () {
      expect(
        _budget(
          prompt: _limit,
          rebuilding: true,
        ).check(image: false, sendsImage: false),
        isA<BudgetTooLong>(),
      );
    });
  });

  group('the replay room: what a fresh chat has left for the text history '
      'once it holds the turn, never negative', () {
    test('after a budget reset at most kHistoryReplayMaxFraction of the '
        'limit', () {
      final cap = (_limit * kHistoryReplayMaxFraction).floor();
      expect(_budget().replayRoom(image: false, afterReset: true), cap);
      expect(
        _budget(prompt: _limit - _reply - kTurnOverheadTokens - 100)
            .replayRoom(image: false, afterReset: true),
        100,
      );
      expect(
        _budget(prompt: _limit - _reply - kTurnOverheadTokens - 100)
            .replayRoom(image: true, afterReset: true),
        0,
        reason: 'the image leaves no room',
      );
      expect(kHistoryReplayMaxFraction, greaterThan(0));
      expect(kHistoryReplayMaxFraction, lessThan(0.5));
    });

    test('for any other rebuild (a stop with an audio turn in the chat, an '
        'interrupted skill call) all of it: the limit less the turn', () {
      final budget = _budget(native: 3000, rebuilding: true);
      final full = budget.replayRoom(image: false, afterReset: false);
      expect(full, budget.limit - budget.needFresh(image: false));
      expect(
        full,
        greaterThan(budget.replayRoom(image: false, afterReset: true)),
        reason: 'more than the reset\'s share',
      );
      expect(
        budget.replayRoom(image: true, afterReset: false),
        full - kImageTokenAllowance,
      );
      final agent = _budget(agent: _reserve, rebuilding: true);
      expect(
        agent.replayRoom(image: false, afterReset: false),
        agent.limit - agent.needFresh(image: false),
        reason: 'an agent chat\'s system prompt and tool rounds come first',
      );
      expect(
        _budget(prompt: _limit).replayRoom(image: false, afterReset: false),
        0,
      );
    });
  });
}
