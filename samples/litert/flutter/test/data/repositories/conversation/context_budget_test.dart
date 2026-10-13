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
}) => ContextBudget(
  nativeTokens: native,
  chatTokens: chat,
  promptTokens: prompt,
  replyTokens: _reply,
  maxTokens: _window,
  agentReserve: agent,
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
}
