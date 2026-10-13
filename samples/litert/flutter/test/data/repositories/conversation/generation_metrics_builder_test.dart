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

import 'package:flutter_edge_ai/flutter_edge_ai.dart' show SessionMetrics;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/conversation/generation_metrics_builder.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/skill_step.dart';

/// A stopwatch the test moves by hand ([advance]); [stop] freezes it.
final class _ManualStopwatch implements Stopwatch {
  Duration _elapsed = Duration.zero;
  bool _running = false;

  void advance(Duration by) {
    if (_running) _elapsed += by;
  }

  @override
  Duration get elapsed => _elapsed;
  @override
  int get elapsedMicroseconds => _elapsed.inMicroseconds;
  @override
  int get elapsedMilliseconds => _elapsed.inMilliseconds;
  @override
  int get elapsedTicks => _elapsed.inMicroseconds;
  @override
  int get frequency => 1000000;
  @override
  bool get isRunning => _running;
  @override
  void reset() => _elapsed = Duration.zero;
  @override
  void start() => _running = true;
  @override
  void stop() => _running = false;
}

const _ms = Duration(milliseconds: 1);

void main() {
  late _ManualStopwatch clock;
  late GenerationMetricsBuilder metrics;

  setUp(() {
    clock = _ManualStopwatch();
    metrics = GenerationMetricsBuilder(imageAttached: false, stopwatch: clock);
  });

  /// [chunks] text chunks, the first at [first], then one every [every].
  void stream(int chunks, {required Duration first, required Duration every}) {
    clock.advance(first);
    for (var i = 0; i < chunks; i++) {
      if (i > 0) clock.advance(every);
      metrics.chunk();
    }
  }

  GenerationMetrics finished({
    bool stopped = false,
    SessionMetrics? before,
    SessionMetrics? after,
    bool rebuildPending = false,
    int chatTokens = 0,
    int toolRounds = 0,
  }) => metrics.finished(
    stopped: stopped,
    stopLatency: null,
    imageSent: false,
    imageResent: null,
    before: before,
    after: after,
    rebuildPending: rebuildPending,
    chatTokens: chatTokens,
    promptTokens: 7,
    toolRounds: toolRounds,
  );

  test('the clock starts with the builder and stops with stop()', () {
    expect(clock.isRunning, isTrue);
    clock.advance(_ms * 5);
    expect(metrics.elapsed, _ms * 5);

    metrics.stop();
    clock.advance(_ms * 5);

    expect(metrics.elapsed, _ms * 5);
  });

  test('chunks: the first sets the time to first token; the rate is the '
      'chunks after the first over the time from the first to the last', () {
    stream(5, first: _ms * 200, every: _ms * 50);
    metrics.stop();

    final m = finished();

    expect(m.timeToFirstToken, _ms * 200);
    expect(m.chunks, 5);
    expect(m.tokensPerSecond, closeTo(20, 1e-9)); // 4 chunks in 200 ms
    expect(m.tokensPerSecondSource, TokenRateSource.chunks);
    expect(m.total, _ms * 400);
    expect(m.stopped, isFalse);
  });

  test('no chunk rate from fewer than two chunks or from chunks at the same '
      'instant', () {
    stream(1, first: _ms * 10, every: _ms);
    expect(finished().tokensPerSecond, isNull);

    final same =
        GenerationMetricsBuilder(
            imageAttached: false,
            stopwatch: _ManualStopwatch(),
          )
          ..chunk()
          ..chunk();
    final m = same.finished(
      stopped: false,
      stopLatency: null,
      imageSent: false,
      imageResent: null,
      before: null,
      after: null,
      rebuildPending: false,
      chatTokens: 0,
      promptTokens: 0,
    );
    expect(m.chunks, 2);
    expect(m.tokensPerSecond, isNull);
  });

  group('the native rate (LiteRT-LM reports the last decode turn)', () {
    final after = SessionMetrics(
      inputTokens: 120,
      totalTokens: 150,
      tokensPerSecond: 31.5,
    );

    test('is used for a turn that ran to the end with text', () {
      stream(3, first: _ms * 100, every: _ms * 10);

      final m = finished(after: after);

      expect(m.tokensPerSecond, 31.5);
      expect(m.tokensPerSecondSource, TokenRateSource.native);
    });

    test('is not used for a stopped turn: the chunk rate is', () {
      stream(3, first: _ms * 100, every: _ms * 10);

      final m = finished(stopped: true, after: after);

      expect(m.stopped, isTrue);
      expect(m.tokensPerSecond, closeTo(100, 1e-9));
      expect(m.tokensPerSecondSource, TokenRateSource.chunks);
    });

    test('is not used for a turn without text', () {
      final m = finished(after: after);

      expect(m.tokensPerSecond, isNull);
      expect(m.tokensPerSecondSource, TokenRateSource.chunks);
    });

    test('is not used when the metrics could not be read', () {
      stream(3, first: _ms * 100, every: _ms * 10);

      expect(finished().tokensPerSecondSource, TokenRateSource.chunks);
    });
  });

  group('prefill: the input-token delta', () {
    final before = SessionMetrics(inputTokens: 100, totalTokens: 140);
    final after = SessionMetrics(inputTokens: 380, totalTokens: 450);

    test('when both counts are known', () {
      expect(finished(before: before, after: after).prefillTokens, 280);
    });

    test('not after a stop (native rebuild pending), not with tool rounds, '
        'not when a count is missing, not when it is not positive', () {
      expect(
        finished(
          before: before,
          after: after,
          rebuildPending: true,
        ).prefillTokens,
        isNull,
      );
      expect(
        finished(before: before, after: after, toolRounds: 1).prefillTokens,
        isNull,
      );
      expect(finished(after: after).prefillTokens, isNull);
      expect(finished(before: before).prefillTokens, isNull);
      expect(finished(before: after, after: after).prefillTokens, isNull);
    });
  });

  test('context tokens: the larger of LiteRT-LM and flutter_edge_ai', () {
    final after = SessionMetrics(totalTokens: 450);

    expect(finished(after: after, chatTokens: 300).contextTokens, 450);
    expect(finished(after: after, chatTokens: 600).contextTokens, 600);
    expect(finished(chatTokens: 300).contextTokens, 300);
  });

  test('the turn facts go into the metrics as given', () {
    final steps = [const SkillLoaded('current-time', found: true, at: _ms)];
    final image = GenerationMetricsBuilder(
      imageAttached: true,
      stopwatch: _ManualStopwatch(),
    );

    final m = image.finished(
      stopped: true,
      stopLatency: _ms * 40,
      imageSent: true,
      imageResent: ImageLoss.stop,
      before: null,
      after: null,
      rebuildPending: false,
      chatTokens: 0,
      promptTokens: 12,
      toolRounds: 1,
      skillSteps: steps,
    );

    expect(m.imageAttached, isTrue);
    expect(m.imageSent, isTrue);
    expect(m.imageResent, ImageLoss.stop);
    expect(m.stopLatency, _ms * 40);
    expect(m.promptTokens, 12);
    expect(m.toolRounds, 1);
    expect(m.skillSteps, same(steps));
  });

  group('context reset', () {
    test('none by default: budget is the reason field\'s default', () {
      final m = finished();

      expect(m.contextReset, isFalse);
      expect(m.contextResetReason, ContextResetReason.budget);
    });

    test('a reset is reported with its reason', () {
      metrics.contextWasReset(ContextResetReason.interruptedSkill);

      final m = finished();

      expect(m.contextReset, isTrue);
      expect(m.contextResetReason, ContextResetReason.interruptedSkill);
    });

    test('the first reset of a turn keeps its reason', () {
      metrics
        ..contextWasReset(ContextResetReason.interruptedSkill)
        ..contextWasReset(ContextResetReason.budget);

      expect(
        finished().contextResetReason,
        ContextResetReason.interruptedSkill,
      );
    });
  });

  test('stopped before sending: no text, no rates, nothing sent, the reset '
      'and the prompt as given', () {
    final image = GenerationMetricsBuilder(
      imageAttached: true,
      stopwatch: clock,
    )..contextWasReset(ContextResetReason.budget);
    clock.advance(_ms * 30);
    image.stop();

    final m = image.stoppedBeforeSending(stopLatency: _ms * 3, promptTokens: 5);

    expect(m.stopped, isTrue);
    expect(m.timeToFirstToken, isNull);
    expect(m.chunks, 0);
    expect(m.tokensPerSecond, isNull);
    expect(m.tokensPerSecondSource, TokenRateSource.chunks);
    expect(m.total, _ms * 30);
    expect(m.stopLatency, _ms * 3);
    expect(m.imageAttached, isTrue);
    expect(m.imageSent, isFalse);
    expect(m.imageResent, isNull);
    expect(m.contextReset, isTrue);
    expect(m.contextResetReason, ContextResetReason.budget);
    expect(m.contextTokens, isNull);
    expect(m.prefillTokens, isNull);
    expect(m.promptTokens, 5);
    expect(m.toolRounds, 0);
    expect(m.skillSteps, isEmpty);
  });
}
