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
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/domain/ports/voice_diagnostics_sink.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_turn_metrics.dart';

/// Records what the publisher hands the overlay.
final class _Sink implements VoiceDiagnosticsSink {
  final List<TurnPhase> phases = [];
  final List<VoiceTurnMetrics> turns = [];
  final List<BargeInMetrics> bargeIns = [];

  @override
  void recordVoicePhase(TurnPhase phase) => phases.add(phase);

  @override
  void recordVoiceTurn(VoiceTurnMetrics metrics) => turns.add(metrics);

  @override
  void recordBargeIn(BargeInMetrics metrics) => bargeIns.add(metrics);
}

void main() {
  group('TurnMetricsBuilder', () {
    test('builds every figure, the gate as peak, threshold and voiced', () {
      final builder =
          TurnMetricsBuilder(
              typed: false,
              captureClose: const Duration(milliseconds: 40),
            )
            ..stt = const Duration(milliseconds: 300)
            ..firstText = const Duration(milliseconds: 500)
            ..firstAudio = const Duration(milliseconds: 700)
            ..sampleRate = 24000
            ..total = const Duration(seconds: 2)
            ..outcome = TurnOutcome.completed
            ..gate = (
              peak: -13.5,
              threshold: -30,
              voiced: const Duration(milliseconds: 900),
            );
      builder.tts.addAll(const [
        Duration(milliseconds: 80),
        Duration(milliseconds: 90),
      ]);

      final m = builder.build();

      expect(m.typed, isFalse);
      expect(m.captureClose, const Duration(milliseconds: 40));
      expect(m.stt, const Duration(milliseconds: 300));
      expect(m.firstText, const Duration(milliseconds: 500));
      expect(m.firstAudio, const Duration(milliseconds: 700));
      expect(m.sampleRate, 24000);
      expect(m.ttsClauses, const [
        Duration(milliseconds: 80),
        Duration(milliseconds: 90),
      ]);
      expect(m.total, const Duration(seconds: 2));
      expect(m.outcome, TurnOutcome.completed);
      expect(m.peakDbfs, -13.5);
      expect(m.gateDbfs, -30);
      expect(m.voiced, const Duration(milliseconds: 900));
    });

    test('a record is a snapshot: later clauses do not change it, and its '
        'clause list cannot be modified', () {
      final builder = TurnMetricsBuilder(typed: true);
      builder.tts.add(const Duration(milliseconds: 80));
      final first = builder.build();

      builder.tts.add(const Duration(milliseconds: 90));

      expect(first.ttsClauses, hasLength(1));
      expect(builder.build().ttsClauses, hasLength(2));
      expect(() => first.ttsClauses.add(Duration.zero), throwsUnsupportedError);
    });

    test('without a gate the gate figures are null', () {
      final m = TurnMetricsBuilder(typed: true).build();

      expect(m.typed, isTrue);
      expect((m.peakDbfs, m.gateDbfs, m.voiced), (null, null, null));
      expect(m.ttsClauses, isEmpty);
    });

    test("the log line names the kind and every figure in ms, 'null' for "
        'the unknown ones', () {
      final voice =
          TurnMetricsBuilder(
              typed: false,
              captureClose: const Duration(milliseconds: 12),
            )
            ..stt = const Duration(milliseconds: 340)
            ..firstText = const Duration(milliseconds: 900)
            ..firstAudio = const Duration(milliseconds: 1200)
            ..total = const Duration(milliseconds: 3000)
            ..outcome = TurnOutcome.completed;
      voice.tts.addAll(const [
        Duration(milliseconds: 80),
        Duration(milliseconds: 95),
      ]);

      expect(
        voice.summary(),
        'voice outcome=completed capture=12ms stt=340ms first_text=900ms '
        'first_audio=1200ms tts=[80, 95]ms total=3000ms',
      );
      expect(
        TurnMetricsBuilder(typed: true).summary(),
        'typed outcome=null capture=nullms stt=nullms first_text=nullms '
        'first_audio=nullms tts=[]ms total=nullms',
      );
    });
  });

  group('TurnMetricsPublisher', () {
    late _Sink sink;
    late TurnMetricsPublisher publisher;

    setUp(() {
      sink = _Sink();
      publisher = TurnMetricsPublisher(sink);
    });

    test('numbers turns in order', () {
      expect(publisher.nextSeq(), 1);
      expect(publisher.nextSeq(), 2);
    });

    test('the newest turn wins: an older turn\'s late figures are dropped, '
        'the shown turn may update its own', () {
      final older = publisher.nextSeq();
      final newer = publisher.nextSeq();
      final olderMetrics = TurnMetricsBuilder(typed: true)
        ..outcome = TurnOutcome.superseded;
      final newerMetrics = TurnMetricsBuilder(typed: false);

      publisher.publish(newer, newerMetrics);
      publisher.publish(older, olderMetrics);
      newerMetrics.outcome = TurnOutcome.completed;
      publisher.publish(newer, newerMetrics);

      expect(sink.turns.map((m) => (m.typed, m.outcome)), [
        (false, null),
        (false, TurnOutcome.completed),
      ]);
    });

    test('an older turn still updates its record until a newer one is '
        'shown', () {
      final older = publisher.nextSeq();
      publisher.nextSeq();

      publisher.publish(older, TurnMetricsBuilder(typed: true));

      expect(sink.turns, hasLength(1));
    });

    test('a capture without a turn is a voice record with its outcome and '
        'gate, numbered after every turn so far', () {
      final running = publisher.nextSeq();

      publisher.publishGate(
        outcome: TurnOutcome.notHeard,
        gate: (peak: -80, threshold: -45, voiced: Duration.zero),
      );
      publisher.publish(running, TurnMetricsBuilder(typed: true));

      final record = sink.turns.single;
      expect(record.typed, isFalse);
      expect(record.outcome, TurnOutcome.notHeard);
      expect(record.peakDbfs, -80);
      expect(record.gateDbfs, -45);
      expect(record.voiced, Duration.zero);
      expect(publisher.nextSeq(), 3);
    });

    test('a capture whose audio was never measured has no gate '
        'figures', () {
      publisher.publishGate(outcome: TurnOutcome.micUnavailable);

      final record = sink.turns.single;
      expect(record.outcome, TurnOutcome.micUnavailable);
      expect(record.peakDbfs, isNull);
    });
  });
}
