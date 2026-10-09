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

import '../models/voice.dart';
import '../ports/voice_diagnostics_sink.dart';

/// The silence gate's figures for one capture.
typedef MetricsGate = ({double peak, double threshold, Duration voiced});

/// One turn's figures, filled in as they become known: the
/// capture close, STT, the first text and audio, each clause's TTS, the
/// total and the outcome, plus the gate's figures of a voice turn.
final class TurnMetricsBuilder {
  TurnMetricsBuilder({required this.typed, this.captureClose});

  final bool typed;
  final Duration? captureClose;
  Duration? stt;
  Duration? firstText;
  Duration? firstAudio;
  int? sampleRate;
  final List<Duration> tts = [];
  Duration? total;
  TurnOutcome? outcome;
  MetricsGate? gate;

  VoiceTurnMetrics build() => VoiceTurnMetrics(
    typed: typed,
    captureClose: captureClose,
    stt: stt,
    firstText: firstText,
    firstAudio: firstAudio,
    sampleRate: sampleRate,
    ttsClauses: List.unmodifiable(tts),
    total: total,
    outcome: outcome,
    peakDbfs: gate?.peak,
    gateDbfs: gate?.threshold,
    voiced: gate?.voiced,
  );

  /// The turn's log line, after `[Voice] turn `.
  String summary() =>
      '${typed ? 'typed' : 'voice'} '
      'outcome=${outcome?.name} '
      'capture=${captureClose?.inMilliseconds}ms '
      'stt=${stt?.inMilliseconds}ms '
      'first_text=${firstText?.inMilliseconds}ms '
      'first_audio=${firstAudio?.inMilliseconds}ms '
      'tts=${tts.map((d) => d.inMilliseconds).toList()}ms '
      'total=${total?.inMilliseconds}ms';
}

/// Where the voice turn machine's figures go for the debug overlay: each
/// turn's record, and each barge-in's cost.
///
/// Newest turn wins: every turn (and every capture that made none) takes a
/// number from [nextSeq], and a record numbered below the one shown is
/// dropped, so a detached turn's late figures never replace the overlay's
/// current ones.
final class TurnMetricsPublisher {
  TurnMetricsPublisher(this._sink);

  final VoiceDiagnosticsSink _sink;
  int _seq = 0;
  int _publishedSeq = 0;

  /// The next turn's number; later numbers win.
  int nextSeq() => ++_seq;

  /// Records [metrics] of the turn numbered [seq], unless a newer turn's
  /// record is already shown.
  void publish(int seq, TurnMetricsBuilder metrics) {
    if (seq < _publishedSeq) return;
    _publishedSeq = seq;
    _sink.recordVoiceTurn(metrics.build());
  }

  /// A capture that made no turn (the mic was unavailable, or nothing was
  /// heard): a voice record with its [outcome] and the [gate] figures, if
  /// the audio was measured.
  void publishGate({required TurnOutcome outcome, MetricsGate? gate}) =>
      publish(
        nextSeq(),
        TurnMetricsBuilder(typed: false)
          ..outcome = outcome
          ..gate = gate,
      );
}
