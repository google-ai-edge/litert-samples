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

import 'package:flutter/foundation.dart';

import '../../data/repositories/speech_repository.dart';
import '../../utils/result.dart';
import '../models/model_id.dart';

/// What [TranscriptJob] reports for one audio turn.
sealed class const TranscriptOutcome(final int turn, final Duration elapsed);

/// The question's words, corrected like a transcript turn's.
final class const TranscriptReady(super.turn, super.elapsed, final String text)
    extends TranscriptOutcome;

/// No words: the recognizer failed or heard nothing; [reason] says which.
final class const TranscriptUnavailable(
  super.turn,
  super.elapsed,
  final String reason,
) extends TranscriptOutcome;

/// One pending transcription: the audio turn it is for, the model its words
/// must come from (null: whichever is loaded when it runs), and where its
/// outcome goes.
typedef _Pending = ({
  int turn,
  ModelId? model,
  void Function(TranscriptOutcome outcome) onDone,
});

/// Audio mode's transcripts of questions the chat model already answered from
/// their audio: Whisper transcribes each turn's recording in the background,
/// for the user's entry and the conversation's text history. Off the hot
/// path: the assistant starts a job only once its turn's reply has played and
/// the turn is idle.
///
/// Several jobs may be pending at once; they go to the recognizer one at a
/// time, in order, the next only once the previous finished (the native
/// worker would otherwise hold every recording at once, and a demo switch
/// would wait for all of them). Each outcome reaches the `onDone` its job
/// was started with, carrying
/// its turn's number, whatever happened since: a mic press, a new question, a
/// press that heard nothing or a mic that failed to start leave a pending job
/// alone. Cancelling would save no work — the native transcription cannot be
/// stopped, and Whisper base encodes its whole 30 s window for any clip — and
/// would only lose the question's words. A pending result is dropped only
/// when the demo is left ([dispose]: nothing more is submitted, so the next
/// demo waits for the one transcription already running at most) or the
/// speech recognizer is switched to another model than the job's
/// ([SpeechRepository.activeStt] becomes a different, non-null one): its
/// words would come from a model being closed, or not come at all.
final class TranscriptJob {
  TranscriptJob({required this._speech, this._recognizer, this._correct}) {
    _speech.activeStt.addListener(_onActiveStt);
  }

  final SpeechRepository _speech;

  /// The model the transcripts must come from (Demo 1: Whisper); null
  /// accepts the one active when a job starts.
  final ModelId? _recognizer;
  final String Function(String transcript)? _correct;

  /// The pending jobs by their serial number, oldest first.
  final Map<int, _Pending> _pending = {};

  /// The recordings of the pending jobs not yet submitted, oldest first.
  final Map<int, Uint8List> _waiting = {};

  /// A transcription is with the recognizer.
  bool _running = false;
  int _serial = 0;
  bool _disposed = false;

  /// The turns whose transcripts are pending, oldest first.
  List<int> get pendingTurns => [for (final job in _pending.values) job.turn];

  /// Transcribes [pcm] (16 kHz mono PCM16) for audio turn [turn], after the
  /// jobs ahead of it; its outcome goes to [onDone] (never after [dispose]).
  /// Jobs still pending go on.
  void start({
    required int turn,
    required Uint8List pcm,
    required void Function(TranscriptOutcome outcome) onDone,
  }) {
    if (_disposed) return;
    final serial = ++_serial;
    final ahead = _pending.length;
    _pending[serial] = (
      turn: turn,
      model: _recognizer ?? _speech.activeStt.value?.id,
      onDone: onDone,
    );
    debugPrint(
      '[Voice] transcript job for turn $turn started'
      '${ahead == 0 ? '' : ' ($ahead ahead of it)'}',
    );
    _waiting[serial] = pcm;
    _next();
  }

  /// Drops every pending result, submits nothing more and stops following
  /// recognizer switches. A transcription already with the recognizer runs
  /// to its end (it cannot be stopped); its result is dropped.
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _speech.activeStt.removeListener(_onActiveStt);
    _drop(_pending.keys.toList(), 'the demo was left');
  }

  /// Submits the oldest waiting job when none is with the recognizer.
  void _next() {
    if (_running || _disposed || _waiting.isEmpty) return;
    final MapEntry(key: serial, value: pcm) = _waiting.entries.first;
    _waiting.remove(serial);
    final turn = _pending[serial]?.turn;
    if (turn == null) return _next();
    _running = true;
    unawaited(
      _run(serial, turn, pcm).whenComplete(() {
        _running = false;
        _next();
      }),
    );
  }

  Future<void> _run(int serial, int turn, Uint8List pcm) async {
    final watch = Stopwatch()..start();
    final Result<String> result;
    try {
      result = await _speech.transcribe(pcm, expected: _recognizer);
    } catch (e, st) {
      // transcribe returns a Result; anything escaping it still ends this
      // job (with its reason) so the queue goes on.
      debugPrint('[Voice] transcript of turn $turn threw: $e\n$st');
      _pending
          .remove(serial)
          ?.onDone(TranscriptUnavailable(turn, watch.elapsed, '$e'));
      return;
    }
    watch.stop();
    final job = _pending.remove(serial);
    if (job == null) {
      debugPrint(
        '[Voice] transcript of turn $turn dropped after '
        '${watch.elapsedMilliseconds}ms (cancelled)',
      );
      return;
    }
    final TranscriptOutcome outcome;
    switch (result) {
      case Ok(:final value):
        final text = (_correct?.call(value) ?? value).trim();
        outcome = text.isEmpty
            ? TranscriptUnavailable(turn, watch.elapsed, 'no words recognized')
            : TranscriptReady(turn, watch.elapsed, text);
      case Error(:final error):
        outcome = TranscriptUnavailable(turn, watch.elapsed, '$error');
    }
    debugPrint(
      '[Voice] transcript of turn $turn ${watch.elapsedMilliseconds}ms: '
      '${switch (outcome) {
        TranscriptReady(:final text) => '"$text"',
        TranscriptUnavailable(:final reason) => 'none ($reason)',
      }}',
    );
    job.onDone(outcome);
  }

  /// A switch to another recognizer (another demo's model) drops the jobs
  /// that need a different one. No recognizer (a switch in progress, an
  /// unload) drops nothing: the transcription waits for the switch and checks
  /// the model when it runs (`transcribe(expected:)`), so the demo's own
  /// switch to its recognizer keeps its jobs.
  void _onActiveStt() {
    final active = _speech.activeStt.value;
    if (active == null) return;
    _drop([
      for (final MapEntry(key: serial, value: job) in _pending.entries)
        if (job.model case final model? when model != active.id) serial,
    ], 'the speech recognizer changed to ${active.modelId}');
  }

  void _drop(List<int> serials, String why) {
    for (final serial in serials) {
      _waiting.remove(serial);
      final job = _pending.remove(serial);
      if (job == null) continue;
      debugPrint(
        '[Voice] transcript job for turn ${job.turn} cancelled ($why)',
      );
    }
  }
}
