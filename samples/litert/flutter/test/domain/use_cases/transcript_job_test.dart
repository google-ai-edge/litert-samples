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
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/speech_repository.dart';
import 'package:litert_edge_demos/data/services/speech/stt_service.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/use_cases/transcript_job.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_speech.dart';

const _whisper = ActiveStt(
  id: ModelId.whisperBase,
  modelId: 'whisper_base_30s_i8',
  loadTime: Duration.zero,
);
const _moonshine = ActiveStt(
  id: ModelId.moonshineTiny,
  modelId: 'moonshine_tiny_5s_f32',
  loadTime: Duration.zero,
);

/// A speech repository whose transcriptions the test answers by hand, and
/// whose active recognizer it switches.
final class _Speech extends SpeechRepository {
  _Speech() : super(stt: fakeSttService(), tts: fakeTtsService());

  final ValueNotifier<ActiveStt?> active = ValueNotifier(_whisper);
  final List<(Uint8List, ModelId?)> calls = [];
  final List<Completer<Result<String>>> pending = [];

  @override
  ValueListenable<ActiveStt?> get activeStt => active;

  @override
  Future<Result<String>> transcribe(Uint8List pcm, {ModelId? expected}) {
    calls.add((pcm, expected));
    final answer = Completer<Result<String>>();
    pending.add(answer);
    return answer.future;
  }
}

void main() {
  late _Speech speech;
  late List<TranscriptOutcome> outcomes;
  late TranscriptJob job;
  final pcm = Uint8List.fromList([1, 2, 3, 4]);

  setUp(() {
    speech = _Speech();
    outcomes = [];
    job = TranscriptJob(
      speech: speech,
      recognizer: ModelId.whisperBase,
      correct: (t) => t.replaceAll('littered', 'LiteRT'),
    );
  });

  tearDown(() => job.dispose());

  void start(int turn) => job.start(turn: turn, pcm: pcm, onDone: outcomes.add);

  test('transcribes with the demo\'s recognizer and reports the corrected, '
      'trimmed words with the turn\'s number and the time', () async {
    start(4);
    expect(job.pendingTurns, [4]);
    expect(speech.calls.single, (pcm, ModelId.whisperBase));
    speech.pending.single.complete(const Result.ok(' What is littered? '));
    await pumpEventQueue();

    final ready = outcomes.single as TranscriptReady;
    expect(ready.turn, 4);
    expect(ready.text, 'What is LiteRT?');
    expect(ready.elapsed.isNegative, isFalse);
    expect(job.pendingTurns, isEmpty);
  });

  test('no words, or a failing recognizer: unavailable, with why', () async {
    start(1);
    speech.pending.last.complete(const Result.ok('   '));
    await pumpEventQueue();
    start(2);
    speech.pending.last.complete(
      const Result.error(SpeechNotReadyException('speech recognizer')),
    );
    await pumpEventQueue();

    expect(outcomes, hasLength(2));
    expect(
      (outcomes[0] as TranscriptUnavailable).reason,
      'no words recognized',
    );
    expect(
      (outcomes[1] as TranscriptUnavailable).reason,
      'The speech recognizer is not loaded',
    );
    expect(outcomes[1].turn, 2);
  });

  test('a newer job does not cancel an older one: both are pending, and each '
      'outcome reaches its own callback with its turn\'s number', () async {
    final first = <TranscriptOutcome>[];
    final second = <TranscriptOutcome>[];
    job
      ..start(turn: 1, pcm: pcm, onDone: first.add)
      ..start(turn: 2, pcm: pcm, onDone: second.add);
    expect(job.pendingTurns, [1, 2]);
    expect(speech.calls, hasLength(1), reason: 'one at a time');

    speech.pending[0].complete(const Result.ok('first'));
    await pumpEventQueue();
    expect(speech.calls, hasLength(2), reason: 'the next once it finished');
    speech.pending[1].complete(const Result.ok('second'));
    await pumpEventQueue();

    expect(first.map((o) => (o.turn, (o as TranscriptReady).text)), [
      (1, 'first'),
    ]);
    expect(second.map((o) => (o.turn, (o as TranscriptReady).text)), [
      (2, 'second'),
    ]);
    expect(job.pendingTurns, isEmpty);
  });

  test('the demo\'s own switch to its recognizer (none -> Whisper) keeps a '
      'pending job: its words arrive', () async {
    speech.active.value = null;
    start(1);
    speech.active.value = _whisper;
    expect(job.pendingTurns, [1]);
    speech.pending.single.complete(const Result.ok('words'));
    await pumpEventQueue();
    expect((outcomes.single as TranscriptReady).text, 'words');
  });

  test('a switch to another recognizer (Whisper -> moonshine, another '
      'demo\'s) drops the pending results; a change that keeps the model, or '
      'no recognizer for a while (a switch in progress), does not', () async {
    start(1);
    start(2);
    speech.active.value = const ActiveStt(
      id: ModelId.whisperBase,
      modelId: 'whisper_base_30s_i8',
      loadTime: Duration.zero,
      warmUpTime: Duration(milliseconds: 5),
    );
    expect(job.pendingTurns, [1, 2], reason: 'the same model');
    speech.active.value = null;
    expect(job.pendingTurns, [1, 2], reason: 'no recognizer yet');
    speech.active.value = _moonshine;
    expect(job.pendingTurns, isEmpty);
    for (final answer in speech.pending) {
      answer.complete(const Result.ok('words'));
    }
    await pumpEventQueue();
    expect(outcomes, isEmpty);
  });

  test('without a recognizer of its own a job belongs to the one active at '
      'its start; one started with none active survives any switch', () async {
    final any = TranscriptJob(speech: speech);
    addTearDown(any.dispose);
    any.start(turn: 1, pcm: pcm, onDone: outcomes.add);
    speech.active.value = null;
    any.start(turn: 2, pcm: pcm, onDone: outcomes.add);
    speech.active.value = _moonshine;
    expect(any.pendingTurns, [2]);
    speech.pending.single.complete(const Result.ok('words'));
    await pumpEventQueue();
    expect(speech.calls.map((c) => c.$2), [null, null]);
    speech.pending.last.complete(const Result.ok('words'));
    await pumpEventQueue();
    expect(outcomes.map((o) => o.turn), [2]);
  });

  test('dispose drops every pending result and stops following switches; '
      'later starts do nothing', () async {
    start(1);
    start(2);
    job.dispose();
    expect(job.pendingTurns, isEmpty);
    for (final answer in speech.pending) {
      answer.complete(const Result.ok('words'));
    }
    await pumpEventQueue();
    expect(outcomes, isEmpty);
    start(3);
    expect(speech.calls, hasLength(1), reason: 'job 2 was never submitted');
    speech.active.value = _moonshine; // no listener left to throw
  });

  test('the jobs go to the recognizer one at a time; leaving the demo '
      '(dispose) after the first started submits nothing more, so the next '
      'demo waits for one transcription at most', () async {
    final recognizer = FakeRecognizer('words')..gate = Completer<void>();
    final real = await loadedSpeech(
      recognizer: recognizer,
      synthesizer: RecordingSynth(),
    );
    final jobs = TranscriptJob(speech: real, recognizer: ModelId.whisperBase);
    for (final turn in [1, 2, 3]) {
      jobs.start(turn: turn, pcm: pcm, onDone: outcomes.add);
    }
    await pumpEventQueue();
    expect(recognizer.calls, 1, reason: 'the second waits for the first');
    expect(jobs.pendingTurns, [1, 2, 3]);

    jobs.dispose();
    recognizer.gate!.complete();
    await pumpEventQueue();

    expect(recognizer.calls, 1);
    expect(outcomes, isEmpty, reason: 'its result is dropped too');
  });
}
