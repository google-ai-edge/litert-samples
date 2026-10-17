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
import 'package:flutter_edge_ai_speech/flutter_edge_ai_speech.dart';

import '../../config/voice_config.dart';
import '../../data/repositories/audio_repository.dart';
import '../../data/repositories/speech_repository.dart';
import '../../utils/pcm.dart';
import '../../utils/result.dart';
import '../models/audio_interruption.dart';
import '../models/model_id.dart';
import '../models/voice.dart';
import '../ports/voice_diagnostics_sink.dart';
import 'push_to_talk_capture.dart';
import 'transcript_job.dart';
import 'turn_responder_factory.dart';
import 'voice_turn_metrics.dart';

/// What the assistant tells its view model, at turn rate (never per token or
/// per level tick; those go through [VoiceAssistant.partialReply] and
/// [VoiceAssistant.inputLevel]).
sealed class const VoiceAssistantEvent<S>();

/// The user's words, once they are known (the transcript, or the typed
/// text). [image]: the picture attached to this turn (Demo 1), for the
/// user's entry.
final class const UserSaid<S>(
  final String text, {
  required final bool typed,
  final Uint8List? image,
}) extends VoiceAssistantEvent<S>;

/// A spoken question sent to the chat model as audio (audio mode), at the
/// start of its turn: no transcript yet. [turn] is the number its transcript
/// arrives with ([TranscriptArrived], [TranscriptFailed]); [duration] the
/// recording's length; [image] as in [UserSaid].
final class const UserSpoke<S>(
  final int turn,
  final Duration duration, {
  final Uint8List? image,
}) extends VoiceAssistantEvent<S>;

/// The words of audio turn [turn], transcribed after its reply played.
final class const TranscriptArrived<S>(final int turn, final String text)
    extends VoiceAssistantEvent<S>;

/// Audio turn [turn] got no transcript: [reason] (the recognizer failed or
/// heard no words), or [interrupted]: a barge-in replaced the turn and its
/// recording went with it ([kInterruptedTranscriptReason]). A dropped
/// transcript (the demo was left, or the speech recognizer switched to
/// another model) reports nothing.
final class const TranscriptFailed<S>(
  final int turn,
  final String reason, {
  final bool interrupted = false,
}) extends VoiceAssistantEvent<S>;

/// [TranscriptFailed.reason] of a barged-in audio question.
const kInterruptedTranscriptReason = 'interrupted';

/// The assistant's reply, committed. [interrupted]: cut short by Stop, a
/// barge-in, an interruption from outside or a failure (the text is what was
/// shown until then; may be empty, except after a failure, which commits
/// only a reply that had text).
final class const AssistantSaid<S>(
  final String text, {
  required final bool interrupted,
}) extends VoiceAssistantEvent<S>;

/// A capture that made no LLM call: "Didn't catch that".
final class const NotHeard<S>(final NotHeardReason reason)
    extends VoiceAssistantEvent<S>;

/// The mic gave no usable audio; [message] says what to do.
final class const MicUnavailable<S>(final String message)
    extends VoiceAssistantEvent<S>;

/// The turn failed (STT, the model, TTS or playback).
final class const TurnFailed<S>(final Object error)
    extends VoiceAssistantEvent<S>;

/// Something outside the app ended a press or a turn ([cause]): the OS or
/// another app took the audio, the output went away, or the app left the
/// foreground on iOS. A turn's reply, if any, was committed (as interrupted
/// when it was cut short) just before.
final class const TurnInterrupted<S>(final InterruptionCause cause)
    extends VoiceAssistantEvent<S>;

/// A responder's side event. [detached]: it belongs to a turn a barge-in
/// already replaced (Demo 1 still records its metrics; Demo 3 must not
/// freeze on its snapshot).
final class const SideEvent<S>(final S event, {required final bool detached})
    extends VoiceAssistantEvent<S>;

/// The push-to-talk turn state machine shared by both demos: capture → gate
/// → `VoiceSession.custom` → streamed reply with clause-by-clause audio,
/// barge-in and Stop. Generic over the demo's
/// side events [S]; the demo plugs in its [TurnResponderFactory].
///
/// Two kinds of spoken turn, chosen per turn by the responder: a transcript
/// turn (STT, then the reply to the words: [UserSaid]) and, when the
/// responder takes the recording itself (`respondToAudio`: a chat model
/// that hears audio), an audio turn (`VoiceSession.runAudioTurn`, no STT:
/// [UserSpoke] at once, the reply, then — once its audio has drained and the
/// turn is idle — the transcript job, whose words arrive as
/// [TranscriptArrived] with the turn's number, also after newer turns began:
/// they never drop it). A press on the reply's playback tail holds the job
/// until no generation runs: the next turn's reply has been generated (or
/// cut by another press), or nothing runs at all. Typed turns are never audio
/// turns.
///
/// One action at a time wins: every user action (mic down, a typed send,
/// Stop, dispose) supersedes whatever an earlier one still awaits, and a
/// barge-in detaches the running turn at once — its audio stops, its partial
/// reply is committed as interrupted — while the session drains it in the
/// background. The next turn starts only after that drain.
///
/// Interruptions from outside ([interrupt]: the audio repository's, and the
/// app leaving the foreground on iOS) end a press or a turn like Stop and
/// say why ([TurnInterrupted]). The audio is taken per turn: a press or a
/// spoken reply activates it, and the end of every press or turn hands it
/// back ([AudioRepository.deactivate]; Android's other apps resume).
class VoiceAssistant<S> {
  /// [correctTranscript]: applied to every transcript (Demo 1's Whisper
  /// corrections, `correctVoiceTranscript`), the transcript job's included.
  /// [onTranscript]: an audio turn's words, before [TranscriptArrived] (Demo
  /// 1 puts them into the conversation's text history).
  VoiceAssistant({
    required this._speech,
    required this._audio,
    required this._responders,
    required this._diagnostics,
    VoiceConfig? config,
    this._recognizer,
    String Function(String transcript)? correctTranscript,
    void Function(int turn, String text)? onTranscript,
  }) : _config = config ?? kVoiceConfig,
       _correct = correctTranscript,
       _onTranscriptText = onTranscript {
    _audioInterruptions = _audio.interruptions.listen(_onAudioInterruption);
  }

  final SpeechRepository _speech;
  final AudioRepository _audio;

  /// The audio repository's interruptions, until [dispose].
  late final StreamSubscription<AudioInterruption> _audioInterruptions;
  final TurnResponderFactory<S> _responders;

  /// Where the debug overlay's voice phase and figures go.
  final VoiceDiagnosticsSink _diagnostics;
  late final TurnMetricsPublisher _metrics = TurnMetricsPublisher(_diagnostics);
  final VoiceConfig _config;
  late final PushToTalkCapture _mic = PushToTalkCapture(
    audio: _audio,
    maxLength: _config.maxUtterance,
  );

  /// The demo's speech recognizer: a voice turn fails fast when another one
  /// is loaded (a failed switch) instead of being transcribed by it. Null
  /// accepts whichever is active (tests of the turn machine).
  final ModelId? _recognizer;
  final String Function(String transcript)? _correct;
  final void Function(int turn, String text)? _onTranscriptText;

  /// Audio turns' transcripts, after their replies.
  late final TranscriptJob _transcripts = TranscriptJob(
    speech: _speech,
    recognizer: _recognizer,
    correct: _correct,
  );

  /// Answered audio turns whose recordings wait before they go to
  /// [_transcripts]: a press on a reply's playback tail opens the next
  /// capture at once, and Whisper's 30 s window must not compete with the
  /// turn it asks (its audio query, its generation, its speech). Released
  /// ([_releaseHeldTranscripts]) as soon as no generation runs: that turn's
  /// reply has been generated in full (only its playback is left), a press
  /// cut its generation (the chat it leaves is rebuilt from the text history
  /// for the next turn, which must find these words there rather than the
  /// placeholder), or nothing runs at all ([_setPhase]). Oldest first.
  final List<(_Turn, Uint8List)> _heldTranscripts = [];

  /// Numbers every turn at release ([TurnRequest.turn]), across every
  /// assistant of the process: a number is never handed out twice. The
  /// conversation's text mirror finds a transcript's entry by it, and the
  /// mirror belongs to the shared chat, which can outlive this assistant (a
  /// demo entered again while the previous reply still stops keeps the
  /// previous visit's chat, its history included): a per-assistant count
  /// would start at 1 again and name that visit's turns.
  static int _turnNumbers = 0;

  final ValueNotifier<TurnPhase> _phase = ValueNotifier(TurnPhase.idle);
  final ValueNotifier<String> _partial = ValueNotifier('');
  // Sync so a commit reaches the view model before the phase change that
  // follows it.
  final StreamController<VoiceAssistantEvent<S>> _events =
      StreamController.broadcast(sync: true);
  bool _speak = true;
  bool _disposed = false;

  /// Bumped by every user action; an action that resumes after an await
  /// with a stale epoch has been superseded and stops there. A press in
  /// progress belongs to the current epoch: every newer action drops it.
  int _epoch = 0;
  _Turn? _turn;

  /// Barged-in turns still draining; the next turn waits for them.
  Future<void>? _pendingInterrupt;

  ValueListenable<TurnPhase> get phase => _phase;

  /// The reply as it streams; empty between turns. Only the streaming bubble
  /// listens to it.
  ValueListenable<String> get partialReply => _partial;

  /// The mic level while listening (0–1), for the meter.
  ValueListenable<double> get inputLevel => _audio.inputLevel;

  Stream<VoiceAssistantEvent<S>> get events => _events.stream;

  /// A turn or a capture is in flight.
  bool get isBusy => _phase.value.isActive;

  /// Whether replies are spoken. Off: the next turns use a silent
  /// synthesizer; a reply playing now is silenced.
  bool get speakReplies => _speak;

  set speakReplies(bool value) {
    if (_speak == value) return;
    _speak = value;
    final turn = _turn;
    if (value || turn == null || turn.muted) return;
    turn.muted = true;
    final playback = turn.playback;
    if (playback != null) _silence(playback, 'mute');
    if (_phase.value == TurnPhase.speaking) _setPhase(TurnPhase.thinking);
  }

  /// Asks for microphone access now (a demo opening), so the
  /// first press does not meet the OS dialog.
  Future<Result<void>> requestMicAccess() => _audio.requestMicAccess();

  /// Starts the audio session and the output engine now, so the first turn
  /// does not pay for it (on macOS starting the output device can be slow).
  /// A failure is logged here and surfaces on the first turn, which prepares
  /// again.
  Future<void> prepareAudio() async {
    if (_disposed) return;
    if (await _audio.prepare() case Error(:final error)) {
      debugPrint(
        '[Voice] audio warm-up failed (the next turn retries): $error',
      );
    }
  }

  /// Opens the mic. During a turn this is the barge-in: playback stops before
  /// anything else, the turn is detached, and the new capture starts.
  ///
  /// [TurnPhase.openingMic] until the capture runs, [TurnPhase.listening]
  /// from then on: the start waits for the audio warm-up, and words spoken
  /// before it are never recorded. A press while a released capture is
  /// still starting takes that start over (the finger slipped and pressed
  /// again): the release waiting for it is superseded.
  Future<Result<void>> micDown() async {
    if (_disposed) return _disposedError();
    if (_mic.pressAgain()) return const Result.ok(null);
    final since = Stopwatch()..start();
    ++_epoch;
    final active = _turn;
    if (active != null) _detach(active, since: since);
    _setPartial('');
    _setPhase(TurnPhase.openingMic);
    final started = await _mic.open(onLimit: () => unawaited(micUp()));
    switch (started) {
      case CaptureOpened(:final released):
        // Released while the mic opened (and not pressed again): the release
        // cancels it right away, so it never shows as listening.
        if (!released) _setPhase(TurnPhase.listening);
        return const Result.ok(null);
      case CaptureDropped():
        // Stop, a typed send or dispose took over while the mic started.
        return const Result.ok(null);
      case CaptureFailed(:final error):
        final message = switch (error) {
          MicAccessException(:final message) ||
          AudioFocusException(:final message) => message,
          _ => 'The microphone did not start: $error',
        };
        _emit(MicUnavailable(message));
        _metrics.publishGate(outcome: TurnOutcome.micUnavailable);
        _setPhase(TurnPhase.error);
        return Result.error(error);
    }
  }

  /// Releases the mic and runs the turn: closes the capture, applies the
  /// gate (too short, all zeros, silent: no STT, no LLM), then STT → reply →
  /// audio. Completes when the turn has ended (after the audio drained), or
  /// as soon as a newer action supersedes it. A release before the mic
  /// opened records nothing ([NotHeardReason.releasedBeforeListening]).
  ///
  /// [image]: attached to this turn (Demo 1), handed to the
  /// responder factory with it.
  Future<TurnResult> micUp({Uint8List? image}) async {
    if (_disposed) return const TurnResult(TurnOutcome.ignored);
    final epoch = _epoch;
    switch (_mic.release()) {
      case null:
        return const TurnResult(TurnOutcome.ignored);
      case ReleasedWhileOpening(:final cancelled):
        return _releasedBeforeStart(cancelled, epoch);
      case ReleasedListening(:final handOver):
        final clock = Stopwatch()..start();
        // At the moment of release, in parallel with closing the capture,
        // the gate and STT (Demo 3 snapshots the frame the user was looking
        // at).
        final prep = _prepare(_request(typed: false, image: image));
        final handle = await handOver;
        if (_disposed || handle == null) {
          prep.discard();
          return const TurnResult(TurnOutcome.superseded);
        }
        _setPhase(TurnPhase.transcribing);
        final stopped = await handle.stop();
        final captureClose = clock.elapsed;
        if (_disposed || epoch != _epoch) {
          prep.discard();
          return const TurnResult(TurnOutcome.superseded);
        }
        switch (stopped) {
          case Error(:final error):
            prep.discard();
            debugPrint('[Voice] capture failed: $error');
            _emit(MicUnavailable(error.toString()));
            _metrics.publishGate(outcome: TurnOutcome.micUnavailable);
            _setPhase(TurnPhase.error);
            return TurnResult(TurnOutcome.micUnavailable, error: error);
          case Ok(:final value):
            return _submit(
              value,
              prep: prep,
              clock: clock,
              epoch: epoch,
              captureClose: captureClose,
            );
        }
    }
  }

  /// A release before the mic opened: nothing was recorded, so no STT and no
  /// turn. Once the start ends its capture is cancelled and the user is told
  /// to wait for Listening — unless the button was pressed again meanwhile
  /// (that press keeps the capture) or a newer action took over.
  Future<TurnResult> _releasedBeforeStart(
    Future<bool> cancelled,
    int epoch,
  ) async {
    if (!await cancelled || _disposed || epoch != _epoch) {
      // Pressed again, Stop / a typed send / dispose, or a failed start
      // (micDown reported it).
      return const TurnResult(TurnOutcome.superseded);
    }
    debugPrint('[Voice] gate: released before the mic opened');
    _emit(const NotHeard(NotHeardReason.releasedBeforeListening));
    _metrics.publishGate(outcome: TurnOutcome.notHeard);
    _setPhase(TurnPhase.idle);
    return const TurnResult(TurnOutcome.notHeard);
  }

  /// The PCM entry point, the same one mic release uses after closing the
  /// capture (integration tests feed fixture audio here). Supersedes a
  /// running turn like a barge-in does.
  Future<TurnResult> submitUtterance(Utterance utterance, {Uint8List? image}) {
    if (_disposed) return Future.value(const TurnResult(TurnOutcome.ignored));
    final clock = Stopwatch()..start();
    final epoch = ++_epoch;
    unawaited(_mic.drop());
    final active = _turn;
    if (active != null) _detach(active, since: clock);
    return _submit(
      utterance,
      prep: _prepare(_request(typed: false, image: image)),
      clock: clock,
      epoch: epoch,
    );
  }

  /// A typed turn through the same pipeline (no STT; the transcript event
  /// echoes [text]). Supersedes a running turn like a barge-in does.
  Future<TurnResult> sendText(String text, {Uint8List? image}) {
    final prompt = text.trim();
    if (_disposed || prompt.isEmpty) {
      return Future.value(const TurnResult(TurnOutcome.ignored));
    }
    final clock = Stopwatch()..start();
    final epoch = ++_epoch;
    unawaited(_mic.drop());
    final active = _turn;
    if (active != null) _detach(active, since: clock);
    return _startTurn(
      prep: _prepare(_request(typed: true, image: image)),
      epoch: epoch,
      clock: clock,
      metrics: TurnMetricsBuilder(typed: true),
      typed: prompt,
    );
  }

  /// The Stop button: closes the mic, or stops the running turn and waits
  /// for its terminal (the partial reply is committed as interrupted).
  /// Bounded by VoiceSession's drain (worst case ≈10 s).
  Future<void> stop() async {
    if (_disposed) return;
    ++_epoch;
    if (_mic.pressed) {
      _setPhase(TurnPhase.idle);
      // A capture still starting is cancelled once it has started.
      await _mic.drop();
      return;
    }
    final turn = _turn;
    if (turn == null) {
      // A release still closing the capture or waiting to start: the epoch
      // bump stops it there.
      if (_phase.value.isActive) _setPhase(TurnPhase.idle);
      return;
    }
    turn.stopRequested = true;
    final playback = turn.playback;
    if (playback != null) _silence(playback, 'stop');
    await turn.voice.interrupt();
    await turn.result.future;
  }

  /// The app left the foreground with the push-to-talk button held: the
  /// press ends without a turn (no STT, no reply). An open capture is
  /// cancelled, one still starting as soon as its start ends; a release
  /// waiting for it, or one that comes later (the pointer cancel), finds
  /// nothing. Returns whether a press was cancelled. A turn already past
  /// its capture (transcribing, thinking, speaking) goes on.
  Future<bool> cancelCapture() async {
    if (_disposed || !_mic.pressed) return false;
    ++_epoch;
    debugPrint('[Voice] the app left the foreground: the capture is cancelled');
    _setPartial('');
    _setPhase(TurnPhase.idle);
    await _mic.drop();
    return true;
  }

  /// Something outside the app ended what runs ([cause]): the OS or another
  /// app took the audio (the audio repository has cancelled the capture and
  /// stopped the playback already), the output went away, or the app left
  /// the foreground on iOS (no GPU work there, and suspension within
  /// seconds). A press ends without a turn; a running turn stops as with
  /// Stop — its playback silenced, its partial reply committed as
  /// interrupted, an audio question still transcribed — and
  /// [TurnInterrupted] says why. With nothing running nothing happens.
  /// Completes once the turn has ended (bounded by VoiceSession's drain,
  /// worst case ≈10 s).
  Future<void> interrupt(InterruptionCause cause) async {
    if (_disposed) return;
    if (_mic.pressed) {
      ++_epoch;
      debugPrint('[Voice] interrupted (${cause.name}): the capture ends');
      _setPartial('');
      _emit(TurnInterrupted(cause));
      _setPhase(TurnPhase.idle);
      await _mic.drop();
      return;
    }
    final turn = _turn;
    if (turn == null) {
      if (!_phase.value.isActive) return;
      // A release still closing its capture or waiting to start: the epoch
      // bump stops it there.
      ++_epoch;
      debugPrint('[Voice] interrupted (${cause.name}) before the turn began');
      _setPartial('');
      _emit(TurnInterrupted(cause));
      _setPhase(TurnPhase.idle);
      return;
    }
    ++_epoch;
    debugPrint('[Voice] interrupted (${cause.name}): the turn stops');
    turn
      ..stopRequested = true
      ..interruptedBy ??= cause;
    final playback = turn.playback;
    if (playback != null) _silence(playback, 'interruption');
    await turn.voice.interrupt();
    await turn.result.future;
  }

  void _onAudioInterruption(AudioInterruption what) => unawaited(
    interrupt(switch (what) {
      AudioInterruption.taken => InterruptionCause.audioTaken,
      AudioInterruption.outputRemoved => InterruptionCause.outputRemoved,
    }),
  );

  /// Ends everything: the mic closes, a running turn is detached and
  /// interrupted. Completes once that turn has drained.
  Future<void> dispose() async {
    if (_disposed) return;
    ++_epoch;
    unawaited(_audioInterruptions.cancel());
    // First: a turn detached now (its reply's playback tail cut) must not
    // hand its recording to a recognizer nobody listens to any more.
    _transcripts.dispose();
    final turn = _turn;
    if (turn != null) _detach(turn);
    _heldTranscripts.clear();
    // Recorded before _disposed blocks phase updates: the overlay must not
    // stay on "speaking" after the demo was left.
    _setPhase(TurnPhase.idle);
    _disposed = true;
    await _mic.drop();
    final pending = _pendingInterrupt;
    await _events.close();
    _phase.dispose();
    _partial.dispose();
    if (pending != null) await pending;
  }

  // ---- Turn pipeline ----

  Future<TurnResult> _submit(
    Utterance utterance, {
    required _Prepared<S> prep,
    required Stopwatch clock,
    required int epoch,
    Duration? captureClose,
  }) {
    final stats = analyzePcm16(utterance.pcm);
    final voice = measureVoice(
      utterance.pcm,
      gateDbfs: _config.silenceGateDbfs,
      aboveFloorDb: _config.aboveFloorDb,
      floorCapDbfs: _config.floorCapDbfs,
      sampleRate: _config.captureSampleRate,
    );
    final audioMs = pcm16Duration(
      utterance.pcm.length,
      _config.captureSampleRate,
    ).inMilliseconds;
    final facts =
        'held=${utterance.held.inMilliseconds}ms audio=${audioMs}ms '
        'peak=${voice.peakFrameDbfs.toStringAsFixed(1)}dBFS '
        'floor=${voice.floorDbfs.toStringAsFixed(1)}dBFS '
        'threshold=${voice.thresholdDbfs.toStringAsFixed(1)}dBFS '
        'voiced=${voice.voiced.inMilliseconds}ms allZero=${stats.allZero}';
    final MetricsGate gate = (
      peak: voice.peakFrameDbfs,
      threshold: voice.thresholdDbfs,
      voiced: voice.voiced,
    );
    final passes =
        utterance.held >= _config.minUtterance &&
        stats.samples > 0 &&
        !stats.allZero &&
        voice.voiced >= _config.minVoiced;
    if (!passes) prep.discard();
    if (utterance.held < _config.minUtterance) {
      debugPrint('[Voice] gate: too short ($facts)');
      return Future.value(_notHeard(NotHeardReason.tooShort, gate));
    }
    if (stats.samples == 0 || stats.allZero) {
      // Digital zeros for a whole press: the OS gave the app no audio
      // (macOS TCC does this instead of failing). Not "Didn't catch that".
      debugPrint('[Voice] gate: no audio from the mic ($facts)');
      final message = stats.samples == 0
          ? '$kMicAccessMessage (no audio arrived in ${utterance.held.inMilliseconds} ms)'
          : '$kMicAccessMessage (the capture was all digital silence)';
      _emit(MicUnavailable(message));
      _metrics.publishGate(outcome: TurnOutcome.micUnavailable, gate: gate);
      _setPhase(TurnPhase.error);
      return Future.value(
        TurnResult(
          TurnOutcome.micUnavailable,
          error: MicAccessException(message),
        ),
      );
    }
    if (voice.voiced < _config.minVoiced) {
      debugPrint('[Voice] gate: silent ($facts)');
      return Future.value(_notHeard(NotHeardReason.silent, gate));
    }
    debugPrint('[Voice] gate: pass ($facts)');
    return _startTurn(
      prep: prep,
      epoch: epoch,
      clock: clock,
      metrics: TurnMetricsBuilder(typed: false, captureClose: captureClose)
        ..gate = gate,
      pcm: utterance.pcm,
    );
  }

  /// Starts the turn, or discards [prep] on every path that does not.
  Future<TurnResult> _startTurn({
    required _Prepared<S> prep,
    required int epoch,
    required Stopwatch clock,
    required TurnMetricsBuilder metrics,
    Uint8List? pcm,
    String? typed,
  }) async {
    try {
      return await _startTurnOrNot(
        prep: prep,
        epoch: epoch,
        clock: clock,
        metrics: metrics,
        pcm: pcm,
        typed: typed,
      );
    } finally {
      // A no-op when the turn started (the responder was built).
      prep.discard();
    }
  }

  Future<TurnResult> _startTurnOrNot({
    required _Prepared<S> prep,
    required int epoch,
    required Stopwatch clock,
    required TurnMetricsBuilder metrics,
    Uint8List? pcm,
    String? typed,
  }) async {
    _setPartial('');
    _setPhase(typed != null ? TurnPhase.thinking : TurnPhase.transcribing);
    // A barged-in turn must reach its terminal before the next starts.
    final pending = _pendingInterrupt;
    if (pending != null) await pending;
    if (_disposed || epoch != _epoch) {
      return const TurnResult(TurnOutcome.superseded);
    }
    final speak = _speak;
    if (speak) {
      // The reply will play: the audio for the turn (Android: the focus,
      // until the turn ends; a voice turn's press took it already).
      final activated = await _audio.activate();
      if (_disposed || epoch != _epoch) {
        return const TurnResult(TurnOutcome.superseded);
      }
      if (activated case Error(:final error)) {
        return _failBeforeTurn(error, metrics);
      }
    }
    final turn = _Turn(
      seq: _metrics.nextSeq(),
      number: prep.turn,
      clock: clock,
      typed: typed != null,
      metrics: metrics,
      image: prep.image,
    );
    final VoiceResponder responder;
    switch (prep.responder((event) => _onSide(turn, event))) {
      case Ok(:final value):
        responder = value;
      case Error(:final error):
        return _failBeforeTurn(error, metrics);
    }
    // The responder decides: one that takes the recording makes this an
    // audio turn (the chat model hears it; no STT before the reply).
    final audio = pcm != null && responder.respondToAudio != null;
    metrics.audio = audio;
    final started = _speech.startTurn(
      pcm: pcm,
      typedText: typed,
      audio: audio,
      expectedStt: _recognizer,
      responder: responder,
      speak: speak,
      onTiming: (timing) => _onTiming(turn, timing),
      correct: _correct,
    );
    switch (started) {
      case Error(:final error):
        return _failBeforeTurn(error, metrics);
      case Ok(:final value):
        turn.voice = value;
    }
    _turn = turn;
    // _drive listens before its first await, in this synchronous block.
    unawaited(_drive(turn));
    if (audio) {
      // No transcript before the reply: the question is the recording. Its
      // entry now, its words after the reply (the transcript job).
      turn
        ..audio = true
        ..pcm = pcm
        ..userCommitted = true;
      final duration = pcm16Duration(pcm.length, _config.captureSampleRate);
      debugPrint(
        '[Voice] audio turn ${turn.number}: ${duration.inMilliseconds}ms '
        'of audio to the chat model (no STT)',
      );
      _emit(UserSpoke(turn.number, duration, image: turn.image));
      _setPhase(TurnPhase.thinking);
    }
    return turn.result.future;
  }

  Future<void> _drive(_Turn turn) async {
    try {
      await for (final event in turn.voice.events) {
        await _onEvent(turn, event);
      }
      if (!turn.result.isCompleted) {
        throw StateError('the voice turn ended without a terminal event');
      }
    } catch (e, st) {
      await _onTurnError(turn, e, st);
    } finally {
      if (identical(_turn, turn)) _turn = null;
      _logTurn(turn);
    }
  }

  Future<void> _onEvent(_Turn turn, VoiceEvent event) async {
    switch (event) {
      case VoiceTranscriptEvent(:final text):
        if (!turn.live) return;
        final said = text.trim();
        // An empty transcript ends with Complete('', ''), no LLM call.
        if (said.isEmpty) return;
        turn.userCommitted = true;
        _emit(UserSaid(said, typed: turn.typed, image: turn.image));
        _setPhase(TurnPhase.thinking);
      case VoiceReplyTextEvent(:final chunk):
        if (!turn.live) return;
        turn.reply.write(chunk);
        turn.metrics.firstText ??= turn.clock.elapsed;
        _setPartial(turn.reply.toString());
      case VoiceReplyAudioEvent(:final pcm, :final sampleRate, :final isFinal):
        if (!turn.live || turn.muted) return;
        // Zero bytes is normal (a non-speech clause, speech off): only real
        // audio starts playback and the speaking phase.
        if (pcm.isNotEmpty) {
          var playback = turn.playback;
          if (playback == null) {
            switch (_audio.beginPlayback(sampleRate)) {
              case Ok(:final value):
                playback = turn.playback = value;
              case Error(:final error):
                throw error;
            }
            turn.metrics
              ..firstAudio = turn.clock.elapsed
              ..sampleRate = sampleRate;
            debugPrint(
              '[Voice] first audio ${turn.clock.elapsedMilliseconds}ms after '
              '${turn.typed ? 'send' : 'release'} ($sampleRate Hz)',
            );
            _setPhase(TurnPhase.speaking);
            _publish(turn);
          }
          playback.enqueue(pcm);
        }
        if (isFinal) turn.playback?.end();
      case VoiceTurnCompleteEvent(:final transcript, :final replyText):
        if (turn.detached) return;
        // An audio turn's transcript is always empty (no STT ran): only a
        // transcript turn ends here when nothing was heard.
        if (!turn.audio && transcript.trim().isEmpty) {
          _emit(const NotHeard(NotHeardReason.emptyTranscript));
          _finish(turn, TurnOutcome.notHeard, TurnPhase.idle);
          return;
        }
        if (replyText.trim().isEmpty) throw const EmptyReplyException();
        // Generated in full: a press from here on cuts only the playback,
        // and the question still gets its words ([_detach]).
        turn.replyDone = true;
        // No generation runs any more: earlier questions held back for this
        // turn go to the recognizer now, beside its playback only.
        _releaseHeldTranscripts();
        final playback = turn.playback;
        if (playback != null) {
          playback.end();
          await playback.drained;
        }
        if (turn.detached || _disposed) return;
        _setPartial('');
        _emit(AssistantSaid(replyText.trim(), interrupted: false));
        _finish(turn, TurnOutcome.completed, TurnPhase.idle);
        _transcribeLater(turn);
      case VoiceTurnInterruptedEvent(:final partialReplyText):
        // A barged-in turn was committed when it was detached.
        if (turn.detached) return;
        _setPartial('');
        if (turn.userCommitted) {
          _emit(AssistantSaid(partialReplyText.trim(), interrupted: true));
        }
        _finish(turn, TurnOutcome.interrupted, TurnPhase.idle);
        // Stop, not a new press: the question still gets its words.
        _transcribeLater(turn);
      case VoiceErrorEvent(:final error):
        // Reserved in flutter_edge_ai_speech 0.5.7: never emitted.
        throw asException(error);
    }
  }

  Future<void> _onTurnError(_Turn turn, Object error, StackTrace st) async {
    if (turn.detached || _disposed) {
      debugPrint('[Voice] a detached turn failed: $error');
      return;
    }
    debugPrint('[Voice] turn failed: $error\n$st');
    final playback = turn.playback;
    if (playback != null) _silence(playback, 'error');
    // A throw from our own handler leaves the session running; a session
    // error has torn it down already (then this is a no-op).
    _trackInterrupt(turn.voice.interrupt());
    final partial = turn.reply.toString().trim();
    _setPartial('');
    // Cut off by the failure: committed as interrupted, so the bubble above
    // the error says "stopped" (its excerpts' chips stay, as after Stop)
    // instead of passing for a complete answer.
    if (partial.isNotEmpty) _emit(AssistantSaid(partial, interrupted: true));
    _emit(TurnFailed(error));
    _finish(turn, TurnOutcome.failed, TurnPhase.error, error: error);
    // The question was asked all the same: its entry still gets its words.
    _transcribeLater(turn);
  }

  /// Barge-in (or a typed send / dispose during a turn): silence first, then
  /// commit what the user saw, then let the session drain in the
  /// background.
  void _detach(_Turn turn, {Stopwatch? since}) {
    if (turn.detached) return;
    turn.detached = true;
    // A recording the transcript job holds already is not this one: pcm is
    // null then.
    if (turn.pcm case final pcm? when turn.audio) {
      if (turn.replyDone) {
        // Only the playback's tail is cut: the question was answered in
        // full, and gets its words as at a normal end, once no generation
        // runs.
        _holdTranscript(turn, pcm);
      } else {
        // A question barged in on during its reply gets no transcript: its
        // recording goes now, and its entry says so instead of looking
        // pending forever.
        _emit(
          TranscriptFailed(
            turn.number,
            kInterruptedTranscriptReason,
            interrupted: true,
          ),
        );
      }
    }
    turn.pcm = null;
    // Its generation was cut: questions held back for it go to the
    // recognizer now, so the chat rebuilt for the next turn replays their
    // words, not the placeholder (the next turn starts only after this one
    // drained).
    if (!turn.replyDone) _releaseHeldTranscripts();
    if (identical(_turn, turn)) _turn = null;
    final playback = turn.playback;
    final wasPlaying = playback != null && !turn.muted;
    Duration? silenced;
    Duration? stopConfirmed;
    Duration? interruptDone;
    // One record per barge-in, filled in as the stop and the drain finish.
    void record() => _diagnostics.recordBargeIn(
      BargeInMetrics(
        wasPlaying: wasPlaying,
        silenced: silenced,
        stopConfirmed: stopConfirmed,
        interruptDone: interruptDone,
      ),
    );
    if (wasPlaying) {
      final stopping = playback.stop(); // the native stop runs here
      silenced = since?.elapsed;
      unawaited(
        stopping.then(
          (_) {
            stopConfirmed = since?.elapsed;
            record();
          },
          onError: (Object e, StackTrace st) =>
              debugPrint('[Voice] stopping playback failed: $e\n$st'),
        ),
      );
    }
    record();
    if (turn.userCommitted) {
      _emit(AssistantSaid(turn.reply.toString().trim(), interrupted: true));
    }
    _setPartial('');
    turn.metrics
      ..outcome = TurnOutcome.superseded
      ..total = turn.clock.elapsed;
    _publish(turn);
    _trackInterrupt(
      turn.voice.interrupt().then((_) {
        interruptDone = since?.elapsed;
        debugPrint(
          '[Voice] barge-in: silenced=${silenced?.inMilliseconds}ms '
          'stop confirmed=${stopConfirmed?.inMilliseconds}ms '
          'interrupt done=${interruptDone?.inMilliseconds}ms',
        );
        record();
      }),
    );
    turn.complete(const TurnResult(TurnOutcome.superseded));
  }

  void _finish(
    _Turn turn,
    TurnOutcome outcome,
    TurnPhase phase, {
    Object? error,
  }) {
    turn.metrics
      ..outcome = outcome
      ..total = turn.clock.elapsed;
    if (identical(_turn, turn)) _turn = null;
    _publish(turn);
    // After the reply's commit, before idle: why it ended there.
    if (turn.interruptedBy case final cause?) _emit(TurnInterrupted(cause));
    _setPhase(phase);
    turn.complete(TurnResult(outcome, error: error));
  }

  TurnResult _failBeforeTurn(Exception error, TurnMetricsBuilder metrics) {
    debugPrint('[Voice] the turn could not start: $error');
    _emit(TurnFailed(error));
    metrics.outcome = TurnOutcome.failed;
    _metrics.publish(_metrics.nextSeq(), metrics);
    _setPhase(TurnPhase.error);
    return TurnResult(TurnOutcome.failed, error: error);
  }

  TurnResult _notHeard(NotHeardReason reason, MetricsGate gate) {
    _emit(NotHeard(reason));
    _metrics.publishGate(outcome: TurnOutcome.notHeard, gate: gate);
    _setPhase(TurnPhase.idle);
    return const TurnResult(TurnOutcome.notHeard);
  }

  void _silence(PlaybackHandle playback, String why) {
    unawaited(
      playback.stop().catchError(
        (Object e, StackTrace st) =>
            debugPrint('[Voice] stopping playback ($why) failed: $e\n$st'),
      ),
    );
  }

  void _trackInterrupt(Future<void> interrupt) {
    final previous = _pendingInterrupt;
    late final Future<void> tracked;
    tracked =
        (previous == null
                ? interrupt
                : Future.wait([previous, interrupt]).then((_) {}))
            .catchError(
              (Object e, StackTrace st) =>
                  debugPrint('[Voice] interrupt failed: $e\n$st'),
            )
            .whenComplete(() {
              if (identical(_pendingInterrupt, tracked)) {
                _pendingInterrupt = null;
              }
            });
    _pendingInterrupt = tracked;
  }

  void _onSide(_Turn turn, S event) =>
      _emit(SideEvent(event, detached: turn.detached));

  void _onTiming(_Turn turn, SpeechTiming timing) {
    switch (timing) {
      case SttTiming(:final elapsed):
        turn.metrics.stt = elapsed;
        debugPrint('[Voice] stt=${elapsed.inMilliseconds}ms');
      case TtsClauseTiming(:final elapsed):
        turn.metrics.tts.add(elapsed);
    }
    if (!turn.detached) _publish(turn);
  }

  TurnRequest _request({required bool typed, Uint8List? image}) =>
      TurnRequest(typed: typed, image: image, turn: ++_turnNumbers);

  _Prepared<S> _prepare(TurnRequest request) {
    try {
      return _Prepared(
        _responders.prepare(request),
        image: request.image,
        turn: request.turn,
      );
    } catch (e, st) {
      debugPrint('[Voice] preparing the turn failed: $e\n$st');
      return _Prepared.failed(e, image: request.image, turn: request.turn);
    }
  }

  // ---- Audio turns' transcripts ----

  /// An audio turn that ended while it was still this assistant's (its reply
  /// played to the end, Stop, or a failure; never a barge-in): its recording
  /// goes to the transcript job, now that nothing plays and the turn is idle.
  /// Later presses and turns do not drop it: its words reach its own entry.
  void _transcribeLater(_Turn turn) {
    final pcm = turn.pcm;
    if (!turn.audio || pcm == null || turn.detached) return;
    _startTranscript(turn, pcm);
  }

  /// [turn]'s recording [pcm] (a question answered in full whose playback
  /// tail a press cut): it is held until no generation runs
  /// ([_heldTranscripts]), then goes to the transcript job like a normal
  /// end's. Neither the next press nor its turn drops it.
  void _holdTranscript(_Turn turn, Uint8List pcm) {
    turn.pcm = null;
    _heldTranscripts.add((turn, pcm));
    // Not reached today (a turn is detached only while one runs), but a
    // recording must never wait for a phase change that does not come.
    if (!_phase.value.isActive) _releaseHeldTranscripts();
  }

  /// No generation runs any more: the held recordings go to the transcript
  /// job, oldest first (ahead of a turn that just ended, which adds its own
  /// after this).
  void _releaseHeldTranscripts() {
    if (_heldTranscripts.isEmpty) return;
    final held = List.of(_heldTranscripts);
    _heldTranscripts.clear();
    for (final (turn, pcm) in held) {
      _startTranscript(turn, pcm);
    }
  }

  /// Hands [turn]'s recording [pcm] to the transcript job, which holds it
  /// from here until its result.
  void _startTranscript(_Turn turn, Uint8List pcm) {
    turn.pcm = null;
    if (_disposed) return;
    _transcripts.start(
      turn: turn.number,
      pcm: pcm,
      onDone: (outcome) => _onTranscript(turn, outcome),
    );
  }

  /// [asked]'s transcript arrived: its figures get the time (shown unless a
  /// newer turn's are), its entry and the text history the words.
  void _onTranscript(_Turn asked, TranscriptOutcome outcome) {
    if (_disposed) return;
    asked.metrics.transcript = outcome.elapsed;
    if (outcome case TranscriptUnavailable(:final reason)) {
      asked.metrics.transcriptError = reason;
    }
    _publish(asked);
    switch (outcome) {
      case TranscriptReady(:final turn, :final text):
        try {
          _onTranscriptText?.call(turn, text);
        } catch (e, st) {
          debugPrint('[Voice] the transcript hook failed: $e\n$st');
        }
        _emit(TranscriptArrived(turn, text));
      case TranscriptUnavailable(:final turn, :final reason):
        _emit(TranscriptFailed(turn, reason));
    }
  }

  // ---- Publishing ----

  void _emit(VoiceAssistantEvent<S> event) {
    if (!_disposed && !_events.isClosed) _events.add(event);
  }

  void _setPhase(TurnPhase phase) {
    if (_disposed) return;
    final ended = _phase.value.isActive && !phase.isActive;
    _phase.value = phase;
    _diagnostics.recordVoicePhase(phase);
    // A press or a turn ended: the audio goes back (Android: the focus, so
    // other apps' audio resumes). The next press or reply takes it again.
    if (ended) unawaited(_audio.deactivate());
    // Nothing runs: held transcripts may start.
    if (!phase.isActive) _releaseHeldTranscripts();
  }

  void _setPartial(String text) {
    if (!_disposed) _partial.value = text;
  }

  /// The turn's figures, unless a newer turn's are already shown.
  void _publish(_Turn turn) => _metrics.publish(turn.seq, turn.metrics);

  void _logTurn(_Turn turn) =>
      debugPrint('[Voice] turn ${turn.metrics.summary()}');

  static Result<void> _disposedError() =>
      Result.error(asException(StateError('VoiceAssistant disposed')));
}

/// A [TurnPreparation] used exactly once: [responder] when the turn runs,
/// [discard] otherwise (later calls are no-ops).
final class _Prepared<S> {
  _Prepared(
    TurnPreparation<S> preparation, {
    required this.image,
    required this.turn,
  }) : _preparation = preparation;

  _Prepared.failed(Object error, {required this.image, required this.turn})
    : _preparation = null,
      _error = error;

  final TurnPreparation<S>? _preparation;

  /// The turn's attached image ([TurnRequest.image]).
  final Uint8List? image;

  /// The turn's number ([TurnRequest.turn]).
  final int turn;
  Object? _error;
  bool _used = false;

  Result<VoiceResponder> responder(void Function(S event) onSide) {
    final preparation = _preparation;
    if (_used || preparation == null) {
      return Result.error(
        asException(_error ?? StateError('the turn was prepared once already')),
      );
    }
    _used = true;
    try {
      return Result.ok(preparation.responder(onSide));
    } catch (e, st) {
      debugPrint('[Voice] building the responder failed: $e\n$st');
      return Result.error(asException(e));
    }
  }

  void discard() {
    if (_used) return;
    _used = true;
    try {
      _preparation?.discard();
    } catch (e, st) {
      debugPrint('[Voice] discarding the prepared turn failed: $e\n$st');
    }
  }
}

/// One running turn.
final class _Turn {
  _Turn({
    required this.seq,
    required this.number,
    required this.clock,
    required this.typed,
    required this.metrics,
    required this.image,
  });

  final int seq;

  /// [TurnRequest.turn]: what its entry and its transcript share.
  final int number;

  /// The chat model hears the recording (no STT before the reply).
  bool audio = false;

  /// An audio turn's recording, kept for the transcript job until the turn
  /// ends; dropped by a barge-in.
  Uint8List? pcm;

  /// The image attached to this turn, shown with the user's entry.
  final Uint8List? image;

  /// Started at release (or Send): the overlay's timings count from here.
  final Stopwatch clock;
  final bool typed;
  final TurnMetricsBuilder metrics;
  late final VoiceTurn voice;
  final StringBuffer reply = StringBuffer();
  final Completer<TurnResult> result = Completer();
  PlaybackHandle? playback;
  bool userCommitted = false;

  /// A barge-in replaced it: its events no longer reach the UI.
  bool detached = false;

  /// The reply was generated in full (only its playback may still run): a
  /// press now does not cost the question its transcript.
  bool replyDone = false;

  /// Stop was pressed: wait for its terminal, ignore further output.
  bool stopRequested = false;

  /// Stopped from outside ([VoiceAssistant.interrupt]): its end says why.
  InterruptionCause? interruptedBy;

  /// "Speak replies" was switched off during it.
  bool muted = false;

  bool get live => !detached && !stopRequested;

  void complete(TurnResult value) {
    if (!result.isCompleted) result.complete(value);
  }
}
