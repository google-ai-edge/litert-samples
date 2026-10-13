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
import 'package:flutter_edge_ai_speech/flutter_edge_ai_speech.dart'
    show VoiceResponder;

import '../../config/model_catalog.dart';
import '../../config/voice_config.dart';
import '../../data/repositories/conversation_repository.dart';
import '../../utils/result.dart';
import '../models/assistant_event.dart';
import '../models/camera_side_event.dart';
import '../models/route_decision.dart';
import '../models/scene_snapshot.dart';
import '../vision/stt_corrections.dart';
import 'camera_prompt.dart';
import 'fast_answer_composer.dart';
import 'question_router.dart';
import 'turn_responder_factory.dart';

/// Captures the frame a question is about; `LiveDetectionRepository.capture`.
typedef CaptureScene = Future<Result<SceneSnapshot>> Function();

/// The snapshot as Gemma's PNG; `LiveDetectionRepository.encodeForLlm`.
typedef SnapshotEncoding = Future<Result<EncodedSnapshot>> Function(
  SceneSnapshot snapshot,
);

/// How long a detailed question waits for the previous turn's chat reset
/// (normally well under 150 ms) before it checks the chat anyway.
const kCameraResetWait = Duration(seconds: 3);

/// Demo 3's LLM step.
///
/// At release it captures the next frame (in parallel with closing the mic
/// and STT). With the transcript it routes:
/// - **fast**: answered from that frame's detection summary with a template,
///   no LLM;
/// - **detailed**: the frame goes to Gemma. The view model freezes on the
///   snapshot when it arrives ([SnapshotTaken] after a detailed
///   [RouteChosen]); the snapshot is encoded (PNG ≤1024 px, mirrored back),
///   asked about through the shared chat with the camera profile, and the
///   reply streams back as text. Each detailed turn is stateless: after
///   it, the chat is reset off the critical path, and the next detailed
///   question waits for that reset first.
///
/// Scoping, as in Demo 1's `ChatTurnResponder`: a turn's stop only stops the
/// chat while that turn's own ask runs, and the reset only applies while the
/// camera profile is still the open one, so neither can reach the next
/// demo's chat.
final class CameraTurnResponder
    implements TurnResponderFactory<CameraSideEvent> {
  CameraTurnResponder({
    required this._capture,
    required this._conversation,
    required this._encode,
    this._router = const QuestionRouter(),
    this._composer = const FastAnswerComposer(),
    this._profile = kCameraProfile,
    this._idleWait = kResponderIdleWait,
    this._resetWait = kCameraResetWait,
  });

  final CaptureScene _capture;
  final ConversationRepository _conversation;
  final SnapshotEncoding _encode;
  final QuestionRouter _router;
  final FastAnswerComposer _composer;
  final ConversationProfile _profile;
  final Duration _idleWait;
  final Duration _resetWait;

  /// The previous detailed turn's reset, while it runs.
  Future<void>? _pendingReset;

  /// Spoken when no frame could be captured; the UI shows why.
  static const cameraNotRunning = "The camera isn't running.";

  /// Spoken when the camera runs but sent no frame in time.
  static const cameraTimedOut =
      "The camera didn't send a frame in time. "
      'Please ask again.';

  /// Spoken when the camera pipeline failed (the screen offers Retry).
  static const cameraFailed =
      'The camera stopped working. Press Retry on '
      'the screen.';

  /// Said before the detector's list when a detailed question comes in and
  /// the chat model has images off.
  static const imagesOff =
      "I can't look at the picture with this chat model, only list what the "
      'detector sees.';

  /// Spoken when a detailed question finds the camera chat not open.
  static const chatNotReady =
      "I can't look at the frame right now: the camera chat isn't ready. "
      'Please ask again in a moment.';

  /// Starts the snapshot now, at release. `request.image` is ignored: the
  /// camera is the image.
  @override
  TurnPreparation<CameraSideEvent> prepare(TurnRequest request) =>
      _CameraPreparation(this, _capture());

  /// Starts the post-turn reset without awaiting it and reports
  /// its time through [onSide]. The next detailed turn waits for it.
  void _resetAfterTurn(void Function(CameraSideEvent event) onSide) {
    final previous = _pendingReset;
    // A typed function: an async closure's future is a Future<Null> at run
    // time, and a void onTimeout on it would throw.
    Future<void> run() async {
      if (previous != null) await previous;
      final watch = Stopwatch()..start();
      final result = await _conversation.reset(ifCurrent: _profile);
      final elapsed = watch.elapsed;
      final error = switch (result) {
        Ok() => null,
        Error(:final error) => error.toString(),
      };
      debugPrint(
        '[CameraTurn] chat reset ${elapsed.inMilliseconds}ms'
        '${error == null ? '' : ' failed: $error'}',
      );
      onSide(CameraChatReset(elapsed: elapsed, error: error));
    }

    late final Future<void> reset;
    reset = run().whenComplete(() {
      if (identical(_pendingReset, reset)) _pendingReset = null;
    });
    _pendingReset = reset;
    // Never throws (a Result); the side event carries a failure.
    unawaited(reset);
  }

  /// Waits (bounded) for the previous turn's reset and for a stopped
  /// generation that is still draining, then says whether the camera
  /// chat is open.
  Future<bool> _chatReady() async {
    final reset = _pendingReset;
    if (reset != null) {
      await reset.timeout(
        _resetWait,
        onTimeout: () => debugPrint(
          '[CameraTurn] the previous reset still runs after '
          '${_resetWait.inSeconds}s',
        ),
      );
    }
    await whenPreviousReplyEnded(
      _conversation.isGenerating,
      wait: _idleWait,
      tag: 'CameraTurn',
    );
    return _conversation.profile == _profile;
  }
}

final class _CameraPreparation implements TurnPreparation<CameraSideEvent> {
  _CameraPreparation(this._owner, this._snapshot);

  final CameraTurnResponder _owner;

  /// Started at release. Never throws (a [Result]); when the turn is
  /// discarded it completes unobserved, and the frame it took was published
  /// to the live view like any other.
  final Future<Result<SceneSnapshot>> _snapshot;

  /// Per-turn latches: a later turn can never un-cancel this one, and this
  /// one's stop never reaches a turn it did not start.
  bool _cancelled = false;
  bool _asking = false;

  @override
  VoiceResponder responder(void Function(CameraSideEvent event) onSide) =>
      VoiceResponder(
        respond: (transcript) => _respond(transcript, onSide),
        stop: () async {
          _cancelled = true;
          // Only while this turn's own ask runs: a late stop (a left demo
          // disposing) must not stop the next turn's or demo's reply.
          if (_asking) await _owner._conversation.stop();
        },
      );

  @override
  void discard() => _cancelled = true;

  Stream<String> _respond(
    String heard,
    void Function(CameraSideEvent event) onSide,
  ) async* {
    if (_cancelled) return;
    final watch = Stopwatch()..start();
    // Moonshine's known misses ("cop" for "cup") are corrected before
    // routing, only towards objects the detector knows.
    final corrected = correctSttHomophones(heard);
    final transcript = corrected.text;
    if (corrected.corrections.isNotEmpty) {
      debugPrint(
        '[CameraTurn] heard "$heard", corrected to "$transcript" '
        '(${corrected.corrections.join(', ')})',
      );
      onSide(TranscriptCorrected(transcript, corrected.corrections));
    }
    final route = _owner._router.classify(transcript);
    onSide(RouteChosen(route, watch.elapsed));
    final result = await _snapshot;
    if (_cancelled) return;
    final SceneSnapshot snapshot;
    switch (result) {
      case Error(:final error):
        onSide(SnapshotFailed(error.toString()));
        yield switch (error) {
          CaptureUnavailableException(kind: CaptureFailure.timedOut) =>
            CameraTurnResponder.cameraTimedOut,
          CaptureUnavailableException(kind: CaptureFailure.failed) =>
            CameraTurnResponder.cameraFailed,
          _ => CameraTurnResponder.cameraNotRunning,
        };
        return;
      case Ok(:final value):
        snapshot = value;
        onSide(SnapshotTaken(value));
    }
    switch (route) {
      case FastRoute():
        final composer = _owner._composer;
        final answer = composer.compose(route, snapshot.summary);
        onSide(
          FastAnswered(answer: answer, basis: composer.basis(snapshot.summary)),
        );
        yield answer;
      case DetailedRoute():
        final capabilities = _owner._conversation.capabilities;
        if (!capabilities.images) {
          // Images off: the frame cannot go to the chat model; the
          // detections can.
          final composer = _owner._composer;
          final listed = composer.compose(
            const FastRoute(FastIntent.inventory, 'images-off'),
            snapshot.summary,
          );
          onSide(
            DetailedUnavailable(
              reason:
                  'Detailed answers are off: ${capabilities.modelName} was '
                  'set up without images.',
              answer: listed,
              basis: composer.basis(snapshot.summary),
            ),
          );
          yield '${CameraTurnResponder.imagesOff} $listed';
          return;
        }
        yield* _detailed(transcript, snapshot, onSide);
    }
  }

  /// The frame to Gemma (the detailed path).
  Stream<String> _detailed(
    String question,
    SceneSnapshot snapshot,
    void Function(CameraSideEvent event) onSide,
  ) async* {
    final ready = await _owner._chatReady();
    if (_cancelled) return;
    if (!ready) {
      onSide(
        CameraChatUnavailable(
          'The camera chat is not open (profile: '
          '${_owner._conversation.profile?.name ?? 'none'})',
        ),
      );
      yield CameraTurnResponder.chatNotReady;
      return;
    }
    final EncodedSnapshot image;
    switch (await _owner._encode(snapshot)) {
      case Ok(:final value):
        image = value;
      case Error(:final error):
        // A turn error: VoiceSession ends the turn with it, and the UI
        // shows it (never an empty reply).
        throw error;
    }
    if (_cancelled) return;
    onSide(FrameSentToGemma(image));
    final prompt = buildCameraPrompt(
      question,
      snapshot.detections,
      mirrored: snapshot.mirrored,
    );
    final conversation = _owner._conversation;
    _asking = true;
    var stopResent = false;
    try {
      // Never break out of this loop: the final AssistantDone carries the
      // stop latency, and cancelling the subscription would skip the chat's
      // end-of-turn bookkeeping.
      await for (final event in conversation.ask(prompt, image: image.png)) {
        switch (event) {
          case AssistantTextDelta(:final text) when !_cancelled:
            yield text;
          case AssistantTextDelta():
            // A stop that landed before native generation began had nothing
            // to cancel; text still arriving means it must be sent again
            // (scoped: this is our own ask).
            if (!stopResent) {
              stopResent = true;
              unawaited(conversation.stop());
            }
          case AssistantContextReset():
            // A stateless camera turn has no history to lose.
            debugPrint('[CameraTurn] the budget guard reset the chat');
          case AssistantDone(:final metrics):
            onSide(DetailedAnswered(metrics));
          case AssistantFailed(:final error):
            // ask() reports failures as events; dropping one would end the
            // turn as an empty reply, a silent failure. VoiceSession turns
            // this into the turn's stream error.
            throw error;
        }
      }
    } finally {
      _asking = false;
      // Stateless turns: a clean chat for the next question, off the
      // critical path. Also after a stop or a failure, so no image or
      // partial reply is left in the chat.
      _owner._resetAfterTurn(onSide);
    }
  }
}
