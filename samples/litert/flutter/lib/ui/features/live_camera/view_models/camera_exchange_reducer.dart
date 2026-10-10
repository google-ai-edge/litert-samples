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

import '../../../../domain/models/assistant_event.dart' show GenerationMetrics;
import '../../../../domain/models/camera_side_event.dart';
import '../../../../domain/models/chat_capabilities.dart';
import '../../../../domain/models/route_decision.dart';
import '../../../../domain/models/scene_snapshot.dart';
import '../../../../domain/models/voice.dart';
import '../../../../domain/use_cases/voice_assistant.dart';
import '../../../core/voice_notices.dart';

/// The last question on screen: what was asked, how it was routed, what was
/// answered.
final class const CameraExchange({
  final String? question,
  final String? answer,

  /// "cat ×2 · remote — detector, no LLM".
  final String? route,

  /// The route needed Gemma.
  final bool detailed = false,

  /// "Didn't catch that", why the frame could not be captured, or why the
  /// answer failed (next to the partial answer when it failed partway).
  final String? notice,
});

/// What Demo 3's voice turns have put on screen, and the current turn's
/// facts the debug overlay records.
final class const CameraExchangeState({
  final CameraExchange exchange = const CameraExchange(),

  /// A turn started (a press, a submitted utterance) and its question has
  /// not arrived yet. The caption still shows the previous exchange (a
  /// barge-in commits its answer into it), but a failure now belongs to the
  /// new turn alone: it replaces the exchange instead of joining it.
  final bool questionPending = false,

  /// The current turn's route (the detailed chip names its rule).
  final RouteChosen? route,

  /// The current turn's snapshot latency, the frame it sent to Gemma and
  /// Gemma's time to first token, for the overlay.
  final Duration? snapshotLatency,
  final EncodedSnapshot? turnImage,
  final Duration? turnTtft,

  /// The last frame sent to Gemma (kept across turns).
  final EncodedSnapshot? sentImage,
}) {
  /// This state with the given fields replaced; a null keeps the field.
  /// [turnTtft] is always kept: the reducer replaces it (also with null) by
  /// building the state in full.
  CameraExchangeState copyWith({
    CameraExchange? exchange,
    bool? questionPending,
    RouteChosen? route,
    Duration? snapshotLatency,
    EncodedSnapshot? turnImage,
    EncodedSnapshot? sentImage,
  }) => CameraExchangeState(
    exchange: exchange ?? this.exchange,
    questionPending: questionPending ?? this.questionPending,
    route: route ?? this.route,
    snapshotLatency: snapshotLatency ?? this.snapshotLatency,
    turnImage: turnImage ?? this.turnImage,
    turnTtft: turnTtft,
    sentImage: sentImage ?? this.sentImage,
  );
}

/// One step of [CameraExchangeReducer]: the next state, and what the view
/// model does about it, in this order — leave the frozen frame, take the
/// state, record the figures, freeze, record the chat reset, rebuild.
final class const CameraExchangeUpdate({
  required final CameraExchangeState state,

  /// Leave the frozen frame: the turn's answer was committed (after playback
  /// drained, or at a barge-in), or the turn failed.
  final bool unfreeze = false,

  /// Gemma's figures for the overlay (also from a turn a barge-in replaced).
  final GenerationMetrics? generation,

  /// The overlay's record of the current camera turn, rewritten.
  final CameraTurnMetrics? turn,

  /// Freeze the view on this snapshot: a detailed turn sends its frame.
  final SceneSnapshot? freeze,

  /// The post-turn chat reset finished (also from a replaced turn): the
  /// overlay records it, and its error is the chat's until the next open.
  final CameraChatReset? chatReset,

  /// The screen changed, or the chat's state did: rebuild.
  final bool notify = true,
});

/// Demo 3's projection of its voice events onto the screen, as a pure
/// function: [reduce] takes the state, one event and the chat model's
/// capabilities, and returns the next state plus the effects for the view
/// model to apply (the freeze, the overlay's records, the rebuild).
///
/// The rules it owns:
/// - a failure after the question keeps this turn's question, route chip and
///   partial answer with the notice; a failure before it
///   ([CameraExchangeState.questionPending]: the recognizer, the audio start,
///   the turn's start) shows only the notice;
/// - the side events of a turn a barge-in replaced (detached) never freeze or
///   touch the screen; their generation figures and their chat reset still
///   count, and only the reset rebuilds;
/// - only a detailed turn that sends its frame (images on) freezes, on its
///   snapshot; the committed answer, a failure and the turn's end
///   ([unfreezesAt]) leave the frozen frame;
/// - a capture that made no LLM call shows only its notice, e.g. a release
///   while the mic was still opening ([NotHeardReason.releasedBeforeListening]).
final class CameraExchangeReducer {
  const CameraExchangeReducer();

  /// A turn started (a press, a submitted utterance): its question is
  /// pending. Nothing on screen changes yet.
  CameraExchangeState turnStarted(CameraExchangeState state) =>
      state.copyWith(questionPending: true);

  /// Reaching [phase] leaves the frozen frame: a turn that ended any other
  /// way (failed, superseded) leaves nothing frozen. Opening the mic,
  /// listening and answering keep it.
  bool unfreezesAt(TurnPhase phase) =>
      phase == TurnPhase.idle || phase == TurnPhase.error;

  CameraExchangeUpdate reduce(
    CameraExchangeState state,
    VoiceAssistantEvent<CameraSideEvent> event,
    ChatCapabilities chat,
  ) {
    final exchange = state.exchange;
    switch (event) {
      case UserSaid(:final text):
        // The new turn's question: its facts start over.
        return CameraExchangeUpdate(
          state: CameraExchangeState(
            exchange: CameraExchange(question: text),
            sentImage: state.sentImage,
          ),
        );
      case AssistantSaid(:final text, :final interrupted):
        // Committed after playback drained, or at a barge-in: either way the
        // frozen frame has served its turn.
        return CameraExchangeUpdate(
          unfreeze: true,
          state: state.copyWith(
            exchange: CameraExchange(
              question: exchange.question,
              answer: interrupted && text.isEmpty ? '(stopped)' : text,
              route: exchange.route,
              detailed: exchange.detailed,
              notice: exchange.notice,
            ),
          ),
        );
      case NotHeard(:final reason):
        return CameraExchangeUpdate(
          state: state.copyWith(
            exchange: CameraExchange(notice: notHeardText(reason)),
          ),
        );
      case MicUnavailable(:final message):
        return CameraExchangeUpdate(
          state: state.copyWith(exchange: CameraExchange(notice: message)),
        );
      case TurnFailed(:final error):
        final notice = 'The answer failed: $error';
        // Failed before its question (STT, the audio start, the turn's
        // start): nothing on screen belongs to this turn. Failed later: the
        // partial answer the user saw was committed just before (Demo 1
        // keeps it too) and stays with the notice; its question reset the
        // exchange, so an answer here is this turn's.
        return CameraExchangeUpdate(
          unfreeze: true,
          state: state.copyWith(
            exchange: state.questionPending
                ? CameraExchange(notice: notice)
                : CameraExchange(
                    question: exchange.question,
                    answer: exchange.answer,
                    route: exchange.route,
                    detailed: exchange.detailed,
                    notice: notice,
                  ),
          ),
        );
      case SideEvent(:final event, :final detached) when detached:
        return _detached(state, event);
      case SideEvent(:final event):
        return _side(state, event, chat);
    }
  }

  /// A barged-in turn's late facts: never freeze or touch the screen, but
  /// its figures and the chat's state still count.
  CameraExchangeUpdate _detached(
    CameraExchangeState state,
    CameraSideEvent event,
  ) => switch (event) {
    DetailedAnswered(:final metrics) => CameraExchangeUpdate(
      state: state,
      generation: metrics,
      notify: false,
    ),
    CameraChatReset() => CameraExchangeUpdate(state: state, chatReset: event),
    TranscriptCorrected() ||
    RouteChosen() ||
    SnapshotTaken() ||
    SnapshotFailed() ||
    FastAnswered() ||
    DetailedUnavailable() ||
    CameraChatUnavailable() ||
    FrameSentToGemma() => CameraExchangeUpdate(state: state, notify: false),
  };

  CameraExchangeUpdate _side(
    CameraExchangeState state,
    CameraSideEvent event,
    ChatCapabilities chat,
  ) {
    final exchange = state.exchange;
    switch (event) {
      case TranscriptCorrected(:final text, :final corrections):
        // The corrected question, with what moonshine heard.
        final heard = corrections.map((c) => "'${c.heard}'").join(', ');
        return CameraExchangeUpdate(
          state: state.copyWith(
            exchange: CameraExchange(question: '$text (heard $heard)'),
          ),
        );
      case RouteChosen():
        return CameraExchangeUpdate(
          state: state.copyWith(
            route: event,
            exchange: switch (event.route) {
              DetailedRoute(:final rule) when chat.images => CameraExchange(
                question: exchange.question,
                route: 'detailed ($rule) — ${chat.modelName}',
                detailed: true,
              ),
              // The chip waits for the answer's basis.
              _ => exchange,
            },
          ),
        );
      case SnapshotTaken(:final snapshot):
        final next = state.copyWith(snapshotLatency: snapshot.latency);
        // RouteChosen comes first: only a detailed turn that sends the
        // frame freezes. A fast turn is recorded with its basis
        // (FastAnswered), one without images with DetailedUnavailable.
        if (next.route?.route is DetailedRoute && chat.images) {
          return CameraExchangeUpdate(
            state: next,
            turn: _turn(next),
            freeze: snapshot,
          );
        }
        return CameraExchangeUpdate(state: next);
      case SnapshotFailed(:final message):
        // Keeps the route chip: the question was routed, the frame failed.
        final next = state.copyWith(
          exchange: CameraExchange(
            question: exchange.question,
            route: exchange.route,
            detailed: exchange.detailed,
            notice: message,
          ),
        );
        return CameraExchangeUpdate(
          state: next,
          turn: _turn(next, snapshotError: message),
        );
      case FastAnswered(:final basis):
        final next = state.copyWith(
          exchange: CameraExchange(
            question: exchange.question,
            route: '$basis — detector, no LLM',
          ),
        );
        return CameraExchangeUpdate(
          state: next,
          turn: _turn(next, basis: basis),
        );
      case DetailedUnavailable(:final basis):
        final next = state.copyWith(
          exchange: CameraExchange(
            question: exchange.question,
            route: '$basis — detector only (detailed answers off)',
          ),
        );
        return CameraExchangeUpdate(
          state: next,
          turn: _turn(next, basis: basis),
        );
      case CameraChatUnavailable(:final message):
        final next = state.copyWith(
          exchange: CameraExchange(
            question: exchange.question,
            route: exchange.route,
            detailed: exchange.detailed,
            notice: message,
          ),
        );
        return CameraExchangeUpdate(
          state: next,
          turn: _turn(next, chatError: message),
        );
      case FrameSentToGemma(:final image):
        final rule = state.route?.route.rule ?? '?';
        final next = state.copyWith(
          sentImage: image,
          turnImage: image,
          exchange: CameraExchange(
            question: exchange.question,
            route:
                'detailed ($rule) — frame #${image.frameId} '
                '${image.width}×${image.height} → Gemma',
            detailed: true,
            notice: exchange.notice,
          ),
        );
        return CameraExchangeUpdate(state: next, turn: _turn(next));
      case DetailedAnswered(:final metrics):
        // Built in full: a reply stopped before its first text has no time
        // to first token, and that null replaces an earlier one.
        final next = CameraExchangeState(
          exchange: state.exchange,
          questionPending: state.questionPending,
          route: state.route,
          snapshotLatency: state.snapshotLatency,
          turnImage: state.turnImage,
          turnTtft: metrics.timeToFirstToken,
          sentImage: state.sentImage,
        );
        return CameraExchangeUpdate(
          state: next,
          generation: metrics,
          turn: _turn(next),
        );
      case CameraChatReset():
        return CameraExchangeUpdate(state: state, chatReset: event);
    }
  }

  /// The overlay's record of [state]'s turn; null before its route.
  CameraTurnMetrics? _turn(
    CameraExchangeState state, {
    String? basis,
    String? snapshotError,
    String? chatError,
  }) {
    final route = state.route;
    if (route == null) return null;
    return CameraTurnMetrics(
      route: route.route,
      routeTime: route.elapsed,
      snapshotLatency: state.snapshotLatency,
      basis: basis,
      snapshotError: snapshotError,
      image: state.turnImage,
      timeToFirstToken: state.turnTtft,
      chatError: chatError,
    );
  }
}
