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
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show
        ErrorResult,
        ImageResult,
        SkillResult,
        TextResult,
        WebviewResult,
        WidgetResult;
import 'package:flutter_edge_ai_speech/flutter_edge_ai_speech.dart'
    show VoiceResponder;

import '../../config/voice_config.dart';
import '../../data/repositories/conversation_repository.dart';
import '../../utils/result.dart';
import '../models/assistant_event.dart';
import '../models/chat_side_event.dart';
import '../models/conversation_history.dart';
import '../models/knowledge.dart';
import '../models/knowledge_query.dart';
import '../models/skill_step.dart';
import '../ports/knowledge_retriever.dart';
import 'prompt_builder.dart';
import 'skill_question_router.dart';
import 'turn_responder_factory.dart';

/// Runs one app intent with JSON parameters (`AppIntentExecutor.run`).
typedef DirectIntentRunner = Future<SkillResult> Function(
  String intent,
  String paramsJson,
);

/// Demo 1's LLM step: the transcript is looked up in the knowledge base — the
/// excerpts that clear the gate go into the prompt, and the retrieval reaches
/// the UI as a [ChatRetrieval] side event — then the prompt goes to the shared
/// chat with the turn's attached image when there is one, and the reply streams
/// back as text. Whether the image is actually sent again is the conversation's
/// call (it re-sends only what the live chat cannot see), and so is the context
/// budget: an excerpt-heavy prompt plus an image can make it start the chat
/// over before the turn.
///
/// Audio mode (a chat model loaded with its audio encoder): a spoken turn's
/// recording is searched as audio (the audio gate), and the spoken question's
/// prompt with its excerpts goes to the chat with the recording itself; the
/// text history records [kSpokenQuestionPlaceholder] until [transcribed]
/// brings the words.
final class ChatTurnResponder implements TurnResponderFactory<ChatSideEvent> {
  /// [retriever]: the knowledge base. Null builds a responder without one
  /// (tests of the plain chat path); the app always passes it, and an
  /// unavailable knowledge base is a visible outcome, not a null here.
  ChatTurnResponder({
    required this._conversation,
    this._retriever,
    this._idleWait = kResponderIdleWait,
    this._skillRouter = const SkillQuestionRouter(),
    this._direct,
  });

  /// Runs a high-confidence action ([SkillQuestionRouter.action])
  /// without the model; null leaves every request to the model.
  final DirectIntentRunner? _direct;

  final ConversationRepository _conversation;
  final KnowledgeRetriever? _retriever;
  final Duration _idleWait;

  /// Live skill questions skip retrieval on an agent chat.
  final SkillQuestionRouter _skillRouter;

  /// Nothing to start at release: the chat needs only the transcript (or
  /// the recording) and this turn's image, captured here so a later change of
  /// the attachment cannot reach this turn.
  @override
  TurnPreparation<ChatSideEvent> prepare(TurnRequest request) =>
      _ChatPreparation(this, request);

  /// Puts an audio turn's transcript (it arrives after the reply) into the
  /// conversation's text history, where a rebuilt chat reads it instead of
  /// [kSpokenQuestionPlaceholder].
  void transcribed(int turn, String text) =>
      _conversation.setHistoryText(turn, text);

  /// The responder of one turn. A spoken turn on a chat model that hears
  /// audio gets `respondToAudio` too, which makes it an audio turn: the
  /// recording is searched in the knowledge base as audio and goes to the
  /// chat model with the excerpts; no transcript is needed before the reply.
  VoiceResponder _build(
    void Function(ChatSideEvent event) onSide,
    TurnRequest request,
  ) {
    final image = request.image;
    // Per-turn latches: a later turn's responder can never un-cancel this
    // one, and this one's stop never reaches a turn it did not start.
    var cancelled = false;
    var asking = false;

    /// Asks the chat with [prompt] (and the recording [audioWav] of an audio
    /// turn) and streams the reply; [retrieval] reaches the UI once the turn
    /// really asks. [historyText]: the question as the text history keeps it.
    Stream<String> ask(
      String prompt,
      Retrieval? retrieval, {
      required String historyText,
      Uint8List? audioWav,
    }) async* {
      await whenPreviousReplyEnded(
        _conversation.isGenerating,
        wait: _idleWait,
        tag: 'ChatTurnResponder',
      );
      if (cancelled) return;
      // Only for a turn that asks: a stopped turn's excerpts were never in
      // any prompt, so they must not show up under its reply.
      if (retrieval != null) onSide(ChatRetrieval(retrieval));
      asking = true;
      var stopResent = false;
      // What the last intent that ran reported, and whether the
      // model said anything. A skill call whose final generation comes back
      // empty is not a failed turn: the intent ran, so its result is what
      // the user hears.
      String? intentResult;
      var spoke = false;
      try {
        // Never break out of this loop: the final AssistantDone carries the
        // stop latency, and cancelling the subscription would skip the
        // chat's end-of-turn bookkeeping.
        await for (final event in _conversation.ask(
          prompt,
          image: image,
          audioWav: audioWav,
          historyText: historyText,
          turn: request.turn,
          // Skill steps are facts for the UI, also after a stop (the
          // tool may have run).
          onStep: (step) {
            if (step case IntentSucceeded(:final result)) {
              intentResult = result;
            }
            onSide(ChatSkillStep(step));
          },
        )) {
          switch (event) {
            case AssistantTextDelta(:final text) when !cancelled:
              if (text.trim().isNotEmpty) spoke = true;
              yield text;
            case AssistantTextDelta():
              // A stop that landed before native generation began had
              // nothing to cancel; text still arriving means it must be
              // sent again (scoped: this is our own ask).
              if (!stopResent) {
                stopResent = true;
                unawaited(_conversation.stop());
              }
            case AssistantContextReset(
              :final reason,
              :final keptTurns,
              :final droppedTurns,
              :final wordlessTurns,
              :final replayed,
            ):
              // Even for a stopped (barged-in) turn: the chat it leaves
              // behind has forgotten everything, and the user must know.
              onSide(
                ChatContextReset(
                  reason: reason,
                  keptTurns: keptTurns,
                  droppedTurns: droppedTurns,
                  wordlessTurns: wordlessTurns,
                  replayed: replayed,
                ),
              );
            case AssistantDone(:final metrics):
              onSide(ChatGenerationDone(metrics));
            case AssistantFailed(:final error):
              // ask() reports failures as events; dropping one would end the
              // turn as an empty reply, a silent failure. VoiceSession turns
              // this into the turn's stream error.
              throw error;
          }
        }
        if (!cancelled && !spoke) {
          // The conversation's text history keeps the same result as the
          // reply (`agent_turn.dart`), so a rebuilt chat replays it.
          if (intentResult case final result? when result.trim().isNotEmpty) {
            debugPrint(
              '[ChatTurnResponder] empty reply after an intent: speaking its '
              'result "$result"',
            );
            yield result;
          }
        }
      } finally {
        asking = false;
      }
    }

    Stream<String> respond(String text) async* {
      if (cancelled) return;
      // The time and live device questions run their intent
      // directly. The device_info text is exact; the model could only get it
      // wrong.
      // With a chat model that has tools off the
      // skills are not in the chat, but these direct intents need no tool
      // call and keep working.
      final direct = _direct;
      if (direct != null &&
          (_conversation.hasSkills || _conversation.skillsNeedTools)) {
        if (_skillRouter.action(text) case final action?) {
          final reply = await _runDirect(action, direct, onSide);
          if (!cancelled) yield reply;
          return;
        }
      }
      final (:prompt, :retrieval) = await _augment(text, image: image);
      if (cancelled) return;
      yield* ask(prompt, retrieval, historyText: text.trim());
    }

    // An audio turn: no words to route, so the time and device questions
    // reach their intents through the agent's tools (an agent chat) or not
    // at all (the mode line says so).
    Stream<String> respondToAudio(Uint8List wav) async* {
      if (cancelled) return;
      final (:prompt, :retrieval) = await _augmentSpoken(wav, image: image);
      if (cancelled) return;
      yield* ask(
        prompt,
        retrieval,
        historyText: kSpokenQuestionPlaceholder,
        audioWav: wav,
      );
    }

    return VoiceResponder(
      respond: respond,
      respondToAudio: !request.typed && _conversation.capabilities.audio
          ? respondToAudio
          : null,
      stop: () async {
        cancelled = true;
        // Only stop the chat while this responder's own ask runs: a late
        // stop (a left demo disposing) must not stop the next demo's turn.
        if (asking) await _conversation.stop();
      },
    );
  }

  /// Runs [action] through [direct], reports it as the usual skill steps,
  /// and returns what to say: the result text, or the part of an error
  /// meant for the user (the rest instructs the model).
  Future<String> _runDirect(
    DirectAction action,
    DirectIntentRunner direct,
    void Function(ChatSideEvent event) onSide,
  ) async {
    final watch = Stopwatch()..start();
    final params = jsonEncode(action.params);
    debugPrint(
      '[ChatTurnResponder] direct ${action.intent} $params (${action.rule})',
    );
    if (_retriever != null) {
      onSide(
        ChatRetrieval(
          Retrieval(
            outcome: RetrievalOutcome.skipped,
            detail: 'direct ${action.intent} (${action.rule})',
          ),
        ),
      );
    }
    onSide(
      ChatSkillStep(IntentCalled(action.intent, params, at: watch.elapsed)),
    );
    final started = watch.elapsed;
    SkillResult result;
    try {
      result = await direct(action.intent, params);
    } catch (e) {
      result = ErrorResult('$e');
    }
    final elapsed = watch.elapsed - started;
    switch (result) {
      case ErrorResult(:final message):
        onSide(
          ChatSkillStep(
            IntentFailed(
              action.intent,
              message,
              elapsed: elapsed,
              at: watch.elapsed,
            ),
          ),
        );
        return userFacing(message);
      case TextResult(:final text):
        onSide(
          ChatSkillStep(
            IntentSucceeded(
              action.intent,
              text,
              elapsed: elapsed,
              at: watch.elapsed,
            ),
          ),
        );
        return text;
      case ImageResult() || WidgetResult() || WebviewResult():
        onSide(
          ChatSkillStep(
            IntentSucceeded(
              action.intent,
              '$result',
              elapsed: elapsed,
              at: watch.elapsed,
            ),
          ),
        );
        return '$result';
    }
  }

  /// The sentences of an intent's error meant for the user: the executor
  /// ends some with instructions for the model ("Tell the user it did not
  /// work.").
  static String userFacing(String message) {
    final sentences = RegExp(r'[^.!?]+[.!?]*')
        .allMatches(message)
        .map((m) => m[0]!.trim())
        .where((s) => s.isNotEmpty)
        .where(
          (s) => !RegExp(
            r'^(Tell the user|Tell them|Do not|Don.t|Call |Otherwise|If the '
            r'user)',
          ).hasMatch(s),
        )
        .toList();
    return sentences.isEmpty ? 'That did not work.' : sentences.join(' ');
  }

  /// The prompt — the question with the excerpts that cleared the gate, or
  /// the question alone — and the retrieval behind it (null without a
  /// retriever). A failed search becomes a [RetrievalOutcome.failed]
  /// retrieval and the turn goes on without excerpts.
  ///
  /// On an agent chat a live skill question (device facts, the time;
  /// [SkillQuestionRouter]) is not searched at all: device
  /// questions clear the gate against the GPU docs, and with excerpts in the
  /// prompt Gemma sometimes answered from them instead of calling the skill.
  /// The overlay shows it as [RetrievalOutcome.skipped].
  ///
  /// An image turn on an agent chat that is not a skill
  /// question ends with [PromptBuilder.directAnswerHint]. Without it Gemma
  /// called `runIntent` (an invented intent, or `camera-watch`) for photo
  /// questions in 5 of 6 first turns, an extra generation each.
  Future<({String prompt, Retrieval? retrieval})> _augment(
    String question, {
    required Uint8List? image,
  }) async {
    final route = _conversation.hasSkills
        ? _skillRouter.classify(question)
        : null;
    final (:prompt, :retrieval) = await _retrieve(question, route);
    final hint = image != null && _conversation.hasSkills && route == null;
    return (
      prompt: hint ? '$prompt${PromptBuilder.directAnswerHint}' : prompt,
      retrieval: retrieval,
    );
  }

  /// An audio turn's prompt and retrieval: the recording searched as audio
  /// (gated at the audio threshold by the retriever), and the spoken
  /// question's prompt ([PromptBuilder.buildSpoken]) with the excerpts that
  /// cleared it. A failed search goes on without excerpts, like a typed one.
  Future<({String prompt, Retrieval? retrieval})> _augmentSpoken(
    Uint8List wav, {
    required Uint8List? image,
  }) async {
    final retriever = _retriever;
    Retrieval? retrieval;
    if (retriever != null) {
      retrieval = switch (await retriever.retrieve(KnowledgeQuery.audio(wav))) {
        Ok(:final value) => value,
        Error(:final error) => Retrieval(
          outcome: RetrievalOutcome.failed,
          detail: error.toString(),
          queryKind: KnowledgeQueryKind.audio,
        ),
      };
    }
    final passages = retrieval?.outcome == RetrievalOutcome.used
        ? retrieval!.passages
        : const <Passage>[];
    return (
      prompt: PromptBuilder.buildSpoken(
        passages,
        skills: _conversation.hasSkills,
        image: image != null,
      ),
      retrieval: retrieval,
    );
  }

  Future<({String prompt, Retrieval? retrieval})> _retrieve(
    String question,
    SkillRoute? route,
  ) async {
    final retriever = _retriever;
    if (retriever == null) return (prompt: question, retrieval: null);
    if (route != null) {
      debugPrint('[ChatTurnResponder] skill question ($route): no retrieval');
      return (
        prompt: question,
        retrieval: Retrieval(
          outcome: RetrievalOutcome.skipped,
          detail: route.toString(),
        ),
      );
    }
    final retrieval = switch (await retriever.retrieve(
      KnowledgeQuery.text(question),
    )) {
      Ok(:final value) => value,
      Error(:final error) => Retrieval(
        outcome: RetrievalOutcome.failed,
        detail: error.toString(),
        queryKind: KnowledgeQueryKind.text,
      ),
    };
    final passages = retrieval.outcome == RetrievalOutcome.used
        ? retrieval.passages
        : const <Passage>[];
    return (
      // On an agent chat the excerpts never stand in for a skill.
      prompt: PromptBuilder.build(
        question,
        passages,
        skills: _conversation.hasSkills,
      ),
      retrieval: retrieval,
    );
  }
}

final class _ChatPreparation implements TurnPreparation<ChatSideEvent> {
  _ChatPreparation(this._factory, this._request);

  final ChatTurnResponder _factory;
  final TurnRequest _request;

  @override
  VoiceResponder responder(void Function(ChatSideEvent event) onSide) =>
      _factory._build(onSide, _request);

  @override
  void discard() {}
}
