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
        Skill,
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
import '../skills/app_intents.dart';
import 'prompt_builder.dart';
import 'skill_question_router.dart';
import 'turn_responder_factory.dart';

/// Runs one intent with JSON parameters, without the model: the app passes
/// `NativeIntentExecutor.handleIntent`, which decodes the parameters as a
/// JSON object (an empty string is `{}`), refuses an intent it has no handler
/// for, runs a built-in intent's own check and the app's validator, then the
/// handler, and returns every failure as an `ErrorResult`. Not the agent
/// loop's `runIntent` path: no agent events, no intent read from the
/// parameters, no hint for a text-only skill.
typedef DirectIntentRunner = Future<SkillResult> Function(
  String intent,
  String paramsJson,
);

/// The skills installed now: the latest scan of the skills folder, what the
/// Skills sheet lists (`SkillCatalog.agentSkills`), whether or not the open
/// chat can call them.
typedef InstalledSkills = List<Skill> Function();

/// One turn's prompt before it meets the context window: the [retrieval]
/// behind it (null without a retriever), the excerpts that cleared the gate
/// ([passages], best first; empty when none did), and the [prompt] with any
/// leading part of them.
typedef _Draft = ({
  Retrieval? retrieval,
  List<Passage> passages,
  String Function(List<Passage> passages) prompt,
});

/// Demo 1's LLM step: the transcript is looked up in the knowledge base — the
/// excerpts that clear the gate go into the prompt, as many as fit the chat
/// model's context window beside the turn (the lowest-scoring go first), and
/// the retrieval reaches the UI as a [ChatRetrieval] side event — then the
/// prompt goes to the shared chat with the turn's attached image when there is
/// one, and the reply streams back as text. Whether the image is actually sent
/// again is the conversation's call (it re-sends only what the live chat cannot
/// see), and so is the context budget: an excerpt-heavy prompt plus an image
/// can make it start the chat over before the turn.
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
    this._installedSkills,
  });

  /// Runs a high-confidence action ([SkillQuestionRouter.action])
  /// without the model, while an installed intent skill names its intent
  /// ([_installedSkills]); null leaves every request to the model.
  final DirectIntentRunner? _direct;

  /// The installed skills whose intents the app may run itself (and skip
  /// retrieval for); null: none, so every request goes to the model.
  final InstalledSkills? _installedSkills;

  final ConversationRepository _conversation;
  final KnowledgeRetriever? _retriever;
  final Duration _idleWait;

  /// Live questions the app recognizes skip retrieval while an installed
  /// intent skill names their intent.
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

    /// Asks the chat with [draft]'s prompt, fitted to the context window
    /// (and the recording [audioWav] of an audio turn), and streams the
    /// reply; the retrieval reaches the UI once the turn really asks.
    /// [historyText]: the question as the text history keeps it.
    Stream<String> ask(
      _Draft draft, {
      required String historyText,
      Uint8List? audioWav,
    }) async* {
      await whenPreviousReplyEnded(
        _conversation.isGenerating,
        wait: _idleWait,
        tag: 'ChatTurnResponder',
      );
      if (cancelled) return;
      // After the wait: the chat a previous turn may have been replacing is
      // the one this turn measures against.
      final (:prompt, :retrieval) = await _fit(
        draft,
        image: image,
        audioWav: audioWav,
      );
      if (cancelled) return;
      // Only for a turn that asks: a stopped turn's excerpts were never in
      // any prompt, so they must not show up under its reply.
      if (retrieval != null) onSide(ChatRetrieval(retrieval));
      asking = true;
      var stopResent = false;
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
          onStep: (step) => onSide(ChatSkillStep(step)),
        )) {
          switch (event) {
            case AssistantTextDelta(:final text) when !cancelled:
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
      } finally {
        asking = false;
      }
    }

    Stream<String> respond(String text) async* {
      if (cancelled) return;
      // A request the app recognizes runs its intent directly (the intent's
      // text is exact; the model could only get it wrong), but only while an
      // installed intent skill names that intent: the shortcut stands in for
      // that skill, and goes when it is deleted or replaced. It does not
      // depend on the chat: with tools off (NPU builds by default) the model
      // cannot call any skill, and without the shortcut it would invent the
      // time or answer device questions from the docs.
      //
      // Known limitation: the shortcut speaks the intent's own text, not the
      // skill's wording. A skill that names current_time but replies its own
      // way (a `french-clock` that answers in French) is bypassed for the
      // phrasings the rules recognize, which get the English intent text;
      // its other phrasings reach the model and the skill.
      final offered = _offeredIntents();
      final direct = _direct;
      if (direct != null) {
        if (_skillRouter.action(text) case final action?) {
          if (offered.contains(action.intent)) {
            final reply = await _runDirect(action, direct, onSide);
            if (!cancelled) yield reply;
            return;
          }
          debugPrint(
            '[ChatTurnResponder] ${action.intent} (${action.rule}) not run '
            'directly: no installed intent skill names it',
          );
        }
      }
      final draft = await _augment(text, image: image, offered: offered);
      if (cancelled) return;
      yield* ask(draft, historyText: text.trim());
    }

    // An audio turn: no words to route, so every skill is reached through
    // the agent's tools (an agent chat) or not at all (the mode line says
    // so).
    Stream<String> respondToAudio(Uint8List wav) async* {
      if (cancelled) return;
      final draft = await _augmentSpoken(wav, image: image);
      if (cancelled) return;
      yield* ask(draft, historyText: kSpokenQuestionPlaceholder, audioWav: wav);
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
  /// and returns what to say: the result text, or the executor's error
  /// message.
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
        return message;
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

  /// The names of the open agent chat's skills, as its system prompt lists
  /// them; empty on a plain chat.
  List<String> _skillNames() => [
    if (_conversation.hasSkills)
      for (final skill in _conversation.skills) skill.name,
  ];

  /// The app intents the installed intent skills name ([intentsOfferedBy]):
  /// the only ones a recognized request may run directly, or skip retrieval
  /// for, with tools on or off. Only skills of type intent count: an
  /// instruction-only skill that merely quotes an intent runs nothing.
  Set<String> _offeredIntents() =>
      intentsOfferedBy(_installedSkills?.call() ?? const <Skill>[]);

  /// The prompt — the question with the excerpts that cleared the gate, or
  /// the question alone — and the retrieval behind it (null without a
  /// retriever). A failed search becomes a [RetrievalOutcome.failed]
  /// retrieval and the turn goes on without excerpts.
  ///
  /// A live question the app recognizes (device facts, the time;
  /// [SkillQuestionRouter]) is not searched at all while an installed intent
  /// skill names its intent ([offered]): device questions clear the gate
  /// against the GPU docs, and with excerpts in the prompt the model can
  /// answer from them instead of calling the skill. The overlay shows it as
  /// [RetrievalOutcome.skipped]. Without such a skill the question retrieves
  /// like any other. (With a direct runner, the app's case, such a question
  /// has already run its intent before this.)
  ///
  /// An image turn on an agent chat that is not such a question ends with
  /// [PromptBuilder.directAnswerHint]. Without a hint the model tends to call
  /// `runIntent` (with an invented intent) for photo questions, an extra
  /// generation each.
  Future<_Draft> _augment(
    String question, {
    required Uint8List? image,
    required Set<String> offered,
  }) async {
    final route = switch (_skillRouter.classify(question)) {
      final route? when offered.contains(route.intent) => route,
      _ => null,
    };
    final skills = _skillNames();
    final draft = await _retrieve(question, route, skills);
    final hint = image != null && skills.isNotEmpty && route == null;
    if (!hint) return draft;
    final directAnswer = PromptBuilder.directAnswerHint(skills);
    return (
      retrieval: draft.retrieval,
      passages: draft.passages,
      prompt: (List<Passage> passages) =>
          '${draft.prompt(passages)}$directAnswer',
    );
  }

  /// An audio turn's prompt and retrieval: the recording searched as audio
  /// (gated at the audio threshold by the retriever), and the spoken
  /// question's prompt ([PromptBuilder.buildSpoken]) with the excerpts that
  /// cleared it. A failed search goes on without excerpts, like a typed one.
  Future<_Draft> _augmentSpoken(
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
    final skills = _skillNames();
    return (
      retrieval: retrieval,
      passages: retrieval?.outcome == RetrievalOutcome.used
          ? retrieval!.passages
          : const <Passage>[],
      prompt: (List<Passage> passages) => PromptBuilder.buildSpoken(
        passages,
        skills: skills,
        image: image != null,
      ),
    );
  }

  Future<_Draft> _retrieve(
    String question,
    SkillRoute? route,
    List<String> skills,
  ) async {
    final retriever = _retriever;
    String plain(List<Passage> _) => question;
    if (retriever == null) {
      return (retrieval: null, passages: const <Passage>[], prompt: plain);
    }
    if (route != null) {
      debugPrint('[ChatTurnResponder] skill question ($route): no retrieval');
      return (
        retrieval: Retrieval(
          outcome: RetrievalOutcome.skipped,
          detail: route.toString(),
        ),
        passages: const <Passage>[],
        prompt: plain,
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
    return (
      retrieval: retrieval,
      passages: retrieval.outcome == RetrievalOutcome.used
          ? retrieval.passages
          : const <Passage>[],
      // On an agent chat the excerpts never stand in for a skill.
      prompt: (List<Passage> passages) =>
          PromptBuilder.build(question, passages, skills: skills),
    );
  }

  /// [draft]'s prompt with as many of its excerpts as fit the chat model's
  /// context window beside the turn (its [image], its recording [audioWav],
  /// the reply allowance and the headroom; the conversation measures,
  /// [ConversationRepository.firstFittingPrompt]): the lowest-scoring go
  /// first, and the retrieval says how many stayed ([Retrieval.keeping]).
  /// Every excerpt stays when all fit, and also when not even the question
  /// without any fits or the fit cannot be measured: the ask then fails as
  /// it would without the fitting (too long, or why the chat cannot take the
  /// turn).
  Future<({String prompt, Retrieval? retrieval})> _fit(
    _Draft draft, {
    required Uint8List? image,
    required Uint8List? audioWav,
  }) async {
    final (:retrieval, :passages, :prompt) = draft;
    final all = prompt(passages);
    if (retrieval == null || passages.isEmpty) {
      return (prompt: all, retrieval: retrieval);
    }
    // Most excerpts first, down to none.
    final prompts = [
      for (var keep = passages.length; keep >= 0; keep--)
        prompt(passages.sublist(0, keep)),
    ];
    switch (await _conversation.firstFittingPrompt(
      prompts,
      image: image,
      audioWav: audioWav,
    )) {
      case Ok(value: 0):
        return (prompt: all, retrieval: retrieval);
      case Ok(value: final fits?):
        final keep = passages.length - fits;
        debugPrint(
          '[ChatTurnResponder] $keep of ${passages.length} excerpts fit the '
          'context window beside the turn',
        );
        return (prompt: prompts[fits], retrieval: retrieval.keeping(keep));
      case Ok(value: null):
        debugPrint(
          '[ChatTurnResponder] not even the question without excerpts fits '
          'the context window; the turn goes as it is',
        );
        return (prompt: all, retrieval: retrieval);
      case Error(:final error):
        debugPrint(
          '[ChatTurnResponder] the excerpts could not be fitted to the '
          'context window ($error); the turn goes as it is',
        );
        return (prompt: all, retrieval: retrieval);
    }
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
