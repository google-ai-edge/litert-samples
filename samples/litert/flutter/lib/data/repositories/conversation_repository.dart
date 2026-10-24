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

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart' show Skill;

import '../../config/model_catalog.dart';
import '../../domain/models/assistant_event.dart';
import '../../domain/models/chat_capabilities.dart';
import '../../domain/models/conversation_history.dart';
import '../../domain/models/skill_step.dart';
import '../../utils/result.dart';

/// The single chat on the loaded model (the model has one chat slot, the
/// `createChat` lane; `openSession` is never used). All turns go through
/// here, one at a time. Both demos share it: the entering demo opens it with
/// its [ConversationProfile], which drops the previous history.
///
/// Images: the caller passes the attached image on every turn and this
/// repository decides whether the live chat needs it sent, so a stop, a
/// failure, a reset or the budget guard never leaves a follow-up question
/// without its picture.
abstract interface class ConversationRepository {
  /// True from the start of a turn to its last event.
  ValueListenable<bool> get isGenerating;

  /// Whether a chat exists (the last [open] succeeded).
  bool get isOpen;

  /// The open chat's profile; null when no chat is open.
  ConversationProfile? get profile;

  /// Builds a fresh chat with [profile]. A running turn is stopped first, and
  /// a stopped turn that is still draining is waited for (bounded), so this
  /// never fails just because a demo was left mid-reply. Rebuilds run one at
  /// a time; a queued open that a later one replaces before it runs is not
  /// built (it fails with [ConversationSupersededException], or shares the
  /// later result when both asked for the same profile).
  ///
  /// If the draining turn does not end within the stop timeout, this fails
  /// and the previous chat stays open with its own profile; open again once
  /// the turn has ended.
  ///
  /// [skills]: with a profile that has a `skillsTemplate`, a non-empty list
  /// makes the chat an agent chat — tools `loadSkill` and `runIntent`, every
  /// skill listed in the system prompt — whose turns run through
  /// `AgentSession`. Otherwise it is ignored (Demo 3 stays plain). A new skill
  /// set needs a new chat: open again.
  Future<Result<void>> open(
    ConversationProfile profile, {
    List<Skill> skills = const [],
  });

  /// The open chat is an agent chat (opened with skills).
  bool get hasSkills;

  /// The open agent chat's skills: the ones its system prompt lists, which
  /// `loadSkill` can load. Empty for a plain chat (a chat model with tools
  /// off included) and when no chat is open.
  List<Skill> get skills;

  /// The open chat was asked for skills but is plain, because the chat model
  /// has tools off: the model cannot call any skill. The requests the app
  /// runs itself still run while an installed skill names their intent.
  bool get skillsNeedTools;

  /// What the loaded chat model allows: images, tools, and its name for the
  /// labels that say why something is off.
  ChatCapabilities get capabilities;

  /// Stops a running turn and closes the chat, but stays usable (unlike
  /// [close]): the chat model is about to be unloaded or replaced. The next
  /// [open] builds on the model loaded then.
  ///
  /// A [ConversationNotReadyException] when the stopped turn still generates
  /// after the stop timeout (a native turn that ignores the stop): then
  /// nothing is closed, and the caller must not close the chat model either.
  Future<Result<void>> release();

  /// Sends [prompt] and streams the reply. The stream ends with exactly one
  /// [AssistantDone] or [AssistantFailed]; failures, a busy chat included,
  /// arrive as [AssistantFailed], never as stream errors. Cancelling the
  /// subscription stops generation too.
  ///
  /// On an agent chat [onStep] receives each skill step as it happens
  /// (loadSkill, runIntent and its result). A tool call's JSON is never spoken:
  /// flutter_edge_ai 2.4.0 takes a call it can read out of the text, and the
  /// app drops the text of one it cannot read (`AgentTurnFold`); prose the
  /// model writes before a call is streamed and spoken. A step is not an
  /// [AssistantEvent], so code that switches over those stays unchanged. A
  /// cancelled subscription does not stop native generation on the agent path:
  /// the repository stops it and drains the turn, keeping [isGenerating] true
  /// until it has ended. Too many tool rounds end the turn as [AssistantFailed]
  /// with a [SkillLoopException], no answer after a loaded skill (after the
  /// agent loop's one retry, or with none when the generations ran out) with an
  /// [EmptyAnswerAfterSkillException].
  ///
  /// [image]: a PNG or JPEG (normally from `normalizeForLlm` or a camera
  /// snapshot). Pass the SAME object every turn while it stays attached: it
  /// is sent only when the live chat cannot see it — the first time, and
  /// again after a failure, an [open]/[reset], a budget reset or a stop on
  /// the NPU (compared with `identical`, see [imageInContext]). On the GPU and
  /// the CPU flutter_edge_ai_litertlm rewinds a stopped turn in place and
  /// keeps it, so it is not sent again; except after a stopped turn that began
  /// during the plugin's recovery from an earlier stop and produced nothing
  /// (it may never have reached the model). When the plugin falls back to a
  /// rebuild from text there (a rewind that fails, a stopped turn that does
  /// not wind down) the image is gone and this repository cannot tell: only
  /// the plugin's log says so. A new object is sent as a new image.
  ///
  /// [audioWav]: the spoken question itself, as a WAV (16 kHz mono PCM16
  /// behind a 44-byte RIFF header), for a chat model that hears audio
  /// ([ChatCapabilities.audio]). It rides on this turn's user message with
  /// [prompt] (which may be empty, or carry the excerpts) and [image]; a
  /// recording goes out once, with its turn, and is never re-sent.
  /// On a chat built without audio support the turn fails with
  /// [ConversationAudioUnsupportedException] before anything reaches the
  /// engine or its history: it never degrades to a text-only turn. A
  /// recording that is not such a WAV, or has no samples, fails the same way
  /// with a [FormatException].
  ///
  /// The engine's history holds the recording (inside `InferenceChat`'s
  /// message list) until the chat is next rebuilt or replaced: at most
  /// ~960 KB per 30 s question (16 kHz mono PCM16), and the context budget
  /// bounds how many audio turns fit between rebuilds (about six of 30 s in
  /// an 8192-token window, ≈ 6 MB). No rebuild keeps one: a rebuild replays
  /// the text history only. A stop on the GPU or the CPU is no rebuild: the
  /// native conversation is rewound in place with its recordings.
  ///
  /// [historyText]: the user's words as the conversation's text mirror
  /// records them (the question without its excerpts, or
  /// [kSpokenQuestionPlaceholder] for an audio turn); [prompt] when null.
  /// [turn]: the voice turn's number, so [setHistoryText] can put the
  /// transcript in later.
  ///
  /// Before each turn a context budget guard compares the tokens in use plus
  /// what this turn needs with the model's window (a recording counts
  /// [kAudioTokensPerSecond] per second); when it would not fit, the chat is
  /// recreated with the same profile first and given back the newest turns of
  /// the text mirror that fit (at most [kHistoryReplayMaxFraction] of the
  /// window); the older ones are dropped ([GenerationMetrics.contextReset] and
  /// [AssistantContextReset.keptTurns] say so). A chat rebuilt after an
  /// interrupted skill call, or after a stop (below), replays every turn that
  /// fits beside the new one; only turns that do not fit are dropped, and
  /// announced the same way. The guard acts before `InferenceChat`'s own trim
  /// ([kContextHeadroomTokens]), which would replay the history recordings
  /// included.
  ///
  /// After a stopped turn flutter_edge_ai_litertlm recovers the native
  /// conversation as the next turn starts. On the GPU and the CPU it rewinds
  /// it in place: the stopped turn (its message, image and recording, and the
  /// partial reply) and every earlier turn stay as they were, unless the
  /// rewind fails or the stopped turn does not wind down: the plugin then
  /// rebuilds it from text without telling the app. LiteRT-LM's token count
  /// holds a rewound exchange twice, so the budget guard errs high after a
  /// stop. On the NPU it always rebuilds it from its own text history, in
  /// which an audio turn's question is missing, so there a turn after a
  /// stopped one, while an audio turn is in the live chat, rebuilds the chat
  /// from the text mirror (`clearHistory(replayHistory:)`) instead.
  Stream<AssistantEvent> ask(
    String prompt, {
    Uint8List? image,
    Uint8List? audioWav,
    void Function(SkillStep step)? onStep,
    String? historyText,
    int? turn,
  });

  /// Which of [prompts] — one turn's prompt in versions with fewer and fewer
  /// knowledge-base excerpts, most first — fits the model's context window:
  /// the index of the first one a fresh chat could hold with the turn's
  /// [image], its recording [audioWav], the reply allowance and, on an agent
  /// chat, the skills reserve, below the window less the headroom. These are
  /// the [ask] budget guard's own figures, counted with the chat model's
  /// tokenizer, so an [ask] with that prompt is not refused as too long. Ok
  /// with null when none fits (an [ask] then fails with
  /// [ConversationTooLongException]). An error when no chat is open, a turn
  /// runs (nothing is counted beside its generation), or the recording is
  /// not a 16 kHz mono PCM16 WAV.
  Future<Result<int?>> firstFittingPrompt(
    List<String> prompts, {
    Uint8List? image,
    Uint8List? audioWav,
  });

  /// Puts [text] (a transcript) as the user's words of voice turn [turn] in
  /// the conversation's text mirror, for the next rebuild. Returns false when
  /// the mirror has no such turn (the chat was opened or reset since, or the
  /// turn never reached the model).
  bool setHistoryText(int turn, String text);

  /// The conversation's text mirror, oldest first: what a rebuilt chat
  /// replays. Empty after [open], [reset] and [release].
  List<HistoryTurn> get history;

  /// The image the live chat can see: the last one a turn sent and that has
  /// not been lost since, as far as the app can tell (a rewind the plugin
  /// silently replaced with a rebuild from text loses it unseen). Null after a
  /// failure, [open], [reset], a budget reset or a stop on the NPU.
  Uint8List? get imageInContext;

  /// Stops the running turn and completes once it has ended. No-op when idle.
  Future<void> stop();

  /// Drops the history by rebuilding the chat with [ifCurrent] — only while
  /// [ifCurrent] is still the profile most recently passed to [open], checked
  /// when the reset's turn in the open queue comes. Otherwise it does nothing
  /// and returns Ok: a late, unawaited reset from a demo the user has left
  /// can never reopen that demo's profile over the next one's.
  Future<Result<void>> reset({required ConversationProfile ifCurrent});

  /// Closes the chat, after any chat still being built: the chat model may
  /// be closed once it returns. Safe to call more than once.
  Future<void> close();
}

/// [ConversationRepository.ask] while a turn is running or before [open].
final class ConversationNotReadyException implements Exception {
  const ConversationNotReadyException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// [ConversationRepository.ask] with an image on a chat built without image
/// support (the engine would drop the image silently).
final class ConversationImageUnsupportedException implements Exception {
  const ConversationImageUnsupportedException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// [ConversationRepository.ask] with a recording on a chat built without
/// audio support (the engine would drop the audio silently, and the model
/// would answer as if nothing was said).
final class ConversationAudioUnsupportedException implements Exception {
  const ConversationAudioUnsupportedException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// The prompt (with its image and the reply allowance) does not fit the
/// model's context window even in a fresh chat.
final class ConversationTooLongException implements Exception {
  const ConversationTooLongException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// An agent turn used up its generations without a plain answer (the model
/// kept calling tools).
final class SkillLoopException implements Exception {
  const SkillLoopException(this.iterations);

  final int iterations;

  @override
  String toString() =>
      'The model kept calling skills for $iterations rounds without '
      'answering';
}

/// An agent turn that ended with no answer right after the model loaded
/// [skillName] (flutter_edge_ai_agent's `EmptyAnswerEvent`): the generation
/// was empty, and so was the one the agent loop asked for again, or there was
/// none because `maxIterations` left no generation for a retry. A failure
/// the user sees, like any empty reply, not an empty bubble.
final class EmptyAnswerAfterSkillException implements Exception {
  const EmptyAnswerAfterSkillException(this.skillName);

  final String skillName;

  @override
  String toString() =>
      'the model returned an empty reply after loading the skill '
      '"$skillName"';
}

/// A queued [ConversationRepository.open] that a later open for another
/// profile replaced before it ran.
final class ConversationSupersededException implements Exception {
  const ConversationSupersededException(this.message);

  final String message;

  @override
  String toString() => message;
}
