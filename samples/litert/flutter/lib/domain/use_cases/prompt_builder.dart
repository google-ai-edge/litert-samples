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

import '../../utils/citations.dart';
import '../models/knowledge.dart';

/// Builds a Demo 1 turn's prompt from the question and the excerpts that
/// cleared the gate, and reads the citations back out of the
/// reply. One place owns the `[n]` format both ways.
///
/// On an agent chat the turn hints name the chat's skills, taken from the
/// skills its system prompt lists (`skills`: their names), and nothing else:
/// no intent, no topic, no tool to call. A skill the app has never seen
/// (dropped into the skills folder at run time) is worded exactly like a
/// bundled one, and a skill is always reached through `loadSkill` and its
/// own instructions.
abstract final class PromptBuilder {
  /// ` (a, b, c)`: the skill names for a hint; empty without any.
  static String _named(List<String> skills) =>
      skills.isEmpty ? '' : ' (${skills.join(', ')})';

  /// On an agent chat, the instruction of a turn with excerpts, typed or
  /// spoken (after "The question is in this audio."): a request one of
  /// [skills] serves goes to that skill, anything else is answered from the
  /// excerpts without a tool call. A per-turn tool restriction does not
  /// exist: `InferenceChat.tools` and `toolChoice` are fixed at creation
  /// (in flutter_edge_ai 2.4.0's `InferenceChat`, `lib/src/chat.dart`,
  /// `toolChoice` is a final field and `tools` a fixed view of the final
  /// `_tools`).
  ///
  /// This wording is new and not measured yet (a probe follows). It replaces
  /// two that named things: the typed "Answer this question directly with no
  /// tool call." (fine while every skill question skipped retrieval; it
  /// silenced a run-time skill whose request cleared the gate) and the spoken
  /// one that named the intents. Their history:
  ///
  /// Typed: without an instruction a knowledge question was pulled into a
  /// `runIntent` call, a ~2.8 s detour with a failed step on screen.
  ///
  /// Spoken, measured 2026-10-14 (audio-prompt probe, macOS, Gemma 4 E2B on
  /// the GPU, agent chat with current-time and device-info, questions spoken
  /// by the app's Inflect voice, a fresh chat per turn) on the questions that
  /// clear the audio gate (live device questions clear it against the
  /// accelerator, backend and memory docs at 0.72–0.79 against 0.69; a KB
  /// question 0.77): the YOLO26n input size (answered from the excerpts,
  /// cited, no call) and five device questions (device_info called):
  ///
  /// | instruction | KB ok | device ok | turns with a wrong call |
  /// |---|---|---|---|
  /// | "… live facts about this device or app, or for the time, use your skills instead of the excerpts" | 0/3 | 7/15 | 8/18 |
  /// | the same, the time named first | 2/3 | 3/15 | 7/18 |
  /// | skills named ("use the device-info skill …") | 2/7 | 24/35 | 6/42 |
  /// | device-info skill named, no time | 1/4 | 14/20 | 4/24 |
  /// | intents named ("call runIntent with device_info … current_time"; replaced) | 3/4 | 17/20 | 1/24 |
  ///
  /// Naming the skills helped against the generic "use your skills", so this
  /// one names them too, from the chat's own list. Empty replies came with
  /// every wording: 1 of 15, 3 of 15, 9 of 35, 3 of 20 and 3 of 20 device
  /// turns ended with an empty reply, 18 of the 19 right after a `loadSkill`
  /// (three of those loaded the current-time skill).
  static String excerptSkills(List<String> skills) =>
      'If the request matches one of your skills${_named(skills)}, load it '
      'and follow it instead of the excerpts; otherwise answer from the '
      'excerpts with no tool call. ';

  /// Appended to an image turn on an agent chat that is not a question the
  /// app recognizes: the picture alone takes no tool, a request one of
  /// [skills] serves still does.
  ///
  /// New wording, not measured yet. It replaces "(A picture is attached.
  /// Answer directly; no tool is needed.)", which silenced every skill on a
  /// picture turn; without any hint Gemma called `runIntent` (an invented
  /// intent, or the camera-watch skill's, since removed) for photo questions
  /// in 5 of 6 first turns.
  static String directAnswerHint(List<String> skills) =>
      '\n\n(A picture is attached. Use one of your skills${_named(skills)} '
      'only when the request matches its description; otherwise answer about '
      'the picture directly with no tool call.)';

  /// [question] unchanged without [passages]; otherwise the excerpts
  /// numbered from 1, the instruction, then the question.
  ///
  /// [skills]: the names of the agent chat's skills (empty: a plain chat).
  /// With some, the excerpts are framed as reference documentation and the
  /// skills come first ([excerptSkills]).
  static String build(
    String question,
    List<Passage> passages, {
    List<String> skills = const [],
  }) {
    if (passages.isEmpty) return question;
    final agent = skills.isNotEmpty;
    final prompt = StringBuffer(
      agent
          ? 'Reference excerpts from the knowledge base (general '
                'documentation):\n\n'
          : 'Excerpts from the knowledge base:\n\n',
    );
    for (var i = 0; i < passages.length; i++) {
      prompt.write('[${i + 1}] ${passages[i].content}\n\n');
    }
    prompt
      ..write(
        '${agent ? excerptSkills(skills) : 'Answer the question. '}Use the '
        'excerpts only if they are relevant and cite each one you use as '
        '${_markers(passages.length)}.\n',
      )
      ..write(
        'If they do not contain the answer, say so briefly and answer '
        'without citations.\n\n',
      )
      ..write('Question: $question');
    return prompt.toString();
  }

  /// A spoken question's prompt (audio mode): the question is the recording
  /// that goes with it, so the prompt carries only the excerpts and the
  /// instruction, never empty (a rebuilt chat replays text, and this is the
  /// text an audio turn has).
  ///
  /// Without [passages]: [spokenQuestion] (on an agent chat
  /// [spokenQuestionWithSkills]). With them: the excerpts numbered from 1, then
  /// the instruction. On an agent chat ([skills]: the names of its skills) a
  /// spoken question cannot be routed before the reply (there are no words
  /// yet), so the excerpts never forbid a skill ([excerptSkills]). [image]: a
  /// picture goes with the turn ([spokenImageHint] on an agent chat).
  static String buildSpoken(
    List<Passage> passages, {
    List<String> skills = const [],
    bool image = false,
  }) {
    final agent = skills.isNotEmpty;
    final hint = agent && image ? spokenImageHint : '';
    if (passages.isEmpty) {
      return '${agent ? spokenQuestionWithSkills(skills) : spokenQuestion}'
          '$hint';
    }
    final prompt = StringBuffer(
      agent
          ? 'Reference excerpts from the knowledge base (general '
                'documentation):\n\n'
          : 'Excerpts from the knowledge base:\n\n',
    );
    for (var i = 0; i < passages.length; i++) {
      prompt.write('[${i + 1}] ${passages[i].content}\n\n');
    }
    prompt
      ..write('The question is in this audio. ')
      ..write(agent ? excerptSkills(skills) : 'Answer it. ')
      ..write(
        'Use the excerpts only if they are relevant and cite each one you use '
        'as ${_markers(passages.length)}.\n',
      )
      ..write(
        'If they do not contain the answer, say so briefly and answer '
        'without citations.',
      )
      ..write(hint);
    return prompt.toString();
  }

  /// A spoken question's whole prompt when no excerpt cleared the gate.
  static const spokenQuestion = 'Answer the question asked in this audio.';

  /// [spokenQuestion] on an agent chat: the request may be a task as well as
  /// a question, one of [skills] is used only when the request matches its
  /// description, and everything else takes no tool.
  ///
  /// This wording is new and not measured yet (a probe follows): the one it
  /// replaces named the bundled skills and what each was for ("Use
  /// current-time only for the time and device-info only for this device or
  /// app …"), which kept a run-time skill out. Measured 2026-10-14
  /// (audio-prompt probe, macOS, Gemma 4 E2B on the GPU, agent chat with
  /// current-time and device-info, questions spoken by the app's Inflect
  /// voice, a fresh chat per turn). Turns with a spurious tool call among
  /// the story, France, "how are you" and joke requests (3 rounds each),
  /// refusals without a call ("I can't tell jokes"), and the time and device
  /// questions:
  ///
  /// | wording | spurious | refusals | time / device |
  /// |---|---|---|---|
  /// | "Answer the question asked in this audio. Use a skill only if the question needs one." | 4/12 | 1/12 | 3/3 / – |
  /// | "Answer the question asked in this audio." | 5/12 | 5/12 | 3/3 / – |
  /// | "Respond to what the user says in this audio." | 5/12 | 3/12 | 3/3 / – |
  /// | "… If it asks for the time or for live facts about this device or app, use your skills; otherwise answer it directly with no tool call." | 3/12 | 1/12 | 3/3 / – |
  /// | "Respond to … Call a tool only if they ask for the time or for live facts about this device or app; otherwise reply directly with no tool call." (2 runs) | 0/24 | 5/24 | 6/6 / 3/3 |
  /// | "Do what the user asks …, or answer their question. Use a skill only if …" | 0/12 | 3/12 | 3/3 / 3/3 |
  /// | skills named, "Respond to …" form | 3/12 | 1/12 | 3/3 / 3/3 |
  /// | "… Use current-time only for the time and device-info only for this device or app …" (replaced) | 0/12 | 2/12 | 3/3 / 3/3 |
  ///
  /// The generic "Use a skill only if …" made Gemma call `runIntent` with
  /// invented intents (`storytelling`, `geographical_capital`) for a story,
  /// a generation each (~1–3 s before the first word); what kept them out
  /// was saying when a skill applies and that everything else takes no tool.
  /// This one says that against each skill's own description.
  static String spokenQuestionWithSkills(List<String> skills) =>
      'Answer the question asked in this audio, or do what it asks. Use one '
      'of your skills${_named(skills)} only when the request matches its '
      'description; otherwise reply directly with no tool call.';

  /// Appended to a spoken question with a picture on an agent chat: like
  /// [directAnswerHint] it leaves the skills open, because the question has
  /// not been routed.
  static const spokenImageHint =
      '\n\n(A picture is attached. Unless the question needs a skill, answer '
      'about it directly with no tool call.)';

  /// The excerpt numbers [reply] cites: `[2]`, `[1, 3]` and ranges such as
  /// `[1-3]` count when every number in the group is in 1..[count] and the
  /// group is not inside backticks (a tensor shape such as `[1, 3, 640,
  /// 640]` is not a citation; see [findCitations]).
  static Set<int> citedNumbers(String reply, int count) => {
    for (final marker in findCitations(reply, count: count)) ...marker.numbers,
  };

  /// `[1]`, `[1] or [2]`, `[1], [2] or [3]`: never invites a number that is
  /// not in the prompt.
  static String _markers(int count) {
    final markers = [for (var i = 1; i <= count; i++) '[$i]'];
    if (markers.length == 1) return markers.single;
    return '${markers.sublist(0, markers.length - 1).join(', ')} or '
        '${markers.last}';
  }
}
