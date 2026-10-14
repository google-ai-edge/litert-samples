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
import '../skills/app_intents.dart';

/// Builds a Demo 1 turn's prompt from the question and the excerpts that
/// cleared the gate, and reads the citations back out of the
/// reply. One place owns the `[n]` format both ways.
abstract final class PromptBuilder {
  /// On an agent chat, the instruction for a turn with excerpts,
  /// worded like the agent template ("answer … directly with no tool call").
  /// Live skill questions never get excerpts (`SkillQuestionRouter` skips
  /// retrieval for them), so a turn that has some is a knowledge question:
  /// without this line one was pulled into a `runIntent` call, a ~2.8 s
  /// detour with a failed step on screen. A per-turn tool restriction does
  /// not exist: `InferenceChat.tools` and `toolChoice`
  /// are fixed at creation (final fields of flutter_edge_ai 2.1.0's
  /// `InferenceChat`, `lib/core/chat.dart`).
  static const noToolCall = 'Answer this question directly with no tool call. ';

  /// Appended to an image turn on an agent chat that is not a
  /// live skill question.
  static const directAnswerHint =
      '\n\n(A picture is attached. Answer directly; no tool is needed.)';

  /// [question] unchanged without [passages]; otherwise the excerpts
  /// numbered from 1, the instruction, then the question.
  ///
  /// [skills]: the chat has agent skills, so the excerpts are framed
  /// as reference documentation and the turn is told it needs no tool call
  /// ([noToolCall]).
  static String build(
    String question,
    List<Passage> passages, {
    bool skills = false,
  }) {
    if (passages.isEmpty) return question;
    final prompt = StringBuffer(
      skills
          ? 'Reference excerpts from the knowledge base (general '
                'documentation):\n\n'
          : 'Excerpts from the knowledge base:\n\n',
    );
    for (var i = 0; i < passages.length; i++) {
      prompt.write('[${i + 1}] ${passages[i].content}\n\n');
    }
    prompt
      ..write(
        '${skills ? noToolCall : 'Answer the question. '}Use the excerpts '
        'only if they are relevant and cite each one you use as '
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
  /// the instruction. On an agent chat ([skills]) a spoken question cannot be
  /// routed to a skill before the reply (there are no words yet), so the
  /// excerpts never forbid a tool call: a question about live device facts or
  /// the time is told which intent answers it instead of them
  /// ([spokenExcerptSkills]). [image]: a picture goes with the turn
  /// ([spokenImageHint] on an agent chat).
  static String buildSpoken(
    List<Passage> passages, {
    bool skills = false,
    bool image = false,
  }) {
    final hint = skills && image ? spokenImageHint : '';
    if (passages.isEmpty) {
      return '${skills ? spokenQuestionWithSkills : spokenQuestion}$hint';
    }
    final prompt = StringBuffer(
      skills
          ? 'Reference excerpts from the knowledge base (general '
                'documentation):\n\n'
          : 'Excerpts from the knowledge base:\n\n',
    );
    for (var i = 0; i < passages.length; i++) {
      prompt.write('[${i + 1}] ${passages[i].content}\n\n');
    }
    prompt
      ..write('The question is in this audio. ')
      ..write(skills ? spokenExcerptSkills : 'Answer it. ')
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

  /// The instruction of a spoken question with excerpts on an agent chat,
  /// after "The question is in this audio.": it names the intent that
  /// answers a device or time question, because live device questions clear
  /// the audio gate against the accelerator, backend and memory docs
  /// (0.72–0.79 against 0.69; a KB question 0.77), so most of them reach the
  /// model with excerpts.
  ///
  /// Measured 2026-10-14 (audio-prompt probe, macOS, Gemma 4 E2B on the GPU,
  /// agent chat with current-time and device-info, questions spoken by the
  /// app's Inflect voice, a fresh chat per turn) on the questions that clear
  /// the gate: the YOLO26n input size (answered from the excerpts, cited, no
  /// call) and five device questions (device_info called):
  ///
  /// | instruction | KB ok | device ok | turns with a wrong call |
  /// |---|---|---|---|
  /// | "… live facts about this device or app, or for the time, use your skills instead of the excerpts" (before) | 0/3 | 7/15 | 8/18 |
  /// | the same, the time named first | 2/3 | 3/15 | 7/18 |
  /// | skills named ("use the device-info skill …") | 2/7 | 24/35 | 6/42 |
  /// | device-info skill named, no time | 1/4 | 14/20 | 4/24 |
  /// | this one (intents named) | 3/4 | 17/20 | 1/24 |
  ///
  /// The old wording sent device questions to current_time and added a
  /// current-time call to the KB answer. Left: 3 of 20 device turns load the
  /// skill and then end with an empty reply (an agent-loop issue, seen with
  /// every wording), and one KB turn still called current_time.
  static const spokenExcerptSkills =
      'If it asks about this device or app itself (its models, backends, '
      'accelerators or memory), call runIntent with ${AppIntent.deviceInfo} '
      'instead of using the excerpts; if it asks for the time, call runIntent '
      'with ${AppIntent.currentTime}. Otherwise answer it directly with no '
      'tool call. ';

  /// A spoken question's whole prompt when no excerpt cleared the gate.
  static const spokenQuestion = 'Answer the question asked in this audio.';

  /// [spokenQuestion] on an agent chat: it says the request may be a task as
  /// well as a question, names the only requests that take a tool, and says
  /// that everything else takes none.
  ///
  /// Measured 2026-10-14 (audio-prompt probe, macOS, Gemma 4 E2B on the GPU,
  /// agent chat with current-time and device-info, questions spoken by the
  /// app's Inflect voice, a fresh chat per turn). Turns with a spurious tool
  /// call among the story, France, "how are you" and joke requests (3 rounds
  /// each), refusals without a call ("I can't tell jokes"), and the time and
  /// device questions:
  ///
  /// | wording | spurious | refusals | time / device |
  /// |---|---|---|---|
  /// | "Answer the question asked in this audio. Use a skill only if the question needs one." (before) | 4/12 | 1/12 | 3/3 / – |
  /// | "Answer the question asked in this audio." | 5/12 | 5/12 | 3/3 / – |
  /// | "Respond to what the user says in this audio." | 5/12 | 3/12 | 3/3 / – |
  /// | "… If it asks for the time or for live facts about this device or app, use your skills; otherwise answer it directly with no tool call." | 3/12 | 1/12 | 3/3 / – |
  /// | "Respond to … Call a tool only if they ask for the time or for live facts about this device or app; otherwise reply directly with no tool call." (2 runs) | 0/24 | 5/24 | 6/6 / 3/3 |
  /// | "Do what the user asks …, or answer their question. Use a skill only if …" | 0/12 | 3/12 | 3/3 / 3/3 |
  /// | skills named, "Respond to …" form | 3/12 | 1/12 | 3/3 / 3/3 |
  /// | this one | 0/12 | 2/12 | 3/3 / 3/3 |
  ///
  /// The old line made Gemma call `runIntent` with invented intents
  /// (`storytelling`, `geographical_capital`) or current_time for a story,
  /// a generation each (~1–3 s before the first word). What is left are
  /// refusals without a call: one story and one joke in 12 turns.
  static const spokenQuestionWithSkills =
      'Answer the question asked in this audio, or do what it asks. Use '
      'current-time only for the time and device-info only for this device or '
      'app (models, backends, accelerators, memory); for anything else reply '
      'directly with no tool call.';

  /// Appended to a spoken question with a picture on an agent chat: unlike
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
