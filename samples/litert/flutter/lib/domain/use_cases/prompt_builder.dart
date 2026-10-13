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
