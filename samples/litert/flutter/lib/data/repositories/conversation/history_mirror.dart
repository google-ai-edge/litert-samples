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

import '../../../domain/models/conversation_history.dart';

/// Where one turn sits in a [HistoryMirror] ([HistoryMirror.begin]).
typedef MirrorSlot = ({int index, int generation});

/// The text mirror of the open chat's conversation: per turn the user's words
/// (the question, never the prompt's excerpts; [kSpokenQuestionPlaceholder]
/// for an audio turn until its transcript arrives) and the reply as it
/// streamed. It holds text only: no image and no recording outlives the turn
/// that sent it here.
///
/// Why it exists: flutter_edge_ai_litertlm (its FFI session) rebuilds the
/// native conversation from the turns' text after a stopped turn, and an
/// audio turn has no text (its prompt carries at most the excerpts), so that
/// rebuild would replay a question nobody can read. A chat that must be
/// rebuilt — after a stop when an audio turn is in it, after a budget reset,
/// after an interrupted skill call — is rebuilt from this mirror instead
/// ([replayText]).
final class HistoryMirror {
  final List<HistoryTurn> _turns = [];

  /// An audio turn reached the live native conversation since it was last
  /// built or rebuilt: a stop now must be followed by a rebuild from here.
  bool audioInLive = false;

  /// The voice turns the live native conversation holds only as
  /// [kSpokenQuestionPlaceholder]: replayed ([replayed]) before their
  /// transcripts arrived.
  final Set<int> _wordlessInLive = {};

  /// A transcript arrived for a spoken question the live native conversation
  /// holds only as the placeholder ([wordlessInLive]): the model would never
  /// see those words, so the next turn rebuilds the chat from here.
  bool replayOutdated = false;

  /// The voice turns the live chat holds only as the placeholder, oldest
  /// first.
  List<int> get wordlessInLive => List.unmodifiable(_wordlessInLive);

  /// Every turn, oldest first.
  List<HistoryTurn> get turns => List.unmodifiable(_turns);

  bool get isEmpty => _turns.isEmpty;

  /// A sent turn: [user] as the history says it, [turn] its voice turn's
  /// number, [audio] whether a recording went with it. Returns the slot
  /// [reply] fills.
  MirrorSlot begin({required String user, int? turn, bool audio = false}) {
    if (audio) audioInLive = true;
    _turns.add(
      HistoryTurn(user: user, assistant: '', turn: turn, audio: audio),
    );
    return (index: _turns.length - 1, generation: _generation);
  }

  /// The reply of the turn in [slot], as far as it got. Ignored when the
  /// mirror was cleared since (the chat was reopened).
  void reply(MirrorSlot slot, String text) {
    final (:index, :generation) = slot;
    if (generation != _generation || index >= _turns.length) return;
    final t = _turns[index];
    _turns[index] = HistoryTurn(
      user: t.user,
      assistant: text,
      turn: t.turn,
      audio: t.audio,
    );
  }

  /// Replaces the user's words of voice turn [turn] (its transcript arrived).
  /// False when no entry has that number (the chat was reopened or reset
  /// since, or the turn never reached the model). Turn numbers are never
  /// reused (`TurnRequest.turn`), so a turn that never reached the model
  /// cannot land on an entry another assistant's turn made in this chat.
  /// When the live chat holds that turn only as the placeholder, the replay
  /// is [replayOutdated].
  bool setUser(int turn, String text) {
    for (var i = _turns.length - 1; i >= 0; i--) {
      final t = _turns[i];
      if (t.turn != turn) continue;
      _turns[i] = HistoryTurn(
        user: text,
        assistant: t.assistant,
        turn: t.turn,
        audio: t.audio,
      );
      if (_wordlessInLive.remove(turn)) replayOutdated = true;
      return true;
    }
    return false;
  }

  /// The live native conversation was given [turns] as text (a rebuild or a
  /// fresh chat's replay, planned before it was sent): it holds no
  /// recording, and the spoken questions among them that had no words are
  /// what a later transcript makes outdated. One whose words arrived while
  /// the replay was planned and sent (the mirror has them now) made it
  /// outdated already.
  void replayed(List<HistoryTurn> turns) {
    audioInLive = false;
    replayOutdated = false;
    _wordlessInLive.clear();
    for (final t in turns) {
      final turn = t.turn;
      if (!t.wordless || turn == null) continue;
      final now = _turns.where((m) => m.turn == turn).lastOrNull;
      if (now != null && !now.wordless) {
        replayOutdated = true;
      } else {
        _wordlessInLive.add(turn);
      }
    }
  }

  /// The live native conversation is a fresh one (recreated): it holds
  /// nothing of this history until a replay.
  void forgetLive() {
    audioInLive = false;
    replayOutdated = false;
    _wordlessInLive.clear();
  }

  /// Bumped by [clear]: a turn that began before it must not write into the
  /// new history.
  int _generation = 0;

  /// A new conversation: nothing to replay.
  void clear() {
    _turns.clear();
    forgetLive();
    _generation++;
  }

  /// The newest replayable turns, oldest first, whose text fits [maxTokens]
  /// by [tokens] (the model's tokenizer). [dropped]: the replayable turns
  /// left out, oldest ones.
  Future<({List<HistoryTurn> turns, int tokens, int dropped})> newestThatFit(
    int maxTokens,
    Future<int> Function(String text) tokens,
  ) async {
    final replayable = [
      for (final t in _turns)
        if (t.replayable) t,
    ];
    if (replayable.isEmpty || maxTokens <= 0) {
      return (
        turns: const <HistoryTurn>[],
        tokens: 0,
        dropped: replayable.length,
      );
    }
    var used = await tokens(replayHeader);
    final kept = <HistoryTurn>[];
    for (final t in replayable.reversed) {
      final cost = await tokens(_lines(t));
      if (used + cost > maxTokens) break;
      used += cost;
      kept.insert(0, t);
    }
    if (kept.isEmpty) {
      return (
        turns: const <HistoryTurn>[],
        tokens: 0,
        dropped: replayable.length,
      );
    }
    return (
      turns: List<HistoryTurn>.unmodifiable(kept),
      tokens: used,
      dropped: replayable.length - kept.length,
    );
  }

  /// What a rebuilt chat is given: one user message, staged before the next
  /// turn's own (flutter_edge_ai_litertlm's FFI session sends what is staged
  /// as one message), so it ends
  /// with a blank line.
  static String replayText(List<HistoryTurn> turns) {
    final text = StringBuffer(replayHeader);
    for (final t in turns) {
      text.write(_lines(t));
    }
    text.write('\n');
    return text.toString();
  }

  /// The replay's first line.
  static const replayHeader = 'The conversation so far, as text:\n';

  static String _lines(HistoryTurn t) =>
      'User: ${_oneLine(t.user)}\nAssistant: ${_oneLine(t.assistant)}\n';

  static String _oneLine(String text) =>
      text.trim().replaceAll(RegExp(r'\s*\n\s*'), ' ');
}
