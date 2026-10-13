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

import '../../../../config/model_catalog.dart' show ConversationProfile;
import '../../../../data/repositories/conversation_repository.dart';
import '../../../../data/repositories/skill_repository.dart';
import '../../../../domain/models/skill_catalog.dart';
import '../../../../utils/result.dart';

/// What the last Reload of the skills found (the Skills sheet shows it).
enum SkillsReload {
  /// The files are as the chat already has them.
  unchanged,

  /// Changed; the chat is being re-opened with them.
  applying,

  /// Changed; applied after the running turn.
  pending,

  /// Changed and applied.
  applied,

  /// Changed, but re-opening the chat failed (Reload tries again).
  failed,
}

/// How [SkillsApplier] reaches the screen: the apply runs as the screen's
/// own command, so the screen shows its progress and failure whoever starts
/// it (a Reload, a rescan, the end of a turn).
abstract interface class SkillsApplyCommands {
  /// The chat is being opened (entering, New conversation, an apply) or the
  /// recognizer switched.
  bool get openingChat;

  /// A turn or a capture is in flight.
  bool get turnRunning;

  /// The apply runs.
  bool get applying;

  /// Runs [SkillsApplier.apply] as the screen's command.
  Future<void> apply();
}

/// Demo 1's runtime skills, applied to its chat: the chat is
/// opened with the latest scan's skills, and re-opened with a changed set —
/// only between turns, when nothing else is opening it or switching the
/// recognizer, because re-opening drops the history (and a re-open and a
/// model load at the same moment would both slow the first turn).
///
/// Every open of the screen's chat goes through [open] or [apply], so the
/// rules below are kept here only:
///
/// - A set whose apply (or open) failed is not retried by itself; a Reload
///   retries it, and so does New conversation, which opens with the latest
///   set.
/// - Another demo's chat is never replaced; with no chat at all (a failed
///   open) the apply is the retry.
/// - A change that arrives while the chat is busy is applied once
///   [commandsChanged] or [turnChanged] finds it idle.
///
/// Owned by the screen's view model, which [close]s it when it leaves.
final class SkillsApplier {
  /// [profile]: the screen's chat profile; a chat open with another one is
  /// another demo's.
  SkillsApplier({
    required this._skills,
    required this._conversation,
    required this._profile,
    required this._commands,
    required this._onChanged,
  }) {
    _skills.catalog.addListener(_onCatalog);
  }

  final SkillRepository _skills;
  final ConversationRepository _conversation;
  final ConversationProfile _profile;
  final SkillsApplyCommands _commands;

  /// [pending] or [lastReload] changed.
  final void Function() _onChanged;

  /// The set the open chat was built with ([SkillCatalog.fingerprint]), and
  /// one whose apply failed (not retried until Reload).
  String? _applied;
  String? _failed;
  SkillsReload? _lastReload;
  bool _closed = false;

  /// The latest skills scan; null before the first one.
  SkillCatalog? get catalog => _skills.catalog.value;

  /// A scan changed the skills and the chat has not been re-opened with
  /// them yet (it waits for the running turn).
  bool get pending {
    final catalog = this.catalog;
    return catalog != null &&
        catalog.fingerprint != _applied &&
        catalog.fingerprint != _failed;
  }

  /// What the last Reload found; null before the first.
  SkillsReload? get lastReload => _lastReload;

  /// Opens the screen's chat (entering, New conversation) with the skills
  /// and records the outcome like [apply]: the set is applied, or failed
  /// (not re-applied by itself; New conversation or Reload retries).
  ///
  /// [rescan]: rescans the skills directory first (entering picks up
  /// dropped-in skills); otherwise the latest scan's skills, none before
  /// the first scan. Closed before the open, it opens nothing.
  Future<Result<void>> open({required bool rescan}) async {
    final catalog = rescan ? await _skills.refresh() : this.catalog;
    // Left (while scanning): a late open would replace the chat the next
    // demo has opened meanwhile.
    if (_closed) return const Result.ok(null);
    return _openWith(catalog);
  }

  /// The Skills sheet's Reload: rescans the skills directory. A changed set
  /// is applied at once when idle, otherwise after the running turn;
  /// unchanged files leave the conversation alone.
  Future<Result<void>> reload() async {
    final catalog = await _skills.refresh();
    if (_closed) return const Result.ok(null);
    // An explicit Reload retries a skill set whose apply failed.
    _failed = null;
    final changed = catalog.fingerprint != _applied;
    maybeApply();
    _lastReload = !changed
        ? SkillsReload.unchanged
        : _commands.applying
        ? SkillsReload.applying
        : SkillsReload.pending;
    _onChanged();
    return const Result.ok(null);
  }

  /// Re-opens the chat with the latest scan's skills. Returns the set
  /// applied (for the chat's notice), null when there was no scan yet, or
  /// why the chat could not be re-opened.
  Future<Result<SkillCatalog?>> apply() async {
    final catalog = this.catalog;
    if (catalog == null) return const Result.ok(null);
    final result = await _openWith(catalog);
    if (_closed) {
      return switch (result) {
        Ok() => const Result.ok(null),
        Error(:final error) => Result.error(error),
      };
    }
    final reloaded =
        _lastReload == SkillsReload.pending ||
        _lastReload == SkillsReload.applying;
    switch (result) {
      case Ok():
        if (reloaded) _lastReload = SkillsReload.applied;
        return Result.ok(catalog);
      case Error(:final error):
        if (reloaded) _lastReload = SkillsReload.failed;
        return Result.error(error);
    }
  }

  /// The one place the screen's chat is opened and the outcome recorded:
  /// [catalog]'s set is applied, or failed (a failed set is not pending, so
  /// nothing retries it by itself). Closed meanwhile, it records nothing.
  Future<Result<void>> _openWith(SkillCatalog? catalog) async {
    final result = await _conversation.open(
      _profile,
      skills: catalog?.agentSkills ?? const [],
    );
    if (_closed) return result;
    switch (result) {
      case Ok():
        _applied = catalog?.fingerprint;
        _failed = null;
      case Error():
        _failed = catalog?.fingerprint;
    }
    return result;
  }

  /// Applies a pending set now, unless the chat is busy: being opened, the
  /// recognizer switching, a turn running, or another demo's chat open.
  /// Busy, it only tells the screen (the set shows as pending).
  void maybeApply() {
    if (_closed || !pending) return;
    final profile = _conversation.profile;
    final busy =
        _commands.openingChat ||
        _commands.turnRunning ||
        (profile != null && profile != _profile);
    if (busy) {
      _onChanged();
      return;
    }
    unawaited(_commands.apply());
  }

  /// One of the commands that open the chat, or the recognizer switch,
  /// started or ended: once none runs, a set that changed meanwhile is
  /// applied.
  void commandsChanged() {
    if (!_commands.openingChat) maybeApply();
  }

  /// A turn started or ended: once it has ended, a set that changed during
  /// it is applied.
  void turnChanged() {
    if (!_commands.turnRunning) maybeApply();
  }

  void _onCatalog() {
    maybeApply();
    _onChanged();
  }

  /// The screen left: nothing is applied any more, and an apply in flight
  /// records nothing.
  void close() {
    if (_closed) return;
    _closed = true;
    _skills.catalog.removeListener(_onCatalog);
  }
}
