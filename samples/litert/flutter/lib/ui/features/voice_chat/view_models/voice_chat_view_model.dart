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

import '../../../../config/demos.dart';
import '../../../../config/model_catalog.dart';
import '../../../../data/repositories/audio_repository.dart'
    show MicAccessException;
import '../../../../data/repositories/conversation_repository.dart';
import '../../../../data/repositories/diagnostics_repository.dart';
import '../../../../data/repositories/image_repository.dart';
import '../../../../data/repositories/skill_repository.dart';
import '../../../../domain/models/chat_entry.dart';
import '../../../../domain/models/chat_side_event.dart';
import '../../../../domain/models/llm_image.dart';
import '../../../../domain/models/model_state.dart' show SttActivation;
import '../../../../domain/models/skill_catalog.dart';
import '../../../../domain/models/skill_step.dart';
import '../../../../domain/models/voice.dart';
import '../../../../domain/use_cases/voice_assistant.dart';
import '../../../../utils/command.dart';
import '../../../../utils/result.dart';
import 'skills_applier.dart';
import 'voice_chat_reducer.dart';

export 'skills_applier.dart' show SkillsReload;
export 'voice_chat_reducer.dart'
    show
        contextResetText,
        interruptedSkillResetText,
        skillAfterInterruptionText;

/// Demo 1's chat screen state: push-to-talk and typed turns, both through
/// the [VoiceAssistant], with an optional picture.
///
/// The picture is sticky: once attached it goes with every turn until it is
/// removed or a new conversation starts, so follow-up questions are about it
/// too. The conversation re-sends it only when the live chat cannot see it.
///
/// Rate discipline: [notifyListeners] fires on commits, phase changes and
/// command state changes. Tokens go to [partialReply] (the streaming bubble)
/// and the mic level to [inputLevel] (the meter's painter).
class VoiceChatViewModel extends ChangeNotifier {
  /// [skills]: the runtime skills the chat is opened with.
  VoiceChatViewModel({
    required this._conversation,
    required this._diagnostics,
    required this._images,
    required this._assistant,
    required this._activateStt,
    required this._skills,
    this._accessSettleCap = const Duration(seconds: 3),
    this._foreground,
  }) {
    open = Command0<void>(_open);
    selectStt = Command0<void>(_selectStt);
    send = Command1<void, String>(_send);
    stop = Command0<void>(_stop);
    newConversation = Command0<void>(_newConversation);
    attach = Command1<void, ImageSourceKind>(_attach);
    reloadSkills = Command0<void>(() => _applier.reload());
    applySkills = Command0<void>(_applySkills);
    requestAccess = Command0<void>(_requestAccess);
    for (final command in _commands) {
      command.addListener(_notify);
    }
    // A skill set that changed while the chat was being (re)opened, or
    // while the recognizer was being switched, is applied once that has
    // finished.
    for (final command in [open, newConversation, applySkills, selectStt]) {
      command.addListener(_onOpenCommand);
    }
    _events = _assistant.events.listen(_onEvent);
    _assistant.phase.addListener(_notify);
    _assistant.phase.addListener(_onPhase);
    _foreground?.addListener(_onForeground);
    // It applies a rescan that changed the skills.
    _applier = SkillsApplier(
      skills: _skills,
      conversation: _conversation,
      profile: kVoiceChatProfile,
      commands: _ViewModelSkillsCommands(this),
      onChanged: _notify,
    );
    // The chat and the recognizer switch (back to Whisper) in parallel.
    unawaited(open.execute());
    unawaited(selectStt.execute());
    // The first question must not pay for the audio device start.
    unawaited(_assistant.prepareAudio());
    // The microphone dialog now, not inside the first turn.
    // The chat is not ready until it is settled (at most the cap).
    _accessCap = Timer(_accessSettleCap, _settleAccess);
    unawaited(requestAccess.execute().whenComplete(_settleAccess));
  }

  final ConversationRepository _conversation;
  final DiagnosticsRepository _diagnostics;
  final ImageRepository _images;

  /// Owned: disposed with this view model.
  final VoiceAssistant<ChatSideEvent> _assistant;
  final SttActivation _activateStt;
  final SkillRepository _skills;

  /// How long entering waits at most for the access requests.
  final Duration _accessSettleCap;

  /// False while the app is out of the foreground on Android and iOS
  /// (`AppForeground`): a held push-to-talk press then ends without a turn.
  /// Null: always in the foreground (desktop keeps the microphone).
  final ValueListenable<bool>? _foreground;
  Timer? _accessCap;
  bool _accessSettled = false;

  late final StreamSubscription<VoiceAssistantEvent<ChatSideEvent>> _events;
  LlmImage? _attachment;
  bool _disposed = false;

  /// What the turns have put on screen, and the running turn's facts; its
  /// steps are mirrored in [liveSteps] for the streaming bubble.
  VoiceChatState _chat = VoiceChatState.empty;
  static const _reducer = VoiceChatReducer();
  final ValueNotifier<List<SkillStep>> _liveSteps = ValueNotifier(const []);

  /// When the chat is re-opened with a changed skill set.
  late final SkillsApplier _applier;

  /// Builds the chat on the loaded model.
  late final Command0<void> open;

  /// Makes this demo's recognizer (Whisper) the active one; a voice turn
  /// asked during the switch waits for it. Executing it again is the Retry.
  late final Command0<void> selectStt;

  /// Sends a typed prompt; the reply streams and (if on) is spoken.
  late final Command1<void, String> send;

  /// Stops the turn (or closes the mic) and keeps the partial reply.
  late final Command0<void> stop;

  /// Clears the screen, the model's history and the attachment.
  late final Command0<void> newConversation;

  /// Picks a picture from the gallery or the camera and attaches it to the
  /// next turns (replacing the current one; a cancelled pick keeps it).
  late final Command1<void, ImageSourceKind> attach;

  /// Rescans the skills directory (the Skills sheet's Reload). A
  /// changed skill set is applied at once when idle, otherwise after the
  /// running turn; unchanged files leave the conversation alone.
  late final Command0<void> reloadSkills;

  /// Re-opens the chat with the latest scan's skills, which drops the
  /// history (the chat says so). Runs by itself when a scan changed them.
  late final Command0<void> applySkills;

  /// Asks for microphone access; runs on entry, and again from
  /// the access bar's Retry.
  late final Command0<void> requestAccess;

  List<Command<void>> get _commands => [
    open,
    selectStt,
    send,
    stop,
    newConversation,
    attach,
    reloadSkills,
    applySkills,
    requestAccess,
  ];

  /// The latest skills scan; null before the first one.
  SkillCatalog? get skillCatalog => _skills.catalog.value;

  /// A scan changed the skills and the chat has not been re-opened with
  /// them yet (it waits for the running turn).
  bool get skillsPending => _applier.pending;

  /// What the last Reload found; null before the first.
  SkillsReload? get lastReload => _applier.lastReload;

  /// The open chat is an agent chat (it was opened with skills).
  bool get hasSkills => _conversation.hasSkills;

  /// Committed conversation, oldest first.
  List<ChatEntry> get entries => _chat.entries;

  /// The reply as it streams in; empty between turns.
  ValueListenable<String> get partialReply => _assistant.partialReply;

  /// The running turn's skill steps as they happen (loadSkill, runIntent,
  /// results); empty between turns. Only the streaming bubble listens.
  ValueListenable<List<SkillStep>> get liveSteps => _liveSteps;

  /// Mic level 0–1 while listening.
  ValueListenable<double> get inputLevel => _assistant.inputLevel;

  TurnPhase get phase => _assistant.phase.value;

  /// The streaming bubble is shown (the reply is being generated or spoken).
  bool get isStreaming =>
      phase == TurnPhase.thinking || phase == TurnPhase.speaking;

  /// A turn runs after the mic was released or a prompt was sent: the
  /// composer shows Stop instead of Send.
  bool get isGenerating =>
      phase == TurnPhase.transcribing || isStreaming || send.running;

  /// The mic button is held but the capture is still starting: nothing is
  /// recorded yet.
  bool get isOpeningMic => phase == TurnPhase.openingMic;

  /// The capture runs: what the user says now is recorded.
  bool get isListening => phase == TurnPhase.listening;

  /// The last failure, shown to the user; cleared by the next turn.
  String? get error => _chat.error;

  /// Microphone access is off (with the platform's settings
  /// path); lasts until a Retry finds it on.
  String? get micAccessError => switch (_micAccess) {
    // Only access: an audio-session failure surfaces on the first turn.
    Error(error: final MicAccessException error) =>
      '$error. Voice input is off; typing still works.',
    _ => null,
  };

  Result<void>? _micAccess;

  /// Why this demo's recognizer (Whisper) could not be made active; lasts
  /// until a retry ([selectStt], or New conversation) succeeds.
  String? get sttError => switch (selectStt.result) {
    Error(:final error) => 'Could not switch the speech recognizer: $error',
    _ => null,
  };

  bool get speakReplies => _assistant.speakReplies;

  /// The picture sent with every turn until removed; null when none.
  LlmImage? get attachment => _attachment;

  /// The camera button is shown only where the platform has one in the
  /// picker (iPhone, Android); macOS picks from the gallery only.
  bool get canUseCamera => _images.supports(ImageSourceKind.camera);

  bool get canUseGallery => _images.supports(ImageSourceKind.gallery);

  /// A pick is not already open. Allowed during a turn: the new picture
  /// applies from the next turn.
  bool get canAttach => !attach.running;

  /// The chat model was loaded with images; off, the photo buttons are disabled
  /// with [photosOffReason].
  bool get photosEnabled => _conversation.capabilities.images;

  /// Why the photo buttons are disabled; null when photos work.
  String? get photosOffReason => photosEnabled
      ? null
      : 'Photos are off: ${_conversation.capabilities.modelName} was set up '
            'without images.';

  /// Why the skills that need tool calls are off (the chat model has tools
  /// off); null when they are on or no skill is installed. The questions
  /// the app recognizes itself (the time, device facts) still work.
  String? get skillsOffReason => _conversation.skillsNeedTools
      ? 'Skills that need tool calls are off: '
            '${_conversation.capabilities.modelName} has tools off. The '
            'time and device facts still work.'
      : null;

  /// This demo's chat exists: opened, or reopened by [newConversation] after
  /// a failed [open]. Not while [open] still runs: the shared chat is then
  /// the previous demo's, possibly still draining a stopped turn.
  ///
  /// Also not before the entry access requests have settled (or the
  /// cap passed), so the camera probe never shares the GPU with a prefill.
  bool get isReady =>
      _accessSettled &&
      !open.running &&
      !applySkills.running &&
      _conversation.profile == kVoiceChatProfile;

  bool get canSend =>
      isReady &&
      !newConversation.running &&
      !send.running &&
      !_assistant.isBusy;

  /// The mic stays enabled during a turn: pressing it is the barge-in.
  bool get canTalk => isReady && !newConversation.running;

  /// Enabled once the first open has finished, either way: after a failed
  /// open it is the way to retry without restarting the app.
  bool get canStartNewConversation =>
      !open.running && !newConversation.running && !applySkills.running;

  /// Push-to-talk press; during a reply it is the barge-in.
  ///
  /// Deliberately not a [Command]: a command ignores `execute` while its
  /// previous run is pending, and a release can be pending for seconds
  /// (behind a draining interrupt), so the next press's release would be
  /// dropped and its capture left open. The assistant serializes presses
  /// and releases itself; failures arrive as events.
  Future<void> pressMic() async {
    _clearError();
    await _assistant.micDown();
  }

  /// Push-to-talk release; completes when that turn has ended or was
  /// superseded. Never guarded (see [pressMic]).
  Future<void> releaseMic() async {
    await _assistant.micUp(image: _attachment?.png);
  }

  /// Leaving the app (Android, iOS) ends a held press without a turn; a
  /// turn past its capture goes on.
  void _onForeground() {
    if (_disposed || (_foreground?.value ?? true)) return;
    unawaited(_assistant.cancelCapture());
  }

  /// Detaches the picture: the next turns go without it (the model may still
  /// remember it from earlier turns).
  void removeAttachment() => _setAttachment(null);

  /// Turns "speak replies" on or off; off also silences a reply playing now.
  void setSpeakReplies({required bool enabled}) {
    _assistant.speakReplies = enabled;
    _notify();
  }

  /// Feeds captured audio through the same entry point mic release uses
  /// (integration tests; the mic is bypassed).
  @visibleForTesting
  Future<TurnResult> submitUtterance(Utterance utterance) {
    _clearError();
    return _assistant.submitUtterance(utterance, image: _attachment?.png);
  }

  /// Entering Demo 1 opens the shared chat with this demo's profile, which
  /// drops whatever the previous demo left in it. It rescans the skills
  /// first: entering the demo picks up dropped-in skills.
  Future<Result<void>> _open() async {
    // A failure is not retried by itself: New conversation or Reload
    // retries.
    final result = await _applier.open(rescan: true);
    if (_disposed) return result;
    if (result case Error(:final error)) {
      _showError('Could not open the chat: $error');
    }
    return result;
  }

  /// Shown until a switch succeeds ([sttError]); voice turns fail fast
  /// meanwhile (the active model is checked at use), typed turns work.
  Future<Result<void>> _selectStt() => _activateStt(Demo.voiceChat.stt);

  Future<Result<void>> _send(String text) async {
    final prompt = text.trim();
    if (prompt.isEmpty) return const Result.ok(null);
    _clearError();
    return _toResult(
      await _assistant.sendText(prompt, image: _attachment?.png),
    );
  }

  Future<Result<void>> _attach(ImageSourceKind source) async {
    if (photosOffReason case final reason?) {
      // The buttons are disabled; a stale tap still must not send a picture
      // the chat would refuse.
      return Result.error(ConversationImageUnsupportedException(reason));
    }
    final result = await _images.pick(source);
    if (_disposed) return const Result.ok(null);
    switch (result) {
      case Ok(:final value):
        // null: cancelled; the current attachment stays.
        if (value != null) _setAttachment(value);
        return const Result.ok(null);
      case Error(:final error):
        _showError('Could not attach the picture: $error');
        return Result.error(error);
    }
  }

  void _setAttachment(LlmImage? image) {
    if (identical(image, _attachment)) return;
    _attachment = image;
    _diagnostics.recordAttachment(image);
    _notify();
  }

  Future<Result<void>> _stop() async {
    await _assistant.stop();
    return const Result.ok(null);
  }

  void _settleAccess() {
    if (_accessSettled) return;
    _accessSettled = true;
    _accessCap?.cancel();
    _notify();
  }

  Future<Result<void>> _requestAccess() async {
    final mic = await _assistant.requestMicAccess();
    if (_disposed) return const Result.ok(null);
    _micAccess = mic;
    _notify();
    return mic;
  }

  Future<Result<void>> _newConversation() async {
    // Before any await: a picture picked while the turn drains
    // belongs to the new conversation, and a view model left mid-open must
    // not touch the overlay afterwards.
    _setAttachment(null);
    if (_assistant.isBusy) await _assistant.stop();
    // Left the screen while the turn drained: a late open would replace the
    // chat the next demo has opened meanwhile.
    if (_disposed) return const Result.ok(null);
    // A fresh chat with this demo's profile and the latest skills; also the
    // retry after a failed first open, and (in parallel, awaited) after a
    // failed recognizer switch, whose error then stays on [sttError].
    final sttRetry = selectStt.result is Error<void>
        ? selectStt.execute()
        : null;
    // A failure marks the set failed: not retried by itself.
    final result = await _applier.open(rescan: false);
    await sttRetry;
    if (_disposed) return result;
    _setChat(VoiceChatState.empty);
    if (result case Error(:final error)) {
      _showError('Could not start a new conversation: $error');
    } else {
      _notify();
    }
    return result;
  }

  // ---- Skills ----

  /// [SkillsApplier.apply] as a command; the chat says it started over, or
  /// why it could not.
  Future<Result<void>> _applySkills() async {
    final applied = await _applier.apply();
    if (!_disposed) {
      switch (applied) {
        case Ok(value: final catalog?):
          _setChat(
            _chat
                .copyWith(clearTurnRetrieval: true, turnSteps: const [])
                .adding(
                  ChatEntry(
                    role: ChatRole.notice,
                    text: skillsReloadedText(catalog),
                  ),
                ),
          );
          _notify();
        case Ok():
          break;
        case Error(:final error):
          _showError('Could not reload the skills: $error');
      }
    }
    return switch (applied) {
      Ok() => const Result.ok(null),
      Error(:final error) => Result.error(error),
    };
  }

  void _onOpenCommand() => _applier.commandsChanged();

  void _onPhase() => _applier.turnChanged();

  void _onEvent(VoiceAssistantEvent<ChatSideEvent> event) {
    final update = _reducer.reduce(_chat, event);
    _setChat(update.state);
    if (update.generation case final metrics?) {
      _diagnostics.recordGeneration(metrics);
    }
    if (update.retrieval case final retrieval?) {
      _diagnostics.recordRetrieval(retrieval);
    }
    if (update.notify) _notify();
  }

  /// Takes [chat]; the streaming bubble follows its steps (one update per
  /// step: the same `const []` while there are none).
  void _setChat(VoiceChatState chat) {
    _chat = chat;
    if (!_disposed) _liveSteps.value = chat.turnSteps;
  }

  /// A new action starts: the last failure goes (no rebuild by itself).
  void _clearError() => _chat = _chat.copyWith(clearError: true);

  static Result<void> _toResult(TurnResult result) => switch (result.outcome) {
    TurnOutcome.failed || TurnOutcome.micUnavailable => Result.error(
      asException(result.error ?? StateError(result.outcome.name)),
    ),
    TurnOutcome.completed ||
    TurnOutcome.interrupted ||
    TurnOutcome.notHeard ||
    TurnOutcome.superseded ||
    TurnOutcome.ignored => const Result.ok(null),
  };

  void _showError(String message) {
    _chat = _chat.showingError(message);
    _notify();
  }

  void _notify() {
    if (!_disposed) notifyListeners();
  }

  @override
  void dispose() {
    _disposed = true;
    _foreground?.removeListener(_onForeground);
    _accessCap?.cancel();
    // The overlay outlives this screen; the picture does not.
    if (_attachment != null) _diagnostics.recordAttachment(null);
    unawaited(_events.cancel());
    _liveSteps.dispose();
    _applier.close();
    _assistant.phase.removeListener(_notify);
    _assistant.phase.removeListener(_onPhase);
    // Detaches a running turn (its audio stops now) and interrupts it; the
    // drain finishes in the background.
    unawaited(_assistant.dispose());
    for (final command in _commands) {
      command
        ..removeListener(_notify)
        ..dispose();
    }
    super.dispose();
  }
}

/// The notice when a changed skill set re-opened the chat.
String skillsReloadedText(SkillCatalog catalog) {
  final n = catalog.skills.length;
  final errors = catalog.errors.length;
  return 'Skills reloaded: $n skill${n == 1 ? '' : 's'}'
      '${errors == 0 ? '' : ', $errors with errors'}. The conversation '
      'started over.';
}

/// [SkillsApplier]'s way to the view model's commands: the apply runs as
/// [VoiceChatViewModel.applySkills], and the chat counts as busy while it is
/// being opened, the recognizer switches, or a turn runs.
final class _ViewModelSkillsCommands implements SkillsApplyCommands {
  _ViewModelSkillsCommands(this._viewModel);

  final VoiceChatViewModel _viewModel;

  @override
  bool get openingChat =>
      _viewModel.open.running ||
      _viewModel.newConversation.running ||
      _viewModel.applySkills.running ||
      _viewModel.selectStt.running;

  @override
  bool get turnRunning => _viewModel._assistant.isBusy;

  @override
  bool get applying => _viewModel.applySkills.running;

  @override
  Future<void> apply() => _viewModel.applySkills.execute();
}
