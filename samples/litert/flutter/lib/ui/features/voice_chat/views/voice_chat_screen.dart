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

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../../../domain/models/llm_image.dart';
import '../../../../domain/models/voice.dart';
import '../../../core/debug_overlay.dart';
import '../../../core/level_meter.dart';
import '../../../core/warning_color.dart';
import '../view_models/voice_chat_view_model.dart';
import 'attachment_bar.dart';
import 'chat_keys.dart';
import 'composer.dart';
import 'message_bubble.dart';
import 'skills_sheet.dart';

/// Demo 1 chat: message list, streaming bubble, phase line,
/// the attachment bar (gallery, camera on phones, the sticky picture), and a
/// composer with the push-to-talk mic, the text field and Send/Stop. App bar:
/// speak replies on/off, diagnostics, and a "More" menu with Skills and New
/// conversation (four icons cut the title to "Voice …" on a phone).
class VoiceChatScreen extends StatelessWidget {
  const VoiceChatScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final viewModel = context.read<VoiceChatViewModel>();
    return ListenableBuilder(
      listenable: viewModel,
      builder: (context, _) => Scaffold(
        appBar: AppBar(
          title: const Text('Voice chat'),
          actions: [
            IconButton(
              key: ChatKeys.speakReplies,
              tooltip: viewModel.speakReplies
                  ? 'Speak replies: on'
                  : 'Speak replies: off',
              isSelected: viewModel.speakReplies,
              onPressed: () =>
                  viewModel.setSpeakReplies(enabled: !viewModel.speakReplies),
              icon: const Icon(Icons.volume_off_outlined),
              selectedIcon: const Icon(Icons.volume_up),
            ),
            const DebugOverlayToggle(),
            _MoreMenu(viewModel: viewModel),
          ],
        ),
        body: Column(
          children: [
            if (!viewModel.isReady && viewModel.open.running)
              const LinearProgressIndicator(),
            Expanded(child: _MessageList(viewModel: viewModel)),
            if (viewModel.sttError case final String sttError)
              _ErrorBar(
                message: sttError,
                retrying: viewModel.selectStt.running,
                onRetry: () => unawaited(viewModel.selectStt.execute()),
              ),
            // Microphone access is off.
            if (viewModel.micAccessError case final String access)
              _ErrorBar(
                message: access,
                retrying: viewModel.requestAccess.running,
                onRetry: () => unawaited(viewModel.requestAccess.execute()),
                messageKey: ChatKeys.accessError,
                retryKey: ChatKeys.retryAccess,
              ),
            if (viewModel.skillsOffReason case final String reason)
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 4, 16, 0),
                child: Text(
                  reason,
                  key: ChatKeys.skillsOff,
                  style: Theme.of(context).textTheme.labelMedium
                      ?.copyWith(color: kWarningColor),
                ),
              ),
            _PhaseLine(phase: viewModel.phase),
            AttachmentBar(
              attachment: viewModel.attachment,
              picking: !viewModel.canAttach,
              showGallery: viewModel.canUseGallery,
              showCamera: viewModel.canUseCamera,
              onGallery: () =>
                  unawaited(viewModel.attach.execute(ImageSourceKind.gallery)),
              onCamera: () =>
                  unawaited(viewModel.attach.execute(ImageSourceKind.camera)),
              onRemove: viewModel.removeAttachment,
              disabledReason: viewModel.photosOffReason,
            ),
            Composer(
              canSend: viewModel.canSend,
              isGenerating: viewModel.isGenerating,
              onSend: (text) => unawaited(viewModel.send.execute(text)),
              onStop: () => unawaited(viewModel.stop.execute()),
              leading: MicButton(
                key: ChatKeys.mic,
                level: viewModel.inputLevel,
                phase: viewModel.phase,
                enabled: viewModel.canTalk,
                onDown: () => unawaited(viewModel.pressMic()),
                onUp: () => unawaited(viewModel.releaseMic()),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// A lasting problem with a Retry: the recognizer switch failed (voice
/// turns cannot run until it is retried; typed turns still work), or
/// microphone access is off.
class _ErrorBar extends StatelessWidget {
  const _ErrorBar({
    required this.message,
    required this.retrying,
    required this.onRetry,
    this.messageKey = ChatKeys.sttError,
    this.retryKey = ChatKeys.retryStt,
  });

  final String message;
  final bool retrying;
  final VoidCallback onRetry;
  final Key messageKey;
  final Key retryKey;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Material(
      color: colors.errorContainer,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 8, 8),
        child: Row(
          spacing: 8,
          children: [
            Expanded(
              child: Text(
                message,
                key: messageKey,
                style: TextStyle(color: colors.onErrorContainer),
              ),
            ),
            FilledButton(
              key: retryKey,
              onPressed: retrying ? null : onRetry,
              child: const Text('Retry'),
            ),
          ],
        ),
      ),
    );
  }
}

class _PhaseLine extends StatelessWidget {
  const _PhaseLine({required this.phase});

  final TurnPhase phase;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final text = switch (phase) {
      TurnPhase.idle => 'Hold the mic to talk, or type a question',
      TurnPhase.openingMic => kOpeningMicLabel,
      TurnPhase.listening => 'Listening… release to send',
      TurnPhase.transcribing => 'Transcribing…',
      TurnPhase.thinking => 'Thinking…',
      TurnPhase.speaking => 'Speaking… press the mic to interrupt',
      TurnPhase.error => 'The last turn failed — try again',
    };
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 4, 16, 0),
      child: Row(
        spacing: 8,
        children: [
          if (phase.isActive)
            const SizedBox.square(
              dimension: 12,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
          Expanded(
            child: Text(
              text,
              key: ChatKeys.phase,
              style: theme.textTheme.labelMedium?.copyWith(
                color: phase == TurnPhase.error
                    ? theme.colorScheme.error
                    : theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _MessageList extends StatelessWidget {
  const _MessageList({required this.viewModel});

  final VoiceChatViewModel viewModel;

  @override
  Widget build(BuildContext context) {
    final entries = viewModel.entries;
    final streaming = viewModel.isStreaming;
    final count = entries.length + (streaming ? 1 : 0);
    if (count == 0) {
      return const Center(child: Text('Hold the mic and ask something.'));
    }
    // Reversed so the newest message sits at the bottom and the list follows
    // the growing reply without a scroll controller.
    return ListView.builder(
      reverse: true,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      itemCount: count,
      itemBuilder: (context, index) {
        if (streaming && index == 0) {
          return StreamingBubble(
            key: ChatKeys.streamingBubble,
            text: viewModel.partialReply,
            steps: viewModel.liveSteps,
          );
        }
        final entry =
            entries[entries.length - 1 - (index - (streaming ? 1 : 0))];
        return MessageBubble(entry: entry);
      },
    );
  }
}

/// The app bar's "More" menu: Skills (with the count, and a badge when some
/// failed to load or a change waits) and New conversation.
class _MoreMenu extends StatelessWidget {
  const _MoreMenu({required this.viewModel});

  final VoiceChatViewModel viewModel;

  @override
  Widget build(BuildContext context) {
    final catalog = viewModel.skillCatalog;
    return PopupMenuButton<void>(
      key: ChatKeys.more,
      tooltip: 'More',
      icon: Badge(
        isLabelVisible:
            catalog != null &&
            (catalog.errors.isNotEmpty || viewModel.skillsPending),
        child: const Icon(Icons.more_vert),
      ),
      itemBuilder: (context) => [
        if (catalog != null)
          PopupMenuItem<void>(
            key: ChatKeys.skills,
            onTap: () => unawaited(showSkillsSheet(context, viewModel)),
            child: ListTile(
              contentPadding: EdgeInsets.zero,
              leading: const Icon(Icons.extension_outlined),
              title: Text(
                catalog.errors.isEmpty
                    ? 'Skills (${catalog.skills.length})'
                    : 'Skills (${catalog.skills.length}, '
                          '${catalog.errors.length} with errors)',
              ),
            ),
          ),
        PopupMenuItem<void>(
          key: ChatKeys.newConversation,
          enabled: viewModel.canStartNewConversation,
          onTap: () => unawaited(viewModel.newConversation.execute()),
          child: const ListTile(
            contentPadding: EdgeInsets.zero,
            leading: Icon(Icons.add_comment_outlined),
            title: Text('New conversation'),
          ),
        ),
      ],
    );
  }
}
