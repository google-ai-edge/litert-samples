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
import 'package:flutter/material.dart';

import '../../../../domain/models/chat_entry.dart';
import '../../../../domain/models/skill_step.dart';
import 'chat_keys.dart';
import 'citation_chips.dart';
import 'skill_steps.dart';

/// A committed message.
class MessageBubble extends StatelessWidget {
  const MessageBubble({super.key, required this.entry});

  final ChatEntry entry;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    if (entry.role == ChatRole.notice) {
      return Padding(
        padding: const EdgeInsets.symmetric(vertical: 6),
        child: Center(
          child: Text(
            entry.text,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: colors.onSurfaceVariant,
              fontStyle: FontStyle.italic,
            ),
          ),
        ),
      );
    }
    final (background, foreground, alignment) = switch (entry.role) {
      ChatRole.user => (
        colors.primaryContainer,
        colors.onPrimaryContainer,
        Alignment.centerRight,
      ),
      ChatRole.assistant => (
        colors.surfaceContainerHighest,
        colors.onSurface,
        Alignment.centerLeft,
      ),
      ChatRole.error || ChatRole.notice => (
        colors.errorContainer,
        colors.onErrorContainer,
        Alignment.centerLeft,
      ),
    };
    return _Bubble(
      alignment: alignment,
      background: background,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        spacing: 6,
        children: [
          // What the agent did comes before what it said.
          if (entry.steps.isNotEmpty)
            SkillStepsPanel(
              steps: entry.steps,
              color: foreground.withValues(alpha: 0.8),
            ),
          if (entry.image case final image?)
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Image.memory(
                image,
                key: ChatKeys.entryImage,
                height: 120,
                cacheHeight: 240,
                gaplessPlayback: true,
                errorBuilder: (context, error, stack) =>
                    const Icon(Icons.broken_image_outlined),
              ),
            ),
          if (entry.text.isNotEmpty)
            Text(entry.text, style: TextStyle(color: foreground)),
          if (entry.interrupted)
            Text(
              'stopped',
              style: Theme.of(context).textTheme.labelSmall
                  ?.copyWith(color: foreground),
            ),
          if (entry.knowledge case final knowledge?)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: KnowledgeChips(knowledge: knowledge),
            ),
        ],
      ),
    );
  }
}

/// The reply while it streams. Only this widget rebuilds per token.
class StreamingBubble extends StatelessWidget {
  const StreamingBubble({super.key, required this.text, this.steps});

  final ValueListenable<String> text;

  /// The running turn's skill steps, shown as they happen — a skill
  /// call is two or three generations with nothing to read or hear until
  /// the answer. Updates once per step.
  final ValueListenable<List<SkillStep>>? steps;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final reply = ValueListenableBuilder<String>(
      valueListenable: text,
      builder: (context, value, _) => Text(
        value.isEmpty ? '…' : value,
        style: TextStyle(color: colors.onSurface),
      ),
    );
    final live = steps;
    return _Bubble(
      alignment: Alignment.centerLeft,
      background: colors.surfaceContainerHighest,
      child: live == null
          ? reply
          : Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                ValueListenableBuilder<List<SkillStep>>(
                  valueListenable: live,
                  builder: (context, value, _) => value.isEmpty
                      ? const SizedBox.shrink()
                      : Padding(
                          padding: const EdgeInsets.only(bottom: 6),
                          child: SkillStepsPanel(
                            steps: value,
                            color: colors.onSurface.withValues(alpha: 0.8),
                          ),
                        ),
                ),
                reply,
              ],
            ),
    );
  }
}

class _Bubble extends StatelessWidget {
  const _Bubble({
    required this.alignment,
    required this.background,
    required this.child,
  });

  final Alignment alignment;
  final Color background;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: alignment,
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 560),
        child: Container(
          margin: const EdgeInsets.symmetric(vertical: 4),
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
          decoration: BoxDecoration(
            color: background,
            borderRadius: BorderRadius.circular(16),
          ),
          child: child,
        ),
      ),
    );
  }
}
