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
import 'package:flutter/material.dart';
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show SkillType;

import '../../../../domain/models/skill_catalog.dart';
import '../../../core/warning_color.dart';
import '../view_models/voice_chat_view_model.dart';

/// Keys for tests.
abstract final class SkillsSheetKeys {
  static const toolsOff = ValueKey('skills-sheet-tools-off');
  static const sheet = ValueKey('skills-sheet');
  static const reload = ValueKey('skills-reload');
  static const status = ValueKey('skills-status');

  static ValueKey<String> skill(String name) => ValueKey('skills-skill-$name');
  static ValueKey<String> error(String path) => ValueKey('skills-error-$path');
}

/// Opens the Skills sheet over the chat.
Future<void> showSkillsSheet(
  BuildContext context,
  VoiceChatViewModel viewModel,
) => showModalBottomSheet<void>(
  context: context,
  isScrollControlled: true,
  showDragHandle: true,
  builder: (context) => SkillsSheet(viewModel: viewModel),
);

/// The runtime skills: where to drop a SKILL.md, the
/// skills the chat runs with, every file that failed and why, and Reload.
/// A changed skill set starts the conversation over, so the sheet says so.
class SkillsSheet extends StatelessWidget {
  const SkillsSheet({super.key, required this.viewModel});

  final VoiceChatViewModel viewModel;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: viewModel,
      builder: (context, _) {
        final theme = Theme.of(context);
        final catalog = viewModel.skillCatalog;
        return DraggableScrollableSheet(
          key: SkillsSheetKeys.sheet,
          expand: false,
          initialChildSize: 0.6,
          minChildSize: 0.3,
          maxChildSize: 0.95,
          builder: (context, scroll) => ListView(
            controller: scroll,
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 24),
            children: [
              Row(
                children: [
                  Expanded(
                    child: Text('Skills', style: theme.textTheme.titleLarge),
                  ),
                  FilledButton.tonalIcon(
                    key: SkillsSheetKeys.reload,
                    onPressed:
                        viewModel.reloadSkills.running ||
                            viewModel.applySkills.running
                        ? null
                        : () => unawaited(viewModel.reloadSkills.execute()),
                    icon:
                        viewModel.reloadSkills.running ||
                            viewModel.applySkills.running
                        ? const SizedBox.square(
                            dimension: 16,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.refresh),
                    label: const Text('Reload'),
                  ),
                ],
              ),
              const SizedBox(height: 4),
              Text(
                _statusText(viewModel),
                key: SkillsSheetKeys.status,
                style: theme.textTheme.bodySmall,
              ),
              if (viewModel.skillsOffReason case final String reason) ...[
                const SizedBox(height: 4),
                Text(
                  reason,
                  key: SkillsSheetKeys.toolsOff,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: kWarningColor,
                  ),
                ),
              ],
              const SizedBox(height: 12),
              if (catalog == null)
                const Text('Scanning the skills folder…')
              else ...[
                _Folder(catalog: catalog),
                const SizedBox(height: 12),
                for (final loaded in catalog.skills) _SkillTile(loaded: loaded),
                if (catalog.errors.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  Text(
                    'Not loaded',
                    style: theme.textTheme.titleSmall?.copyWith(
                      color: theme.colorScheme.error,
                    ),
                  ),
                  for (final error in catalog.errors) _ErrorTile(error: error),
                ],
              ],
            ],
          ),
        );
      },
    );
  }

  static String _statusText(VoiceChatViewModel vm) {
    if (vm.applySkills.running) {
      return 'Reloading the chat with the new skills…';
    }
    if (vm.skillsPending) {
      return 'Changed: applied when the current reply finishes.';
    }
    return switch (vm.lastReload) {
      SkillsReload.unchanged => 'Reload found no changes.',
      SkillsReload.applied =>
        'Reloaded: the conversation started over with the new skills.',
      SkillsReload.failed =>
        'Changed, but the chat did not reload. Reload to try again.',
      SkillsReload.applying || SkillsReload.pending || null =>
        'Applying changed skills starts the conversation over. Add a folder '
            'with a SKILL.md, then Reload.',
    };
  }
}

class _Folder extends StatelessWidget {
  const _Folder({required this.catalog});

  final SkillCatalog catalog;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    if (catalog.storeError case final error?) {
      return Text(
        'No skills folder: $error',
        style: TextStyle(color: theme.colorScheme.error),
      );
    }
    final hint = switch (defaultTargetPlatform) {
      TargetPlatform.android => 'adb push <skill-folder> to this folder.',
      TargetPlatform.iOS => 'Files app › On My iPhone › this app › skills.',
      TargetPlatform.macOS => 'Copy a skill folder here in Finder.',
      _ => 'Copy a skill folder here.',
    };
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      spacing: 2,
      children: [
        SelectableText(
          catalog.directory ?? '–',
          style: theme.textTheme.labelSmall?.copyWith(
            fontFamily: 'Menlo',
            fontFamilyFallback: const ['monospace', 'Courier'],
          ),
        ),
        Text(hint, style: theme.textTheme.labelSmall),
      ],
    );
  }
}

class _SkillTile extends StatelessWidget {
  const _SkillTile({required this.loaded});

  final LoadedSkill loaded;

  @override
  Widget build(BuildContext context) {
    final skill = loaded.skill;
    return ListTile(
      key: SkillsSheetKeys.skill(skill.name),
      contentPadding: EdgeInsets.zero,
      dense: true,
      leading: Icon(
        skill.type == SkillType.intent
            ? Icons.bolt_outlined
            : Icons.article_outlined,
      ),
      title: Text(skill.name),
      subtitle: Text('${skill.description}\n${loaded.path}'),
      isThreeLine: true,
    );
  }
}

class _ErrorTile extends StatelessWidget {
  const _ErrorTile({required this.error});

  final SkillLoadError error;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return ListTile(
      key: SkillsSheetKeys.error(error.path),
      contentPadding: EdgeInsets.zero,
      dense: true,
      leading: Icon(Icons.error_outline, color: colors.error),
      title: Text(error.path),
      subtitle: Text(error.message, style: TextStyle(color: colors.error)),
    );
  }
}
