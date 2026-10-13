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

import 'package:flutter/material.dart';

import '../../../../domain/models/knowledge.dart';

/// Keys for tests.
abstract final class KnowledgeChipKeys {
  /// The chip of excerpt [number] (1-based, as cited in the reply).
  static ValueKey<String> citation(int number) =>
      ValueKey('kb-citation-$number');

  /// The chip that says the knowledge base was unavailable or failed.
  static const status = ValueKey('kb-status');
}

/// What each chip says before the similarity: "Title › Section", but only
/// the section when the reply's excerpts share a title (a long title
/// ellipsized three chips into the same text on a phone), and a part
/// number when two excerpts share the section too ("Inputs (2)").
List<String> citationLabels(List<Passage> passages) {
  final titles = <String, int>{};
  for (final p in passages) {
    titles[p.title] = (titles[p.title] ?? 0) + 1;
  }
  final sections = <String, int>{};
  for (final p in passages) {
    final key = '${p.title}\u0000${p.section}';
    sections[key] = (sections[key] ?? 0) + 1;
  }
  final seen = <String, int>{};
  return [
    for (final p in passages)
      if (titles[p.title]! < 2)
        p.label
      else ...[
        () {
          final key = '${p.title}\u0000${p.section}';
          final n = seen[key] = (seen[key] ?? 0) + 1;
          final name = p.section.isEmpty ? p.title : p.section;
          return sections[key]! > 1 ? '$name ($n)' : name;
        }(),
      ],
  ];
}

/// Under an assistant reply: one chip per excerpt the prompt
/// carried, "Title › Section · 0.62" ([citationLabels]), the cited ones
/// highlighted and the uncited dimmed; a tap opens the excerpt and its
/// source. An unavailable knowledge base or a failed search is one chip
/// saying so. Below the gate there is nothing to show here (the overlay says
/// "below gate").
class KnowledgeChips extends StatelessWidget {
  const KnowledgeChips({super.key, required this.knowledge});

  final ReplyKnowledge knowledge;

  @override
  Widget build(BuildContext context) {
    final retrieval = knowledge.retrieval;
    final colors = Theme.of(context).colorScheme;
    return switch (retrieval.outcome) {
      RetrievalOutcome.belowGate ||
      RetrievalOutcome.skipped => const SizedBox.shrink(),
      RetrievalOutcome.unavailable => _StatusChip(
        icon: Icons.menu_book_outlined,
        text: 'Knowledge base unavailable: ${retrieval.detail ?? 'unknown'}',
        foreground: colors.onSurfaceVariant,
      ),
      RetrievalOutcome.failed => _StatusChip(
        icon: Icons.error_outline,
        text: 'Knowledge base search failed: ${retrieval.detail ?? 'unknown'}',
        foreground: colors.error,
      ),
      RetrievalOutcome.used => Wrap(
        spacing: 6,
        runSpacing: 6,
        children: [
          for (final (i, label) in citationLabels(retrieval.passages).indexed)
            _CitationChip(
              key: KnowledgeChipKeys.citation(i + 1),
              number: i + 1,
              passage: retrieval.passages[i],
              label: label,
              cited: knowledge.cited.contains(i + 1),
            ),
        ],
      ),
    };
  }
}

class _CitationChip extends StatelessWidget {
  const _CitationChip({
    super.key,
    required this.number,
    required this.passage,
    required this.label,
    required this.cited,
  });

  final int number;
  final Passage passage;
  final String label;
  final bool cited;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Opacity(
      opacity: cited ? 1 : 0.55,
      child: ActionChip(
        visualDensity: VisualDensity.compact,
        backgroundColor: cited ? colors.secondaryContainer : null,
        avatar: CircleAvatar(
          backgroundColor: cited ? colors.secondary : colors.outline,
          foregroundColor: cited ? colors.onSecondary : colors.surface,
          child: Text('$number'),
        ),
        label: Text(
          '$label · ${passage.similarity.toStringAsFixed(2)}',
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
        tooltip: cited ? 'Cited as [$number]' : 'In the prompt, not cited',
        onPressed: () => showDialog<void>(
          context: context,
          builder: (context) =>
              _PassageDialog(number: number, passage: passage),
        ),
      ),
    );
  }
}

class _PassageDialog extends StatelessWidget {
  const _PassageDialog({required this.number, required this.passage});

  final int number;
  final Passage passage;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    // The stored chunk starts with its label; the title already shows it.
    final prefix = '${passage.label}\n\n';
    final body = passage.content.startsWith(prefix)
        ? passage.content.substring(prefix.length)
        : passage.content;
    return AlertDialog(
      title: Text('[$number] ${passage.label}'),
      content: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          spacing: 12,
          children: [
            SelectableText(body),
            Text(
              '${passage.doc} · similarity '
              '${passage.similarity.toStringAsFixed(3)}',
              style: theme.textTheme.labelSmall,
            ),
            if (passage.source case final source?)
              SelectableText(
                source,
                style: theme.textTheme.labelSmall?.copyWith(
                  color: theme.colorScheme.primary,
                ),
              ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Close'),
        ),
      ],
    );
  }
}

class _StatusChip extends StatelessWidget {
  const _StatusChip({
    required this.icon,
    required this.text,
    required this.foreground,
  });

  final IconData icon;
  final String text;
  final Color foreground;

  @override
  Widget build(BuildContext context) {
    return Chip(
      key: KnowledgeChipKeys.status,
      visualDensity: VisualDensity.compact,
      avatar: Icon(icon, size: 16, color: foreground),
      label: Text(
        text,
        maxLines: 2,
        overflow: TextOverflow.ellipsis,
        style: TextStyle(color: foreground),
      ),
    );
  }
}
