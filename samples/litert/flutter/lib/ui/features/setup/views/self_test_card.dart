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
import 'package:flutter/services.dart';

import '../view_models/self_test_view_model.dart';

/// Keys for tests.
abstract final class SelfTestKeys {
  static const card = ValueKey('self-test-card');
  static const run = ValueKey('self-test-run');
  static const copy = ValueKey('self-test-copy');
  static const report = ValueKey('self-test-report');
  static const result = ValueKey('self-test-result');
}

/// "Run self-test" on the Models screen: the `--selftest` steps inside the
/// app (Android has no command line), its progress, and the report with a
/// Copy button.
class SelfTestCard extends StatelessWidget {
  const SelfTestCard({super.key, required this.viewModel});

  final SelfTestViewModel viewModel;

  Future<void> _copy(BuildContext context, String text) async {
    final messenger = ScaffoldMessenger.maybeOf(context);
    String message;
    try {
      await Clipboard.setData(ClipboardData(text: text));
      message = 'Self-test report copied to the clipboard';
    } on PlatformException catch (e) {
      message = 'Could not copy the report: ${e.message ?? e.code}';
    }
    messenger?.showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: viewModel,
      builder: (context, _) {
        final theme = Theme.of(context);
        final caption = theme.textTheme.bodySmall?.copyWith(
          color: theme.colorScheme.onSurfaceVariant,
        );
        final outcome = viewModel.outcome;
        const mono = TextStyle(fontFamily: 'monospace', fontSize: 11);
        return Card(
          key: SelfTestKeys.card,
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              spacing: 8,
              children: [
                Text('SELF-TEST', style: theme.textTheme.labelMedium),
                Text(
                  'The --selftest steps in the app: hardware probe, detector, '
                  'cats golden, the chat model on its backend with a short '
                  'generation and tok/s, memory, audio. The chat model is '
                  'unloaded for the run and loaded again after it.',
                  style: caption,
                ),
                Row(
                  spacing: 8,
                  children: [
                    FilledButton.tonal(
                      key: SelfTestKeys.run,
                      onPressed: viewModel.canRun
                          ? viewModel.run.execute
                          : null,
                      child: Text(
                        viewModel.run.running ? 'Running…' : 'Run self-test',
                      ),
                    ),
                    if (outcome != null)
                      TextButton.icon(
                        key: SelfTestKeys.copy,
                        onPressed: () =>
                            unawaited(_copy(context, outcome.text)),
                        icon: const Icon(Icons.copy, size: 18),
                        label: const Text('Copy'),
                      ),
                  ],
                ),
                if (viewModel.blockedReason case final reason?)
                  Text(reason, style: caption),
                if (viewModel.run.running)
                  ValueListenableBuilder(
                    valueListenable: viewModel.progress,
                    builder: (context, lines, _) => Text(
                      lines.length > 6
                          ? lines.sublist(lines.length - 6).join('\n')
                          : lines.join('\n'),
                      style: mono,
                    ),
                  ),
                if (viewModel.error case final error?)
                  Text(error, style: TextStyle(color: theme.colorScheme.error)),
                if (outcome != null) ...[
                  Text(
                    outcome.passed ? 'PASS' : 'FAIL',
                    key: SelfTestKeys.result,
                    style: theme.textTheme.titleSmall?.copyWith(
                      color: outcome.passed ? null : theme.colorScheme.error,
                    ),
                  ),
                  if (outcome.reportPath case final path?)
                    Text('Saved to $path', style: caption),
                  _ReportBox(text: outcome.text, style: mono),
                ],
              ],
            ),
          ),
        );
      },
    );
  }
}

/// The report in its own fixed-width columns: scrolled both ways with the
/// scroll bars always shown, so long lines are visibly cut by the box, not
/// by the screen (on a phone they looked truncated). Wrapping would break
/// the columns.
class _ReportBox extends StatefulWidget {
  const _ReportBox({required this.text, required this.style});

  final String text;
  final TextStyle? style;

  @override
  State<_ReportBox> createState() => _ReportBoxState();
}

class _ReportBoxState extends State<_ReportBox> {
  final _vertical = ScrollController();
  final _horizontal = ScrollController();

  @override
  void dispose() {
    _vertical.dispose();
    _horizontal.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => ConstrainedBox(
    constraints: const BoxConstraints(maxHeight: 360),
    child: Scrollbar(
      controller: _vertical,
      thumbVisibility: true,
      child: SingleChildScrollView(
        controller: _vertical,
        child: Scrollbar(
          controller: _horizontal,
          thumbVisibility: true,
          notificationPredicate: (n) => n.depth == 0,
          child: SingleChildScrollView(
            controller: _horizontal,
            scrollDirection: Axis.horizontal,
            padding: const EdgeInsets.only(bottom: 12, right: 12),
            child: SelectableText(
              widget.text,
              key: SelfTestKeys.report,
              style: widget.style,
            ),
          ),
        ),
      ),
    ),
  );
}
