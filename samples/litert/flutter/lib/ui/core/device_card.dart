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

import '../../domain/hardware/device_summary.dart';
import 'warning_color.dart';

/// Keys for tests.
abstract final class DeviceCardKeys {
  static const card = ValueKey('device-card');
  static const copy = ValueKey('device-card-copy');
}

/// "This device": chip, RAM, GPU and API, and where each model runs with
/// how that is known. Green = confirmed, amber = inferred or requested,
/// red = mismatch, software GPU or failure.
/// "Copy diagnostics" puts [report]'s text on the clipboard.
class DeviceCard extends StatelessWidget {
  const DeviceCard({super.key, required this.summary, required this.report});

  final DeviceSummary summary;

  /// Builds the report when Copy is pressed (it samples memory then).
  final String Function() report;

  Future<void> _copy(BuildContext context) async {
    final messenger = ScaffoldMessenger.maybeOf(context);
    String message;
    try {
      await Clipboard.setData(ClipboardData(text: report()));
      message = 'Diagnostics copied to the clipboard';
    } on PlatformException catch (e) {
      debugPrint('[DeviceCard] copy failed: $e');
      message = 'Could not copy the diagnostics: ${e.message ?? e.code}';
    }
    messenger?.showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      key: DeviceCardKeys.card,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 8, 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          spacing: 4,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    'THIS DEVICE',
                    style: theme.textTheme.labelMedium,
                  ),
                ),
                TextButton.icon(
                  key: DeviceCardKeys.copy,
                  onPressed: () => unawaited(_copy(context)),
                  icon: const Icon(Icons.copy, size: 18),
                  label: const Text('Copy diagnostics'),
                ),
              ],
            ),
            Text(summary.title, style: theme.textTheme.titleSmall),
            for (final line in summary.device) _Line(line: line),
            if (summary.models.isNotEmpty) const Divider(height: 12),
            for (final line in summary.models) _Line(line: line),
          ],
        ),
      ),
    );
  }
}

class _Line extends StatelessWidget {
  const _Line({required this.line});

  final SummaryLine line;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final color = switch (line.tone) {
      SummaryTone.neutral => null,
      SummaryTone.confirmed =>
        theme.brightness == Brightness.dark
            ? const Color(0xFF81C784)
            : const Color(0xFF2E7D32),
      SummaryTone.caution => kWarningColor,
      SummaryTone.error => theme.colorScheme.error,
    };
    final style = theme.textTheme.bodySmall;
    return Padding(
      padding: const EdgeInsets.only(right: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 116,
            child: Text(
              line.label,
              style: style?.copyWith(fontWeight: FontWeight.w600),
            ),
          ),
          Expanded(
            child: Text(line.value, style: style?.copyWith(color: color)),
          ),
          if (line.tag case final tag?)
            Padding(
              padding: const EdgeInsets.only(left: 8),
              child: Text(
                tag,
                style: style?.copyWith(
                  color: color,
                  fontStyle: FontStyle.italic,
                ),
              ),
            ),
        ],
      ),
    );
  }
}
