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
import 'package:provider/provider.dart';

import '../../../../domain/models/model_id.dart';
import '../../../core/debug_overlay.dart';
import '../../../core/device_card.dart';
import '../../../core/warning_color.dart';
import '../view_models/chat_model_view_model.dart';
import '../view_models/self_test_view_model.dart';
import '../view_models/setup_view_model.dart';
import 'chat_model_section.dart';
import 'self_test_card.dart';

/// Keys for tests.
abstract final class SetupKeys {
  /// A load failure's Retry.
  static const retry = ValueKey('setup-retry');
  static const continueSetup = ValueKey('setup-continue');

  static ValueKey<String> row(ModelId id) => ValueKey('setup-row-${id.name}');
}

/// The Models screen: the Chat model card (the
/// one model the app does not ship), then one row per model with its size,
/// status, progress, error and Retry. On the first run it moves on to home
/// once every required model is ready; opened from home it stays.
class SetupScreen extends StatefulWidget {
  const SetupScreen({super.key, this.onReady});

  /// First run: called once, when every required model is ready.
  final VoidCallback? onReady;

  @override
  State<SetupScreen> createState() => _SetupScreenState();
}

class _SetupScreenState extends State<SetupScreen> {
  late final SetupViewModel _viewModel = context.read<SetupViewModel>();

  /// The "Chat model" card's view model.
  late final ChatModelViewModel _chatModel = context.read<ChatModelViewModel>();

  /// The "Run self-test" card's view model.
  late final SelfTestViewModel _selfTest = context.read<SelfTestViewModel>();
  bool _handedOver = false;

  @override
  void initState() {
    super.initState();
    _viewModel.addListener(_onChanged);
    _onChanged();
  }

  @override
  void dispose() {
    _viewModel.removeListener(_onChanged);
    super.dispose();
  }

  void _onChanged() {
    final onReady = widget.onReady;
    if (onReady == null || _handedOver || !_viewModel.allReady) return;
    _handedOver = true;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) onReady();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          _viewModel.mode == SetupMode.firstRun ? 'Set up models' : 'Models',
        ),
        actions: const [DebugOverlayToggle()],
      ),
      body: ListenableBuilder(
        listenable: _viewModel,
        builder: (context, _) => ListView(
          padding: const EdgeInsets.all(16),
          children: [
            // The chat model first: the only model the app does not ship.
            ChatModelSection(viewModel: _chatModel),
            if (_viewModel.canContinue)
              Align(
                alignment: Alignment.centerLeft,
                child: FilledButton.tonal(
                  key: SetupKeys.continueSetup,
                  onPressed: _viewModel.prepare.execute,
                  child: const Text('Continue'),
                ),
              ),
            for (final row in _viewModel.rows)
              _ModelRowTile(
                key: SetupKeys.row(row.spec.id),
                row: row,
                onRetryLoad: _viewModel.prepare.execute,
              ),
            if (_viewModel.device case final device?)
              DeviceCard(summary: device, report: _viewModel.diagnosticsReport),
            SelfTestCard(viewModel: _selfTest),
          ],
        ),
      ),
    );
  }
}

Color? _colorOf(Tone tone, ThemeData theme) => switch (tone) {
  Tone.normal => null,
  Tone.warning => kWarningColor,
  Tone.error => theme.colorScheme.error,
};

class _ModelRowTile extends StatelessWidget {
  const _ModelRowTile({
    super.key,
    required this.row,
    required this.onRetryLoad,
  });

  final ModelRow row;
  final VoidCallback onRetryLoad;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final spec = row.spec;
    final progress = row.progress;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          spacing: 8,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    row.title ??
                        (spec.required
                            ? spec.displayName
                            : '${spec.displayName} (optional)'),
                    style: theme.textTheme.titleMedium,
                  ),
                ),
                // Nothing to size yet (no chat model chosen).
                if (row.bytes > 0) Text(SetupViewModel.sizeLabel(row.bytes)),
              ],
            ),
            Text(
              row.status,
              style: row.tone == Tone.normal
                  ? null
                  : TextStyle(
                      color: _colorOf(row.tone, theme),
                      fontWeight: FontWeight.w600,
                    ),
            ),
            if (row.source case final source?)
              Text(
                source,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            if (progress != null)
              LinearProgressIndicator(value: progress.isNaN ? null : progress),
            // A failure's message in red; what to do next (no chat model
            // yet) in the row's own tone.
            if (row.detail case final detail?)
              Text(
                detail,
                style: TextStyle(
                  color: row.tone == Tone.warning
                      ? _colorOf(row.tone, theme)
                      : theme.colorScheme.error,
                ),
              ),
            if (row.canRetryLoad)
              Align(
                alignment: Alignment.centerRight,
                child: FilledButton(
                  key: SetupKeys.retry,
                  onPressed: onRetryLoad,
                  child: const Text('Retry'),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
