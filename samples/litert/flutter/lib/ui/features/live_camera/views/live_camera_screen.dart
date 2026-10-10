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

import '../../../../domain/models/detection.dart';
import '../../../../domain/models/live_state.dart';
import '../../../../domain/models/voice.dart';
import '../../../core/debug_overlay.dart';
import '../../../core/detection_painter.dart';
import '../../../core/level_meter.dart';
import '../../../core/warning_color.dart';
import '../view_models/live_camera_view_model.dart';
import 'camera_settings_sheet.dart';
import 'live_preview.dart';

/// Keys for tests.
abstract final class LiveCameraKeys {
  static const preview = ValueKey('live-camera-preview');
  static const boxes = ValueKey('live-camera-boxes');
  static const status = ValueKey('live-camera-status');
  static const blackFrames = ValueKey('live-camera-black-frames');
  static const retryLive = ValueKey('live-camera-retry-live');
  static const retryChat = ValueKey('live-camera-retry-chat');
  static const mic = ValueKey('live-camera-mic');
  static const caption = ValueKey('live-camera-caption');
  static const answer = ValueKey('live-camera-answer');
  static const routeChip = ValueKey('live-camera-route');
  static const phase = ValueKey('live-camera-phase');
  static const frozen = ValueKey('live-camera-frozen');
  static const frozenLabel = ValueKey('live-camera-frozen-label');
  static const retryStt = ValueKey('live-camera-retry-stt');
  static const detailedOff = ValueKey('live-camera-detailed-off');
  static const micAccessError = ValueKey('live-camera-mic-access-error');
  static const retryMicAccess = ValueKey('live-camera-retry-mic-access');
  static const settings = ValueKey('live-camera-settings');
  static const useNetworkCamera = ValueKey('live-camera-use-network-camera');
  static const detectorFailure = ValueKey('live-camera-detector-failure');
  static const otherBackend = ValueKey('live-camera-other-backend');
}

/// Demo 3: the live preview with detector boxes, the live status, and
/// push-to-talk questions about the scene: a caption with the question, the
/// streaming answer and its route chip, and the mic with its level meter.
class LiveCameraScreen extends StatelessWidget {
  const LiveCameraScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final viewModel = context.read<LiveCameraViewModel>();
    return Scaffold(
      appBar: AppBar(
        title: const Text('Live camera'),
        actions: [
          if (viewModel.settingsAvailable)
            _SettingsButton(viewModel: viewModel),
          const DebugOverlayToggle(),
        ],
      ),
      body: ListenableBuilder(
        listenable: viewModel,
        builder: (context, _) => Column(
          children: [
            Expanded(
              child: ClipRect(
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    LivePreview(
                      key: LiveCameraKeys.preview,
                      preview: viewModel.preview,
                    ),
                    // Repaints at the detection rate without rebuilding, and
                    // without repainting the preview under it.
                    RepaintBoundary(
                      child: IgnorePointer(
                        child: CustomPaint(
                          key: LiveCameraKeys.boxes,
                          painter: DetectionPainter(
                            frames: viewModel.frames,
                            mirror: viewModel.mirrorBoxes,
                          ),
                        ),
                      ),
                    ),
                    // Over the live view while a detailed answer is about
                    // one frame; rebuilds only on freeze and unfreeze.
                    _FrozenLayer(viewModel: viewModel),
                    Positioned(
                      left: 12,
                      right: 12,
                      bottom: 12,
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        spacing: 8,
                        children: [
                          _Caption(viewModel: viewModel),
                          _BlackFramesWarning(viewModel: viewModel),
                          _LiveStatus(viewModel: viewModel),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
            _VoiceBar(viewModel: viewModel),
          ],
        ),
      ),
    );
  }
}

/// The frame a detailed answer is about: the snapshot's
/// pixels with its own boxes, cover-fitted like the preview (and flipped
/// like it when the preview and frames are mirrored differently), labelled
/// "Answering about this frame". A tap goes back to the live view; the
/// answer goes on.
class _FrozenLayer extends StatelessWidget {
  const _FrozenLayer({required this.viewModel});

  final LiveCameraViewModel viewModel;

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<FrozenFrame?>(
      valueListenable: viewModel.frozen,
      builder: (context, frozen, _) {
        if (frozen == null) return const SizedBox.shrink();
        return GestureDetector(
          key: LiveCameraKeys.frozen,
          behavior: HitTestBehavior.opaque,
          onTap: viewModel.unfreeze,
          child: Stack(
            fit: StackFit.expand,
            children: [
              Transform.flip(
                flipX: frozen.flip,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    const ColoredBox(color: Colors.black),
                    RawImage(image: frozen.image, fit: BoxFit.cover),
                    CustomPaint(
                      painter: DetectionPainter(
                        frames: viewModel.frozenBoxes,
                        mirror: false,
                      ),
                    ),
                  ],
                ),
              ),
              const Positioned(
                top: 12,
                left: 12,
                child: _Chip(
                  chipKey: LiveCameraKeys.frozenLabel,
                  text: 'Answering about this frame · tap for live',
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

/// The live pipeline's state as a chip, with a Retry on failure.
class _LiveStatus extends StatelessWidget {
  const _LiveStatus({required this.viewModel});

  final LiveCameraViewModel viewModel;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return ValueListenableBuilder<LiveState>(
      valueListenable: viewModel.liveState,
      builder: (context, state, _) {
        if (viewModel.detectorReloading) {
          return _Chip(
            text:
                'Loading the detector on the '
                '${viewModel.backend.name.toUpperCase()}…',
          );
        }
        // A detector that failed on its backend: the reason, and the other
        // backend as the explicit way out (never switched by itself).
        if (viewModel.detectorFailure case final failure?) {
          return _DetectorFailureCard(viewModel: viewModel, failure: failure);
        }
        if (viewModel.settingsActionError case final String error) {
          return _Failure(message: error, viewModel: viewModel);
        }
        if (viewModel.startError case final String error) {
          return _Failure(message: error, viewModel: viewModel);
        }
        final detector = viewModel.detectorLabel ?? 'detector';
        return switch (state) {
          LiveStopped() when viewModel.startLive.running => _Chip(
            text: viewModel.sourceIsNetwork
                ? 'Connecting to the network camera…'
                : 'Starting…',
          ),
          LiveStopped() => const SizedBox.shrink(),
          LiveStarting() => _Chip(
            text: viewModel.sourceIsNetwork
                ? 'Connecting to the network camera…'
                : 'Starting the frame source…',
          ),
          LiveRunning(:final source) when viewModel.sourceIsNetwork =>
            // The camera's own rate and size, at the stats rate (≤4 Hz).
            ValueListenableBuilder<LiveStats>(
              valueListenable: viewModel.stats,
              builder: (context, stats, _) => _Chip(
                text: [
                  source,
                  if (stats.sourceWidth > 0)
                    '${stats.sourceWidth}×${stats.sourceHeight}',
                  if (stats.sourceWidth > 0)
                    '${stats.sourceFps.toStringAsFixed(0)} fps',
                  detector,
                ].join(' · '),
                color: viewModel.detectorOnCpu ? kWarningColor : null,
              ),
            ),
          LiveRunning(:final source) => _Chip(
            text: '$source · $detector',
            color: viewModel.detectorOnCpu ? kWarningColor : null,
          ),
          LivePaused(:final reason) => _Chip(
            text: 'Detector paused ($reason)',
            color: colors.tertiaryContainer,
          ),
          LiveFailed(:final message) => _Failure(
            message: message,
            viewModel: viewModel,
          ),
        };
      },
    );
  }
}

/// An amber chip while the camera delivers black frames (the pipeline keeps
/// running). Rebuilds only when the warning turns on or off.
class _BlackFramesWarning extends StatelessWidget {
  const _BlackFramesWarning({required this.viewModel});

  final LiveCameraViewModel viewModel;

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<bool>(
      valueListenable: viewModel.blackFrames,
      builder: (context, black, _) => black
          ? Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: _Chip(
                chipKey: LiveCameraKeys.blackFrames,
                text: viewModel.blackFramesWarning,
                color: kWarningColor,
              ),
            )
          : const SizedBox.shrink(),
    );
  }
}

class _Chip extends StatelessWidget {
  const _Chip({
    required this.text,
    this.color,
    this.chipKey = LiveCameraKeys.status,
  });

  final String text;
  final Color? color;
  final Key chipKey;

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: Alignment.bottomLeft,
      child: DecoratedBox(
        key: chipKey,
        decoration: BoxDecoration(
          color: color ?? Colors.black.withValues(alpha: 0.6),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          child: Text(
            text,
            style: TextStyle(
              color: color == null ? Colors.white : Colors.black,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      ),
    );
  }
}

/// A live failure with its one explicit action: Retry (Reconnect for a
/// network camera). When the device camera failed and the source is the
/// user's to choose, it also offers the network camera — first on Linux,
/// where boards often have no camera.
class _Failure extends StatelessWidget {
  const _Failure({required this.message, required this.viewModel});

  final String message;
  final LiveCameraViewModel viewModel;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final retry = viewModel.startLive.running ? null : viewModel.retryStart;
    final network = viewModel.offerNetworkCamera
        ? () => showCameraSettings(context, viewModel)
        : null;
    final prominent = network != null && viewModel.preferNetworkCamera;
    return Card(
      color: colors.errorContainer,
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          mainAxisSize: MainAxisSize.min,
          spacing: 8,
          children: [
            Text(
              message,
              key: LiveCameraKeys.status,
              style: TextStyle(color: colors.onErrorContainer),
            ),
            Wrap(
              alignment: WrapAlignment.end,
              spacing: 8,
              runSpacing: 8,
              children: [
                if (network != null)
                  prominent
                      ? FilledButton.icon(
                          key: LiveCameraKeys.useNetworkCamera,
                          onPressed: network,
                          icon: const Icon(Icons.wifi),
                          label: const Text('Use a network camera'),
                        )
                      : OutlinedButton.icon(
                          key: LiveCameraKeys.useNetworkCamera,
                          onPressed: network,
                          icon: const Icon(Icons.wifi),
                          label: const Text('Network camera…'),
                        ),
                prominent
                    ? OutlinedButton(
                        key: LiveCameraKeys.retryLive,
                        onPressed: retry,
                        child: Text(viewModel.retryLabel),
                      )
                    : FilledButton(
                        key: LiveCameraKeys.retryLive,
                        onPressed: retry,
                        child: Text(viewModel.retryLabel),
                      ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

/// The detector failed or is unavailable: the reason, and — when it failed
/// on a backend — the other one as the one explicit action ("Run detector
/// on CPU"), unless the build fixes the backend.
class _DetectorFailureCard extends StatelessWidget {
  const _DetectorFailureCard({required this.viewModel, required this.failure});

  final LiveCameraViewModel viewModel;
  final DetectorFailure failure;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final backend = failure.backend;
    final onGpu = backend == DetectorBackend.gpu.name;
    final other = onGpu ? DetectorBackend.cpu : DetectorBackend.gpu;
    final lock = viewModel.backendLock;
    return Card(
      key: LiveCameraKeys.detectorFailure,
      color: colors.errorContainer,
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          mainAxisSize: MainAxisSize.min,
          spacing: 8,
          children: [
            Text(
              backend == null
                  ? 'The detector is not available: ${failure.message}'
                  : 'The detector failed on the ${backend.toUpperCase()}: '
                        '${failure.message}',
              key: LiveCameraKeys.status,
              style: TextStyle(color: colors.onErrorContainer),
            ),
            if (backend == null)
              const SizedBox.shrink()
            else if (lock != null)
              Text(
                'The backend is fixed by the build ($lock).',
                style: TextStyle(color: colors.onErrorContainer),
              )
            else if (viewModel.canChooseBackend)
              Align(
                alignment: Alignment.centerRight,
                child: FilledButton(
                  key: LiveCameraKeys.otherBackend,
                  onPressed: viewModel.applying
                      ? null
                      : () => viewModel.runDetectorOn(other),
                  child: Text(
                    onGpu ? 'Run detector on CPU' : 'Run detector on the GPU',
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

/// Opens Demo 3's settings: a labelled button on Linux (the network camera
/// is the usual source there), an icon elsewhere.
class _SettingsButton extends StatelessWidget {
  const _SettingsButton({required this.viewModel});

  final LiveCameraViewModel viewModel;

  @override
  Widget build(BuildContext context) {
    void open() => showCameraSettings(context, viewModel);
    return viewModel.preferNetworkCamera
        ? TextButton.icon(
            key: LiveCameraKeys.settings,
            onPressed: open,
            icon: const Icon(Icons.videocam_outlined),
            label: const Text('Camera'),
          )
        : IconButton(
            key: LiveCameraKeys.settings,
            tooltip: 'Camera and detector',
            onPressed: open,
            icon: const Icon(Icons.tune),
          );
  }
}

/// The last question, the answer as it streams (only this text rebuilds
/// per token), the route chip and any notice.
class _Caption extends StatelessWidget {
  const _Caption({required this.viewModel});

  final LiveCameraViewModel viewModel;

  @override
  Widget build(BuildContext context) {
    final exchange = viewModel.exchange;
    final active = viewModel.phase.isActive;
    if (!active &&
        exchange.question == null &&
        exchange.answer == null &&
        exchange.notice == null) {
      return const SizedBox.shrink();
    }
    final theme = Theme.of(context);
    return DecoratedBox(
      key: LiveCameraKeys.caption,
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.65),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          spacing: 6,
          children: [
            if (exchange.question case final String question)
              Text(
                '“$question”',
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: Colors.white70,
                  fontStyle: FontStyle.italic,
                ),
              ),
            ValueListenableBuilder<String>(
              valueListenable: viewModel.partialReply,
              builder: (context, partial, _) {
                final text = partial.isNotEmpty ? partial : exchange.answer;
                if (text == null) return const SizedBox.shrink();
                return Text(
                  text,
                  key: LiveCameraKeys.answer,
                  style: theme.textTheme.titleMedium?.copyWith(
                    color: Colors.white,
                  ),
                );
              },
            ),
            if (exchange.route case final String route)
              DecoratedBox(
                key: LiveCameraKeys.routeChip,
                decoration: BoxDecoration(
                  color: exchange.detailed
                      ? theme.colorScheme.tertiaryContainer
                      : theme.colorScheme.secondaryContainer,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 4,
                  ),
                  child: Text(route, style: theme.textTheme.labelMedium),
                ),
              ),
            if (exchange.notice case final String notice)
              Text(
                notice,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: kWarningColor,
                ),
              ),
          ],
        ),
      ),
    );
  }
}

/// The mic, the turn's phase, and the camera chat's state (with a Retry
/// when it failed to open).
class _VoiceBar extends StatelessWidget {
  const _VoiceBar({required this.viewModel});

  final LiveCameraViewModel viewModel;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final phase = switch (viewModel.phase) {
      TurnPhase.idle => 'Hold the mic and ask about the scene',
      TurnPhase.openingMic => kOpeningMicLabel,
      TurnPhase.listening => 'Listening… release to ask',
      TurnPhase.transcribing => 'Transcribing…',
      TurnPhase.thinking => 'Answering…',
      TurnPhase.speaking => 'Speaking… press the mic to interrupt',
      TurnPhase.error => 'The last question failed — try again',
    };
    return SafeArea(
      top: false,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        child: Row(
          spacing: 12,
          children: [
            MicButton(
              key: LiveCameraKeys.mic,
              level: viewModel.inputLevel,
              phase: viewModel.phase,
              enabled: viewModel.canTalk,
              onDown: () => unawaited(viewModel.pressMic()),
              onUp: () => unawaited(viewModel.releaseMic()),
            ),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    phase,
                    key: LiveCameraKeys.phase,
                    style: theme.textTheme.bodyMedium,
                  ),
                  switch (viewModel) {
                    LiveCameraViewModel(:final String chatError) => Text(
                      chatError,
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: theme.colorScheme.error,
                      ),
                    ),
                    LiveCameraViewModel(
                      chatReady: true,
                      :final String detailedOffReason,
                    ) =>
                      Text(
                        detailedOffReason,
                        key: LiveCameraKeys.detailedOff,
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: kWarningColor,
                        ),
                      ),
                    LiveCameraViewModel(chatReady: true) => Text(
                      'Chat ready · simple questions use the detector only; '
                      'detailed ones send the frame to '
                      '${viewModel.chatModelName}',
                      style: theme.textTheme.bodySmall,
                    ),
                    _ => Text(
                      'Opening the camera chat…',
                      style: theme.textTheme.bodySmall,
                    ),
                  },
                  if (viewModel.sttError case final String sttError)
                    Text(
                      sttError,
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: theme.colorScheme.error,
                      ),
                    ),
                  if (viewModel.micAccessError case final String micError)
                    Text(
                      micError,
                      key: LiveCameraKeys.micAccessError,
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: theme.colorScheme.error,
                      ),
                    ),
                ],
              ),
            ),
            if (viewModel.chatError != null)
              FilledButton(
                key: LiveCameraKeys.retryChat,
                onPressed: viewModel.open.running
                    ? null
                    : viewModel.open.execute,
                child: const Text('Retry'),
              ),
            if (viewModel.micAccessError != null)
              FilledButton(
                key: LiveCameraKeys.retryMicAccess,
                onPressed: viewModel.micAccess.running
                    ? null
                    : viewModel.micAccess.execute,
                child: const Text('Retry mic'),
              ),
            if (viewModel.sttError != null)
              FilledButton(
                key: LiveCameraKeys.retryStt,
                onPressed: viewModel.selectStt.running
                    ? null
                    : viewModel.selectStt.execute,
                child: const Text('Retry STT'),
              ),
          ],
        ),
      ),
    );
  }
}
