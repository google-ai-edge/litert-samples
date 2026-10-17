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

import '../../../../domain/models/camera_source.dart';
import '../../../../domain/models/detection.dart';
import '../view_models/live_camera_view_model.dart';

/// Keys for tests.
abstract final class CameraSettingsKeys {
  static const sheet = ValueKey('camera-settings-sheet');
  static const deviceCamera = ValueKey('camera-settings-device');
  static const networkCamera = ValueKey('camera-settings-network');
  static const url = ValueKey('camera-settings-url');
  static const gpu = ValueKey('camera-settings-gpu');
  static const cpu = ValueKey('camera-settings-cpu');
  static const apply = ValueKey('camera-settings-apply');
  static const clearMemory = ValueKey('camera-settings-clear-memory');
  static const memorySwitch = ValueKey('camera-settings-memory-switch');
  static const memoryDefaultReason = ValueKey(
    'camera-settings-memory-default-reason',
  );
}

/// Opens Demo 3's settings: the camera source and the detector backend.
/// Applying runs the view model's commands; the sheet itself decides
/// nothing.
Future<void> showCameraSettings(
  BuildContext context,
  LiveCameraViewModel viewModel,
) => showModalBottomSheet<void>(
  context: context,
  isScrollControlled: true,
  showDragHandle: true,
  builder: (context) => Padding(
    padding: EdgeInsets.only(bottom: MediaQuery.viewInsetsOf(context).bottom),
    child: CameraSettingsSheet(viewModel: viewModel),
  ),
);

/// The sheet's form. Camera: "Device camera" / "Network camera (URL)" with
/// the MJPEG URL; Detector: "GPU (default)" / "CPU (slower)". A setting the
/// build fixes (`FRAME_SOURCE`, `DETECTOR_BACKEND`) is shown locked with
/// the define that fixes it.
class CameraSettingsSheet extends StatefulWidget {
  const CameraSettingsSheet({super.key, required this.viewModel});

  final LiveCameraViewModel viewModel;

  @override
  State<CameraSettingsSheet> createState() => _CameraSettingsSheetState();
}

class _CameraSettingsSheetState extends State<CameraSettingsSheet> {
  late CameraSourceKind _kind = widget.viewModel.sourceChoice.kind;
  late DetectorBackend _backend = widget.viewModel.backend;
  late final TextEditingController _url = TextEditingController(
    text: widget.viewModel.sourceChoice.networkUrl,
  );
  String? _urlError;

  @override
  void dispose() {
    _url.dispose();
    super.dispose();
  }

  void _apply() {
    final error = widget.viewModel.applySettings(
      kind: _kind,
      networkUrl: _url.text,
      backend: _backend,
    );
    if (error != null) {
      setState(() => _urlError = error);
      return;
    }
    Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    final vm = widget.viewModel;
    final theme = Theme.of(context);
    final sourceLock = vm.sourceLock;
    final backendLock = vm.backendLock;
    return SafeArea(
      child: SingleChildScrollView(
        key: CameraSettingsKeys.sheet,
        padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          mainAxisSize: MainAxisSize.min,
          spacing: 12,
          children: [
            Text('Camera', style: theme.textTheme.titleMedium),
            SegmentedButton<CameraSourceKind>(
              segments: const [
                ButtonSegment(
                  value: CameraSourceKind.device,
                  icon: Icon(Icons.photo_camera_outlined),
                  label: Text(
                    'Device camera',
                    key: CameraSettingsKeys.deviceCamera,
                  ),
                ),
                ButtonSegment(
                  value: CameraSourceKind.network,
                  icon: Icon(Icons.wifi),
                  label: Text(
                    'Network camera (URL)',
                    key: CameraSettingsKeys.networkCamera,
                  ),
                ),
              ],
              selected: {_kind},
              onSelectionChanged: sourceLock != null
                  ? null
                  : (selection) => setState(() => _kind = selection.single),
            ),
            if (sourceLock != null)
              Text(
                'Fixed by the build: $sourceLock',
                style: theme.textTheme.bodySmall,
              )
            else if (_kind == CameraSourceKind.network)
              TextField(
                key: CameraSettingsKeys.url,
                controller: _url,
                keyboardType: TextInputType.url,
                autocorrect: false,
                decoration: InputDecoration(
                  labelText: 'MJPEG stream URL',
                  hintText: kNetworkCameraUrlExample,
                  helperText:
                      'A phone with the free IP Webcam app on the same '
                      'Wi-Fi: tap "Start server", then use the address it '
                      'shows followed by /video. A user and password in the '
                      'URL (http://user:password@…) are saved on this device '
                      'and sent in clear over http.',
                  helperMaxLines: 5,
                  errorText: _urlError,
                  errorMaxLines: 3,
                  border: const OutlineInputBorder(),
                ),
                onChanged: (_) {
                  if (_urlError != null) setState(() => _urlError = null);
                },
                onSubmitted: (_) => _apply(),
              ),
            const SizedBox(height: 4),
            Text('Detector', style: theme.textTheme.titleMedium),
            SegmentedButton<DetectorBackend>(
              segments: const [
                ButtonSegment(
                  value: DetectorBackend.gpu,
                  icon: Icon(Icons.memory),
                  label: Text('GPU (default)', key: CameraSettingsKeys.gpu),
                ),
                ButtonSegment(
                  value: DetectorBackend.cpu,
                  icon: Icon(Icons.developer_board),
                  label: Text('CPU (slower)', key: CameraSettingsKeys.cpu),
                ),
              ],
              selected: {_backend},
              onSelectionChanged: vm.canChooseBackend
                  ? (selection) => setState(() => _backend = selection.single)
                  : null,
            ),
            Text(
              backendLock != null
                  ? 'Fixed by the build: $backendLock'
                  : 'The GPU runs YOLO26n in about 4 ms per frame on a Mac, '
                        'about 20 ms on recent phones. Choose the '
                        'CPU only when the GPU fails (e.g. on a Raspberry '
                        'Pi): it is much slower, and it is never chosen for '
                        'you.',
              style: theme.textTheme.bodySmall,
            ),
            if (vm.memoryStatus != null)
              ListenableBuilder(
                // The switch changes the view model, not this sheet's state.
                listenable: vm,
                builder: (context, _) => _MemorySettings(viewModel: vm),
              ),
            if (vm.settingsError case final String error)
              Text(
                error,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.error,
                ),
              ),
            FilledButton(
              key: CameraSettingsKeys.apply,
              onPressed: _apply,
              child: const Text('Apply'),
            ),
          ],
        ),
      ),
    );
  }
}

/// The sheet's visual memory part: the switch (applied and saved at once,
/// unlike the camera and detector, which wait for Apply), what it costs,
/// and "Clear memory".
class _MemorySettings extends StatelessWidget {
  const _MemorySettings({required this.viewModel});

  final LiveCameraViewModel viewModel;

  @override
  Widget build(BuildContext context) {
    final vm = viewModel;
    final theme = Theme.of(context);
    final enabled = vm.memoryEnabled;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      mainAxisSize: MainAxisSize.min,
      spacing: 8,
      children: [
        const SizedBox(height: 4),
        SwitchListTile(
          key: CameraSettingsKeys.memorySwitch,
          contentPadding: EdgeInsets.zero,
          title: Text('Visual memory', style: theme.textTheme.titleMedium),
          subtitle: switch (vm.memoryDefaultReason) {
            final reason? => Text(
              reason,
              key: CameraSettingsKeys.memoryDefaultReason,
            ),
            null => null,
          },
          value: enabled ?? false,
          onChanged: enabled == null || vm.setMemoryEnabled.running
              ? null
              : (on) => vm.setMemoryEnabled.execute(on),
        ),
        Text(
          'Frames the embedder remembers while the camera runs, so a '
          'question like "Where was the yellow mug?" can find them. Kept in '
          'memory only, and forgotten when you leave this demo or change '
          'the camera. Turning it on the first time loads the embedder\'s '
          'image encoder (a few seconds, about 300–450 MB more memory) until '
          'the app closes.',
          style: theme.textTheme.bodySmall,
        ),
        if (vm.memorySettingError case final String error)
          Text(
            error,
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.error,
            ),
          ),
        if (enabled ?? false)
          Align(
            alignment: Alignment.centerLeft,
            child: OutlinedButton.icon(
              key: CameraSettingsKeys.clearMemory,
              onPressed: () {
                vm.clearMemory();
                ScaffoldMessenger.maybeOf(context)?.showSnackBar(
                  const SnackBar(content: Text('Memory cleared')),
                );
              },
              icon: const Icon(Icons.delete_sweep_outlined),
              label: const Text('Clear memory'),
            ),
          ),
      ],
    );
  }
}
