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

import '../../../../config/demos.dart';
import '../../../../config/model_catalog.dart';
import '../../../../data/repositories/hardware_repository.dart';
import '../../../../domain/hardware/device_summary.dart';
import '../../../../domain/models/knowledge.dart';
import '../../../../domain/models/model_id.dart';
import '../../../../domain/models/model_state.dart';

/// One launcher tile: whether the demo can open, and why not.
final class const DemoTile({
  required final Demo demo,
  required final bool available,

  /// The demo's line under its title; Demo 1 names the loaded chat model
  /// and where it runs (`Gemma 4 E2B on the GPU`).
  required final String subtitle,

  /// "Ready", or the first missing model's reason, shown on the tile.
  required final String status,

  /// Ready, but a model runs in an explicitly chosen degraded mode (the
  /// detector on the CPU) or the knowledge base is unavailable: the status
  /// is shown in amber.
  final bool warning = false,
});

/// The launcher. Availability comes from the model states and each demo's
/// required models ([Demo.models]); a demo with a knowledge base also shows
/// its state ([Demo.knowledgeBase]: never blocking). Nothing here loads or
/// opens anything.
class HomeViewModel extends ChangeNotifier {
  HomeViewModel({required this._models, this._knowledge, this._hardware}) {
    _models.addListener(notifyListeners);
    _knowledge?.addListener(notifyListeners);
    _hardwareChanges?.addListener(notifyListeners);
  }

  /// Every model's state (`ModelStates.states`).
  final ValueListenable<Map<ModelId, ModelState>> _models;
  final ValueListenable<KnowledgeStatus>? _knowledge;

  /// The "This device" card's source; no card without it.
  final HardwareRepository? _hardware;
  late final Listenable? _hardwareChanges = _hardware?.changes;

  /// The "This device" card; null when there is no hardware repository.
  DeviceSummary? get device => _hardware?.summary();

  /// The text "Copy diagnostics" puts on the clipboard.
  String diagnosticsReport() => _hardware?.report() ?? '';

  /// Shown under the tiles: the chat is shared, so a switch drops history.
  static const historyNote =
      'Opening a demo starts a fresh conversation; '
      'Voice chat does not keep its chat history across a switch.';

  List<DemoTile> get tiles => [for (final demo in Demo.values) _tile(demo)];

  String _subtitle(Demo demo) {
    if (demo != Demo.voiceChat) return demo.subtitle;
    return switch (_models.value[ModelId.chat]) {
      ModelReady(:final info) =>
        'Talk or type to ${info.chat?.name ?? kDefineChatModel.name} '
            'on the ${info.backend.toUpperCase()}',
      _ => demo.subtitle,
    };
  }

  DemoTile _tile(Demo demo) {
    final states = _models.value;
    final subtitle = _subtitle(demo);
    ({String message, String backend})? detectorFailure;
    for (final id in demo.models) {
      final state = states[id] ?? const ModelPending();
      // Demo 3 still opens when the detector failed on its backend: its
      // screen shows why and the way out (run it on the CPU, or back on the
      // GPU). A Pi 5's GPU may refuse YOLO26n; the camera and voice still
      // work. A bad file keeps the tile disabled: no backend helps.
      if (demo == Demo.liveCamera && id == ModelId.yolo26n) {
        if (state case ModelFailed(:final message, :final backend?)) {
          detectorFailure = (message: message, backend: backend);
          continue;
        }
      }
      final blocker = switch (state) {
        ModelReady() => null,
        ModelUnavailable(:final reason) => reason,
        ModelFailed(:final message) => '${_slotName(id)}: $message',
        ModelPending() ||
        ModelInstalling() ||
        ModelLoading() ||
        ModelWarmingUp() => '${_slotName(id)} is still loading',
      };
      if (blocker != null) {
        return DemoTile(
          demo: demo,
          available: false,
          subtitle: subtitle,
          status: blocker,
        );
      }
    }
    if (detectorFailure case (:final message, :final backend)) {
      final other = backend == 'gpu' ? 'the CPU' : 'the GPU';
      return DemoTile(
        demo: demo,
        available: true,
        subtitle: subtitle,
        status:
            'The detector failed on the ${backend.toUpperCase()}: open Live '
            'camera to run it on $other · $message',
        warning: true,
      );
    }
    final explicitCpu = [
      for (final id in demo.models)
        if (states[id] case ModelReady(:final info) when info.explicitCpu)
          id.spec.displayName,
    ];
    final notes = [
      if (explicitCpu.isNotEmpty) '${explicitCpu.join(', ')} on CPU (chosen)',
    ];
    var warning = explicitCpu.isNotEmpty;
    if (demo.knowledgeBase) {
      switch (_knowledge?.value) {
        case null || KnowledgeReady():
          break;
        case KnowledgeWaiting():
          notes.add('knowledge base loading…');
        case KnowledgeIndexing(:final percent, :final prebuiltSkipped):
          // Embedding on the device: the prebuilt index was not used.
          notes.add(
            'knowledge base indexing $percent%'
            '${prebuiltSkipped == null ? '' : ' (prebuilt index not used)'}',
          );
        case KnowledgeUnavailable(:final reason):
          notes.add('knowledge base unavailable: $reason');
          warning = true;
        case KnowledgeFailed(:final message):
          notes.add('knowledge base failed: $message');
          warning = true;
      }
    }
    return DemoTile(
      demo: demo,
      available: true,
      subtitle: subtitle,
      status: ['Ready', ...notes].join(' · '),
      warning: warning,
    );
  }

  /// The chat slot holds whichever `.litertlm` was chosen, so its errors
  /// name the slot.
  static String _slotName(ModelId id) =>
      id == ModelId.chat ? 'The chat model' : id.spec.displayName;

  @override
  void dispose() {
    _models.removeListener(notifyListeners);
    _knowledge?.removeListener(notifyListeners);
    _hardwareChanges?.removeListener(notifyListeners);
    super.dispose();
  }
}
