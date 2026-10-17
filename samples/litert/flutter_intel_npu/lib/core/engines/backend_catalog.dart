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

import 'dart:io';

import 'package:flutter_gemma/flutter_gemma.dart';

enum BackendAvailability { available, unavailable }

/// What this host can *attempt*. Availability here is a static, per-OS fact;
/// whether the runtime actually lands on the requested accelerator is only
/// known after load via `InferenceModel.activeBackend`.
class BackendStatus {
  const BackendStatus({
    required this.backend,
    required this.availability,
    this.reason,
  });

  final PreferredBackend backend;
  final BackendAvailability availability;

  /// Human-readable reason when unavailable (rendered in the table).
  final String? reason;

  bool get isAvailable => availability == BackendAvailability.available;
}

class BackendCatalog {
  const BackendCatalog();

  BackendStatus status(PreferredBackend backend) {
    switch (backend) {
      case PreferredBackend.cpu:
      case PreferredBackend.gpu:
        return BackendStatus(
          backend: backend,
          availability: BackendAvailability.available,
        );
      case PreferredBackend.npu:
        if (Platform.isWindows) {
          // Requires Intel Lunar Lake / Panther Lake; the runtime decides.
          return BackendStatus(
            backend: backend,
            availability: BackendAvailability.available,
            reason: 'Intel Lunar/Panther Lake only - check activeBackend',
          );
        }
        return BackendStatus(
          backend: backend,
          availability: BackendAvailability.unavailable,
          reason: 'Windows + Intel Lunar/Panther Lake only',
        );
    }
  }
}

String backendLabel(PreferredBackend? b) => switch (b) {
      PreferredBackend.cpu => 'CPU',
      PreferredBackend.gpu => 'GPU',
      PreferredBackend.npu => 'NPU',
      null => '—',
    };
