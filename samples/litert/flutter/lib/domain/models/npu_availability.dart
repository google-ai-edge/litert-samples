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

/// Whether `PreferredBackend.npu` can reach an NPU on this device, by the
/// app's copy of flutter_edge_ai_litertlm's private NPU check
/// (`npuAvailabilityFor`): the plugin is not asked, it has no public API for
/// it.
sealed class const NpuAvailability();

/// The check has not answered yet: on Windows its device check runs a
/// process, which never runs on the UI isolate's path. The NPU is not
/// offered until it answers.
final class const NpuPending() extends NpuAvailability;

/// This build carries an NPU stack for this device and the device answers:
/// Qualcomm's FastRPC opens (Android, Linux arm64), or an Intel NPU is
/// present and its dispatch is beside the executable (Windows x64).
/// Necessary, not sufficient: whether a given file runs on this SoC only a
/// load tells. [soc] is the SoC the OS names (Android `ro.soc.model`),
/// [device] the NPU device it lists (Windows: `Intel(R) AI Boost, driver
/// 32.0.100.4778`), [note] what could not be checked (the load decides).
final class const NpuAvailable({
  final String? soc,
  final String? device,
  final String? note,
}) extends NpuAvailability;

/// The app's copy of the plugin's check finds no reachable NPU here (an NPU
/// load would fail with flutter_edge_ai_litertlm's
/// `NpuDispatchUnavailableException`); [reason] is worded as the plugin's
/// private check words it, or as the app's own Windows device check does.
final class const NpuUnavailable(final String reason, {final String? soc})
    extends NpuAvailability;

/// `available (SoC SM8650)`, `available (Intel(R) AI Boost, driver
/// 32.0.100.4778)`, `being checked` or `unavailable: reason`.
String describeNpu(NpuAvailability npu) => switch (npu) {
  NpuPending() => 'being checked',
  NpuAvailable(:final soc, :final device, :final note) => switch ([
    if (soc != null) 'SoC $soc',
    ?device,
    ?note,
  ]) {
    [] => 'available',
    final named => 'available (${named.join('; ')})',
  },
  NpuUnavailable(:final reason, :final soc) =>
    'unavailable: $reason${soc == null ? '' : ' (SoC $soc)'}',
};
