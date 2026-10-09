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

/// Whether flutter_edge_ai would accept `PreferredBackend.npu` on this
/// device.
sealed class const NpuAvailability();

/// The NPU dispatch stack ships here and (Android) the device's Qualcomm
/// FastRPC library opens. Necessary, not sufficient: the model must also be
/// compiled for this SoC, which only a load can tell. [soc] is the SoC the
/// OS names (Android `ro.soc.model`), when it names one.
final class const NpuAvailable({final String? soc}) extends NpuAvailability;

/// flutter_edge_ai drops `npu` from its candidates here (so a request would
/// run on the GPU or CPU); [reason] is its own wording.
final class const NpuUnavailable(final String reason, {final String? soc})
    extends NpuAvailability;

/// `available (SoC SM8650)` or `unavailable: <reason>`.
String describeNpu(NpuAvailability npu) => switch (npu) {
  NpuAvailable(:final soc) =>
    'available${soc == null ? '' : ' (SoC $soc)'}: the model must be '
        'compiled for this SoC',
  NpuUnavailable(:final reason, :final soc) =>
    'unavailable: $reason${soc == null ? '' : ' (SoC $soc)'}',
};
