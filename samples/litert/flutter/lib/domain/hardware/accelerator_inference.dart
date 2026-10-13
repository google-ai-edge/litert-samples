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

import '../models/accelerator_evidence.dart';
import '../models/hardware_profile.dart';

/// The GPU API a model's GPU backend uses on [platform] when no log line
/// says: the only GPU accelerator each platform ships (Metal on Apple,
/// the LiteRT-LM native bundle's WebGPU accelerator on Linux and Windows,
/// OpenCL first on
/// Android). Always [EvidenceSource.inferred].
String? shippedGpuApi(HostPlatform platform) => switch (platform) {
  HostPlatform.macos || HostPlatform.ios => 'Metal',
  HostPlatform.linux => 'WebGPU/Vulkan',
  HostPlatform.windows => 'WebGPU/D3D12',
  HostPlatform.android => 'OpenCL',
  HostPlatform.other => null,
};

/// The GPU a GPU backend most likely opens, from the probe alone. On Linux
/// the Vulkan list decides (Dawn enumerates the same ICDs): a real discrete
/// GPU first, then an integrated one; when Vulkan lists only a software
/// device, that one (and [software] is true). Without `vulkaninfo`, the
/// first GPU the probe found that is not a VM display; without one, the ICD
/// files: only lavapipe/SwiftShader installed means a software GPU. Null
/// when nothing names one.
({String name, bool software})? likelyAdapter(HardwareProfile profile) {
  if (profile.vulkan case final devices? when devices.isNotEmpty) {
    final real = devices.where((d) => !d.isSoftware).toList();
    for (final type in const ['DISCRETE_GPU', 'INTEGRATED_GPU']) {
      for (final d in real) {
        if (d.type == type) return (name: d.name, software: false);
      }
    }
    if (real.isNotEmpty) return (name: real.first.name, software: false);
    return (name: devices.first.name, software: true);
  }
  for (final gpu in profile.gpus) {
    switch (gpu.kind) {
      case GpuKind.virtualDisplay:
        continue;
      case GpuKind.software:
        return (name: gpu.name, software: true);
      case GpuKind.discrete || GpuKind.integrated || GpuKind.unknown:
        return (name: gpu.name, software: false);
    }
  }
  final icds = profile.vulkanIcds;
  if (icds.isNotEmpty && icds.every(isSoftwareVulkanIcd)) {
    return (
      name: 'software Vulkan only (ICDs: ${icds.join(', ')})',
      software: true,
    );
  }
  return null;
}

/// One model's [AcceleratorEvidence] from what the runtime [reported] (null:
/// it cannot report, so the backend is only [requested]), the probe and the
/// native log lines of that model's load. A log fact beats an inference; a
/// software adapter is flagged either way.
AcceleratorEvidence inferEvidence({
  required String requested,
  required String? reported,
  required HardwareProfile? hardware,
  NativeLogEvidence log = const NativeLogEvidence(),
}) {
  final actual = reported ?? requested;
  final backendSource = reported == null
      ? EvidenceSource.requested
      : EvidenceSource.api;
  if (actual != 'gpu') {
    return AcceleratorEvidence(
      requested: requested,
      actual: actual,
      backendSource: backendSource,
      api: actual.toUpperCase(),
      apiSource: backendSource,
      cpuDelegate: log.cpuDelegate,
      samplerOnCpu: log.samplerOnCpu,
      noGpuLogged: log.noGpu,
      logLines: log.lines,
    );
  }

  final (api, apiSource) = switch (log.api) {
    final logged? => (logged, EvidenceSource.log),
    null => (
      hardware == null ? null : shippedGpuApi(hardware.platform),
      EvidenceSource.inferred,
    ),
  };

  final String? adapter;
  final EvidenceSource adapterSource;
  var software = log.softwareGpu;
  if (log.adapter case final line?) {
    adapter = line.adapterType == null
        ? line.name
        : '${line.name} (${line.adapterType})';
    adapterSource = EvidenceSource.log;
  } else {
    final likely = hardware == null ? null : likelyAdapter(hardware);
    adapter = likely?.name;
    adapterSource = EvidenceSource.inferred;
    if (likely != null && likely.software) software = true;
  }

  return AcceleratorEvidence(
    requested: requested,
    actual: actual,
    backendSource: backendSource,
    api: api,
    apiSource: apiSource,
    adapter: adapter,
    adapterSource: adapterSource,
    softwareGpu: software,
    cpuDelegate: log.cpuDelegate,
    samplerOnCpu: log.samplerOnCpu,
    noGpuLogged: log.noGpu,
    logLines: log.lines,
  );
}
