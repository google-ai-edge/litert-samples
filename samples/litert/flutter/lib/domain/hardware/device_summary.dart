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

import '../audio/audio_device_checks.dart' show deviceCheckText;
import '../models/accelerator_evidence.dart';
import '../models/audio_devices.dart';
import '../models/hardware_profile.dart';
import 'diagnostics_report.dart' show ModelDiagnostics, formatBytes;

/// How a "This device" line is coloured: green when confirmed,
/// amber when requested or inferred, red for a mismatch, a software GPU or a
/// failure.
enum SummaryTone { neutral, confirmed, caution, error }

/// One line of the card.
final class const SummaryLine({
  required final String label,
  required final String value,
  final SummaryTone tone = SummaryTone.neutral,

  /// `confirmed`, `inferred`, `requested`: shown at the end of the line.
  final String? tag,
});

/// The card's content, ready to lay out.
final class const DeviceSummary({
  /// `Mac16,8 · macOS 26.5.1 (25F71)`.
  required final String title,
  required final List<SummaryLine> device,
  required final List<SummaryLine> models,
});

/// The card from the probe, the audio checks and the model rows. Pure.
DeviceSummary buildDeviceSummary({
  required HardwareProfile? hardware,
  required String? hardwareNote,
  required List<ModelDiagnostics> models,
  AudioDeviceStatus? audio,
}) {
  final hw = hardware;
  if (hw == null) {
    return DeviceSummary(
      title: 'Probing the hardware…',
      device: [
        if (hardwareNote != null)
          SummaryLine(
            label: 'Probe',
            value: hardwareNote,
            tone: SummaryTone.error,
          ),
        ..._audioLines(audio),
      ],
      models: [for (final m in models) ..._modelLines(m)],
    );
  }
  final cpu = hw.cpu;
  final pe = cpu.performanceCores != null && cpu.efficiencyCores != null
      ? ' (${cpu.performanceCores}P+${cpu.efficiencyCores}E)'
      : '';
  final gpus = hw.gpus.where((g) => g.kind != GpuKind.virtualDisplay).toList();
  final softwareVulkan =
      hw.vulkan != null &&
      hw.vulkan!.isNotEmpty &&
      hw.vulkan!.every((d) => d.isSoftware);
  return DeviceSummary(
    title: [?hw.machine, hw.os].join(' · '),
    device: [
      SummaryLine(
        label: 'Chip',
        value: [
          cpu.model,
          '${cpu.cores} cores$pe',
          if (hw.memory case final mem?) 'RAM ${formatBytes(mem.totalBytes)}',
          if (hw.libc case final l? when hw.platform == HostPlatform.linux) l,
        ].join(' · '),
        // Taken from a board name, not named by Android.
        tag: hw.soc?.inferred == true ? 'inferred' : null,
      ),
      if (gpus.isEmpty)
        SummaryLine(
          label: 'GPU',
          // A phone always has one: the probe could not name it.
          value: hw.platform == HostPlatform.android
              ? 'not identified (see Copy diagnostics)'
              : 'none found',
          tone: SummaryTone.caution,
        )
      else
        for (final g in gpus)
          SummaryLine(
            label: 'GPU',
            value: [g.name, ?g.api].join(' · '),
            tone: g.inferred ? SummaryTone.caution : SummaryTone.neutral,
            tag: g.inferred ? 'inferred' : null,
          ),
      if (hw.vulkan case final devices? when !softwareVulkan)
        for (final d in devices.where((d) => !d.isSoftware))
          SummaryLine(
            label: 'Vulkan',
            value: [d.name, ?d.apiVersion].join(' · '),
          ),
      if (softwareVulkan)
        SummaryLine(
          label: 'Vulkan',
          value: '${hw.vulkan!.map((d) => d.name).join(', ')}: software only',
          tone: SummaryTone.error,
        ),
      if (hw.jetson case final j?)
        SummaryLine(
          label: 'Jetson',
          value: [
            if (j.l4tRelease case final r?) 'L4T R$r',
            if (j.jetpack case final v?) 'JetPack $v',
            'power ${j.powerMode ?? 'unknown'}',
          ].join(' · '),
        ),
      for (final n in hw.npuHints) SummaryLine(label: 'NPU', value: n.text),
      if (hw.audio case final a?)
        switch (a.server) {
          final server? => SummaryLine(label: 'Sound', value: server.name),
          null => SummaryLine(
            label: 'Sound',
            value: a.problem ?? 'no sound server',
            tone: SummaryTone.error,
          ),
        },
      ..._audioLines(audio),
    ],
    models: [for (final m in models) ..._modelLines(m)],
  );
}

/// The model's line, and for the chat model a second one with which model
/// it is and how it is set up.
List<SummaryLine> _modelLines(ModelDiagnostics m) => [
  _modelLine(m),
  if (m.chat case final c?)
    SummaryLine(
      label: '',
      value: [
        c.custom ? '${c.name} (your own)' : c.name,
        c.capabilityLine,
        if (c.shortSha case final sha?) 'sha256 $sha',
      ].join(' · '),
      tone: c.custom ? SummaryTone.caution : SummaryTone.neutral,
    ),
];

/// `Mic` and `Speaker`: the device names the voice demos use, or why one is
/// unusable (red).
List<SummaryLine> _audioLines(AudioDeviceStatus? audio) => [
  if (audio != null) ...[
    _deviceLine('Mic', audio.input),
    _deviceLine('Speaker', audio.output),
  ],
];

SummaryLine _deviceLine(String label, DeviceCheck check) => SummaryLine(
  label: label,
  value: deviceCheckText(check),
  tone: switch (check) {
    DeviceUnchecked() => SummaryTone.neutral,
    DeviceReady(:final caution) =>
      caution ? SummaryTone.caution : SummaryTone.neutral,
    DeviceUnavailable() => SummaryTone.error,
  },
);

SummaryLine _modelLine(ModelDiagnostics m) {
  final e = m.evidence;
  if (e == null) {
    return SummaryLine(
      label: m.name,
      value: m.state,
      tone: m.state.startsWith('failed')
          ? SummaryTone.error
          : SummaryTone.neutral,
    );
  }
  final value = [
    (e.backendSource == EvidenceSource.requested
            ? e.requested.toUpperCase()
            : '${e.requested.toUpperCase()} → ${e.actual.toUpperCase()}') +
        // The detector on the CPU by the user's (or the build's) choice.
        (m.explicitCpu ? ' (chosen)' : ''),
    if (e.actual == 'gpu' && e.api != null) e.api!,
    ?e.adapter,
    if (e.softwareGpu) 'SOFTWARE GPU',
    if (e.mismatch) 'MISMATCH',
  ].join(' · ');
  final sources = [
    e.backendSource,
    if (e.actual == 'gpu') e.apiSource,
    if (e.actual == 'gpu' && e.adapter != null) e.adapterSource,
  ];
  final tag = sources.contains(EvidenceSource.requested)
      ? 'requested'
      : sources.contains(EvidenceSource.inferred)
      ? 'inferred'
      : 'confirmed';
  return SummaryLine(
    label: m.name,
    value: value,
    tag: tag,
    tone: e.isError
        ? SummaryTone.error
        : tag == 'confirmed' && !m.explicitCpu
        ? SummaryTone.confirmed
        : SummaryTone.caution,
  );
}
