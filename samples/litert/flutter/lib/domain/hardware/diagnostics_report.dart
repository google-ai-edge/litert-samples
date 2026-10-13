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

/// The plain-text diagnostics report ("Copy diagnostics") and the one-line
/// logs. Pure: golden-tested.
library;

import '../audio/audio_device_checks.dart' show deviceCheckText;
import '../models/accelerator_evidence.dart';
import '../models/audio_devices.dart';
import '../models/hardware_profile.dart';
import '../models/model_state.dart' show ChatModelFacts;

/// One model's row in the report.
final class const ModelDiagnostics({
  /// `Gemma 4 E2B`.
  required final String name,

  /// `ready`, `loading`, `failed: …`, `unavailable: …`.
  required final String state,
  final AcceleratorEvidence? evidence,

  /// The runtime's own label (`GPU fp32 full`, `CPU · 768-d`).
  final String? detail,
  final Duration? loadTime,
  final Duration? warmUpTime,

  /// On the CPU because a flag asked for it (`DETECTOR_BACKEND=cpu`): amber
  /// on the card even though confirmed.
  final bool explicitCpu = false,

  /// The chat model's facts: which model, its source, checksum, context,
  /// images and tools. Null for the others.
  final ChatModelFacts? chat,
});

/// Everything the report shows.
final class const DiagnosticsInput({
  required final DateTime generatedAt,
  required final BuildInfo build,
  final HardwareProfile? hardware,

  /// Why [hardware] is missing (probe failed or still running).
  final String? hardwareNote,
  final List<ModelDiagnostics> models = const [],
  final MemorySnapshot? memory,

  /// The voice demos' input and output as last checked; null leaves the
  /// AUDIO block out.
  final AudioDeviceStatus? audio,

  /// The native log tap's description.
  required final String nativeLog,
});

const _width = 11;

String _row(String label, String value) => '${label.padRight(_width)}$value';

/// The whole report.
String formatDiagnostics(DiagnosticsInput input) {
  final b = input.build;
  final lines = <String>[
    'LiteRT Demos diagnostics',
    _row('generated', _utc(input.generatedAt)),
    _row('app', '${b.appVersion} · ${b.buildMode} build'),
    _row('flutter', '${b.flutterVersion ?? 'unknown'} · Dart ${b.dartVersion}'),
    _row('packages', _packages(b)),
    _row('native log', input.nativeLog),
    '',
    'DEVICE',
    if (input.hardware case final hw?)
      ...deviceLines(hw)
    else
      _row('probe', input.hardwareNote ?? 'not run'),
    if (input.audio case final a?) ...[
      '',
      'AUDIO',
      _row('input', _deviceCheckRow(a.input)),
      _row('output', _deviceCheckRow(a.output)),
    ],
    '',
    'MODELS',
    if (input.models.isEmpty) _row('none', 'no model state yet'),
    for (final m in input.models) ..._modelLines(m),
    '',
    'MEMORY',
    _row(
      'now',
      input.memory == null ? 'not sampled' : memoryText(input.memory!),
    ),
  ];
  return '${lines.join('\n')}\n';
}

String _deviceCheckRow(DeviceCheck c) => switch (c) {
  DeviceUnavailable() => 'UNAVAILABLE · ${deviceCheckText(c)}',
  DeviceReady(caution: true) => '${deviceCheckText(c)} · check it',
  _ => deviceCheckText(c),
};

String _packages(BuildInfo b) =>
    [for (final MapEntry(:key, :value) in b.packages.entries) '$key $value']
        .join(' · ');

String _utc(DateTime t) {
  final s = t.toUtc().toIso8601String();
  final dot = s.indexOf('.');
  return '${dot < 0 ? s.replaceFirst('Z', '') : s.substring(0, dot)}Z';
}

List<String> _modelLines(ModelDiagnostics m) {
  final times = [
    if (m.loadTime case final t?) 'load ${seconds(t)}',
    if (m.warmUpTime case final t?) 'warm-up ${seconds(t)}',
  ];
  final e = m.evidence;
  return [
    m.name,
    _row('  state', [m.state, ...times].join(' · ')),
    if (e != null) ...evidenceRows(e).map((r) => _row('  ${r.$1}', r.$2)),
    if (m.detail case final d?) _row('  detail', d),
    if (m.chat case final c?) ...chatModelRows(c).map((r) => _row(r.$1, r.$2)),
    if (e != null)
      for (final line in e.logLines) _row('  log', line),
  ];
}

/// The chat model's report rows (also the self-test's): name, source, full
/// SHA-256, context (asked vs built), images, tools. The backend row above
/// already shows requested → actual (`activeBackend`).
List<(String, String)> chatModelRows(ChatModelFacts c) => [
  (
    '  model',
    '${c.name} (${c.custom ? 'your own .litertlm' : 'GEMMA_MODEL_PATH'}, type '
        '${c.modelType})',
  ),
  ('  source', c.source),
  (
    '  sha256',
    c.sha256 == null
        ? 'unknown (${c.custom ? 'not computed yet' : 'GEMMA_MODEL_PATH'})'
        : '${c.sha256} (${c.checksumMatched ? 'matches the published one' : 'computed at import, nothing published to compare'})',
  ),
  (
    '  context',
    c.contextTokens == c.requestedContext
        ? '${c.contextTokens} tokens'
        : '${c.contextTokens} tokens (asked ${c.requestedContext}; the engine '
              'raised it)',
  ),
  ('  images', c.images ? 'on' : 'off'),
  ('  tools', c.tools ? 'on' : 'off'),
];

/// The device block, one `label  value` line per fact (also the self-test's).
List<String> deviceLines(HardwareProfile p) {
  final cpu = p.cpu;
  final cores = [
    '${cpu.cores} cores',
    if (cpu.performanceCores != null && cpu.efficiencyCores != null)
      '(${cpu.performanceCores}P+${cpu.efficiencyCores}E)',
    if (cpu.presentCores case final present?) 'online of $present',
  ].join(' ');
  return [
    _row(
      'os',
      [p.os, if (p.kernel case final k?) 'kernel $k', ?p.libc].join(' · '),
    ),
    if (p.machine case final m?) _row('machine', m),
    if (p.soc case final soc?)
      _row(
        'soc',
        '${soc.label} · ${[?soc.manufacturer, soc.model].join(' ')} from '
            '${soc.source}',
      ),
    _row('cpu', [cpu.model, cores, ?cpu.architecture].join(' · ')),
    if (p.memory case final mem?)
      _row(
        'ram',
        [
          '${formatBytes(mem.totalBytes)} total',
          if (mem.availableBytes case final a?) '${formatBytes(a)} available',
        ].join(' · '),
      ),
    if (p.gpus.isEmpty)
      _row(
        'gpu',
        // A phone always has one: the probe could not name it (the card says
        // the same).
        p.platform == HostPlatform.android ? 'not identified' : 'none found',
      ),
    for (final g in p.gpus) _row('gpu', _gpuText(g)),
    if (p.vulkan case final devices?)
      for (final d in devices) _row('vulkan', _vulkanText(d)),
    if (p.vulkanNote case final note?) _row('vulkan', note),
    if (p.jetson case final j?) _row('jetson', _jetsonText(j)),
    for (final n in p.npuHints) _row('npu', '${n.text} · information only'),
    if (p.audio case final a?) ...audioSystemLines(a),
    for (final n in p.notes) _row('note', n),
  ];
}

/// Linux's sound server, its microphones and sinks (monitors only when one
/// is the default input), and the ALSA card count.
List<String> audioSystemLines(AudioSystem a) {
  final server = a.server;
  final cards = a.alsaCards == null ? null : 'ALSA cards ${a.alsaCards}';
  return [
    _row(
      'audio',
      server == null
          ? ['no sound server (see the note)', ?cards].join(' · ')
          : [
              server.name,
              if (server.version case final v?) 'server $v',
              ?cards,
            ].join(' · '),
    ),
    if (server != null) ...[
      if (a.microphones.isEmpty) _row('audio in', 'no microphone'),
      for (final d in a.sources)
        if (!d.isMonitor || d.isDefault) _row('audio in', _audioDeviceText(d)),
      if (a.sinks.isEmpty) _row('audio out', 'no sink'),
      for (final d in a.sinks) _row('audio out', _audioDeviceText(d)),
    ],
  ];
}

String _audioDeviceText(AudioDevice d) => [
  d.name,
  if (d.isDefault) 'default',
  if (d.isMonitor) 'monitor (not a microphone)',
  d.id,
].join(' · ');

String _gpuText(GpuInfo g) => [
  g.name,
  ?g.api,
  if (g.driver case final d?) 'driver $d',
  if (g.kind == GpuKind.software) 'software',
  g.inferred ? 'inferred from ${g.source}' : g.source,
].join(' · ');

String _vulkanText(VulkanDevice d) => [
  d.name,
  d.type,
  if (d.apiVersion case final v?) 'Vulkan $v',
  ?d.driver,
  if (d.isSoftware) 'SOFTWARE (runs on the CPU)',
].join(' · ');

String _jetsonText(JetsonInfo j) => [
  j.model,
  ?j.soc,
  if (j.l4tRelease case final r?) 'L4T R$r',
  if (j.jetpack case final v?) 'JetPack $v',
  'power mode ${j.powerMode ?? 'unknown'}',
].join(' · ');

/// `backend`, `api`, `adapter` and flag rows for [e].
List<(String, String)> evidenceRows(AcceleratorEvidence e) => [
  (
    'backend',
    e.backendSource == EvidenceSource.requested
        ? '${e.requested} (requested · not reportable)'
        : '${e.requested} → ${e.actual} (${e.backendSource.label})'
              '${e.mismatch ? ' · MISMATCH' : ''}',
  ),
  if (e.api case final api?) ('api', '$api (${e.apiSource.label})'),
  if (e.adapter case final a?)
    (
      'adapter',
      '$a (${e.adapterSource.label})'
          '${e.softwareGpu ? ' · SOFTWARE GPU (error)' : ''}',
    )
  else if (e.actual == 'gpu')
    ('adapter', 'unknown'),
  if (e.cpuDelegate) ('note', 'part of the load uses the XNNPACK CPU delegate'),
  if (e.samplerOnCpu) ('note', 'GPU sampler unavailable: sampling on the CPU'),
  if (e.noGpuLogged)
    ('note', 'the log says a GPU accelerator could not be loaded'),
];

/// One line: `gpu → gpu (confirmed: API) · Metal (inferred) · Apple M4 Pro
/// (inferred)`.
String evidenceText(AcceleratorEvidence e) => [
  for (final (label, value) in evidenceRows(e))
    if (label == 'backend' || label == 'note') value else '$label $value',
].join(' · ');

/// `[Hardware] …` at startup.
String hardwareStartupLine(HardwareProfile p, BuildInfo b) {
  final cpu = p.cpu;
  final pe = cpu.performanceCores != null && cpu.efficiencyCores != null
      ? ' ${cpu.performanceCores}P+${cpu.efficiencyCores}E'
      : ' ${cpu.cores}c';
  final gpu = p.gpus.where((g) => g.kind != GpuKind.virtualDisplay).firstOrNull;
  return [
    '[Hardware] ${p.os}',
    ?p.machine,
    '${cpu.model}$pe',
    if (gpu != null)
      'GPU ${gpu.name}${gpu.api == null ? '' : ' ${gpu.api}'}'
          '${gpu.inferred ? ' (inferred)' : ''}',
    if (p.memory case final mem?) formatBytes(mem.totalBytes),
    ?p.libc,
    'app ${b.appVersion} ${b.buildMode}',
    _packages(b),
  ].join(' · ');
}

/// `[Hardware] chat req=gpu act=gpu(confirmed: API) api=Metal(inferred) …`
/// after a load.
String evidenceLogLine(String key, AcceleratorEvidence e) => [
  '[Hardware] $key req=${e.requested}',
  'act=${e.actual}(${e.backendSource.label})',
  if (e.api case final api?) 'api=$api(${e.apiSource.label})',
  if (e.adapter case final a?) 'adapter="$a"(${e.adapterSource.label})',
  if (e.softwareGpu) 'SOFTWARE_GPU',
  if (e.mismatch) 'MISMATCH',
].join(' ');

/// `avail 9.10 GB · rss 182 MB · peak rss 190 MB`.
String memoryText(MemorySnapshot m) => [
  'available ${m.availableBytes == null ? 'n/a' : formatBytes(m.availableBytes!)}',
  'rss ${formatBytes(m.rssBytes)}',
  'peak rss ${formatBytes(m.peakRssBytes)}',
].join(' · ');

/// Binary units: `24.0 GB`, `182 MB`, `512 KB`.
String formatBytes(int bytes) {
  const k = 1024;
  final abs = bytes.abs();
  if (abs >= k * k * k) return '${(bytes / (k * k * k)).toStringAsFixed(2)} GB';
  if (abs >= k * k) return '${(bytes / (k * k)).toStringAsFixed(0)} MB';
  if (abs >= k) return '${(bytes / k).toStringAsFixed(0)} KB';
  return '$bytes B';
}

/// `3.21 s`.
String seconds(Duration d) =>
    '${(d.inMicroseconds / 1e6).toStringAsFixed(2)} s';
