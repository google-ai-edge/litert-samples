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

import '../domain/hardware/diagnostics_report.dart'
    show deviceLines, formatBytes, seconds;
import '../domain/models/hardware_profile.dart';
import 'self_test_runner.dart';

const kSelfTestBegin = '===== SELFTEST BEGIN =====';
const kSelfTestEnd = '===== SELFTEST END =====';

const _w = 11;

/// The memory table's step column.
const _label = 22;

String _row(String label, String value) => '${label.padRight(_w)}$value';

/// The delimited block printed to stdout and written to the report file.
/// Pure. [reportPath] is where the file goes (null: it could not be written).
String formatSelfTest(SelfTestReport r, {String? reportPath}) {
  final counts = <StepStatus, int>{};
  for (final s in r.steps) {
    counts[s.status] = (counts[s.status] ?? 0) + 1;
  }
  final b = r.build;
  final o = r.options;
  String file(ModelFileChoice f, String backend) => f.path == null
      ? 'none (${f.source}): ${f.problem ?? 'no file'}'
      : '${f.path} (${f.source}) · backend $backend';
  final detectorLabel = r.detectorFile.bundled
      ? 'detector (bundled)'
      : 'detector';
  final lines = <String>[
    kSelfTestBegin,
    _row(
      'result',
      [
        r.passed
            ? r.softwareGpuAllowed
                  ? 'PASS (software GPU allowed: GPU code path only)'
                  : 'PASS'
            : 'FAIL',
        if (r.unhandledErrors.isNotEmpty)
          '${r.unhandledErrors.length} unhandled error(s)',
        for (final status in StepStatus.values)
          if (counts[status] case final n?) '$n ${status.name}',
        'exit ${r.exitCode}',
      ].join(' · '),
    ),
    _row('started', '${_utc(r.startedAt)} · took ${seconds(r.elapsed)}'),
    _row(
      'build',
      [
        'app ${b.appVersion} ${b.buildMode}',
        'flutter ${b.flutterVersion ?? 'unknown'}',
        'Dart ${b.dartVersion}',
        for (final MapEntry(:key, :value) in b.packages.entries) '$key $value',
      ].join(' · '),
    ),
    _row(
      'chat model',
      r.chatModel.file.path == null
          ? '${r.chatModel.config.name}: none (${r.chatModel.file.source}): '
                '${r.chatModel.file.problem ?? 'no file'}'
          : '${r.chatModel.config.name} (${r.chatModel.settingsSource}) · '
                '${r.chatModel.file.path} (${r.chatModel.file.source}) · '
                '${r.chatModel.settingsLine}',
    ),
    if (r.chatModel.sha256 case final sha?) _row('sha256', sha),
    if (r.detectorFile.bundled)
      '$detectorLabel ${r.detectorFile.path} · backend ${o.detectorBackend.name}'
    else
      _row('detector', file(r.detectorFile, o.detectorBackend.name)),
    _row('image', r.imageLabel),
    if (o.detectorCpuRetry || o.allowSoftwareGpu || o.skipAudio)
      _row(
        'options',
        [
          if (o.detectorCpuRetry) '--detector-cpu-retry',
          if (o.allowSoftwareGpu) '--allow-software-gpu',
          if (o.skipAudio) '--skip-audio (no step 6)',
        ].join(' '),
      ),
    _row('native log', r.nativeLog),
    _row('report', reportPath ?? 'not written'),
    for (final e in r.unhandledErrors) _row('unhandled', e),
    '--- device ---',
    if (r.hardware case final hw?) ...deviceLines(hw) else 'probe failed',
    '--- steps ---',
    for (final s in r.steps) ...[
      '${s.status.label}  ${s.id.padRight(3)} ${s.title.padRight(48)} '
          '${seconds(s.elapsed)}',
      for (final d in s.details) '           $d',
    ],
    '--- memory (available = what the OS can hand out; on a Jetson the GPU '
        'shares it) ---',
    '${'step'.padRight(_label)}${'available before → after'.padRight(30)}'
        '${'Δ available'.padRight(14)}${'rss after'.padRight(12)}peak rss',
    for (final s in r.steps)
      if (s.before != null && s.after != null)
        _memoryRow('${s.id} ${_short(s.title)}', s.before!, s.after!),
    kSelfTestEnd,
  ];
  return '${lines.join('\n')}\n';
}

String _memoryRow(String label, MemorySnapshot before, MemorySnapshot after) {
  final a0 = before.availableBytes;
  final a1 = after.availableBytes;
  final avail = a0 == null || a1 == null
      ? 'n/a'
      : '${formatBytes(a0)} → ${formatBytes(a1)}';
  final delta = a0 == null || a1 == null ? 'n/a' : _signed(a1 - a0);
  return '${label.padRight(_label)}${avail.padRight(30)}${delta.padRight(14)}'
      '${formatBytes(after.rssBytes).padRight(12)}'
      '${formatBytes(after.peakRssBytes)}';
}

String _signed(int bytes) =>
    bytes < 0 ? '-${formatBytes(-bytes)}' : '+${formatBytes(bytes)}';

/// The first two words of a step title (`detector load`).
String _short(String title) => title.split(' ').take(2).join(' ');

String _utc(DateTime t) {
  final s = t.toUtc().toIso8601String();
  final dot = s.indexOf('.');
  return '${dot < 0 ? s.replaceFirst('Z', '') : s.substring(0, dot)}Z';
}
