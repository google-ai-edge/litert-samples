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
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';

/// One reading from the power/utilisation sidecar (`tools/power_sidecar`).
/// Watts are null when the platform cannot report them (macOS, or Windows
/// with VBS/Memory Integrity on); loads are always there on Windows.
class PowerSample {
  const PowerSample({
    required this.at,
    this.cpuPackageW,
    this.cpuCoresW,
    this.cpuPlatformW,
    this.gpuW,
    this.cpuLoad,
    this.gpu3dLoad,
    this.gpuComputeLoad,
    this.npuLoad,
    this.lhmError,
  });

  final DateTime at;
  final double? cpuPackageW;
  final double? cpuCoresW;
  final double? cpuPlatformW;
  final double? gpuW;
  final double? cpuLoad;
  final double? gpu3dLoad;
  final double? gpuComputeLoad;
  final double? npuLoad;

  /// Why the sidecar reports no watts at all: LibreHardwareMonitor did not
  /// load (no -Lib, wrong path, not elevated). Null when it loaded.
  final String? lhmError;

  bool get hasWatts => cpuPackageW != null || gpuW != null;
  bool get hasLoads => cpuLoad != null || gpu3dLoad != null || npuLoad != null;

  factory PowerSample.fromJson(Map<String, Object?> j) {
    double? d(String k) => (j[k] as num?)?.toDouble();
    return PowerSample(
      at: DateTime.tryParse(j['ts'] as String? ?? '') ?? DateTime.now(),
      cpuPackageW: d('cpu_package_w'),
      cpuCoresW: d('cpu_cores_w'),
      cpuPlatformW: d('cpu_platform_w'),
      gpuW: d('gpu_w'),
      cpuLoad: d('cpu_load'),
      gpu3dLoad: d('gpu_3d_load'),
      gpuComputeLoad: d('gpu_compute_load'),
      npuLoad: d('npu_load'),
      lhmError: j['lhm_error'] as String?,
    );
  }
}

/// Averages over a window (one benchmark run, one narrator phase).
class PowerSummary {
  const PowerSummary({
    required this.samples,
    this.cpuPackageW,
    this.gpuW,
    this.cpuLoad,
    this.gpu3dLoad,
    this.gpuComputeLoad,
    this.npuLoad,
  });

  final int samples;
  final double? cpuPackageW;
  final double? gpuW;
  final double? cpuLoad;
  final double? gpu3dLoad;
  final double? gpuComputeLoad;
  final double? npuLoad;

  bool get hasWatts => cpuPackageW != null || gpuW != null;

  static PowerSummary of(Iterable<PowerSample> xs) {
    final list = xs.toList();
    double? avg(double? Function(PowerSample) pick) {
      final v = list.map(pick).whereType<double>().toList();
      if (v.isEmpty) return null;
      return double.parse((v.reduce((a, b) => a + b) / v.length).toStringAsFixed(2));
    }

    return PowerSummary(
      samples: list.length,
      cpuPackageW: avg((s) => s.cpuPackageW),
      gpuW: avg((s) => s.gpuW),
      cpuLoad: avg((s) => s.cpuLoad),
      gpu3dLoad: avg((s) => s.gpu3dLoad),
      gpuComputeLoad: avg((s) => s.gpuComputeLoad),
      npuLoad: avg((s) => s.npuLoad),
    );
  }

  Map<String, Object?> toJson() => {
        'samples': samples,
        'cpu_package_w': cpuPackageW,
        'gpu_w': gpuW,
        'cpu_load': cpuLoad,
        'gpu_3d_load': gpu3dLoad,
        'gpu_compute_load': gpuComputeLoad,
        'npu_load': npuLoad,
      };
}

/// What the HUD/table shows. `auto` = watts when the sidecar reports any,
/// otherwise utilisation — so turning Memory Integrity off later switches
/// the demo to real watts without a config change.
enum PowerMode { auto, watts, utilization }

String formatPower(PowerSummary? p, PowerMode mode) {
  if (p == null || p.samples == 0) return 'n/a';
  final showWatts = switch (mode) {
    PowerMode.watts => true,
    PowerMode.utilization => false,
    PowerMode.auto => p.hasWatts,
  };
  String w(double? v) => v == null ? '—' : '${v.toStringAsFixed(1)} W';
  String pct(double? v) => v == null ? '—' : '${v.round()} %';
  if (showWatts) return 'CPU ${w(p.cpuPackageW)} · GPU ${w(p.gpuW)}';
  return 'CPU ${pct(p.cpuLoad)} · GPU ${pct(p.gpu3dLoad)} · NPU ${pct(p.npuLoad)}';
}

/// A stream of samples plus a rolling "latest".
abstract class PowerSource extends ChangeNotifier {
  PowerSample? get latest;
  Stream<PowerSample> get stream;
  String get label;
  Future<void> start();
  Future<void> stop();

  /// Collects samples while [body] runs and returns their average.
  Future<(T, PowerSummary)> measure<T>(Future<T> Function() body) async {
    final buf = <PowerSample>[];
    final sub = stream.listen(buf.add);
    try {
      final r = await body();
      return (r, PowerSummary.of(buf));
    } finally {
      await sub.cancel();
    }
  }
}

/// No sidecar (macOS, or the file is absent).
class NullPowerSource extends PowerSource {
  @override
  PowerSample? get latest => null;
  @override
  Stream<PowerSample> get stream => const Stream.empty();
  @override
  String get label => 'n/a';
  @override
  Future<void> start() async {}
  @override
  Future<void> stop() async {}
}

/// Polls the JSON file the sidecar rewrites every second.
class FilePowerSource extends PowerSource {
  FilePowerSource(this.path, {this.interval = const Duration(milliseconds: 500)});

  final String path;
  final Duration interval;
  final _ctl = StreamController<PowerSample>.broadcast();
  Timer? _timer;
  PowerSample? _latest;
  DateTime? _lastTs;
  DateTime? _receivedAt;

  /// A sample older than this is a sidecar that stopped (or a file left by
  /// an earlier session), not the machine's current state.
  static const stale = Duration(seconds: 5);
  int _errors = 0;

  /// Unreadable polls (a half-written file between sidecar writes).
  int get readErrors => _errors;

  bool get _fresh => _receivedAt != null && DateTime.now().difference(_receivedAt!) < stale;

  @override
  PowerSample? get latest => _fresh ? _latest : null;
  @override
  Stream<PowerSample> get stream => _ctl.stream;
  @override
  String get label {
    final s = latest;
    if (s == null) return _latest == null ? 'no sidecar ($path)' : 'sidecar stopped ($path)';
    if (s.hasWatts) return 'lhm watts';
    if (s.lhmError != null) return 'loads only (LibreHardwareMonitor: ${s.lhmError})';
    return 'loads only (RAPL blocked, e.g. by Memory Integrity)';
  }

  @override
  Future<void> start() async {
    _timer ??= Timer.periodic(interval, (_) => _poll());
    _poll();
  }

  void _poll() {
    try {
      final f = File(path);
      if (!f.existsSync()) return;
      final j = jsonDecode(f.readAsStringSync());
      if (j is! Map<String, Object?>) return;
      final s = PowerSample.fromJson(j);
      if (_lastTs == s.at) return; // same sample as last time
      _lastTs = s.at;
      _latest = s;
      _receivedAt = DateTime.now();
      _ctl.add(s);
      notifyListeners();
    } catch (_) {
      _errors++; // a half-written file between the sidecar's writes
    }
  }

  @override
  Future<void> stop() async {
    _timer?.cancel();
    _timer = null;
  }

  @override
  void dispose() {
    _timer?.cancel();
    _ctl.close();
    super.dispose();
  }
}

PowerSource powerSourceFor({String? file}) {
  if (file != null && file.isNotEmpty) return FilePowerSource(file);
  // Where tools/power_sidecar writes by default.
  final programData = Platform.environment['ProgramData'];
  if (Platform.isWindows && programData != null) {
    return FilePowerSource('$programData\\intel_npu_demo\\power.json');
  }
  return NullPowerSource();
}
