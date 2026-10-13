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

import 'dart:convert';
import 'dart:io';
import 'dart:isolate';

import 'package:flutter/foundation.dart';

import '../../../domain/hardware/soc_table.dart';
import '../../../domain/models/hardware_profile.dart';
import 'system_access.dart';

/// The Android system properties the app reads. `ro.soc.*` exists from Android
/// 12 (`Build.SOC_*`); Android 11 has only the board.
const kAndroidPropertyKeys = [
  'ro.soc.manufacturer',
  'ro.soc.model',
  'ro.board.platform',
  'ro.hardware',
  'ro.product.manufacturer',
  'ro.product.model',
  'ro.build.version.release',
  'ro.build.version.sdk',
];

/// Android system properties, read once.
abstract interface class AndroidProperties {
  /// The property's value; null when it is unset, empty or unreadable.
  String? operator [](String key);

  /// Why no property could be read (`getprop` did not run); null when it
  /// ran.
  String? get error;
}

/// Properties from a map (tests, and the parsed `getprop` output).
final class MapAndroidProperties implements AndroidProperties {
  const MapAndroidProperties(this._values, {this.error});

  final Map<String, String> _values;

  @override
  final String? error;

  @override
  String? operator [](String key) => switch (_values[key]?.trim()) {
    final v? when v.isNotEmpty => v,
    _ => null,
  };
}

/// `getprop`'s dump format, one `[key]: [value]` per line (what `adb shell
/// getprop` prints, and what [GetpropAndroidProperties] makes it print).
/// Other lines are skipped.
Map<String, String> parseGetpropDump(String text) {
  final line = RegExp(r'^\[([^\]]+)\]: \[(.*)\]\s*$', multiLine: true);
  return {for (final m in line.allMatches(text)) m.group(1)!: m.group(2)!};
}

/// Runs a process to completion, synchronously. Null when it cannot start.
typedef SyncProcessRunner = ProcessOutput? Function(
  String executable,
  List<String> arguments,
);

/// `Process.runSync`; null when the executable cannot start.
ProcessOutput? runProcessSync(String executable, List<String> arguments) {
  try {
    final result = Process.runSync(
      executable,
      arguments,
      stdoutEncoding: const Utf8Codec(allowMalformed: true),
      stderrEncoding: const Utf8Codec(allowMalformed: true),
    );
    return ProcessOutput(
      exitCode: result.exitCode,
      stdout: '${result.stdout}',
      stderr: '${result.stderr}',
    );
  } on ProcessException catch (e) {
    debugPrint('[Hardware] $executable could not start: $e');
    return null;
  }
}

/// The device's properties: ONE `sh -c` that runs `getprop` for each of
/// [kAndroidPropertyKeys] and prints the dump format, run on first access
/// and kept (properties starting `ro.` never change while the app runs).
/// One process instead of one per key; per key, not a full `getprop` dump,
/// which makes SELinux log a denial for every property area an app may not
/// read.
///
/// Synchronous on purpose: the NPU gate (`probeNpu`) is synchronous. The
/// app's startup reads them on a worker isolate first
/// ([loadAndroidProperties]), so the main isolate normally finds them cached.
final class GetpropAndroidProperties implements AndroidProperties {
  GetpropAndroidProperties({this._run = runProcessSync});

  final SyncProcessRunner _run;
  late final MapAndroidProperties _read = readAndroidProperties(run: _run);

  @override
  String? operator [](String key) => _read[key];

  @override
  String? get error => _read.error;
}

/// Runs the one `sh -c` over [kAndroidPropertyKeys] and parses it. Every
/// value empty is an error too: the script itself always exits 0 (its last
/// command is an `echo`), so a missing or denied `getprop` shows only as
/// empty values and its stderr. Logs the time it took.
MapAndroidProperties readAndroidProperties({
  SyncProcessRunner run = runProcessSync,
  bool log = true,
}) {
  final watch = Stopwatch()..start();
  // The keys are constants: nothing from outside reaches the shell.
  final script =
      'for k in ${kAndroidPropertyKeys.join(' ')}; do '
      r'echo "[$k]: [$(getprop "$k")]"; done';
  final out = run('/system/bin/sh', ['-c', script]);
  final String? error;
  var values = const <String, String>{};
  String stderrOf(ProcessOutput o) =>
      o.stderr.trim().isEmpty ? '' : ': ${o.stderr.trim()}';
  if (out == null) {
    error = '/system/bin/sh could not start';
  } else if (out.exitCode != 0) {
    error = 'getprop exited with ${out.exitCode}${stderrOf(out)}';
  } else {
    values = parseGetpropDump(out.stdout);
    final named = values.values.any((v) => v.trim().isNotEmpty);
    error = named ? null : 'getprop printed no values${stderrOf(out)}';
  }
  if (log) {
    debugPrint(
      '[Hardware] getprop ${error ?? '${values.length} keys'} in '
      '${watch.elapsedMilliseconds} ms',
    );
  }
  return MapAndroidProperties(values, error: error);
}

AndroidProperties? _device;

/// This device's properties, shared by the hardware probe and the NPU gate
/// so `getprop` runs once per process. Call it on Android only.
AndroidProperties androidProperties() => _device ??= GetpropAndroidProperties();

/// [androidProperties], read on a worker isolate (dart:io works there): the
/// startup probe awaits this, so the shell's fork and exec never block the
/// main isolate. Later synchronous reads (the NPU gate) hit the cache.
Future<AndroidProperties> loadAndroidProperties() async {
  if (_device case final cached?) return cached;
  final watch = Stopwatch()..start();
  // Logged here: a worker isolate's prints do not reach the Android log.
  final read = await Isolate.run(() => readAndroidProperties(log: false));
  debugPrint(
    '[Hardware] getprop ${read.error ?? 'read'} in '
    '${watch.elapsedMilliseconds} ms (worker isolate)',
  );
  return _device ??= read;
}

/// The SoC [props] name: `ro.soc.model` (with `ro.soc.manufacturer`), else
/// a board the app's table knows (`ro.board.platform`, then `ro.hardware`),
/// else the raw board. Null when none of them is set.
SocInfo? socFromProperties(AndroidProperties props) {
  final manufacturer = props['ro.soc.manufacturer'];
  if (props['ro.soc.model'] case final model?) {
    return SocInfo(
      manufacturer: manufacturer,
      model: model,
      source: 'ro.soc.model',
      name: lookupSoc(model)?.name,
    );
  }
  final boards = [
    for (final key in const ['ro.board.platform', 'ro.hardware'])
      if (props[key] case final value?) (key: key, value: value),
  ];
  for (final board in boards) {
    if (lookupBoard(board.value) case (:final code, :final spec)?) {
      return SocInfo(
        manufacturer: manufacturer,
        model: code,
        source: '${board.key} ${board.value}',
        name: spec.name,
      );
    }
  }
  if (boards.firstOrNull case final board?) {
    return SocInfo(
      manufacturer: manufacturer,
      model: 'board ${board.value}',
      source: board.key,
    );
  }
  return null;
}
