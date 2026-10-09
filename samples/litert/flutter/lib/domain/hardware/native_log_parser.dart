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
import '../models/hardware_profile.dart' show isSoftwareGpuName;

final _adapter = RegExp(
  r'Selected adapter: (.+?), arch=([^,]*), vendor=([^,]*), '
  r'backend=([^,]*), adapterType=(.+?)\s*$',
);
final _registered = RegExp(r'RegisterAccelerator: ptr=\S+, name=GPU ([\w ]+)');
final _dynamicallyLoaded = RegExp(
  r'Dynamically loaded GPU accelerator\((.+?)\) registered',
);
final _metalDevice = RegExp(r'Created (a )?Metal device');
const _webGpuEnvironment = 'Created a WebGPU environment';
const _noGpu = 'GPU accelerator could not be loaded';
const _cpuDelegate = 'XNNPACK delegate for CPU';
const _samplerOnCpu = 'GPU sampler unavailable';

/// Reads LiteRT / LiteRT-LM native log lines. Patterns
/// are matched anywhere in a line, so glog (`I0000 … file.cc:12]`),
/// `INFO: [file.cc:12]` and logcat prefixes all work; CRLF endings are
/// fine. `Failed to create OpenCL context` is harmless on Apple and ignored.
NativeLogEvidence parseNativeLog(Iterable<String> lines) {
  AdapterLine? adapter;
  String? api;
  var noGpu = false;
  var cpuDelegate = false;
  var samplerOnCpu = false;
  var software = false;
  final matched = <String>[];
  for (final raw in lines) {
    final line = raw.trimRight();
    var hit = true;
    if (_adapter.firstMatch(line) case final m?) {
      String? field(int i) => m.group(i)!.trim().isEmpty ? null : m.group(i);
      adapter = AdapterLine(
        name: m.group(1)!.trim(),
        arch: field(2),
        vendor: field(3),
        backend: field(4),
        adapterType: field(5),
      );
      if (adapter.adapterType == 'CPU' || isSoftwareGpuName(adapter.name)) {
        software = true;
      }
    } else if (_registered.firstMatch(line) case final m?) {
      api ??= _apiName(m.group(1)!);
    } else if (_dynamicallyLoaded.firstMatch(line) case final m?) {
      api ??= _apiName(m.group(1)!);
    } else if (_metalDevice.hasMatch(line)) {
      api ??= 'Metal';
    } else if (line.contains(_webGpuEnvironment)) {
      api ??= 'WebGPU';
    } else if (line.contains(_noGpu)) {
      noGpu = true;
    } else if (line.contains(_cpuDelegate)) {
      cpuDelegate = true;
    } else if (line.contains(_samplerOnCpu)) {
      samplerOnCpu = true;
    } else {
      hit = false;
    }
    if (hit) matched.add(line);
  }
  // The adapter line names the API Dawn runs on: the most specific answer.
  if (adapter?.backend case final backend?) api = 'WebGPU/$backend';
  return NativeLogEvidence(
    adapter: adapter,
    api: api,
    noGpu: noGpu,
    cpuDelegate: cpuDelegate,
    samplerOnCpu: samplerOnCpu,
    softwareGpu: software,
    lines: List.unmodifiable(matched),
  );
}

/// `Metal`, `WebGPU` or `OpenCL` from an accelerator name or library path;
/// the text itself when it is none of them.
String _apiName(String text) {
  final lower = text.toLowerCase();
  if (lower.contains('metal')) return 'Metal';
  if (lower.contains('webgpu')) return 'WebGPU';
  if (lower.contains('opencl')) return 'OpenCL';
  return text.trim();
}

final _problem = RegExp(
  r'^\s*[EFW]\d{4} |\bERROR\b|\bWARNING\b|Unhandled Exception|failed',
  caseSensitive: false,
);

/// The last [max] error and warning lines of a native log window (glog
/// `E`/`F`/`W` prefixes, `ERROR`, `WARNING`, `failed`, Dart's `Unhandled
/// Exception`): what explains a failed load when no evidence pattern
/// matched.
List<String> nativeProblemLines(Iterable<String> lines, {int max = 20}) {
  final hits = [
    for (final raw in lines)
      if (_problem.hasMatch(raw)) raw.trimRight(),
  ];
  return hits.length <= max ? hits : hits.sublist(hits.length - max);
}
