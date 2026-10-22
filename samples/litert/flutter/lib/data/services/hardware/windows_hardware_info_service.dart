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
import 'dart:ffi' show Abi;
import 'dart:io' show Platform;

import '../../../domain/models/hardware_profile.dart';
import 'hardware_info_service.dart';
import 'system_access.dart';

/// What [WindowsHardwareInfoService] asks Windows PowerShell 5.1 (the one
/// every Windows 10 and 11 has), as one JSON object: the CPU
/// (`Win32_Processor`), the display adapters with their driver versions
/// (`Win32_VideoController`), the OS (`Win32_OperatingSystem`), the RAM and
/// the machine (`Win32_ComputerSystem`) and the present compute accelerators,
/// the class Windows files NPUs under (`Get-PnpDevice -Class
/// ComputeAccelerator`). Not `wmic`: Windows 11 24H2 removed it.
///
/// `Remove-TypeData System.Array` keeps Windows PowerShell's `ConvertTo-Json`
/// from writing an array as `{"value": […], "Count": n}`; `@(…)` keeps a
/// single device a one-element array. Errors (no ComputeAccelerator device
/// at all) and progress records are silenced, so stdout holds the JSON only;
/// [parseWindowsProbeOutput] still skips anything before it. The display
/// adapters are `null` when their query fails, so an empty list means
/// Windows lists none.
const kWindowsProbeScript = r'''
$ErrorActionPreference = 'SilentlyContinue'
$ProgressPreference = 'SilentlyContinue'
Remove-TypeData System.Array
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
$cs = Get-CimInstance Win32_ComputerSystem
try {
  $gpus = @(Get-CimInstance Win32_VideoController -ErrorAction Stop |
    Select-Object Name, DriverVersion, AdapterCompatibility, VideoProcessor, PNPDeviceID)
} catch { $gpus = $null }
[pscustomobject]@{
  cpu = @(Get-CimInstance Win32_Processor |
    Select-Object Name, NumberOfCores, NumberOfLogicalProcessors)
  gpus = $gpus
  os = Get-CimInstance Win32_OperatingSystem |
    Select-Object Caption, Version, BuildNumber, OSArchitecture
  ram = $cs.TotalPhysicalMemory
  machine = $cs | Select-Object Manufacturer, Model
  npu = @(Get-PnpDevice -Class ComputeAccelerator -PresentOnly |
    Select-Object FriendlyName, Status, InstanceId)
} | ConvertTo-Json -Depth 4 -Compress
''';

/// `-EncodedCommand`'s argument for [script]: its UTF-16LE bytes in base64,
/// so no quote in the script meets the Windows command-line quoting rules.
String encodePowerShellCommand(String script) => base64.encode([
  for (final unit in script.codeUnits) ...[unit & 0xff, unit >> 8],
]);

/// Windows: one Windows PowerShell call ([kWindowsProbeScript]), through
/// [ProcessRunner] so tests feed it the JSON two real PCs printed. A missing
/// or silent PowerShell leaves the OS and core count from `dart:io` and says
/// why in the notes.
final class WindowsHardwareInfoService implements HardwareInfoService {
  WindowsHardwareInfoService({
    this._processes = const LocalProcessRunner(),
    String? powershell,
    String? architecture,
    String? fallbackOs,
    int? fallbackCores,
  }) : _powershell = powershell ?? _systemPowerShell(),
       _architecture = architecture ?? _currentArchitecture(),
       _fallbackOs =
           fallbackOs ??
           '${Platform.operatingSystem} ${Platform.operatingSystemVersion}',
       _fallbackCores = fallbackCores ?? Platform.numberOfProcessors;

  final ProcessRunner _processes;
  final String _powershell;
  final String _architecture;
  final String _fallbackOs;
  final int _fallbackCores;

  /// A cold Windows PowerShell plus four CIM queries can take several
  /// seconds.
  static const _timeout = Duration(seconds: 20);

  /// Windows PowerShell under `%SystemRoot%`, not whatever `PATH` finds
  /// first.
  static String _systemPowerShell() =>
      switch (Platform.environment['SystemRoot']) {
        final root? when root.isNotEmpty =>
          '$root\\System32\\WindowsPowerShell\\v1.0\\powershell.exe',
        _ => 'powershell.exe',
      };

  static String _currentArchitecture() => switch (Abi.current()) {
    Abi.windowsX64 => 'x86_64',
    Abi.windowsArm64 => 'arm64',
    final other => other.toString(),
  };

  @override
  Future<HardwareProfile> probe() async {
    final out = await _processes.run(_powershell, [
      '-NoLogo',
      '-NoProfile',
      '-NonInteractive',
      '-EncodedCommand',
      encodePowerShellCommand(kWindowsProbeScript),
    ], timeout: _timeout);
    final cim = out == null ? null : parseWindowsProbeOutput(out.stdout);
    if (cim == null) {
      final error = out == null ? '' : _firstLine(out.stderr);
      final why = out == null
          ? 'PowerShell did not run (not found, or no answer within '
                '${_timeout.inSeconds} s)'
          : 'The PowerShell probe printed no JSON (exit ${out.exitCode}'
                '${error.isEmpty ? '' : ': $error'})';
      return HardwareProfile(
        platform: HostPlatform.windows,
        os: _fallbackOs,
        cpu: CpuInfo(
          model: 'unknown',
          cores: _fallbackCores,
          architecture: _architecture,
        ),
        notes: ['$why: the chip, GPU, NPU and RAM are not probed.'],
      );
    }
    return windowsProfileFrom(
      cim,
      architecture: _architecture,
      fallbackOs: _fallbackOs,
      fallbackCores: _fallbackCores,
    );
  }
}

String _firstLine(String text) =>
    text.trim().split('\n').first.trim().replaceAll('\r', '');

/// The JSON object [stdout] holds: the first line that starts with `{` (after
/// a byte-order mark and blanks) up to the last `}`. A BOM, a `#< CLIXML`
/// line (PowerShell's serialized progress or errors when its streams are
/// merged) or any other text before it is skipped. Null when there is no
/// object.
Map<String, Object?>? parseWindowsProbeOutput(String stdout) {
  final start = RegExp('^[\\uFEFF\\s]*\\{', multiLine: true).firstMatch(stdout);
  final end = stdout.lastIndexOf('}');
  if (start == null || end < start.end) return null;
  try {
    final value = jsonDecode(stdout.substring(start.end - 1, end + 1));
    return value is Map<String, Object?> ? value : null;
  } on FormatException {
    return null;
  }
}

/// The profile from the probe's JSON (pure; the tests' seam). [fallbackOs]
/// and [fallbackCores] stand in for facts the JSON lacks.
HardwareProfile windowsProfileFrom(
  Map<String, Object?> cim, {
  required String architecture,
  required String fallbackOs,
  required int fallbackCores,
}) {
  final notes = <String>[];

  final cpus = _records(cim['cpu']);
  final cpuName = cpus.map((c) => _string(c['Name'])).nonNulls.firstOrNull;
  var cores = 0;
  for (final c in cpus) {
    cores +=
        _int(c['NumberOfLogicalProcessors']) ?? _int(c['NumberOfCores']) ?? 0;
  }
  if (cpuName == null) notes.add('Win32_Processor names no CPU.');

  final os = _records(cim['os']).firstOrNull;
  final caption = _string(os?['Caption']);
  final version = _string(os?['Version']);
  final build = _string(os?['BuildNumber']);
  final osText = switch ((caption, version ?? build)) {
    (final c?, final v?) => '$c ($v)',
    (final c?, null) => c,
    (null, final v?) => 'Windows ($v)',
    (null, null) => fallbackOs,
  };

  final ram = _int(cim['ram']);
  if (ram == null) {
    notes.add('Win32_ComputerSystem.TotalPhysicalMemory is not readable.');
  }

  final listed = cim['gpus'] != null;
  final records = _records(cim['gpus']);
  final gpus = [for (final g in records) ?_gpu(g)];
  // Hardware first: the first GPU that is not a VM display is the one a GPU
  // backend is taken to open (likelyAdapter), and Windows may list WARP
  // before the real adapter.
  final ordered = [
    ...gpus.where((g) => _rank(g.kind) == 0),
    ...gpus.where((g) => _rank(g.kind) == 1),
    ...gpus.where((g) => _rank(g.kind) == 2),
  ];
  // An adapter record without a name leaves the GPU unknown, not absent.
  final complete = listed && gpus.length == records.length;
  notes.addAll(windowsGpuNotes(ordered, listed: complete));
  // No display adapter but a VM's (a server, a VM without a GPU): Direct3D 12
  // still opens its own software adapter, WARP, and a GPU backend runs on it.
  if (complete && ordered.every((g) => g.kind == GpuKind.virtualDisplay)) {
    ordered.add(
      const GpuInfo(
        name: 'Microsoft Basic Render Driver',
        source: 'Win32_VideoController lists no GPU: Direct3D 12 opens WARP',
        inferred: true,
        kind: GpuKind.software,
      ),
    );
  }

  final npus = <NpuHint>[];
  for (final d in _records(cim['npu'])) {
    final name = _string(d['FriendlyName']) ?? 'unnamed compute accelerator';
    final status = _string(d['Status']);
    final pci = _pciId(_string(d['InstanceId']));
    final details = [
      if (pci != null) 'PCI $pci',
      'status ${status ?? 'unknown'}',
    ];
    npus.add(NpuHint('$name (${details.join(', ')})'));
    if (status != null && status != 'OK') {
      notes.add(
        'The NPU $name reports status $status: its driver is not working '
        '(see Device Manager, Neural processors).',
      );
    }
  }

  return HardwareProfile(
    platform: HostPlatform.windows,
    os: osText,
    machine: _machine(_records(cim['machine']).firstOrNull),
    cpu: CpuInfo(
      model: cpuName ?? 'unknown',
      cores: cores > 0 ? cores : fallbackCores,
      architecture: architecture,
    ),
    memory: ram == null ? null : MemoryInfo(totalBytes: ram),
    gpus: List.unmodifiable(ordered),
    npuHints: List.unmodifiable(npus),
    notes: List.unmodifiable(notes),
  );
}

/// One `Win32_VideoController`. WARP and the basic display driver are
/// [GpuKind.software]; Remote Desktop's and Hyper-V's adapters are VM
/// displays; the rest is a GPU of unknown kind (CIM does not say integrated
/// or discrete).
GpuInfo? _gpu(Map<String, Object?> g) {
  final name = _string(g['Name']);
  if (name == null) return null;
  final pci = _pciId(_string(g['PNPDeviceID']));
  return GpuInfo(
    name: name,
    source: pci == null
        ? 'Win32_VideoController'
        : 'Win32_VideoController, PCI $pci',
    kind: isSoftwareGpuName(name)
        ? GpuKind.software
        : _isVirtualDisplay(name)
        ? GpuKind.virtualDisplay
        : GpuKind.unknown,
    driver: _string(g['DriverVersion']),
  );
}

bool _isVirtualDisplay(String name) => RegExp(
  r'Microsoft Remote Display Adapter|Microsoft Hyper-V Video',
  caseSensitive: false,
).hasMatch(name);

int _rank(GpuKind kind) => switch (kind) {
  GpuKind.discrete || GpuKind.integrated || GpuKind.unknown => 0,
  GpuKind.software => 1,
  GpuKind.virtualDisplay => 2,
};

/// Warnings about what a GPU backend would open on Windows. [listed]: the
/// display adapter query answered with named adapters (false: it failed, or
/// named an adapter only partly, so [gpus] may lack one).
List<String> windowsGpuNotes(List<GpuInfo> gpus, {bool listed = true}) {
  final real = gpus.where(
    (g) => g.kind != GpuKind.software && g.kind != GpuKind.virtualDisplay,
  );
  if (real.isNotEmpty) return const [];
  final software = gpus.where((g) => g.kind == GpuKind.software).toList();
  if (software.isNotEmpty) {
    return [
      'Windows lists only ${software.map((g) => g.name).join(', ')}: no GPU '
          'driver is installed, and a GPU backend would run on the CPU '
          '(WARP). Install the GPU vendor\'s driver.',
    ];
  }
  if (!listed) {
    return const ['Win32_VideoController is not readable: the GPU is unknown.'];
  }
  return const [
    'No GPU found (Win32_VideoController lists none, or only a remote or VM '
        'display): a GPU backend would run on the CPU (WARP).',
  ];
}

/// `Manufacturer Model`, without the placeholders firmware leaves in them.
String? _machine(Map<String, Object?>? cs) {
  String? clean(Object? value) {
    final v = _string(value);
    if (v == null) return null;
    if (RegExp(
      r'O\.E\.M\.|System (manufacturer|Product Name)|Default string',
      caseSensitive: false,
    ).hasMatch(v)) {
      return null;
    }
    return v;
  }

  final vendor = clean(cs?['Manufacturer']);
  final model = clean(cs?['Model']);
  if (model == null) return vendor;
  if (vendor == null || model.contains(vendor)) return model;
  return '$vendor $model';
}

/// `PCI\VEN_8086&DEV_643E&…` → `8086:643e` (sysfs's spelling, as Linux
/// prints it).
String? _pciId(String? instanceId) {
  if (instanceId == null) return null;
  final m = RegExp(
    r'VEN_([0-9A-F]{4})&DEV_([0-9A-F]{4})',
    caseSensitive: false,
  ).firstMatch(instanceId);
  if (m == null) return null;
  return '${m.group(1)!.toLowerCase()}:${m.group(2)!.toLowerCase()}';
}

/// The records of a JSON value: an array of objects, one object (a single
/// device, should `@(…)` ever be lost), or Windows PowerShell's
/// `{"value": […], "Count": n}` wrapper of an array.
List<Map<String, Object?>> _records(Object? value) => switch (value) {
  {'value': final List<Object?> list} => _records(list),
  final Map<String, Object?> one => [one],
  final List<Object?> list => [
    for (final item in list)
      if (item is Map<String, Object?>) item,
  ],
  _ => const [],
};

String? _string(Object? value) => switch (value) {
  final String s when s.trim().isNotEmpty => s.trim(),
  _ => null,
};

int? _int(Object? value) => switch (value) {
  final int i => i,
  final num n => n.toInt(),
  final String s => int.tryParse(s.trim()),
  _ => null,
};
