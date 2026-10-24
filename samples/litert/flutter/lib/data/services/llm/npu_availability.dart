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
import 'dart:ffi';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart'
    show LiteRtLmNpuTarget;
import 'package:path/path.dart' as p;

import '../../../domain/models/npu_availability.dart';
import '../hardware/android_props.dart';
import '../hardware/system_access.dart';
import '../hardware/windows_hardware_info_service.dart'
    show encodePowerShellCommand, parseWindowsProbeOutput;

/// The app's copy of flutter_edge_ai_litertlm 1.17.0's private NPU gate for
/// the stacks this app builds with. The plugin decides per vendor
/// (`chooseNpuVendor` in `lib/src/ffi/npu_vendor_host.dart`): the vendors of
/// the host's ABI (`NpuVendor.forAbi`), and for each whether the build carries
/// its stack (`npuStackOf`), then whether the device reaches it
/// (`npuUnreachable` in `lib/src/ffi/backend_preference.dart`). It mirrors a
/// private check because the plugin has no public API for "would a load reach
/// the NPU here" (a plugin gap): none of these is exported.
///
/// The stacks here, by [abi] ([kAppNpuStacks]):
/// - `android_arm64`: Qualcomm, which this app opts in to
///   (`qualcomm_npu: true` under `hooks: user_defines` in pubspec.yaml),
///   reached when `libcdsprpc.so` opens. Google Tensor and MediaTek are
///   opt-in too and not enabled, so the plugin never offers them here.
/// - `linux_arm64`: Qualcomm (the same opt-in), reached when the board's
///   compute DSP and FastRPC library answer.
/// - `windows_x64`: Intel, in every Windows x64 build, when its dispatch
///   `LiteRtDispatch.dll` is beside the executable. The plugin has no
///   Windows device probe (a bundle compiled for the Intel NPU is not refused
///   on a PC without one), so the app adds its own: [windows], what Windows
///   lists ([listWindowsComputeAccelerators]); [NpuPending] until it is in.
///   Only a listing without an Intel device makes the NPU unavailable: a
///   listing that could not be made leaves the decision to the load.
/// - Every other ABI (macOS, iOS, Linux x64, Windows arm64): no NPU dispatch
///   stack ships.
///
/// The build's stack is taken from the app's opt-ins, not read back from
/// `NativeAssetsManifest.json` as the plugin reads it at the load.
///
/// Only for the Models screen, which offers `npu` only when this says
/// [NpuAvailable]. A load does not ask it: on Android and Linux arm64 the
/// strict engine refuses an NPU it cannot reach itself
/// (`NpuDispatchUnavailableException`); on Windows the plugin has no device
/// probe, so the load itself tells. Whether a
/// file compiled for one SoC runs on this one only a load tells; whether
/// this build runs a file's NPU vendor here at all is [npuBuildRefusal].
NpuAvailability npuAvailabilityFor(
  String abi,
  NpuProbes probes, {
  String? soc,
  WindowsNpuListing? windows,
}) => switch (abi) {
  'android_arm64' => switch (probes.androidFastRpc()) {
    null => NpuAvailable(soc: soc),
    final error => NpuUnavailable(
      'this device has no Qualcomm FastRPC (libcdsprpc.so did not open: '
      '$error), so the bundled NPU stack cannot run here; this build has no '
      'Google Tensor or MediaTek NPU stack (both opt-in)',
      soc: soc,
    ),
  },
  'linux_arm64' => switch (probes.linuxQualcommNpu()) {
    null => NpuAvailable(soc: soc),
    final error => NpuUnavailable(
      'the bundled Qualcomm NPU stack cannot run here: $error',
      soc: soc,
    ),
  },
  'windows_x64' => switch (probes.windowsIntelDispatch()) {
    final missing? => NpuUnavailable(missing),
    null => switch (windows) {
      null => const NpuPending(),
      final listing => intelNpuFrom(listing),
    },
  },
  _ => NpuUnavailable('no NPU dispatch stack ships for $abi', soc: soc),
};

/// The NPU vendor this build runs on each ABI: flutter_edge_ai_litertlm's
/// `NpuVendor.abis`, narrowed to the stacks the app bundles (Qualcomm by its
/// opt-in, Intel in every Windows x64 build).
const kAppNpuStacks = {
  'android_arm64': 'Qualcomm',
  'linux_arm64': 'Qualcomm',
  'windows_x64': 'Intel',
};

/// Why this build cannot run a file compiled for the [target] NPU on [abi],
/// or null when it can (whether the device reaches the NPU is
/// [npuAvailabilityFor]; whether the SoC matches only a load tells). The
/// plugin refuses such a file on every backend before any engine
/// (`unsupportedNpu`), or finds no stack for it, and only once the running
/// model was unloaded: the app refuses it up front, worded as the plugin's
/// `_checkNpuVendor` words it. The vendor is the compiler's stamp
/// ([LiteRtLmNpuTarget.isQualcomm], [LiteRtLmNpuTarget.isIntel], else its
/// [LiteRtLmNpuTarget.vendor]).
String? npuBuildRefusal(LiteRtLmNpuTarget target, String abi) {
  final vendor = target.isQualcomm
      ? 'Qualcomm'
      : target.isIntel
      ? 'Intel'
      : null;
  if (vendor == null) {
    return switch (target.vendor.toLowerCase()) {
      // Their stacks run on Android only: off it, the plugin's own words.
      'google' || 'mediatek' when abi != 'android_arm64' =>
        'compiled for the $target NPU, which runs on android_arm64 only, and '
            'this is $abi',
      'google' =>
        'compiled for the $target NPU, and this build has no Google '
            'Tensor NPU stack (opt-in, not enabled)',
      'mediatek' =>
        'compiled for the $target NPU, and this build has no '
            'MediaTek NPU stack (opt-in, not enabled)',
      _ =>
        'compiled for the $target NPU, and flutter_edge_ai_litertlm runs '
            'Qualcomm, Google Tensor, MediaTek and Intel NPU builds only',
    };
  }
  if (kAppNpuStacks[abi] == vendor) return null;
  final runsOn = [
    for (final MapEntry(key: abi, value: stack) in kAppNpuStacks.entries)
      if (stack == vendor) abi,
  ];
  return 'compiled for the $target NPU, which this build runs on '
      '${runsOn.join(' and ')} only, and this is $abi';
}

/// This process's ABI as `Abi.current()` spells it (`android_arm64`, …).
String currentAbi() => '${Abi.current()}';

/// What the NPU check asks of the host synchronously, one member per probe:
/// each answer is a property of a device a test host is not, so tests pass a
/// fake. Windows' device listing is asynchronous:
/// [listWindowsComputeAccelerators].
abstract interface class NpuProbes {
  /// Android: null when Qualcomm's FastRPC library `libcdsprpc.so` opens in
  /// this process, else the dlopen error.
  String? androidFastRpc();

  /// Linux arm64: null when this machine can run the Qualcomm NPU stack, else
  /// why not.
  String? linuxQualcommNpu();

  /// Windows x64: null when the Intel dispatch is beside the executable, else
  /// why not, in the plugin's words (`checkIntelNpuStack`).
  String? windowsIntelDispatch();
}

/// What Windows lists of the present compute accelerators (the class it files
/// NPUs under), or why they could not be listed.
sealed class const WindowsNpuListing();

/// The present compute accelerators ([kWindowsNpuScript]): none when there
/// is none.
final class const ComputeAcceleratorsListed(
  final List<ComputeAccelerator> devices,
) extends WindowsNpuListing;

/// The listing could not be made: PowerShell did not run or answer in time,
/// printed no JSON, or the query failed ([reason]).
final class const ComputeAcceleratorsNotListed(final String reason)
    extends WindowsNpuListing;

/// One present compute accelerator, as `Get-PnpDevice` names it.
final class const ComputeAccelerator({
  /// `Intel(R) AI Boost`.
  final String? name,

  /// The device status, an enum name in every display language: `OK`,
  /// `Error`, `Degraded`, `Unknown`.
  final String? status,

  /// `PCI\VEN_8086&DEV_643E&…`.
  required final String instanceId,

  /// `32.0.100.4778`; null when Windows did not say.
  final String? driverVersion,
}) {
  /// Intel's PCI vendor id.
  bool get isIntel => instanceId.toUpperCase().startsWith(r'PCI\VEN_8086');
}

/// The Intel NPU in [listing]: available with its name and driver when one is
/// present (with a note when its status is not OK), unavailable only when the
/// listing names none, and available with a note when there is no listing
/// (the load decides).
NpuAvailability intelNpuFrom(WindowsNpuListing listing) {
  final List<ComputeAccelerator> devices;
  switch (listing) {
    case ComputeAcceleratorsNotListed(:final reason):
      return NpuAvailable(note: 'not checked: $reason; the load decides');
    case ComputeAcceleratorsListed(devices: final listed):
      devices = listed;
  }
  final intel = [
    for (final device in devices)
      if (device.isIntel) device,
  ];
  if (intel.isEmpty) {
    return const NpuUnavailable(
      'no Intel NPU found: Windows lists no present Intel compute accelerator',
    );
  }
  final npu = intel.where((d) => d.status == 'OK').firstOrNull ?? intel.first;
  return NpuAvailable(
    device: [
      npu.name ?? 'Intel NPU',
      if (npu.driverVersion case final version?) 'driver $version',
    ].join(', '),
    note: npu.status == 'OK'
        ? null
        : 'status ${npu.status ?? 'not reported'}: its driver may not be '
              'working; the load decides',
  );
}

/// The Windows PowerShell 5.1 query of the device check, as one JSON object:
/// the present compute accelerators (`Get-PnpDevice -Class
/// ComputeAccelerator -PresentOnly`: a stale, disconnected entry is not
/// present) with their driver version, `listed` false and the `error` when
/// the query failed. `Get-PnpDevice` reports "none" as an ObjectNotFound
/// error: that is an empty list, not a failure. Its `Status` is an enum name
/// in every display language. `Remove-TypeData System.Array` and `@(…)` as in
/// the hardware probe (`kWindowsProbeScript`).
const kWindowsNpuScript = r'''
$ProgressPreference = 'SilentlyContinue'
Remove-TypeData System.Array -ErrorAction SilentlyContinue
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
$listed = $true
$err = $null
$npu = @()
try {
  $npu = @(Get-PnpDevice -Class ComputeAccelerator -PresentOnly -ErrorAction Stop |
    Select-Object FriendlyName, Status, InstanceId,
      @{ n = 'DriverVersion'; e = {
        (Get-PnpDeviceProperty -InstanceId $_.InstanceId `
          -KeyName DEVPKEY_Device_DriverVersion -ErrorAction SilentlyContinue).Data } })
} catch {
  if ($_.CategoryInfo.Category -ne 'ObjectNotFound') {
    $listed = $false
    $err = "$($_.Exception.Message)"
  }
}
[pscustomobject]@{ listed = $listed; error = $err; npu = $npu } |
  ConvertTo-Json -Depth 4 -Compress
''';

/// A cold Windows PowerShell can take several seconds.
const kWindowsNpuTimeout = Duration(seconds: 20);

/// Runs [kWindowsNpuScript] through [runner] (killed after [timeout]) and
/// reads its JSON ([parseWindowsNpuListing]).
Future<WindowsNpuListing> listWindowsComputeAccelerators({
  ProcessRunner runner = const LocalProcessRunner(),
  String? powershell,
  Duration timeout = kWindowsNpuTimeout,
}) async {
  final ProcessOutput? out;
  try {
    out = await runner.run(powershell ?? _systemPowerShell(), [
      '-NoLogo',
      '-NoProfile',
      '-NonInteractive',
      '-EncodedCommand',
      encodePowerShellCommand(kWindowsNpuScript),
    ], timeout: timeout);
  } on Object catch (e) {
    return ComputeAcceleratorsNotListed('PowerShell could not run ($e)');
  }
  if (out == null) {
    return ComputeAcceleratorsNotListed(
      'PowerShell did not run or did not answer within ${timeout.inSeconds} s',
    );
  }
  return parseWindowsNpuListing(out);
}

/// What [out] (the query's output, or the hardware probe's, whose JSON has
/// the same `npu` array) lists.
WindowsNpuListing parseWindowsNpuListing(ProcessOutput out) {
  final json = parseWindowsProbeOutput(out.stdout);
  final npu = json?['npu'];
  if (json == null || npu is! List) {
    final error = out.stderr.trim().split('\n').first.trim();
    return ComputeAcceleratorsNotListed(
      'PowerShell printed no device list (exit ${out.exitCode}'
      '${error.isEmpty ? '' : ': $error'})',
    );
  }
  if (json['listed'] == false) {
    return ComputeAcceleratorsNotListed(
      'Get-PnpDevice failed (${json['error'] ?? 'no reason given'})',
    );
  }
  String? text(Object? value) =>
      value is String && value.trim().isNotEmpty ? value.trim() : null;
  return ComputeAcceleratorsListed([
    for (final device in npu)
      if (device is Map<String, Object?>)
        if (text(device['InstanceId']) case final instanceId?)
          ComputeAccelerator(
            name: text(device['FriendlyName']),
            status: text(device['Status']),
            instanceId: instanceId,
            driverVersion: text(device['DriverVersion']),
          ),
  ]);
}

/// Windows PowerShell under `%SystemRoot%`, not whatever `PATH` finds first.
String _systemPowerShell() => switch (Platform.environment['SystemRoot']) {
  final root? when root.isNotEmpty => p.join(
    root,
    'System32',
    'WindowsPowerShell',
    'v1.0',
    'powershell.exe',
  ),
  _ => 'powershell.exe',
};

/// The probes of this process.
final class PlatformNpuProbes implements NpuProbes {
  const PlatformNpuProbes();

  @override
  String? androidFastRpc() {
    try {
      DynamicLibrary.open('libcdsprpc.so');
      return null;
    } on Object catch (e) {
      return '$e';
    }
  }

  /// flutter_edge_ai_litertlm 1.17.0's `_linuxQualcommNpuError` (private:
  /// mirrored, as above), asked only on linux_arm64. The device files are
  /// opened read-only and closed at once: that maps no memory and opens no
  /// DSP session.
  @override
  String? linuxQualcommNpu() {
    for (final node in const ['/dev/fastrpc-cdsp', '/dev/dma_heap/system']) {
      try {
        File(node).openSync().closeSync();
      } on FileSystemException catch (e) {
        final errno = e.osError?.errorCode;
        if (errno == 2 && node == '/dev/fastrpc-cdsp') {
          return 'this machine has no Qualcomm compute DSP ($node does not '
              'exist)';
        }
        if (errno == 2) continue; // An older kernel without DMA-BUF heaps.
        return errno == 13
            ? '$node is not accessible to this user. Add the user to group '
                  'fastrpc (sudo usermod -aG fastrpc \$USER) and log in again'
            : '$node could not be opened: $e';
      }
    }
    try {
      DynamicLibrary.open('libcdsprpc.so.1');
    } on Object catch (e) {
      return "Qualcomm's FastRPC library libcdsprpc.so.1 did not open ($e); on "
          'Ubuntu it comes with the qcom-fastrpc1 package';
    }
    return null;
  }

  /// flutter_edge_ai_litertlm 1.17.0's `checkIntelNpuStack` over the
  /// executable's directory, with its wording.
  @override
  String? windowsIntelDispatch() {
    final dir = p.dirname(Platform.resolvedExecutable);
    return File(p.join(dir, 'LiteRtDispatch.dll')).existsSync()
        ? null
        : 'LiteRtDispatch.dll is not beside the executable ($dir). Every '
              'Windows x64 build with flutter_edge_ai_litertlm puts the Intel '
              'NPU stack there; ship all the DLLs the build put next to the '
              'executable with the app';
  }
}

NpuAvailability? _probed;
Future<NpuAvailability?>? _loading;

/// [npuAvailabilityFor] on this device, kept once it has answered (the answer
/// cannot change): one `dlopen('libcdsprpc.so')` on Android, a few file opens
/// on Linux arm64, nothing elsewhere. On Windows x64 it never runs a process:
/// it is [NpuPending] until [loadNpu]'s listing is in (and starts it).
NpuAvailability probeNpu() {
  if (_probed case final npu?) return npu;
  final npu = _probeHere();
  if (npu is NpuPending) {
    unawaited(loadNpu());
    return npu;
  }
  return _probed = _logged(npu);
}

/// Lists Windows' compute accelerators asynchronously (a process with a
/// deadline, never on the UI isolate's path) and keeps [probeNpu]'s answer;
/// the answer once it is in, null where nothing is listed (every other ABI:
/// [probeNpu] answers at once there). One run per process; never throws.
Future<NpuAvailability?> loadNpu() => _loading ??= () async {
  if (currentAbi() != 'windows_x64') return null;
  if (_probed case final npu?) return npu;
  final watch = Stopwatch()..start();
  final listing = await listWindowsComputeAccelerators();
  debugPrint('[NPU] Windows listed in ${watch.elapsedMilliseconds} ms');
  return _probed ??= _logged(_probeHere(windows: listing));
}();

NpuAvailability _probeHere({WindowsNpuListing? windows}) => npuAvailabilityFor(
  currentAbi(),
  const PlatformNpuProbes(),
  soc: Platform.isAndroid ? _androidSoc() : null,
  windows: windows,
);

NpuAvailability _logged(NpuAvailability npu) {
  debugPrint('[NPU] ${currentAbi()}: ${describeNpu(npu)}');
  return npu;
}

/// The SoC Android names (`QTI SM8650`), from the properties the hardware
/// probe reads too (`androidProperties`: one `getprop` run per process).
/// Null when `getprop` cannot be run or names no SoC.
String? _androidSoc() {
  final props = androidProperties();
  if (props.error case final error?) {
    debugPrint('[NPU] getprop failed: $error');
  }
  return switch (socFromProperties(props)) {
    final soc? => [?soc.manufacturer, soc.model].join(' '),
    null => null,
  };
}
