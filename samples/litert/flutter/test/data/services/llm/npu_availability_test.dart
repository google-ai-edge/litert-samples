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

import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart'
    show LiteRtLmNpuTarget;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/hardware/system_access.dart';
import 'package:litert_edge_demos/data/services/hardware/windows_hardware_info_service.dart'
    show encodePowerShellCommand;
import 'package:litert_edge_demos/data/services/llm/npu_availability.dart';
import 'package:litert_edge_demos/domain/models/npu_availability.dart';

/// The app's copy of flutter_edge_ai_litertlm 1.17.0's private NPU gate
/// (`chooseNpuVendor`, `npuStackOf`, `npuUnreachable`, `_checkNpuVendor`) for
/// the stacks this app builds with, and its own Windows device check.
void main() {
  /// What Windows PowerShell printed for the hardware probe on the two PCs:
  /// the same `npu` array (`Get-PnpDevice -Class ComputeAccelerator
  /// -PresentOnly`) the device check reads.
  ProcessOutput fixture(String name) => ProcessOutput(
    exitCode: 0,
    stdout: File('test_assets/windows/$name').readAsStringSync(),
  );

  /// The check's own JSON with [npu] as its device list.
  ProcessOutput listed(List<Map<String, Object?>> npu) => ProcessOutput(
    exitCode: 0,
    stdout: jsonEncode({'listed': true, 'error': null, 'npu': npu}),
  );

  const aiBoost = {
    'FriendlyName': 'Intel(R) AI Boost',
    'Status': 'OK',
    'InstanceId':
        r'PCI\VEN_8086&DEV_643E&SUBSYS_72708086&REV_04\3&11583659&0&58',
    'DriverVersion': '32.0.100.4778',
  };

  NpuAvailability windows(ProcessOutput? out, {String? dispatch}) =>
      npuAvailabilityFor(
        'windows_x64',
        _Probes(dispatch: dispatch),
        windows: out == null
            ? const ComputeAcceleratorsNotListed(
                'PowerShell did not run or did not answer within 20 s',
              )
            : parseWindowsNpuListing(out),
      );

  group('Android (arm64): the Qualcomm stack the app opts in to', () {
    test('available when FastRPC opens, with the SoC', () {
      final npu = npuAvailabilityFor(
        'android_arm64',
        const _Probes(),
        soc: 'QTI SM8750',
      );
      expect(describeNpu(npu), 'available (SoC QTI SM8750)');
    });

    test(
      "without FastRPC: the plugin's wording with the dlopen error, and "
      'the opt-in stacks this build leaves out (Google Tensor, MediaTek)',
      () {
        final npu = npuAvailabilityFor(
          'android_arm64',
          const _Probes(
            fastRpc: 'dlopen failed: library "libcdsprpc.so" not found',
          ),
          soc: 'Google Tensor G5',
        );
        final reason = (npu as NpuUnavailable).reason;
        expect(reason, startsWith('this device has no Qualcomm FastRPC'));
        expect(reason, contains('library "libcdsprpc.so" not found'));
        expect(
          reason,
          endsWith(
            'this build has no Google Tensor or MediaTek NPU stack (both '
            'opt-in)',
          ),
        );
        expect(npu.soc, 'Google Tensor G5');
      },
    );
  });

  group('Linux', () {
    test('arm64 offers the NPU when the Qualcomm probe passes', () {
      expect(
        npuAvailabilityFor('linux_arm64', const _Probes()),
        isA<NpuAvailable>(),
      );
    });

    test("arm64 says why the NPU is missing, in the plugin's words around "
        "the probe's", () {
      final npu = npuAvailabilityFor(
        'linux_arm64',
        const _Probes(
          linux:
              '/dev/fastrpc-cdsp is not accessible to this user. Add the user '
              'to group fastrpc',
        ),
      );
      expect(
        (npu as NpuUnavailable).reason,
        'the bundled Qualcomm NPU stack cannot run here: /dev/fastrpc-cdsp is '
        'not accessible to this user. Add the user to group fastrpc',
      );
    });
  });

  test('macOS, iOS, Linux x64 and Windows arm64 ship no NPU stack; no probe '
      'runs', () {
    for (final abi in [
      'macos_arm64',
      'ios_arm64',
      'linux_x64',
      'windows_arm64',
      'android_x64',
    ]) {
      final npu = npuAvailabilityFor(abi, const _NoProbe());
      expect(
        (npu as NpuUnavailable).reason,
        'no NPU dispatch stack ships for $abi',
      );
    }
  });

  group('Windows x64: the Intel stack, and an Intel NPU Windows lists', () {
    test('until the listing is in: pending, never a process on this path', () {
      expect(
        npuAvailabilityFor('windows_x64', const _Probes()),
        isA<NpuPending>(),
      );
      expect(describeNpu(const NpuPending()), 'being checked');
    });

    test('Lunar Lake and Panther Lake (what the two PCs printed): '
        'available, named', () {
      expect(
        describeNpu(windows(fixture('lnl-cim.json'))),
        'available (Intel(R) AI Boost)',
      );
      expect(
        describeNpu(windows(fixture('ptl-cim.json'))),
        'available (Intel(R) NPU)',
      );
    });

    test('with its driver version when Windows gives one', () {
      expect(
        describeNpu(windows(listed([aiBoost]))),
        'available (Intel(R) AI Boost, driver 32.0.100.4778)',
      );
    });

    test('a stale entry listed beside the present one (Status Unknown): the '
        'present one counts; the query asks for present devices only', () {
      final stale = {
        'FriendlyName': 'Intel(R) AI Boost',
        'Status': 'Unknown',
        'InstanceId': r'PCI\VEN_8086&DEV_7D1D&SUBSYS_88C81043&REV_04\3&1&0&58',
      };
      expect(
        describeNpu(windows(listed([stale, aiBoost]))),
        'available (Intel(R) AI Boost, driver 32.0.100.4778)',
      );
      expect(kWindowsNpuScript, contains('-PresentOnly'));
    });

    test('an Intel NPU whose status is not OK: available, with a note (the '
        'load decides)', () {
      final npu = windows(
        listed([
          {...aiBoost, 'Status': 'Error'},
        ]),
      );
      expect(
        describeNpu(npu),
        'available (Intel(R) AI Boost, driver 32.0.100.4778; status Error: '
        'its driver may not be working; the load decides)',
      );
    });

    test('no Intel device in the list: the one case that is unavailable', () {
      const reason =
          'no Intel NPU found: Windows lists no present Intel compute '
          'accelerator';
      expect((windows(listed([])) as NpuUnavailable).reason, reason);
      expect(
        (windows(
          listed([
            {
              'FriendlyName': 'AMD IPU Device',
              'Status': 'OK',
              'InstanceId': r'PCI\VEN_1022&DEV_1502\4&1',
            },
          ]),
        ) as NpuUnavailable).reason,
        reason,
      );
    });

    test('a listing that could not be made is never a dead end: available, '
        '"not checked", the load decides', () {
      for (final (out, why) in [
        (null, 'PowerShell did not run or did not answer within 20 s'),
        (
          const ProcessOutput(
            exitCode: 1,
            stdout: '',
            stderr: 'Get-PnpDevice : not recognized\r\nmore',
          ),
          'PowerShell printed no device list (exit 1: Get-PnpDevice : not '
              'recognized)',
        ),
        (
          ProcessOutput(
            exitCode: 0,
            stdout: jsonEncode({
              'listed': false,
              'error': 'Zugriff verweigert',
              'npu': <Object?>[],
            }),
          ),
          'Get-PnpDevice failed (Zugriff verweigert)',
        ),
        (
          const ProcessOutput(exitCode: 0, stdout: '{"cpu": []}'),
          'PowerShell printed no device list (exit 0)',
        ),
      ]) {
        final npu = windows(out);
        expect(npu, isA<NpuAvailable>(), reason: why);
        expect(
          describeNpu(npu),
          'available (not checked: $why; the load decides)',
        );
      }
    });

    test('the query tells "Windows lists none" from "the query failed": '
        'only the second is not listed', () {
      // The catch sees only a terminating error: without -ErrorAction Stop a
      // failed query would read as an empty list, the one unavailable case.
      expect(
        kWindowsNpuScript,
        contains(
          'Get-PnpDevice -Class ComputeAccelerator -PresentOnly '
          '-ErrorAction Stop',
        ),
      );
      // "None" is Get-PnpDevice's ObjectNotFound error: the list stays empty
      // and listed. Any other error is a failed query, with its message.
      expect(kWindowsNpuScript, contains(r'$listed = $true'));
      expect(
        kWindowsNpuScript,
        contains(r'''
} catch {
  if ($_.CategoryInfo.Category -ne 'ObjectNotFound') {
    $listed = $false
    $err = "$($_.Exception.Message)"
  }
}'''),
      );
      expect(
        kWindowsNpuScript,
        contains(r'listed = $listed; error = $err; npu = $npu'),
      );
    });

    test("the dispatch DLL missing: unavailable with the plugin's reason, "
        'whatever is listed', () {
      const missing =
          r'LiteRtDispatch.dll is not beside the executable (C:\app). Every '
          'Windows x64 build with flutter_edge_ai_litertlm puts the Intel NPU '
          'stack there; ship all the DLLs the build put next to the '
          'executable with the app';
      expect(
        (windows(
          listed([aiBoost]),
          dispatch: missing,
        ) as NpuUnavailable).reason,
        missing,
      );
    });

    test('the listing: one PowerShell call with the encoded query and a '
        'deadline; no answer is "not listed"', () async {
      final runner = _PowerShell(listed([aiBoost]));
      final listing = await listWindowsComputeAccelerators(
        runner: runner,
        powershell: 'powershell.exe',
      );
      expect(
        (listing as ComputeAcceleratorsListed).devices.single.name,
        'Intel(R) AI Boost',
      );
      expect(runner.calls.single, [
        'powershell.exe',
        '-NoLogo',
        '-NoProfile',
        '-NonInteractive',
        '-EncodedCommand',
        encodePowerShellCommand(kWindowsNpuScript),
      ]);
      expect(runner.timeouts.single, kWindowsNpuTimeout);

      final silent = await listWindowsComputeAccelerators(
        runner: _PowerShell(null),
        powershell: 'powershell.exe',
        timeout: const Duration(seconds: 3),
      );
      expect(
        (silent as ComputeAcceleratorsNotListed).reason,
        'PowerShell did not run or did not answer within 3 s',
      );
    });
  });

  group("whether this build runs a file's NPU vendor here at all", () {
    const qualcomm = LiteRtLmNpuTarget(vendor: 'Qualcomm', soc: 'SM8750');
    const intel = LiteRtLmNpuTarget(vendor: 'IntelOpenVINO', soc: 'PTL');
    const tensor = LiteRtLmNpuTarget(vendor: 'Google', soc: 'Tensor_G5');

    test('Qualcomm on Android and Linux arm64, Intel on Windows x64', () {
      expect(npuBuildRefusal(qualcomm, 'android_arm64'), isNull);
      expect(npuBuildRefusal(qualcomm, 'linux_arm64'), isNull);
      expect(npuBuildRefusal(intel, 'windows_x64'), isNull);
    });

    test('a Qualcomm build on a PC', () {
      expect(
        npuBuildRefusal(qualcomm, 'windows_x64'),
        'compiled for the Qualcomm SM8750 NPU, which this build runs on '
        'android_arm64 and linux_arm64 only, and this is windows_x64',
      );
    });

    test('an Intel build on a phone or a board', () {
      for (final abi in ['android_arm64', 'linux_arm64']) {
        expect(
          npuBuildRefusal(intel, abi),
          'compiled for the IntelOpenVINO PTL NPU, which this build runs on '
          'windows_x64 only, and this is $abi',
        );
      }
    });

    test('a Google Tensor build on a phone: its stack is opt-in and not in '
        'this build', () {
      expect(
        npuBuildRefusal(tensor, 'android_arm64'),
        'compiled for the Google Tensor_G5 NPU, and this build has no Google '
        'Tensor NPU stack (opt-in, not enabled)',
      );
    });

    test('a Google Tensor or MediaTek build off Android: their stacks run '
        'on Android only', () {
      for (final abi in ['windows_x64', 'linux_arm64', 'macos_arm64']) {
        expect(
          npuBuildRefusal(tensor, abi),
          'compiled for the Google Tensor_G5 NPU, which runs on '
          'android_arm64 only, and this is $abi',
        );
        expect(
          npuBuildRefusal(
            const LiteRtLmNpuTarget(vendor: 'MediaTek', soc: 'MT6991'),
            abi,
          ),
          'compiled for the MediaTek MT6991 NPU, which runs on android_arm64 '
          'only, and this is $abi',
        );
      }
    });

    test('a vendor the plugin has no dispatch for', () {
      expect(
        npuBuildRefusal(
          const LiteRtLmNpuTarget(vendor: 'Acme', soc: 'X1'),
          'android_arm64',
        ),
        'compiled for the Acme X1 NPU, and flutter_edge_ai_litertlm runs '
        'Qualcomm, Google Tensor, MediaTek and Intel NPU builds only',
      );
    });
  });

  test('this test host (macOS/Linux x64 CI): unavailable at once, nothing '
      'to list', () async {
    expect(probeNpu(), isA<NpuUnavailable>());
    expect(await loadNpu(), isNull);
  });
}

/// Every probe answering as given; null passes.
final class _Probes implements NpuProbes {
  const _Probes({this.fastRpc, this.linux, this.dispatch});

  final String? fastRpc;
  final String? linux;
  final String? dispatch;

  @override
  String? androidFastRpc() => fastRpc;

  @override
  String? linuxQualcommNpu() => linux;

  @override
  String? windowsIntelDispatch() => dispatch;
}

/// A host on which no probe may run.
final class _NoProbe implements NpuProbes {
  const _NoProbe();

  @override
  String? androidFastRpc() => fail('no Android probe');

  @override
  String? linuxQualcommNpu() => fail('no Linux probe');

  @override
  String? windowsIntelDispatch() => fail('no Windows dispatch check');
}

/// PowerShell answering [out] (null: not installed, or killed at the
/// deadline), recording each call.
final class _PowerShell implements ProcessRunner {
  _PowerShell(this.out);

  final ProcessOutput? out;
  final calls = <List<String>>[];
  final timeouts = <Duration>[];

  @override
  Future<ProcessOutput?> run(
    String executable,
    List<String> arguments, {
    Duration timeout = const Duration(seconds: 10),
    Map<String, String>? environment,
  }) async {
    calls.add([executable, ...arguments]);
    timeouts.add(timeout);
    return out;
  }
}
