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

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/hardware/system_access.dart';
import 'package:litert_edge_demos/data/services/hardware/windows_hardware_info_service.dart';
import 'package:litert_edge_demos/domain/hardware/accelerator_inference.dart';
import 'package:litert_edge_demos/domain/hardware/diagnostics_report.dart';
import 'package:litert_edge_demos/domain/models/hardware_profile.dart';

/// What two real PCs and a VM without a GPU printed for
/// [kWindowsProbeScript]'s queries (test_assets/README.md).
final _lunarLake = File('test_assets/windows/lnl-cim.json').readAsStringSync();
final _pantherLake = File('test_assets/windows/ptl-cim.json')
    .readAsStringSync();
final _gpulessVm = File('test_assets/windows/vm-cim.json').readAsStringSync();

/// Records the call and answers with [output] (null: PowerShell missing).
final class _PowerShell implements ProcessRunner {
  _PowerShell(this.output);

  final ProcessOutput? output;
  final calls = <List<String>>[];

  @override
  Future<ProcessOutput?> run(
    String executable,
    List<String> arguments, {
    Duration timeout = const Duration(seconds: 10),
    Map<String, String>? environment,
  }) async {
    calls.add([executable, ...arguments]);
    return output;
  }
}

Future<HardwareProfile> _probe(String stdout, {int exitCode = 0}) =>
    WindowsHardwareInfoService(
      processes: _PowerShell(ProcessOutput(exitCode: exitCode, stdout: stdout)),
      powershell: 'powershell.exe',
      architecture: 'x86_64',
      fallbackOs: 'windows "Windows 11 Pro" 10.0 (Build 26200)',
      fallbackCores: 4,
    ).probe();

HardwareProfile _fromJson(String json) => windowsProfileFrom(
  jsonDecode(json) as Map<String, Object?>,
  architecture: 'x86_64',
  fallbackOs: 'windows',
  fallbackCores: 4,
);

void main() {
  test('Lunar Lake (Core Ultra 7 258V): CPU, cores, RAM, OS, the Arc 140V '
      'with its driver, the NPU with its PCI id; a GPU load is inferred on '
      'the Arc over WebGPU/D3D12, not refused', () async {
    final p = await _probe(_lunarLake);
    expect(p.platform, HostPlatform.windows);
    expect(p.os, 'Microsoft Windows 11 Pro (10.0.26200)');
    expect(p.kernel, isNull);
    expect(p.libc, isNull);
    expect(p.machine, isNull, reason: 'the fixture has no machine record');
    expect(p.cpu.model, 'Intel(R) Core(TM) Ultra 7 258V');
    expect(p.cpu.cores, 8);
    expect(p.cpu.architecture, 'x86_64');
    expect(p.memory!.totalBytes, 33905037312);
    expect(p.memory!.availableBytes, isNull);
    expect(p.gpus, hasLength(1));
    final gpu = p.gpus.single;
    expect(gpu.name, 'Intel(R) Arc(TM) 140V GPU (16GB)');
    expect(gpu.driver, '32.0.101.8826');
    expect(gpu.kind, GpuKind.unknown);
    expect(gpu.source, 'Win32_VideoController');
    expect(gpu.inferred, isFalse);
    expect(p.vulkan, isNull);
    expect(
      [for (final n in p.npuHints) n.text],
      ['Intel(R) AI Boost (PCI 8086:643e, status OK)'],
    );
    expect(p.notes, isEmpty);

    expect(likelyAdapter(p), (
      name: 'Intel(R) Arc(TM) 140V GPU (16GB)',
      software: false,
    ));
    final evidence = inferEvidence(
      requested: 'gpu',
      reported: 'gpu',
      hardware: p,
    );
    expect(evidence.api, 'WebGPU/D3D12');
    expect(evidence.adapter, 'Intel(R) Arc(TM) 140V GPU (16GB)');
    expect(softwareGpuReason(evidence), isNull);
  });

  test('the report names CPU, RAM, GPU with its driver, NPU and OS as on '
      'Linux and macOS', () async {
    final p = await _probe(_lunarLake);
    expect(deviceLines(p), [
      'os         Microsoft Windows 11 Pro (10.0.26200)',
      'cpu        Intel(R) Core(TM) Ultra 7 258V · 8 cores · x86_64',
      'ram        31.58 GB total',
      'gpu        Intel(R) Arc(TM) 140V GPU (16GB) · driver 32.0.101.8826 · '
          'Win32_VideoController',
      'npu        Intel(R) AI Boost (PCI 8086:643e, status OK) · information '
          'only',
    ]);
  });

  test('Panther Lake (Core Ultra X7 358H): the Arc B390 and the NPU', () async {
    final p = await _probe(_pantherLake);
    expect(p.os, 'Microsoft Windows 11 Pro (10.0.26200)');
    expect(p.cpu.model, 'Intel(R) Core(TM) Ultra X7 358H');
    expect(p.cpu.cores, 16);
    expect(p.memory!.totalBytes, 68173721600);
    expect(p.gpus.single.name, 'Intel(R) Arc(TM) B390 GPU');
    expect(p.gpus.single.driver, '32.0.101.8864');
    expect(
      [for (final n in p.npuHints) n.text],
      ['Intel(R) NPU (PCI 8086:b03e, status OK)'],
    );
    expect(p.notes, isEmpty);
  });

  group('the output around the JSON', () {
    test('a byte-order mark, a CLIXML block or other lines before it are '
        'skipped', () async {
      const clixml =
          '#< CLIXML\r\n<Objs Version="1.1.0.1" '
          'xmlns="http://schemas.microsoft.com/powershell/2004/04"><Obj '
          'S="progress" RefId="0"><TN RefId="0"><T>System.Management.'
          'Automation.PSCustomObject</T></TN></Obj></Objs>\r\n';
      for (final stdout in [
        '\uFEFF$_lunarLake',
        '$clixml$_lunarLake',
        '\uFEFF$clixml\uFEFF$_lunarLake',
        '** WARNING: something else first\r\n$_pantherLake',
      ]) {
        final p = await _probe(stdout);
        expect(p.cpu.model, startsWith('Intel(R) Core(TM) Ultra'));
        expect(p.npuHints, hasLength(1));
      }
    });

    test('no JSON: the OS and cores from dart:io, and the exit code and '
        "PowerShell's first error line in the note", () async {
      final runner = _PowerShell(
        const ProcessOutput(
          exitCode: 1,
          stdout: '',
          stderr: 'Get-CimInstance : Access denied\r\nAt line:1 char:1\r\n',
        ),
      );
      final p = await WindowsHardwareInfoService(
        processes: runner,
        powershell: 'powershell.exe',
        architecture: 'x86_64',
        fallbackOs: 'windows "Windows 11 Pro" 10.0 (Build 26200)',
        fallbackCores: 4,
      ).probe();
      expect(p.platform, HostPlatform.windows);
      expect(p.os, 'windows "Windows 11 Pro" 10.0 (Build 26200)');
      expect(p.cpu.model, 'unknown');
      expect(p.cpu.cores, 4);
      expect(p.gpus, isEmpty);
      expect(p.notes, [
        'The PowerShell probe printed no JSON (exit 1: Get-CimInstance : '
            'Access denied): the chip, GPU, NPU and RAM are not probed.',
      ]);
    });

    test('PowerShell missing or silent: said so', () async {
      final p = await WindowsHardwareInfoService(
        processes: _PowerShell(null),
        powershell: 'powershell.exe',
        architecture: 'x86_64',
        fallbackOs: 'windows',
        fallbackCores: 2,
      ).probe();
      expect(p.cpu.cores, 2);
      expect(p.notes.single, startsWith('PowerShell did not run'));
    });

    test('a truncated object is no JSON', () {
      expect(parseWindowsProbeOutput(_lunarLake.substring(0, 200)), isNull);
      expect(parseWindowsProbeOutput('nothing here'), isNull);
      expect(parseWindowsProbeOutput('[1, 2]'), isNull);
    });
  });

  test('one call: Windows PowerShell with the script base64-encoded as '
      'UTF-16LE (no quoting), CIM and PnP queries, never wmic', () async {
    final runner = _PowerShell(ProcessOutput(exitCode: 0, stdout: _lunarLake));
    await WindowsHardwareInfoService(
      processes: runner,
      powershell: r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe',
      architecture: 'x86_64',
    ).probe();
    expect(runner.calls, hasLength(1));
    final call = runner.calls.single;
    expect(call.take(5), [
      r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe',
      '-NoLogo',
      '-NoProfile',
      '-NonInteractive',
      '-EncodedCommand',
    ]);
    final bytes = base64.decode(call[5]);
    final units = [
      for (var i = 0; i < bytes.length; i += 2) bytes[i] | bytes[i + 1] << 8,
    ];
    final script = String.fromCharCodes(units);
    expect(script, kWindowsProbeScript);
    // A failed adapter query is null, not an empty list (which infers WARP):
    // the global SilentlyContinue would otherwise turn it into [].
    expect(script, contains('Win32_VideoController -ErrorAction Stop'));
    expect(script, contains(r'catch { $gpus = $null }'));
    for (final query in [
      'Win32_Processor',
      'Win32_VideoController',
      'Win32_OperatingSystem',
      'Win32_ComputerSystem',
      'Get-PnpDevice -Class ComputeAccelerator -PresentOnly',
      'ConvertTo-Json',
    ]) {
      expect(script, contains(query));
    }
    expect(script.toLowerCase(), isNot(contains('wmic')));
  });

  group('JSON shapes Windows PowerShell can print', () {
    test('a single record as an object, an array in its {value, Count} '
        'wrapper, numbers as strings, a missing NPU list', () {
      final p = _fromJson('''
{"cpu": {"Name": "Intel(R) Core(TM) Ultra 7 258V  ", "NumberOfCores": 8,
         "NumberOfLogicalProcessors": "8"},
 "gpus": {"value": [{"Name": "Intel(R) Arc(TM) 140V GPU (16GB)",
                     "DriverVersion": "32.0.101.8826"}], "Count": 1},
 "os": {"Caption": "Microsoft Windows 11 Home", "Version": "10.0.26100"},
 "ram": "16777216000"}
''');
      expect(p.cpu.model, 'Intel(R) Core(TM) Ultra 7 258V');
      expect(p.cpu.cores, 8);
      expect(p.gpus.single.name, 'Intel(R) Arc(TM) 140V GPU (16GB)');
      expect(p.os, 'Microsoft Windows 11 Home (10.0.26100)');
      expect(p.memory!.totalBytes, 16777216000);
      expect(p.npuHints, isEmpty);
    });

    test('two sockets: their logical cores add up; no core count at all '
        'falls back to dart:io', () {
      final two = _fromJson('''
{"cpu": [{"Name": "Intel(R) Xeon(R) Gold", "NumberOfLogicalProcessors": 16},
         {"Name": "Intel(R) Xeon(R) Gold", "NumberOfLogicalProcessors": 16}]}
''');
      expect(two.cpu.cores, 32);
      final none = _fromJson('{"cpu": []}');
      expect(none.cpu.model, 'unknown');
      expect(none.cpu.cores, 4);
      expect(
        none.notes,
        containsAll([
          'Win32_Processor names no CPU.',
          'Win32_ComputerSystem.TotalPhysicalMemory is not readable.',
          'Win32_VideoController is not readable: the GPU is unknown.',
        ]),
      );
      expect(none.gpus, isEmpty, reason: 'no answer: no GPU inferred');
      expect(none.os, 'windows', reason: 'no OS record: the fallback');
    });

    test('the machine from Win32_ComputerSystem, without firmware '
        'placeholders; the GPU source carries its PCI id', () {
      final p = _fromJson(r'''
{"machine": {"Manufacturer": "ASUSTeK COMPUTER INC.",
             "Model": "ASUS Zenbook S 14 UX5406SA"},
 "gpus": [{"Name": "Intel(R) Arc(TM) 140V GPU (16GB)",
           "PNPDeviceID": "PCI\\VEN_8086\u0026DEV_64A0\u0026SUBSYS_22128086\u0026REV_04\\3\u002611583659\u00260\u002610"}]}
''');
      expect(p.machine, 'ASUSTeK COMPUTER INC. ASUS Zenbook S 14 UX5406SA');
      expect(p.gpus.single.source, 'Win32_VideoController, PCI 8086:64a0');
      final placeholder = _fromJson('''
{"machine": {"Manufacturer": "System manufacturer",
             "Model": "System Product Name"}}
''');
      expect(placeholder.machine, isNull);
    });
  });

  group('software and virtual adapters', () {
    test('WARP only (no GPU driver): a software GPU, a note, and a GPU load '
        'is refused as a software rasterizer', () {
      final p = _fromJson('''
{"gpus": [{"Name": "Microsoft Basic Display Adapter",
           "DriverVersion": "10.0.26100.1"}]}
''');
      expect(p.gpus.single.kind, GpuKind.software);
      expect(p.notes, [
        'Win32_Processor names no CPU.',
        'Win32_ComputerSystem.TotalPhysicalMemory is not readable.',
        'Windows lists only Microsoft Basic Display Adapter: no GPU driver is '
            'installed, and a GPU backend would run on the CPU (WARP). '
            "Install the GPU vendor's driver.",
      ]);
      expect(likelyAdapter(p), (
        name: 'Microsoft Basic Display Adapter',
        software: true,
      ));
      final evidence = inferEvidence(
        requested: 'gpu',
        reported: 'gpu',
        hardware: p,
      );
      expect(
        softwareGpuReason(evidence),
        'The GPU is a software rasterizer: Microsoft Basic Display Adapter '
        '(inferred). It runs on the CPU, far slower than the CPU backend.',
      );
      expect(deviceLines(p), contains(contains('· software ·')));
    });

    test('WARP or a Remote Desktop adapter listed before the real GPU: the '
        'real GPU comes first and is the one a GPU load opens', () {
      final p = _fromJson('''
{"gpus": [{"Name": "Microsoft Remote Display Adapter"},
          {"Name": "Microsoft Basic Render Driver"},
          {"Name": "Intel(R) Arc(TM) B390 GPU", "DriverVersion": "32.0.101.8864"}]}
''');
      expect(
        [for (final g in p.gpus) (g.name, g.kind)],
        [
          ('Intel(R) Arc(TM) B390 GPU', GpuKind.unknown),
          ('Microsoft Basic Render Driver', GpuKind.software),
          ('Microsoft Remote Display Adapter', GpuKind.virtualDisplay),
        ],
      );
      expect(windowsGpuNotes(p.gpus), isEmpty);
      expect(likelyAdapter(p), (
        name: 'Intel(R) Arc(TM) B390 GPU',
        software: false,
      ));
    });

    test('no display adapter listed (a VM without a GPU): WARP is inferred, '
        'and a GPU load on it is refused as a software rasterizer', () {
      for (final json in [
        _gpulessVm,
        '{"gpus": [{"Name": "Microsoft Remote Display Adapter"}]}',
      ]) {
        final p = _fromJson(json);
        final warp = p.gpus.last;
        expect(
          (warp.name, warp.kind, warp.inferred),
          ('Microsoft Basic Render Driver', GpuKind.software, true),
          reason: json,
        );
        expect(
          p.notes,
          contains(
            'No GPU found (Win32_VideoController lists none, or only a remote '
            'or VM display): a GPU backend would run on the CPU (WARP).',
          ),
          reason: json,
        );
        final evidence = inferEvidence(
          requested: 'gpu',
          reported: 'gpu',
          hardware: p,
        );
        expect(
          softwareGpuReason(evidence),
          'The GPU is a software rasterizer: Microsoft Basic Render Driver '
          '(inferred). It runs on the CPU, far slower than the CPU backend.',
          reason: json,
        );
      }
    });

    test('the display adapter query failed (null), or named an adapter '
        'only partly: no GPU is inferred and a GPU load is not refused on '
        'that account', () {
      for (final json in const [
        '{"gpus": null}',
        '{"gpus": [{"Name": "  "}]}',
      ]) {
        final p = _fromJson(json);
        expect(p.gpus, isEmpty, reason: json);
        expect(
          p.notes,
          contains(
            'Win32_VideoController is not readable: the GPU is unknown.',
          ),
          reason: json,
        );
        expect(likelyAdapter(p), isNull, reason: json);
      }
    });
  });

  test('an NPU whose driver is not working: its status in the hint and a '
      'note', () {
    final p = _fromJson(r'''
{"npu": [{"FriendlyName": "Intel(R) AI Boost", "Status": "Error",
          "InstanceId": "PCI\\VEN_8086\u0026DEV_643E\\3"}]}
''');
    expect(
      p.npuHints.single.text,
      'Intel(R) AI Boost (PCI 8086:643e, status '
      'Error)',
    );
    expect(
      p.notes,
      contains(
        'The NPU Intel(R) AI Boost reports status Error: its driver is not '
        'working (see Device Manager, Neural processors).',
      ),
    );
  });

  test('the script stays ASCII: Windows PowerShell reads it as UTF-16 '
      'code units, and a stray character would end up in the command', () {
    expect(kWindowsProbeScript.codeUnits.every((c) => c < 0x80), isTrue);
    expect(
      encodePowerShellCommand('A\u20AC'),
      base64.encode([0x41, 0, 0xac, 0x20]),
    );
  });
}
