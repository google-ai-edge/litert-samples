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

// tool/linux/run.sh, the launcher shipped next to the Linux bundle:
// shellcheck when it is installed, and runs against stub tools (pactl,
// parecord, vulkaninfo, gst-inspect-1.0, the app) on a PATH that holds
// nothing else, so the real machine's tools never leak in.
@TestOn('linux || mac-os')
library;

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

const _script = 'tool/linux/run.sh';

/// The coreutils the script uses, linked into a directory of their own.
const _tools = [
  'bash',
  'env',
  'sed',
  'awk',
  'wc',
  'tr',
  'head',
  'date',
  'dirname',
  'tee',
  'mkdir',
  'cat',
];

String? _which(String tool) {
  final which = Process.runSync('/bin/sh', ['-c', 'command -v $tool']);
  final path = (which.stdout as String).trim();
  return path.isEmpty ? null : path;
}

void _stub(Directory dir, String name, String body) {
  final file = File('${dir.path}/$name')
    ..writeAsStringSync('#!/bin/sh\n$body\n');
  Process.runSync('chmod', ['+x', file.path]);
}

final class _Bundle {
  _Bundle(this.root) : bin = Directory('${root.path}/bin') {
    bin.createSync();
    for (final tool in _tools) {
      final path = _which(tool);
      if (path == null) throw StateError('$tool not found on this machine');
      Link('${bin.path}/$tool').createSync(path);
    }
    // coreutils' timeout (Linux; macOS has none): the script uses it when
    // present.
    if (_which('timeout') case final path?) {
      Link('${bin.path}/timeout').createSync(path);
    }
    File(_script).copySync('${root.path}/run.sh');
    Process.runSync('chmod', ['+x', '${root.path}/run.sh']);
    _stub(root, 'litert_edge_demos', 'echo "app got: \$*"\nexit 3');
  }

  final Directory root;
  final Directory bin;

  Future<ProcessResult> run(
    List<String> args, {
    Map<String, String> env = const {},
  }) => Process.run(
    '${bin.path}/bash',
    ['${root.path}/run.sh', ...args],
    // The board's device-tree model: a file the tests write, absent by default
    // (so a test host that is a Raspberry Pi reads no board either).
    environment: {
      'PATH': bin.path,
      'HOME': root.path,
      'RUN_SH_DT_MODEL': '${root.path}/model',
      ...env,
    },
    includeParentEnvironment: false,
  );
}

void main() {
  late Directory temp;
  late _Bundle bundle;

  setUp(() {
    temp = Directory.systemTemp.createTempSync('run_sh_test');
    bundle = _Bundle(temp);
  });
  tearDown(() => temp.deleteSync(recursive: true));

  test('shellcheck is clean (when installed)', () {
    final which = Process.runSync('/bin/sh', ['-c', 'command -v shellcheck']);
    final shellcheck = (which.stdout as String).trim();
    if (shellcheck.isEmpty) {
      markTestSkipped('shellcheck not installed; the dry runs below still run');
      return;
    }
    final result = Process.runSync(shellcheck, ['-s', 'bash', _script]);
    expect(result.exitCode, 0, reason: '${result.stdout}${result.stderr}');
  });

  test('a healthy machine: every check ok, llvmpipe flagged next to the real '
      'GPU; the dry run prints the app with every argument', () async {
    _stub(bundle.bin, 'pactl', r'''
case "$*" in
  info) echo 'Server Name: PulseAudio (on PipeWire 1.0.5)' ;;
  'list short sources') printf '47\talsa_output.pci.analog-stereo.monitor\tPipeWire\ts32le 2ch 48000Hz\tSUSPENDED\n48\talsa_input.pci.analog-stereo\tPipeWire\ts32le 2ch 48000Hz\tSUSPENDED\n' ;;
  'list short sinks') printf '46\talsa_output.pci.analog-stereo\tPipeWire\ts32le 2ch 48000Hz\tSUSPENDED\n' ;;
esac''');
    _stub(bundle.bin, 'parecord', 'exit 0');
    _stub(bundle.bin, 'vulkaninfo', r'''
echo 'Devices:'
echo '	deviceName         = NVIDIA GeForce RTX 3060'
echo '	deviceName         = llvmpipe (LLVM 15.0.7, 256 bits)' ''');
    _stub(bundle.bin, 'gst-inspect-1.0', 'exit 0');

    final result = await bundle.run(
      const ['--selftest', '--gemma=/m/my model.litertlm'],
      env: const {'RUN_SH_DRY_RUN': '1'},
    );
    final out = result.stdout as String;
    expect(result.exitCode, 0, reason: out);
    expect(
      out,
      contains('  ok    sound server: PulseAudio (on PipeWire 1.0.5)'),
    );
    expect(out, contains('  ok    microphones: 1'));
    expect(out, contains('  ok    outputs: 1'));
    expect(out, contains('  ok    parecord'));
    expect(out, contains('  ok    Vulkan device: NVIDIA GeForce RTX 3060'));
    expect(
      out,
      contains(
        '  WARN  software Vulkan device: llvmpipe (LLVM 15.0.7, 256 bits) '
        '(the CPU, not a GPU)',
      ),
    );
    expect(out, isNot(contains('no hardware GPU')));
    expect(out, contains('  ok    GStreamer v4l2src'));
    expect(
      out,
      contains(
        r'would run: '
        '${temp.path}/litert_edge_demos --selftest '
        r'--gemma=/m/my\ model.litertlm',
      ),
    );
    expect(
      File('${temp.path}/run.log').readAsStringSync(),
      contains('sound server: PulseAudio'),
      reason: 'everything also goes to run.log',
    );
  });

  test('a broken machine: what to install or start, for every check', () async {
    _stub(bundle.bin, 'pactl', r'''
case "$*" in
  info) echo 'Connection failure: Connection refused' >&2; exit 1 ;;
esac''');
    _stub(bundle.bin, 'gst-inspect-1.0', 'exit 1');

    final result = await bundle.run(
      const [],
      env: const {'RUN_SH_DRY_RUN': '1'},
    );
    final out = result.stdout as String;
    expect(result.exitCode, 0, reason: out);
    expect(
      out,
      contains(
        '  WARN  no sound server: Connection failure: Connection refused',
      ),
    );
    expect(out, contains('systemctl --user start pipewire pipewire-pulse'));
    expect(out, contains('  WARN  parecord not found'));
    expect(out, contains('sudo apt install pulseaudio-utils'));
    expect(out, contains('  WARN  vulkaninfo not found'));
    expect(out, contains('  WARN  GStreamer v4l2src is missing'));
    expect(out, contains('sudo apt install gstreamer1.0-plugins-good'));
  });

  test(
    'a hung sound server: pactl is cut off after 5 s, the launcher goes on',
    () async {
      if (_which('timeout') == null) {
        markTestSkipped('no coreutils timeout here (macOS): nothing to bound');
        return;
      }
      _stub(bundle.bin, 'pactl', 'exec /bin/sleep 60');
      final watch = Stopwatch()..start();
      final out =
          (await bundle.run(
                const [],
                env: const {'RUN_SH_DRY_RUN': '1'},
              )).stdout
              as String;
      expect(watch.elapsed, lessThan(const Duration(seconds: 20)));
      expect(
        out,
        contains('  WARN  no sound server: pactl info did not answer'),
      );
      expect(out, contains('would run:'));
    },
    timeout: const Timeout(Duration(seconds: 40)),
  );

  test('the network camera\'s JPEG decoder: ok when the bundle ships '
      'libturbojpeg.so.0; otherwise the system\'s copy, or a warning with '
      'the package to install', () async {
    final hasSystemCopy = [
      for (final dir in [
        '/usr/lib',
        '/usr/lib64',
        '/usr/local/lib',
        if (Directory('/usr/lib').existsSync())
          for (final d in Directory(
            '/usr/lib',
          ).listSync().whereType<Directory>())
            d.path,
      ])
        File('$dir/libturbojpeg.so.0').existsSync(),
    ].any((found) => found);
    final missing =
        (await bundle.run(const [], env: const {'RUN_SH_DRY_RUN': '1'})).stdout
            as String;
    if (hasSystemCopy) {
      expect(missing, contains('  ok    libturbojpeg.so.0 (/usr/'));
    } else {
      expect(
        missing,
        contains('  WARN  libturbojpeg.so.0 not found: the network camera'),
      );
      expect(missing, contains('sudo apt install libturbojpeg'));
    }

    Directory('${temp.path}/lib').createSync();
    File('${temp.path}/lib/libturbojpeg.so.0').writeAsStringSync('');
    final shipped =
        (await bundle.run(const [], env: const {'RUN_SH_DRY_RUN': '1'})).stdout
            as String;
    expect(shipped, contains('  ok    libturbojpeg.so.0 (shipped in lib/)'));
  });

  test('only monitors and the dummy sink: no microphone, no output', () async {
    _stub(bundle.bin, 'pactl', r'''
case "$*" in
  info) echo 'Server Name: pulseaudio' ;;
  'list short sources') printf '0\tauto_null.monitor\tmodule-null-sink.c\ts16le 2ch 44100Hz\tSUSPENDED\n' ;;
  'list short sinks') printf '0\tauto_null\tmodule-null-sink.c\ts16le 2ch 44100Hz\tSUSPENDED\n' ;;
esac''');
    final out =
        (await bundle.run(const [], env: const {'RUN_SH_DRY_RUN': '1'})).stdout
            as String;
    expect(out, contains('  WARN  no recording device (only monitors'));
    expect(
      out,
      contains('  WARN  no audio output (no sink, or only the dummy'),
    );
  });

  test('not a dry run: the shell is replaced by the app, which gets every '
      'argument and whose exit code is the script\'s; RUN_LOG= turns the log '
      'off', () async {
    final result = await bundle.run(
      const ['--selftest', '--skip-audio'],
      env: const {'RUN_LOG': ''},
    );
    expect(result.exitCode, 3);
    expect(result.stdout, contains('app got: --selftest --skip-audio'));
    expect(File('${temp.path}/run.log').existsSync(), isFalse);
  });

  group('V3D_WEBGPU_OVERRIDE', () {
    void board(String model) =>
        File('${temp.path}/model').writeAsStringSync('$model\u0000');

    test('a Raspberry Pi 5: set to 1 for the app, and said so', () async {
      board('Raspberry Pi 5 Model B Rev 1.0');
      final result = await bundle.run(
        const ['--selftest'],
        env: const {'RUN_SH_DRY_RUN': '1'},
      );
      final out = result.stdout as String;
      expect(
        out,
        contains(
          '  ok    Raspberry Pi 5 Model B Rev 1.0: V3D_WEBGPU_OVERRIDE=1',
        ),
      );
      expect(
        out,
        contains(
          'would run: V3D_WEBGPU_OVERRIDE=1 ${temp.path}/litert_edge_demos '
          '--selftest',
        ),
      );
    });

    test('a value the caller set is kept, and said so', () async {
      board('Raspberry Pi 5 Model B Rev 1.0');
      final out =
          (await bundle.run(
                const [],
                env: const {'RUN_SH_DRY_RUN': '1', 'V3D_WEBGPU_OVERRIDE': '0'},
              )).stdout
              as String;
      expect(
        out,
        contains(
          '  ok    Raspberry Pi 5 Model B Rev 1.0: V3D_WEBGPU_OVERRIDE=0 '
          '(kept as set)',
        ),
      );
      expect(out, contains('would run: V3D_WEBGPU_OVERRIDE=0 '));
      expect(out, isNot(contains('V3D_WEBGPU_OVERRIDE=1')));
    });

    test('an empty value counts as unset', () async {
      board('Raspberry Pi 5 Model B Rev 1.0');
      final out =
          (await bundle.run(
                const [],
                env: const {'RUN_SH_DRY_RUN': '1', 'V3D_WEBGPU_OVERRIDE': ''},
              )).stdout
              as String;
      expect(out, contains('would run: V3D_WEBGPU_OVERRIDE=1 '));
    });

    test('another board, or none: not set, and nothing on stderr', () async {
      board('Raspberry Pi 4 Model B Rev 1.5');
      final out =
          (await bundle.run(
                const [],
                env: const {'RUN_SH_DRY_RUN': '1'},
              )).stdout
              as String;
      expect(out, isNot(contains('V3D_WEBGPU_OVERRIDE')));
      // No device tree, as on most machines; without a log, stderr stays apart.
      File('${temp.path}/model').deleteSync();
      final result = await bundle.run(
        const [],
        env: const {'RUN_SH_DRY_RUN': '1', 'RUN_LOG': ''},
      );
      expect(result.stdout, isNot(contains('V3D_WEBGPU_OVERRIDE')));
      expect(result.stderr, isEmpty);
    });

    test('the app itself gets it in its environment', () async {
      board('Raspberry Pi 5 Model B Rev 1.0');
      _stub(
        bundle.root,
        'litert_edge_demos',
        r'echo "app env: V3D_WEBGPU_OVERRIDE=${V3D_WEBGPU_OVERRIDE:-unset}"',
      );
      final out =
          (await bundle.run(const [], env: const {'RUN_LOG': ''})).stdout
              as String;
      expect(out, contains('app env: V3D_WEBGPU_OVERRIDE=1'));
    });
  });
}
