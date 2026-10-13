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

import 'package:flutter_edge_ai/flutter_edge_ai.dart' show PreferredBackend;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/selftest/self_test_options.dart';
import 'package:litert_edge_demos/utils/result.dart';

SelfTestOptions _ok(List<String> args, [Map<String, String> env = const {}]) =>
    switch (SelfTestOptions.parse(args, env)) {
      Ok(:final value) => value,
      final other => fail('expected options, got $other'),
    };

String _error(List<String> args) => switch (SelfTestOptions.parse(args, {})) {
  Error(:final error) => error.toString(),
  final other => fail('expected an error, got $other'),
};

void main() {
  test('no --selftest and no SELFTEST=1: the normal app', () {
    expect(SelfTestOptions.parse(const [], const {}), isNull);
    expect(SelfTestOptions.parse(const ['--gemma=/x'], const {}), isNull);
    expect(SelfTestOptions.parse(const [], const {'SELFTEST': '0'}), isNull);
  });

  test("defaults: the chat model's own backend, GPU for the detector, no "
      'paths, no CPU retry', () {
    final o = _ok(const ['--selftest']);
    expect(o.gemmaBackend, isNull, reason: 'gpu for Gemma 4 E2B, else saved');
    expect(o.detectorBackend, DetectorBackend.gpu);
    expect(o.gemmaPath, isNull);
    expect(o.detectorCpuRetry, isFalse);
    expect(o.skipAudio, isFalse, reason: 'the audio step runs by default');
    expect(o.timeout, const Duration(minutes: 30));
    expect(_ok(const [], const {'SELFTEST': '1'}).gemmaPath, isNull);
  });

  test('--gemma-backend=npu parses (it fails later, with the reason, where '
      'flutter_edge_ai has no NPU stack)', () {
    expect(
      _ok(const ['--selftest', '--gemma-backend=npu']).gemmaBackend,
      PreferredBackend.npu,
    );
  });

  test('every option; macOS -NS… arguments are ignored', () {
    final o = _ok(const [
      '-NSDocumentRevisionsDebugMode',
      'YES',
      '--selftest',
      '--gemma=/m/gemma.litertlm',
      '--detector=/m/yolo.tflite',
      '--image=/m/cats.jpg',
      '--gemma-backend=cpu',
      '--detector-backend=cpu',
      '--detector-cpu-retry',
      '--allow-software-gpu',
      '--skip-audio',
      '--out=/tmp/st.txt',
      '--timeout=600',
    ]);
    expect(o.gemmaPath, '/m/gemma.litertlm');
    expect(o.detectorPath, '/m/yolo.tflite');
    expect(o.imagePath, '/m/cats.jpg');
    expect(o.outPath, '/tmp/st.txt');
    expect(o.gemmaBackend, PreferredBackend.cpu);
    expect(o.detectorBackend, DetectorBackend.cpu);
    expect(o.detectorCpuRetry, isTrue);
    expect(o.allowSoftwareGpu, isTrue);
    expect(o.skipAudio, isTrue);
    expect(o.timeout, const Duration(minutes: 10));
  });

  test(
    'a typo or bad value is an error with the usage, never another test',
    () {
      expect(
        _error(const ['--selftest', '--gemma-backed=cpu']),
        contains('unknown option'),
      );
      expect(
        _error(const ['--selftest', '--gemma-backend=tpu']),
        contains('npu, gpu or cpu'),
      );
      expect(
        _error(const ['--selftest', '--detector-backend=tpu']),
        contains('gpu or cpu'),
      );
      expect(_error(const ['--selftest', '--gemma']), contains('needs =VALUE'));
      expect(
        _error(const ['--selftest', '--detector-cpu-retry=yes']),
        contains('takes no value'),
      );
      expect(
        _error(const ['--selftest', '--skip-audio=1']),
        contains('takes no value'),
      );
      expect(
        _error(const ['--selftest', '--timeout=soon']),
        contains('whole seconds'),
      );
      expect(_error(const ['--selftest', '--out=']), contains(kSelfTestUsage));
    },
  );
}
