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

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/hardware/android_props.dart';
import 'package:litert_edge_demos/data/services/hardware/system_access.dart';

import '../../../support/android_devices.dart';

void main() {
  group('parseGetpropDump', () {
    test('reads every [key]: [value] line, empty values included', () {
      final values = parseGetpropDump(kAndroid11Kona);
      expect(values['ro.board.platform'], 'kona');
      expect(values['ro.build.version.sdk'], '30');
      expect(values['ro.soc.model'], '');
      expect(values, hasLength(8));
    });

    test('a full `adb shell getprop` dump: values with spaces, brackets '
        'inside, CRLF endings; other lines are skipped', () {
      final values = parseGetpropDump(
        '[dalvik.vm.heapsize]: [512m]\r\n'
        'garbage line\r\n'
        '[ro.build.fingerprint]: [samsung/e1quew/e1q:14/UP1A.231005.007/'
        'S921USQU1AXB7:user/release-keys]\r\n'
        '[ro.product.marketname]: [Galaxy S24 [5G]]\r\n'
        '[ro.soc.model]: [SM8650]\r\n',
      );
      expect(values['ro.soc.model'], 'SM8650');
      expect(values['ro.product.marketname'], 'Galaxy S24 [5G]');
      expect(values['dalvik.vm.heapsize'], '512m');
      expect(values, hasLength(4));
    });

    test('nothing parseable is an empty map', () {
      expect(parseGetpropDump(''), isEmpty);
      expect(parseGetpropDump('getprop: not found\n'), isEmpty);
    });
  });

  test('MapAndroidProperties: empty and blank values are unset', () {
    final props = propsOf(kAndroid11Kona);
    expect(props['ro.soc.model'], isNull);
    expect(props['ro.board.platform'], 'kona');
    expect(props['ro.missing'], isNull);
    expect(props.error, isNull);
  });

  group('GetpropAndroidProperties', () {
    test('ONE process for all keys, run on first access and kept; the '
        'script asks getprop for exactly the known keys', () {
      final calls = <List<String>>[];
      final props = GetpropAndroidProperties(
        run: (exe, args) {
          calls.add([exe, ...args]);
          return const ProcessOutput(exitCode: 0, stdout: kGalaxyS24Snapdragon);
        },
      );
      expect(calls, isEmpty, reason: 'lazy');
      expect(props['ro.soc.model'], 'SM8650');
      expect(props['ro.product.model'], 'SM-S921U');
      expect(props.error, isNull);
      expect(calls, hasLength(1));
      final [exe, flag, script] = calls.single;
      expect(exe, '/system/bin/sh');
      expect(flag, '-c');
      for (final key in kAndroidPropertyKeys) {
        expect(script, contains(key));
      }
      expect(script, contains(r'getprop "$k"'));
    });

    test('sh cannot start: every property is null and the error says why', () {
      final props = GetpropAndroidProperties(run: (_, _) => null);
      expect(props['ro.soc.model'], isNull);
      expect(props.error, '/system/bin/sh could not start');
    });

    test('a failing exit code carries stderr', () {
      final props = GetpropAndroidProperties(
        run: (_, _) => const ProcessOutput(
          exitCode: 126,
          stdout: '',
          stderr: 'getprop: Permission denied\n',
        ),
      );
      expect(
        props.error,
        'getprop exited with 126: getprop: Permission denied',
      );
      expect(props['ro.product.model'], isNull);
    });

    test('an exit 0 without a single property line is an error too', () {
      final props = GetpropAndroidProperties(
        run: (_, _) => const ProcessOutput(exitCode: 0, stdout: '\n'),
      );
      expect(props.error, 'getprop printed no values');
    });

    test('getprop missing or denied: the script still exits 0 with every '
        'value empty; that is an error with its stderr', () {
      final stdout = [for (final key in kAndroidPropertyKeys) '[$key]: []']
          .join('\n');
      final props = GetpropAndroidProperties(
        run: (_, _) => ProcessOutput(
          exitCode: 0,
          stdout: '$stdout\n',
          stderr: 'sh: getprop: not found\n' * 8,
        ),
      );
      expect(props.error, startsWith('getprop printed no values: '));
      expect(props.error, contains('getprop: not found'));
      expect(props['ro.soc.model'], isNull);
    });
  });

  group('socFromProperties', () {
    test('ro.soc.model with its manufacturer, named from the table', () {
      final soc = socFromProperties(propsOf(kGalaxyS24Snapdragon))!;
      expect(soc.manufacturer, 'QTI');
      expect(soc.model, 'SM8650');
      expect(soc.source, 'ro.soc.model');
      expect(soc.label, 'Snapdragon 8 Gen 3 (SM8650)');
    });

    test('Exynos', () {
      final soc = socFromProperties(propsOf(kGalaxyS24Exynos))!;
      expect(soc.label, 'Exynos 2400 (s5e9945)');
    });

    test('Android 11: a known board stands for its part', () {
      final soc = socFromProperties(propsOf(kAndroid11Kona))!;
      expect(soc.model, 'SM8250');
      expect(soc.source, 'ro.board.platform kona');
      expect(soc.label, 'Snapdragon 865/865+/870 (SM8250)');
      expect(soc.inferred, isTrue);
    });

    test('an unknown part is kept as Android names it, without a name', () {
      final soc = socFromProperties(propsOf(kMediatekPhone))!;
      expect(soc.name, isNull);
      expect(soc.label, 'Mediatek MT6989');
    });

    test('an unknown board is reported raw; nothing at all is null', () {
      final soc = socFromProperties(propsOf('[ro.board.platform]: [taro]\n'))!;
      expect(soc.model, 'board taro');
      expect(soc.name, isNull);
      expect(socFromProperties(propsOf('')), isNull);
    });
  });
}
