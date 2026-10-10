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

import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/frames/engine_image_decoder.dart';
import 'package:litert_edge_demos/data/services/frames/jpeg_decoder.dart';
import 'package:litert_edge_demos/data/services/frames/turbojpeg.dart';

import '../../../support/mjpeg.dart';

/// Mean and largest absolute difference per channel byte (alpha included).
({double mean, int max}) diff(Uint8List a, Uint8List b) {
  expect(a.length, b.length);
  var sum = 0;
  var max = 0;
  for (var i = 0; i < a.length; i++) {
    final d = (a[i] - b[i]).abs();
    sum += d;
    if (d > max) max = d;
  }
  return (mean: sum / a.length, max: max);
}

/// TurboJPEG on this machine (Homebrew's on macOS, the system's on Linux);
/// the CPU-decoder tests are skipped, with this reason, without it.
final String? _noTurboJpeg = () {
  final failures = <String>[];
  final lib = TurboJpegLibrary.open(turboJpegCandidates(), failures: failures);
  return lib == null
      ? 'libturbojpeg not found (${failures.join('; ')}): '
            'brew install jpeg-turbo / sudo apt install libturbojpeg'
      : null;
}();

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('turboJpegScale: down while the long side stays ≥ 960', () {
    for (final (w, h, num, denom) in const [
      (640, 480, 1, 1),
      (1280, 720, 1, 1),
      (1600, 1200, 1, 1),
      (1920, 1080, 1, 2),
      (2000, 1500, 1, 2),
      (2560, 1440, 1, 2),
      (3840, 2160, 1, 4),
      (4000, 3000, 1, 4),
      (7680, 4320, 1, 8),
      (1080, 1920, 1, 2), // portrait: the long side counts
    ]) {
      test('${w}x$h → $num/$denom', () {
        expect(turboJpegScale(w, h, minLongSide: 960), (
          num: num,
          denom: denom,
        ));
      });
    }
  });

  test('candidates: the bundle copy first on Linux, Homebrew on macOS, none '
      'on the phones', () {
    expect(
      turboJpegCandidates(
        operatingSystem: 'linux',
        executable: '/opt/app/litert_edge_demos',
      ),
      ['/opt/app/lib/libturbojpeg.so.0', 'libturbojpeg.so.0'],
    );
    expect(
      turboJpegCandidates(operatingSystem: 'macos', executable: '/x/app'),
      everyElement(endsWith('libturbojpeg.0.dylib')),
    );
    expect(
      turboJpegCandidates(operatingSystem: 'android', executable: '/x/app'),
      isEmpty,
    );
    expect(
      turboJpegCandidates(operatingSystem: 'ios', executable: '/x/app'),
      isEmpty,
    );
  });

  group('no libturbojpeg: the engine codec, and on Linux it says so', () {
    test(
      'Linux: slow, with the reason (the library path that failed)',
      () async {
        final decoder = await openJpegDecoder(
          platform: TargetPlatform.linux,
          candidates: const ['/nonexistent/libturbojpeg.so.0'],
        );
        expect(decoder, isA<EngineJpegDecoder>());
        expect(decoder.slow, isTrue);
        expect(
          decoder.fallbackReason,
          contains('/nonexistent/libturbojpeg.so.0'),
        );
        await decoder.close();
      },
    );

    test(
      'macOS, Android, iOS: the engine codec is the designed decoder',
      () async {
        for (final platform in const [
          TargetPlatform.macOS,
          TargetPlatform.android,
          TargetPlatform.iOS,
        ]) {
          final decoder = await openJpegDecoder(
            platform: platform,
            candidates: const [],
          );
          expect(decoder, isA<EngineJpegDecoder>(), reason: platform.name);
          expect(decoder.slow, isFalse, reason: platform.name);
        }
      },
    );

    test('a library without the TurboJPEG functions is not used', () async {
      // libc opens, but has no tjInitDecompress.
      final libc = Platform.isMacOS
          ? '/usr/lib/libSystem.B.dylib'
          : 'libc.so.6';
      final failures = <String>[];
      expect(TurboJpegLibrary.open([libc], failures: failures), isNull);
      expect(failures.single, contains(libc));
    });
  });

  group('TurboJPEG', () {
    testWidgets('equals the engine decode of the cats JPEG within JPEG '
        'decoder noise', (tester) async {
      await tester.runAsync(() async {
        final cats = catsJpeg();
        final lib = TurboJpegLibrary.open(turboJpegCandidates())!;
        final session = lib.session();
        addTearDown(session.close);
        final image = session.decode(cats, minLongSide: 960);
        expect((image.width, image.height), (640, 480));
        expect((image.scaleNum, image.scaleDenom), (1, 1));

        final reference = await decodeEncodedImage(cats, maxSide: 1280);
        reference.image.dispose();
        final d = diff(image.rgba, reference.rgba);
        debugPrint(
          'TURBOJPEG cats 640x480 vs engine: mean=${d.mean} max=${d.max}',
        );
        // Two conforming decoders differ in IDCT and chroma upsampling only.
        expect(d.mean, lessThan(1.5));
        expect(d.max, lessThan(48));
        // Opaque alpha, like the engine's.
        for (var i = 3; i < image.rgba.length; i += 4 * 997) {
          expect(image.rgba[i], 255);
        }
      });
    }, skip: _noTurboJpeg != null);

    testWidgets('scales a 2000×1500 JPEG to 1/2 at decode, close to the '
        "engine's downscale", (tester) async {
      await tester.runAsync(() async {
        final big = File('test_assets/images/big_2000x1500.jpg')
            .readAsBytesSync();
        final session = TurboJpegLibrary.open(turboJpegCandidates())!.session();
        addTearDown(session.close);
        final image = session.decode(big, minLongSide: 960);
        expect((image.width, image.height), (1000, 750));
        expect((image.sourceWidth, image.sourceHeight), (2000, 1500));
        expect((image.scaleNum, image.scaleDenom), (1, 2));
        final reference = await decodeEncodedImage(big, maxSide: 1000);
        reference.image.dispose();
        final d = diff(image.rgba, reference.rgba);
        debugPrint('TURBOJPEG big 1/2 vs engine downscale: mean=${d.mean}');
        expect(d.mean, lessThan(6), reason: 'two resamplers, same picture');
      });
    }, skip: _noTurboJpeg != null);

    test('a header warning is not fatal: a JFIF revision TurboJPEG does not '
        'know still decodes', () {
      final quirky = Uint8List.fromList(catsJpeg());
      final app0 = _indexOf(quirky, const [0xFF, 0xE0]);
      expect(String.fromCharCodes(quirky.sublist(app0 + 4, app0 + 8)), 'JFIF');
      quirky[app0 + 9] = 3; // JFIF major version 1 → 3: a header warning
      final session = TurboJpegLibrary.open(turboJpegCandidates())!.session();
      addTearDown(session.close);
      final image = session.decode(quirky, minLongSide: 960);
      expect((image.width, image.height), (640, 480));
    }, skip: _noTurboJpeg != null);

    test('a header claiming 40000×40000 is refused, not allocated', () {
      final huge = Uint8List.fromList(catsJpeg());
      final sof = _indexOf(huge, const [0xFF, 0xC0]);
      // 40000×40000: even at 1/8 it is 5000×5000, over the cap.
      huge.setRange(sof + 5, sof + 9, const [0x9C, 0x40, 0x9C, 0x40]);
      final session = TurboJpegLibrary.open(turboJpegCandidates())!.session();
      addTearDown(session.close);
      expect(
        () => session.decode(huge, minLongSide: 960),
        throwsA(
          isA<TurboJpegException>().having(
            (e) => e.message,
            'message',
            contains('over the 4096×4096 limit'),
          ),
        ),
      );
      // The session still works after the refusal.
      expect(session.decode(catsJpeg(), minLongSide: 960).width, 640);
    }, skip: _noTurboJpeg != null);

    test('not a JPEG: a TurboJpegException with the library\'s reason', () {
      final session = TurboJpegLibrary.open(turboJpegCandidates())!.session();
      addTearDown(session.close);
      expect(
        () => session.decode(
          Uint8List.fromList(List.filled(64, 7)),
          minLongSide: 960,
        ),
        throwsA(
          isA<TurboJpegException>().having(
            (e) => e.message,
            'message',
            startsWith('Not a JPEG'),
          ),
        ),
      );
      expect(
        () => session.decode(Uint8List(0), minLongSide: 960),
        throwsA(isA<TurboJpegException>()),
      );
    }, skip: _noTurboJpeg != null);

    test('the worker-isolate decoder: frames back as RGBA, errors per frame, '
        'nothing after close; per-frame times', () async {
      final cats = catsJpeg();
      final big = File('test_assets/images/big_2000x1500.jpg')
          .readAsBytesSync();
      final decoder = await openJpegDecoder(platform: TargetPlatform.linux);
      expect(decoder, isA<TurboJpegDecoder>());
      expect(decoder.slow, isFalse);
      expect(decoder.description, startsWith('TurboJPEG (worker isolate'));

      Future<double> averageMs(Uint8List jpeg, int n) async {
        var micros = 0;
        for (var i = 0; i < n; i++) {
          micros += (await decoder.decode(
            jpeg,
            maxSide: 1280,
          )).decodeTime.inMicroseconds;
        }
        return micros / n / 1000;
      }

      final frame = await decoder.decode(cats, maxSide: 1280);
      expect((frame.width, frame.height), (640, 480));
      expect(frame.rgba.length, 640 * 480 * 4);
      expect(frame.image, isNull, reason: 'the preview is made separately');
      final scaled = await decoder.decode(big, maxSide: 1280);
      expect((scaled.width, scaled.height), (1000, 750));
      debugPrint(
        'TURBOJPEG_BENCH cats 640x480 ${(await averageMs(cats, 20)).toStringAsFixed(2)} ms · '
        'big 2000x1500→1000x750 ${(await averageMs(big, 20)).toStringAsFixed(2)} ms '
        '(per frame, decode only, worker isolate)',
      );

      await expectLater(
        decoder.decode(Uint8List.fromList([1, 2, 3]), maxSide: 1280),
        throwsA(isA<TurboJpegException>()),
      );
      // One bad frame does not end the decoder.
      expect((await decoder.decode(cats, maxSide: 1280)).width, 640);

      await decoder.close();
      await expectLater(
        decoder.decode(cats, maxSide: 1280),
        throwsA(isA<TurboJpegException>()),
      );
    }, skip: _noTurboJpeg != null);
  });
}

/// The first index of [pattern] in [bytes]; fails the test when absent.
int _indexOf(Uint8List bytes, List<int> pattern) {
  outer:
  for (var i = 0; i + pattern.length <= bytes.length; i++) {
    for (var j = 0; j < pattern.length; j++) {
      if (bytes[i + j] != pattern[j]) continue outer;
    }
    return i;
  }
  fail('pattern $pattern not found');
}
