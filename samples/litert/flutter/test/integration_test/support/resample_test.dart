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

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/utils/pcm.dart';

import '../../../integration_test/support/resample.dart';

void main() {
  group('resample', () {
    test('24 kHz to 16 kHz keeps a 1 kHz sine: length, frequency, '
        'amplitude and phase', () {
      final input = _sine(hz: 1000, rate: 24000, samples: 24000);
      final out = resample(input, fromRate: 24000, toRate: 16000);

      expect(out.length, 16000, reason: '1 s stays 1 s');
      expect(_frequency(out, 16000), closeTo(1000, 0.5));
      // Zero phase and unity gain: sample k is the sine at k/16000 s. The
      // edges (the filter half-length, ~40 output samples) are skipped.
      final ideal = _sine(hz: 1000, rate: 16000, samples: 16000);
      expect(_maxError(out, ideal, skip: 200), lessThan(5e-4));
    });

    test('16 kHz to 24 kHz: the other direction', () {
      final input = _sine(hz: 1000, rate: 16000, samples: 16000);
      final out = resample(input, fromRate: 16000, toRate: 24000);

      expect(out.length, 24000);
      expect(_frequency(out, 24000), closeTo(1000, 0.5));
      final ideal = _sine(hz: 1000, rate: 24000, samples: 24000);
      expect(_maxError(out, ideal, skip: 300), lessThan(5e-4));
    });

    test('nothing above the new Nyquist folds back into the speech band', () {
      // 10 kHz at 24 kHz would alias to 6 kHz at 16 kHz.
      final input = _sine(hz: 10000, rate: 24000, samples: 24000);
      final out = resample(input, fromRate: 24000, toRate: 16000);

      final ratio = _rms(out, skip: 200) / _rms(input, skip: 300);
      expect(ratio, lessThan(1e-3), reason: 'at least 60 dB down');
    });

    test('length is ceil(n * 2 / 3); equal rates return a copy', () {
      for (final n in [0, 1, 2, 3, 4, 7, 24001]) {
        final out = resample(Float64List(n), fromRate: 24000, toRate: 16000);
        expect(out.length, (n * 2 + 2) ~/ 3, reason: 'n=$n');
      }
      final input = Float64List.fromList([1, -2, 3]);
      final same = resample(input, fromRate: 16000, toRate: 16000);
      expect(same, input);
      expect(identical(same, input), isFalse);
    });

    test('rates must be positive', () {
      expect(
        () => resample(Float64List(4), fromRate: 0, toRate: 16000),
        throwsArgumentError,
      );
      expect(
        () => resample(Float64List(4), fromRate: 24000, toRate: -1),
        throwsArgumentError,
      );
    });
  });

  group('resamplePcm16', () {
    test('1 s of a 1 kHz tone at 24 kHz: 32000 bytes at 16 kHz, 1 kHz', () {
      final pcm = sineTonePcm16(
        hz: 1000,
        duration: const Duration(seconds: 1),
        sampleRate: 24000,
      );
      final out = resamplePcm16(pcm, fromRate: 24000, toRate: 16000);

      expect(out.length, 32000);
      expect(_frequency(_samples(out), 16000), closeTo(1000, 0.5));
    });

    test('a full-scale input clamps at the edges instead of wrapping', () {
      // DC at +32767 overshoots at both ends (Gibbs); a wrapped sample
      // would turn negative.
      final pcm = ByteData(2400 * 2);
      for (var i = 0; i < 2400; i++) {
        pcm.setInt16(i * 2, 32767, Endian.little);
      }
      final out = _samples(
        resamplePcm16(pcm.buffer.asUint8List(), fromRate: 24000, toRate: 16000),
      );

      expect(out.length, 1600);
      expect(out.every((s) => s >= 0), isTrue, reason: 'no sample wrapped');
      for (var i = 200; i < out.length - 200; i++) {
        expect(out[i], greaterThan(32700), reason: 'sample $i');
      }
    });

    test('an odd byte count is refused', () {
      expect(
        () => resamplePcm16(Uint8List(3), fromRate: 24000, toRate: 16000),
        throwsArgumentError,
      );
    });
  });
}

Float64List _sine({
  required int hz,
  required int rate,
  required int samples,
  double amplitude = 0.5,
}) => Float64List.fromList([
  for (var i = 0; i < samples; i++)
    amplitude * math.sin(2 * math.pi * hz * i / rate),
]);

Float64List _samples(Uint8List pcm) {
  final view = ByteData.sublistView(pcm);
  return Float64List.fromList([
    for (var i = 0; i < pcm.length ~/ 2; i++)
      view.getInt16(i * 2, Endian.little).toDouble(),
  ]);
}

/// The frequency from the rising zero crossings between [skip] samples at
/// each end, each crossing placed by linear interpolation.
double _frequency(Float64List s, int rate, {int skip = 200}) {
  var first = 0.0;
  var last = 0.0;
  var crossings = 0;
  for (var i = skip + 1; i < s.length - skip; i++) {
    if (s[i - 1] < 0 && s[i] >= 0) {
      final t = (i - 1 + s[i - 1] / (s[i - 1] - s[i])) / rate;
      if (crossings == 0) first = t;
      last = t;
      crossings++;
    }
  }
  expect(crossings, greaterThan(1), reason: 'no periodic signal');
  return (crossings - 1) / (last - first);
}

double _maxError(Float64List a, Float64List b, {required int skip}) {
  var worst = 0.0;
  for (var i = skip; i < a.length - skip; i++) {
    worst = math.max(worst, (a[i] - b[i]).abs());
  }
  return worst;
}

double _rms(Float64List s, {required int skip}) {
  var sum = 0.0;
  for (var i = skip; i < s.length - skip; i++) {
    sum += s[i] * s[i];
  }
  return math.sqrt(sum / (s.length - 2 * skip));
}
