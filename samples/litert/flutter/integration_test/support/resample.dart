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

// A resampler for the regenerated question clips
// (tools/make_question_audio_test.dart): the app's speech synthesizer speaks
// at 24 kHz, the recognizers hear 16 kHz. Test tooling only: the app itself
// never resamples (it records at 16 kHz and plays at the synthesizer's rate).

import 'dart:math' as math;
import 'dart:typed_data';

/// Stopband attenuation of the anti-aliasing filter, dB.
const kResampleStopbandDb = 80.0;

/// Where the passband ends, as a fraction of the lower of the two Nyquist
/// frequencies; the stopband starts at that Nyquist. 24 kHz to 16 kHz: flat
/// to 7 kHz, at least [kResampleStopbandDb] down from 8 kHz, so nothing above
/// the new Nyquist folds back into the speech band.
const kResamplePassbandFraction = 0.875;

/// [input], samples at [fromRate] Hz in any scale, at [toRate] Hz.
///
/// With L/M = [toRate]/[fromRate] in lowest terms (2/3 for 24 kHz to
/// 16 kHz): upsampled by L (zeros between the samples), low-passed by a
/// Kaiser-windowed sinc at the upsampled rate, decimated by M. Computed in
/// polyphase form: each output sample sums only the taps that meet an input
/// sample. The filter is centred (zero phase), so output sample k is the
/// signal at time k/[toRate] and a clip keeps its timing; there are
/// ceil(n * L / M) of them. Outside [input] the signal is zero. Equal rates
/// return a copy.
Float64List resample(
  Float64List input, {
  required int fromRate,
  required int toRate,
}) {
  if (fromRate <= 0) {
    throw ArgumentError.value(fromRate, 'fromRate', 'must be positive');
  }
  if (toRate <= 0) {
    throw ArgumentError.value(toRate, 'toRate', 'must be positive');
  }
  if (fromRate == toRate) return Float64List.fromList(input);
  final divisor = _gcd(fromRate, toRate);
  final up = toRate ~/ divisor;
  final down = fromRate ~/ divisor;
  final taps = _lowPass(fromRate: fromRate, toRate: toRate, up: up);
  final delay = (taps.length - 1) ~/ 2;
  final n = input.length;
  final out = Float64List((n * up + down - 1) ~/ down);
  for (var k = 0; k < out.length; k++) {
    // The upsampled stream at index k*down, filtered by the centred taps:
    // sum over j of taps[j] * upsampled[centre - j], where only indices that
    // are multiples of [up] hold an input sample.
    final centre = k * down + delay;
    var sum = 0.0;
    for (var j = centre % up; j < taps.length; j += up) {
      final m = centre - j;
      if (m < 0) break;
      final i = m ~/ up;
      if (i < n) sum += taps[j] * input[i];
    }
    out[k] = sum;
  }
  return out;
}

/// [resample] on 16-bit little-endian mono PCM. The result is rounded and
/// clamped to the 16-bit range (a full-scale input overshoots a little at
/// its edges, and a wrapped sample would be a loud click). Throws an
/// [ArgumentError] for an odd byte count.
Uint8List resamplePcm16(
  Uint8List pcm, {
  required int fromRate,
  required int toRate,
}) {
  if (pcm.length.isOdd) {
    throw ArgumentError.value(
      pcm.length,
      'pcm',
      '16-bit PCM has an even number of bytes',
    );
  }
  final view = ByteData.sublistView(pcm);
  final samples = Float64List(pcm.length ~/ 2);
  for (var i = 0; i < samples.length; i++) {
    samples[i] = view.getInt16(i * 2, Endian.little).toDouble();
  }
  final resampled = resample(samples, fromRate: fromRate, toRate: toRate);
  final out = ByteData(resampled.length * 2);
  for (var i = 0; i < resampled.length; i++) {
    final s = math.max(-32768, math.min(32767, resampled[i].round()));
    out.setInt16(i * 2, s, Endian.little);
  }
  return out.buffer.asUint8List();
}

/// The Kaiser-windowed sinc low-pass at the upsampled rate (fromRate * up):
/// cutoff halfway through the transition band, length and window shape from
/// Kaiser's formulas for [kResampleStopbandDb]. Odd length (a whole-sample
/// centre); scaled to a total gain of [up], so every polyphase branch passes
/// DC at unity (zero stuffing keeps one sample in [up]).
Float64List _lowPass({
  required int fromRate,
  required int toRate,
  required int up,
}) {
  final upRate = (fromRate * up).toDouble();
  final stop = math.min(fromRate, toRate) / 2;
  final pass = stop * kResamplePassbandFraction;
  // Cycles per upsampled sample.
  final cutoff = (pass + stop) / 2 / upRate;
  // Radians per upsampled sample.
  final transition = 2 * math.pi * (stop - pass) / upRate;
  const attenuation = kResampleStopbandDb;
  const beta = 0.1102 * (attenuation - 8.7);
  var length = ((attenuation - 7.95) / (2.285 * transition)).ceil() + 1;
  if (length.isEven) length++;
  final mid = (length - 1) / 2;
  final windowNorm = _besselI0(beta);
  final taps = Float64List(length);
  var total = 0.0;
  for (var j = 0; j < length; j++) {
    final t = j - mid;
    final x = 2 * cutoff * t;
    final sinc = t == 0 ? 1.0 : math.sin(math.pi * x) / (math.pi * x);
    final r = t / mid;
    final window =
        _besselI0(beta * math.sqrt(math.max(0.0, 1 - r * r))) / windowNorm;
    final tap = 2 * cutoff * sinc * window;
    taps[j] = tap;
    total += tap;
  }
  final gain = up / total;
  for (var j = 0; j < length; j++) {
    taps[j] *= gain;
  }
  return taps;
}

/// The zeroth-order modified Bessel function of the first kind, by its
/// power series (converges fast for the window's arguments, x <= ~8).
double _besselI0(double x) {
  final half = x / 2;
  var term = 1.0;
  var sum = 1.0;
  for (var k = 1; k < 100; k++) {
    term *= half / k;
    final squared = term * term;
    sum += squared;
    if (squared < sum * 1e-16) break;
  }
  return sum;
}

int _gcd(int a, int b) {
  var x = a;
  var y = b;
  while (y != 0) {
    final t = x % y;
    x = y;
    y = t;
  }
  return x;
}
