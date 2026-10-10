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

import 'package:litert_edge_demos/domain/models/voice.dart';

/// 16 kHz mono PCM16 test signals.
const kTestRate = 16000;

int _samples(Duration d) => d.inMicroseconds * kTestRate ~/ 1000000;

/// A sine at [amplitude] of full scale (0.3 ≈ −13 dBFS RMS): "speech".
Uint8List tone(
  Duration duration, {
  double amplitude = 0.3,
  double hz = 440,
  int rate = kTestRate,
}) {
  final n = duration.inMicroseconds * rate ~/ 1000000;
  final data = ByteData(n * 2);
  for (var i = 0; i < n; i++) {
    final v = (amplitude * 32767 * math.sin(2 * math.pi * hz * i / rate))
        .round();
    data.setInt16(i * 2, v, Endian.little);
  }
  return data.buffer.asUint8List();
}

/// Room noise of ±[peak] LSB (±3 ≈ −83 dBFS): quiet, but not digital zero.
Uint8List quiet(Duration duration, {int peak = 3}) {
  final n = _samples(duration);
  final data = ByteData(n * 2);
  final random = math.Random(7);
  for (var i = 0; i < n; i++) {
    data.setInt16(i * 2, random.nextInt(2 * peak + 1) - peak, Endian.little);
  }
  return data.buffer.asUint8List();
}

/// One loud 20 ms burst (a click, a bump) in [duration] of room noise.
Uint8List click(Duration duration) {
  final pcm = quiet(duration);
  final burst = tone(const Duration(milliseconds: 20), amplitude: 0.5);
  final at = (pcm.length ~/ 2) & ~1;
  pcm.setRange(at, at + burst.length, burst);
  return pcm;
}

/// Constant noise of ±[peak] LSB: ±700 ≈ −38 dBFS RMS, a noisy room.
Uint8List noise(Duration duration, {int peak = 700}) =>
    quiet(duration, peak: peak);

/// Digital silence: what macOS hands an app the mic is blocked for.
Uint8List zeros(Duration duration) => Uint8List(_samples(duration) * 2);

/// A one-second press with a tone: passes the gate.
Utterance speechUtterance({Duration length = const Duration(seconds: 1)}) =>
    Utterance(pcm: tone(length), held: length);
