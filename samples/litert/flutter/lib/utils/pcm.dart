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

/// Floor of every dBFS figure here: digital silence reports this, not -∞.
const kPcmFloorDbfs = -96.0;

/// Loudness facts about a 16-bit little-endian mono PCM buffer.
final class const PcmStats({
  /// Whole samples (a trailing odd byte is ignored).
  required final int samples,

  /// Every sample is exactly 0: no audio reached the app (macOS TCC, a dead
  /// device), as opposed to a quiet room, which always has some noise.
  required final bool allZero,

  /// RMS of the loudest frame, dBFS ([kPcmFloorDbfs] for silence or no
  /// samples). The silence gate compares this.
  required final double peakFrameDbfs,
});

/// [PcmStats] of [pcm], with RMS taken over frames of [frameSamples] samples
/// (320 = 20 ms at 16 kHz). Reads through [ByteData] so any buffer offset
/// works; about 1 ms per second of audio.
PcmStats analyzePcm16(Uint8List pcm, {int frameSamples = 320}) {
  final view = ByteData.sublistView(pcm);
  final samples = pcm.length ~/ 2;
  var allZero = true;
  var peakMeanSquare = 0.0;
  var sum = 0.0;
  var inFrame = 0;
  for (var i = 0; i < samples; i++) {
    final s = view.getInt16(i * 2, Endian.little);
    if (s != 0) allZero = false;
    sum += s * s;
    if (++inFrame == frameSamples || i == samples - 1) {
      final meanSquare = sum / inFrame;
      if (meanSquare > peakMeanSquare) peakMeanSquare = meanSquare;
      sum = 0;
      inFrame = 0;
    }
  }
  return PcmStats(
    samples: samples,
    allZero: allZero,
    peakFrameDbfs: _dbfs(peakMeanSquare),
  );
}

/// RMS of the whole buffer, dBFS. For the input level meter (one value per
/// mic chunk).
double rmsDbfs(Uint8List pcm) {
  final samples = pcm.length ~/ 2;
  if (samples == 0) return kPcmFloorDbfs;
  final view = ByteData.sublistView(pcm);
  var sum = 0.0;
  for (var i = 0; i < samples; i++) {
    final s = view.getInt16(i * 2, Endian.little);
    sum += s * s;
  }
  return _dbfs(sum / samples);
}

/// Maps dBFS onto 0–1 for a meter: [floorDbfs] and below are 0, 0 dBFS is 1.
double levelFromDbfs(double dbfs, {double floorDbfs = -60}) =>
    ((dbfs - floorDbfs) / -floorDbfs).clamp(0.0, 1.0);

double _dbfs(double meanSquare) {
  if (meanSquare <= 0) return kPcmFloorDbfs;
  final db = 10 * math.log(meanSquare / (32768.0 * 32768.0)) / math.ln10;
  return math.max(db, kPcmFloorDbfs);
}

/// How much of a capture is louder than speech must be (the silence gate).
final class const VoiceActivity({
  /// RMS of the loudest frame, dBFS.
  required final double peakFrameDbfs,

  /// The capture's noise floor: the 10th-percentile frame RMS, dBFS.
  required final double floorDbfs,

  /// What a frame had to reach to count as voiced, dBFS.
  required final double thresholdDbfs,

  /// Frames at or above the threshold, and their total duration.
  required final int voicedFrames,
  required final Duration voiced,
});

/// Voiced audio in [pcm] (16-bit LE mono at [sampleRate]), frame by frame
/// ([frameSamples], 20 ms at 16 kHz). A frame is voiced when its RMS reaches
/// both [gateDbfs] and the noise floor + [aboveFloorDb]; the floor-relative
/// part is capped at [floorCapDbfs] so a capture that is speech from end to
/// end (no quiet frames to measure the floor on) still counts as speech. A
/// click is a frame or two; the constant noise of a noisy room sits at its
/// own floor.
VoiceActivity measureVoice(
  Uint8List pcm, {
  required double gateDbfs,
  double aboveFloorDb = 10,
  double floorCapDbfs = -30,
  int frameSamples = 320,
  int sampleRate = 16000,
}) {
  final view = ByteData.sublistView(pcm);
  final samples = pcm.length ~/ 2;
  final frames = <double>[];
  for (var start = 0; start < samples; start += frameSamples) {
    final end = math.min(start + frameSamples, samples);
    var sum = 0.0;
    for (var i = start; i < end; i++) {
      final s = view.getInt16(i * 2, Endian.little);
      sum += s * s;
    }
    frames.add(_dbfs(sum / (end - start)));
  }
  if (frames.isEmpty) {
    return VoiceActivity(
      peakFrameDbfs: kPcmFloorDbfs,
      floorDbfs: kPcmFloorDbfs,
      thresholdDbfs: gateDbfs,
      voicedFrames: 0,
      voiced: Duration.zero,
    );
  }
  final sorted = [...frames]..sort();
  final floor = sorted[(sorted.length - 1) ~/ 10];
  final threshold = math.max(
    gateDbfs,
    math.min(floor + aboveFloorDb, floorCapDbfs),
  );
  final voicedFrames = frames.where((db) => db >= threshold).length;
  return VoiceActivity(
    peakFrameDbfs: sorted.last,
    floorDbfs: floor,
    thresholdDbfs: threshold,
    voicedFrames: voicedFrames,
    voiced: Duration(
      microseconds: voicedFrames * frameSamples * 1000000 ~/ sampleRate,
    ),
  );
}

/// [duration] of a [hz] sine as 16-bit little-endian mono PCM at
/// [sampleRate], peak [amplitude] of full scale (0.5: RMS −9.0 dBFS), with
/// 5 ms fades so it starts and ends without a click.
Uint8List sineTonePcm16({
  required int hz,
  required Duration duration,
  required int sampleRate,
  double amplitude = 0.5,
}) {
  final samples = duration.inMicroseconds * sampleRate ~/ 1000000;
  final fade = sampleRate ~/ 200;
  final out = ByteData(samples * 2);
  for (var i = 0; i < samples; i++) {
    final edge = math.min(i, samples - 1 - i);
    final gain = edge < fade ? edge / fade : 1.0;
    final v = amplitude * gain * math.sin(2 * math.pi * hz * i / sampleRate);
    out.setInt16(i * 2, (v * 32767).round(), Endian.little);
  }
  return out.buffer.asUint8List();
}

/// Duration of [bytes] of 16-bit mono PCM at [sampleRate].
Duration pcm16Duration(int bytes, int sampleRate) =>
    Duration(microseconds: (bytes ~/ 2) * 1000000 ~/ sampleRate);
