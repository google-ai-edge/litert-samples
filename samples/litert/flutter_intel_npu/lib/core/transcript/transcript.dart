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

import 'package:flutter/services.dart' show rootBundle;

/// One line of a timestamped transcript: `[mm:ss] Speaker: text`.
class TranscriptSegment {
  const TranscriptSegment({
    required this.index,
    required this.timestamp,
    required this.speaker,
    required this.text,
  });

  final int index;
  final Duration timestamp;
  final String speaker;
  final String text;

  String get stamp => formatStamp(timestamp);

  static String formatStamp(Duration d) {
    final m = d.inMinutes.toString().padLeft(2, '0');
    final s = (d.inSeconds % 60).toString().padLeft(2, '0');
    return '$m:$s';
  }
}

/// A reference the model made to the source. [segment] is the line with
/// exactly that timestamp; null when no line has it — a stamp the model made
/// up, which the viewer shows greyed out instead of linking it to a nearby
/// line it never quoted.
class Citation {
  const Citation({required this.stamp, required this.segment});
  final String stamp;
  final TranscriptSegment? segment;
}

class Transcript {
  Transcript(this.segments, {required this.source});

  final List<TranscriptSegment> segments;
  final String source;

  static final _line = RegExp(r'^\[(\d{1,2}):(\d{2})\]\s*([^:]+):\s*(.*)$');
  static final _stampInText = RegExp(r'\[?\b(\d{1,2}):(\d{2})\b\]?');

  /// Parses `[mm:ss] Speaker: text` lines; other lines are appended to the
  /// previous segment so multi-line utterances survive.
  static Transcript parse(String raw, {required String source}) {
    final segs = <TranscriptSegment>[];
    for (final line in raw.split('\n')) {
      final m = _line.firstMatch(line.trimRight());
      if (m != null) {
        segs.add(TranscriptSegment(
          index: segs.length,
          timestamp: Duration(minutes: int.parse(m.group(1)!), seconds: int.parse(m.group(2)!)),
          speaker: m.group(3)!.trim(),
          text: m.group(4)!.trim(),
        ));
      } else if (segs.isNotEmpty && line.trim().isNotEmpty) {
        final last = segs.removeLast();
        segs.add(TranscriptSegment(
          index: last.index,
          timestamp: last.timestamp,
          speaker: last.speaker,
          text: '${last.text}\n${line.trim()}',
        ));
      }
    }
    return Transcript(segs, source: source);
  }

  static Future<Transcript> fromFile(String path) async =>
      parse(await File(path).readAsString(), source: path);

  static Future<Transcript> fromAsset(String asset) async =>
      parse(await rootBundle.loadString(asset), source: asset);

  /// The text as the model sees it (exactly the lines shown in the viewer, so
  /// a quoted line can be found by the reader).
  String get asPromptText =>
      segments.map((s) => '[${s.stamp}] ${s.speaker}: ${s.text}').join('\n');

  Duration get duration => segments.isEmpty ? Duration.zero : segments.last.timestamp;

  /// Segment at or immediately before [t].
  TranscriptSegment? segmentAt(Duration t) {
    TranscriptSegment? best;
    for (final s in segments) {
      if (s.timestamp <= t) {
        best = s;
      } else {
        break;
      }
    }
    return best;
  }

  late final Map<String, TranscriptSegment> _byStamp = {
    for (final s in segments.reversed) s.stamp: s, // the first line wins
  };

  /// Every `mm:ss` the model wrote, resolved against the transcript.
  List<Citation> citationsIn(String answer) {
    final seen = <String>{};
    final out = <Citation>[];
    for (final m in _stampInText.allMatches(answer)) {
      final min = int.parse(m.group(1)!), sec = int.parse(m.group(2)!);
      if (sec >= 60) continue;
      final stamp = '${min.toString().padLeft(2, '0')}:${sec.toString().padLeft(2, '0')}';
      if (!seen.add(stamp)) continue;
      out.add(Citation(stamp: stamp, segment: _byStamp[stamp]));
    }
    return out;
  }
}
