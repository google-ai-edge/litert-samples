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
import 'dart:typed_data';

/// A meeting transcript read from a plain text file. The text is what the
/// model reads; nothing in it is known to the app in advance.
class MeetingInput {
  const MeetingInput({required this.title, required this.text, required this.source});

  /// The first line of the transcript, or the file name.
  final String title;

  final String text;

  /// The file it came from.
  final String source;

  /// The extensions [read] understands.
  static const extensions = ['txt'];

  static Future<MeetingInput> read(String path) async {
    final bytes = await File(path).readAsBytes();
    return parse(path, bytes);
  }

  /// [path] only names the meeting and must end in `.txt`. Every error is a
  /// [FormatException] that starts with the file's name.
  static MeetingInput parse(String path, Uint8List bytes) {
    final name = path.split(RegExp(r'[\\/]')).last;
    if (!name.toLowerCase().endsWith('.txt')) throw FormatException('$name: not a transcript (.txt)');
    final text = _utf8(name, bytes).replaceAll('\r\n', '\n').replaceAll('\r', '\n').trim();
    if (text.isEmpty) throw FormatException('$name: no text in it');
    final first = text
        .split('\n')
        .first
        // "MEETING TRANSCRIPT — "Weekly sync"": the kind of file is not the name.
        .replaceFirst(RegExp(r'^(meeting )?(transcript|minutes|notes)\s*[:—–-]\s*', caseSensitive: false), '')
        .replaceAll(RegExp(r'^["“”\x27]+|["“”\x27]+$'), '')
        .trim();
    final title = first.isNotEmpty && first.length <= 120 && !_cue.hasMatch(first) ? first : name;
    return MeetingInput(title: title, text: text, source: path);
  }

  /// A line that is an utterance (`[mm:ss] …`), not a title.
  static final _cue = RegExp(r'^\[\d+:\d\d\]');

  /// UTF-8, a byte order mark dropped. Anything else is refused: decoding it
  /// anyway would hand the model replacement characters instead of names.
  static String _utf8(String name, Uint8List b) {
    final body = b.length >= 3 && b[0] == 0xEF && b[1] == 0xBB && b[2] == 0xBF ? b.sublist(3) : b;
    try {
      return utf8.decode(body);
    } on FormatException catch (e) {
      throw FormatException('$name: not UTF-8 text (bad byte at ${e.offset}); save it as UTF-8');
    }
  }
}
