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
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/frames/mjpeg_parser.dart';

import '../../../support/mjpeg.dart';

/// Feeds [stream] to a fresh parser in chunks of [sizes] (cycled).
List<Uint8List> feed(
  String boundary,
  List<int> stream, {
  List<int> sizes = const [1 << 20],
  MjpegParser? parser,
}) {
  final p = parser ?? MjpegParser(boundary);
  final out = <Uint8List>[];
  var at = 0;
  var i = 0;
  while (at < stream.length) {
    final end = math.min(stream.length, at + sizes[i++ % sizes.length]);
    out.addAll(p.add(stream.sublist(at, end)));
    at = end;
  }
  return out;
}

void main() {
  final a = syntheticJpeg(1);
  final b = syntheticJpeg(2);
  final c = syntheticJpeg(3);

  group('parseContentType', () {
    test('media type lower-cased, parameters unquoted', () {
      final ct = parseContentType(
        'Multipart/X-Mixed-Replace; Boundary="--myboundary" ; q=1',
      );
      expect(ct.mediaType, 'multipart/x-mixed-replace');
      expect(ct.parameters, {'boundary': '--myboundary', 'q': '1'});
    });

    test('IP Webcam and ffmpeg styles', () {
      expect(
        parseContentType(
          'multipart/x-mixed-replace;boundary=Ba4oTvQMY8ew04N8dcnM',
        ).parameters['boundary'],
        'Ba4oTvQMY8ew04N8dcnM',
      );
      expect(
        parseContentType('multipart/x-mixed-replace;boundary=ffmpeg')
            .parameters['boundary'],
        'ffmpeg',
      );
    });
  });

  group('MjpegParser', () {
    test('parts with Content-Length come out byte for byte, in order', () {
      final stream = [
        ...mjpegPart('frame', a),
        ...mjpegPart('frame', b),
        ...mjpegPart('frame', c),
      ];
      final images = feed('frame', stream);
      expect(images, [a, b, c]);
    });

    test('the boundary parameter may carry the leading dashes itself', () {
      // `boundary=--myboundary`, body lines `--myboundary` (non-conforming
      // but common), and the conforming `----myboundary`.
      final loose = [
        ...latin1.encode('--myboundary\r\nContent-Type: image/jpeg\r\n'),
        ...latin1.encode('Content-Length: ${a.length}\r\n\r\n'),
        ...a,
        ...latin1.encode('\r\n'),
      ];
      expect(feed('--myboundary', loose), [a]);
      expect(feed('--myboundary', mjpegPart('--myboundary', a)), [a]);
      expect(feed('myboundary', mjpegPart('myboundary', a)), [a]);
    });

    test('chunks split anywhere (byte by byte, odd sizes) give the same '
        'images', () {
      final stream = [
        ...mjpegPart('frame', a),
        ...mjpegPart('frame', b, contentLength: false),
        ...mjpegPart('frame', c),
      ];
      expect(feed('frame', stream, sizes: const [1]), [a, b, c]);
      expect(feed('frame', stream, sizes: const [7, 3, 50, 1, 2]), [a, b, c]);
      expect(feed('frame', stream, sizes: const [64]), [a, b, c]);
    });

    test('without Content-Length the JPEG ends at its EOI, not at the '
        "EXIF thumbnail's", () {
      final stream = [
        ...mjpegPart('frame', a, contentLength: false),
        ...mjpegPart('frame', b, contentLength: false),
      ];
      final images = feed('frame', stream);
      expect(images, [a, b]);
      // The image is complete as soon as its EOI arrives: no waiting for
      // the next boundary (one frame less latency).
      final parser = MjpegParser('frame');
      final first = mjpegPart('frame', a, contentLength: false);
      final upToEoi = first.length - 2; // without the trailing CRLF
      expect(parser.add(first.sublist(0, upToEoi)), [a]);
    });

    test('the real cats JPEG without Content-Length is found intact', () {
      final cats = catsJpeg();
      final stream = [
        ...mjpegPart('frame', cats, contentLength: false),
        ...mjpegPart('frame', cats, contentLength: false),
      ];
      final images = feed('frame', stream, sizes: const [4096]);
      expect(images.length, 2);
      expect(images.first, cats);
    });

    test('a garbage prefix before the first boundary is skipped', () {
      final garbage = [
        ...latin1.encode('HTTP junk\r\nX-Stuff: 1\r\n\r\n'),
        ...List.generate(3000, (i) => (i * 37) & 0xFF),
      ];
      final parser = MjpegParser('frame');
      final images = feed(
        'frame',
        [...garbage, ...mjpegPart('frame', a), ...mjpegPart('frame', b)],
        sizes: const [100],
        parser: parser,
      );
      expect(images, [a, b]);
      expect(parser.garbageBytes, greaterThan(2000));
    });

    test('ffmpeg mpjpeg style: lower-case header names, LF-only lines', () {
      final stream = [
        ...mjpegPart(
          'ffmpeg',
          a,
          lengthHeader: 'Content-length',
          typeHeader: 'Content-type',
        ),
        ...mjpegPart('ffmpeg', b, eol: '\n'),
      ];
      expect(feed('ffmpeg', stream), [a, b]);
    });

    test('a part that is not image/jpeg is skipped and counted', () {
      final parser = MjpegParser('frame');
      final images = feed('frame', [
        ...mjpegPart('frame', utf8.encode('{"x":1}'), contentType: 'text/json'),
        ...mjpegPart('frame', a),
      ], parser: parser);
      expect(images, [a]);
      expect(parser.skippedParts, 1);
      expect(parser.images, 1);
    });

    test('a wrong Content-Length resynchronises at the next boundary', () {
      final short = [
        ...latin1.encode('--frame\r\nContent-Type: image/jpeg\r\n'),
        ...latin1.encode('Content-Length: 10\r\n\r\n'),
        ...a,
        ...latin1.encode('\r\n'),
      ];
      final images = feed('frame', [...short, ...mjpegPart('frame', b)]);
      expect(images.length, 2);
      expect(images.first.length, 10, reason: 'the truncated part');
      expect(images.last, b);
    });

    test('the closing boundary is not a part', () {
      final stream = [
        ...mjpegPart('frame', a),
        ...latin1.encode('--frame--\r\n'),
      ];
      expect(feed('frame', stream), [a]);
    });

    test('no boundary in 1 MB: not an MJPEG stream', () {
      final html = utf8.encode('<html>${'x' * (1100 * 1024)}</html>');
      expect(
        () => feed('frame', html, sizes: const [64 * 1024]),
        throwsA(
          isA<MjpegStreamException>().having(
            (e) => e.message,
            'message',
            contains('not an MJPEG stream'),
          ),
        ),
      );
    });

    test('an oversized Content-Length fails fast', () {
      final head = latin1.encode(
        '--frame\r\nContent-Type: image/jpeg\r\n'
        'Content-Length: ${kMjpegMaxPartBytes + 1}\r\n\r\n',
      );
      expect(() => feed('frame', head), throwsA(isA<MjpegStreamException>()));
    });

    test('a part without Content-Length that is not a JPEG is skipped', () {
      final parser = MjpegParser('frame');
      final images = feed('frame', [
        ...mjpegPart('frame', utf8.encode('hello'), contentLength: false),
        ...mjpegPart('frame', a),
      ], parser: parser);
      expect(images, [a]);
      expect(parser.skippedParts, 1);
    });

    test('detect: the boundary comes from the first line (ffmpeg -listen '
        'sends application/octet-stream)', () {
      final parser = MjpegParser.detect();
      final images = feed(
        '',
        [
          ...mjpegPart('ffmpeg', a, lengthHeader: 'Content-length'),
          ...mjpegPart('ffmpeg', b, lengthHeader: 'Content-length'),
        ],
        sizes: const [5],
        parser: parser,
      );
      expect(images, [a, b]);
      expect(parser.detectedBoundary, 'ffmpeg');
    });

    test('detect: a body that does not start with a boundary line fails', () {
      expect(
        () => MjpegParser.detect().add(utf8.encode('<html>\n<body>')),
        throwsA(isA<MjpegStreamException>()),
      );
      expect(
        () => MjpegParser.detect().add(catsJpeg()),
        throwsA(isA<MjpegStreamException>()),
      );
    });

    test('an empty boundary is rejected', () {
      expect(() => MjpegParser('--'), throwsA(isA<MjpegStreamException>()));
    });
  });
}
