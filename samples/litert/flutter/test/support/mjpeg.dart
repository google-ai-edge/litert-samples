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

/// The cats fixture (640×480 JPEG).
Uint8List catsJpeg() => File('test_assets/cats.jpg').readAsBytesSync();

/// A structurally valid JPEG for the parser (not decodable): SOI, an APP1
/// segment whose payload holds an embedded thumbnail with its own SOI/EOI,
/// a DQT-like segment, SOS, entropy data with stuffed bytes (FF 00), a
/// restart marker and fill bytes, a second scan (progressive-style), EOI.
/// [tag] makes images distinguishable.
Uint8List syntheticJpeg(int tag) {
  final b = BytesBuilder()
    ..add([0xFF, 0xD8]) // SOI
    ..add(
      _segment(0xE1, [
        ...ascii.encode('Exif'),
        0, 0,
        0xFF, 0xD8, 0xFF, 0xD9, // a thumbnail's SOI/EOI inside the payload
        tag,
      ]),
    )
    ..add(_segment(0xDB, List.filled(65, tag)))
    ..add(_segment(0xDA, [1, 1, 0, 0, 63, 0]))
    ..add([0x12, tag, 0xFF, 0x00, 0x34, 0xFF, 0xD3, 0x56, 0xFF, 0x00])
    ..add([0xFF, 0xFF]) // fill before the next marker
    ..add(_segment(0xC4, [0, 1, 2, 3]))
    ..add(_segment(0xDA, [1, 1, 0, 0, 63, 0]))
    ..add([0x78, 0xFF, 0x00, tag])
    ..add([0xFF, 0xD9]); // EOI
  return b.toBytes();
}

List<int> _segment(int code, List<int> payload) {
  final length = payload.length + 2;
  return [0xFF, code, length >> 8, length & 0xFF, ...payload];
}

/// One multipart part: `--boundary`, headers, a blank line, [body], CRLF.
List<int> mjpegPart(
  String boundary,
  List<int> body, {
  bool contentLength = true,
  String contentType = 'image/jpeg',
  String lengthHeader = 'Content-Length',
  String typeHeader = 'Content-Type',
  String eol = '\r\n',
}) {
  final head = StringBuffer('--$boundary$eol')
    ..write('$typeHeader: $contentType$eol');
  if (contentLength) head.write('$lengthHeader: ${body.length}$eol');
  head.write(eol);
  return [...latin1.encode(head.toString()), ...body, ...latin1.encode(eol)];
}

/// An HTTP server on 127.0.0.1 serving [handler] for every request.
Future<HttpServer> serve(Future<void> Function(HttpRequest) handler) async {
  final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
  server.listen((request) async {
    try {
      await handler(request);
    } catch (_) {
      // The client went away mid-response: expected when a source stops.
    }
  });
  return server;
}

/// Writes MJPEG parts of [frames] in a loop, one every [interval], until the
/// client disconnects or [count] parts were sent; then keeps the connection
/// open (a stalled camera) unless [closeAfter].
Future<void> streamMjpeg(
  HttpRequest request,
  List<Uint8List> frames, {
  String boundary = 'frame',
  Duration interval = const Duration(milliseconds: 40),
  int count = 1 << 30,
  bool contentLength = true,
  int chunkSize = 0,
  bool closeAfter = false,
  Duration holdOpen = const Duration(seconds: 30),
  void Function()? onDisconnect,
}) async {
  final response = request.response
    ..headers.set(
      'Content-Type',
      'multipart/x-mixed-replace; boundary=$boundary',
    )
    ..bufferOutput = false;
  try {
    // Headers go out at once, like a camera's, even before the first part
    // (dart:io sends them with the first body bytes: a blank line).
    response.add(const [13, 10]);
    await response.flush();
    for (var i = 0; i < count; i++) {
      final part = mjpegPart(
        boundary,
        frames[i % frames.length],
        contentLength: contentLength,
      );
      if (chunkSize <= 0) {
        response.add(part);
      } else {
        for (var at = 0; at < part.length; at += chunkSize) {
          response.add(
            part.sublist(at, (at + chunkSize).clamp(0, part.length)),
          );
          await response.flush();
        }
      }
      await response.flush();
      await Future<void>.delayed(interval);
    }
    if (closeAfter) {
      await response.close();
    } else {
      await Future<void>.delayed(holdOpen);
    }
  } on Object {
    // The client went away (the source stopped or failed).
    onDisconnect?.call();
  }
}
