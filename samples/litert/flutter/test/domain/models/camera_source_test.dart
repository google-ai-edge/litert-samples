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
import 'package:litert_edge_demos/domain/models/camera_source.dart';
import 'package:litert_edge_demos/utils/result.dart';

String errorOf(Result<Uri> result) => switch (result) {
  Ok(:final value) => fail('expected an error, got $value'),
  Error(:final error) => error.toString(),
};

void main() {
  group('parseNetworkCameraUrl', () {
    test('an IP Webcam URL', () {
      final uri = (parseNetworkCameraUrl(
        ' http://192.168.1.23:8080/video ',
      ) as Ok<Uri>).value;
      expect((uri.host, uri.port, uri.path), ('192.168.1.23', 8080, '/video'));
    });

    test('a missing scheme means http://', () {
      final uri =
          (parseNetworkCameraUrl('192.168.1.23:8080/video') as Ok<Uri>).value;
      expect(uri.toString(), 'http://192.168.1.23:8080/video');
    });

    test("the example's placeholders must be replaced", () {
      expect(
        errorOf(parseNetworkCameraUrl(kNetworkCameraUrlExample)),
        contains('Replace 192.168.x.x'),
      );
    });

    test('empty, not http, not a URL', () {
      expect(errorOf(parseNetworkCameraUrl('  ')), contains('Enter'));
      expect(
        errorOf(parseNetworkCameraUrl('rtsp://192.168.1.23:8554/live')),
        contains('starts with http://'),
      );
      expect(errorOf(parseNetworkCameraUrl('http://')), contains('not a URL'));
    });

    test('what was typed is not echoed: it may hold a password', () {
      final error = errorOf(parseNetworkCameraUrl('http://admin:secret@'));
      expect(error, contains('not a URL'));
      expect(error, isNot(contains('secret')));
    });
  });

  test('the label shows host and port, never the path or a password', () {
    expect(
      networkCameraLabel(Uri.parse('http://user:pw@192.168.1.23:8080/video')),
      'Network camera · 192.168.1.23:8080',
    );
    expect(
      networkCameraLabel(Uri.parse('http://cam.local/video')),
      'Network camera · cam.local:80',
    );
  });

  test('CameraSourceKind round-trips its persisted name', () {
    for (final kind in CameraSourceKind.values) {
      expect(CameraSourceKind.tryParse(kind.name), kind);
    }
    expect(CameraSourceKind.tryParse('webcam'), isNull);
  });
}
