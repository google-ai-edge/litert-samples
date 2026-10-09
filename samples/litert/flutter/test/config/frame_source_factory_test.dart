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
import 'package:litert_edge_demos/config/dependencies.dart'
    show createFrameSource;
import 'package:litert_edge_demos/data/services/frames/camera_frame_source.dart';
import 'package:litert_edge_demos/data/services/frames/fixture_frame_source.dart';
import 'package:litert_edge_demos/data/services/frames/frame_source.dart';
import 'package:litert_edge_demos/data/services/frames/network_frame_source.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/utils/result.dart';

/// The app's frame-source factory: which source each spec opens.
void main() {
  FrameSource sourceFor(FrameSourceSpec spec) =>
      switch (createFrameSource(spec)) {
        Ok(:final value) => value,
        Error(:final error) => fail('$spec: $error'),
      };

  test('the device camera spec opens the camera source', () {
    expect(sourceFor(const CameraSourceSpec()), isA<CameraFrameSource>());
  });

  test('a fixture spec opens the slideshow', () {
    expect(
      sourceFor(const FixtureSourceSpec(['/fixtures'], mirrored: true)),
      isA<FixtureFrameSource>(),
    );
  });

  test('a network spec opens the MJPEG source for its URL', () async {
    final source = sourceFor(
      NetworkSourceSpec(Uri.parse('http://192.168.1.23:8080/video')),
    );
    expect(
      source,
      isA<NetworkFrameSource>().having(
        (s) => s.label,
        'label',
        'Network camera · 192.168.1.23:8080',
      ),
    );
    await source.stop();
  });

  test('every spec gets a new source (sources are single-use)', () {
    const spec = CameraSourceSpec();
    expect(sourceFor(spec), isNot(same(sourceFor(spec))));
  });
}
