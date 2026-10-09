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

import 'dart:typed_data';

import 'package:litert_edge_demos/data/services/detector/frame_message.dart';
import 'package:litert_edge_demos/data/services/frames/frame_source.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';

/// A [FrameView] over given planes.
final class TestFrame implements FrameView {
  TestFrame(
    this.width,
    this.height,
    this.format,
    this.rotationDeg,
    this.planes,
  );

  /// A 640×480 RGBA frame, like one fixture slide: black, or every byte
  /// [fill] (a grey of luma [fill]).
  factory TestFrame.rgba({int width = 640, int height = 480, int fill = 0}) =>
      TestFrame(width, height, FramePixelFormat.rgba8888, 0, [
        FramePlane(
          bytes: Uint8List(width * height * 4)
            ..fillRange(0, width * height * 4, fill),
          bytesPerRow: width * 4,
          bytesPerPixel: 4,
        ),
      ]);

  @override
  final int width;
  @override
  final int height;
  @override
  final FramePixelFormat format;
  @override
  final int rotationDeg;
  @override
  final List<FramePlane> planes;
}

/// A black 640×480 RGBA frame, copied for the worker.
FrameMessage rgbaFrame({int frameId = 1}) =>
    FrameMessage.copyOf(TestFrame.rgba(), frameId: frameId);
