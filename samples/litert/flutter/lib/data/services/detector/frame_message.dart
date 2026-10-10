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

import 'dart:isolate';
import 'dart:typed_data';

import '../../../domain/models/frame_source_info.dart';
import '../frames/frame_source.dart';

/// Where one plane sits in [FrameMessage]'s buffer.
final class const PlaneLayout({
  required final int offset,
  required final int length,
  required final int bytesPerRow,
  required final int bytesPerPixel,
});

/// A frame copied for the detector worker: the planes are packed into one
/// [TransferableTypedData], so the bytes cross to the worker without a second
/// copy.
final class FrameMessage {
  FrameMessage._({
    required this.frameId,
    required this.format,
    required this.width,
    required this.height,
    required this.rotationDeg,
    required this.planes,
    required this._data,
  });

  /// Copies [view]'s planes. Call it inside the source's callback: the view
  /// is not valid after it returns.
  factory FrameMessage.copyOf(FrameView view, {required int frameId}) {
    final layouts = <PlaneLayout>[];
    var offset = 0;
    for (final plane in view.planes) {
      layouts.add(
        PlaneLayout(
          offset: offset,
          length: plane.bytes.length,
          bytesPerRow: plane.bytesPerRow,
          bytesPerPixel: plane.bytesPerPixel,
        ),
      );
      offset += plane.bytes.length;
    }
    return FrameMessage._(
      frameId: frameId,
      format: view.format,
      width: view.width,
      height: view.height,
      rotationDeg: view.rotationDeg,
      planes: List.unmodifiable(layouts),
      data: TransferableTypedData.fromList([
        for (final plane in view.planes) plane.bytes,
      ]),
    );
  }

  final int frameId;
  final FramePixelFormat format;
  final int width;
  final int height;
  final int rotationDeg;
  final List<PlaneLayout> planes;
  final TransferableTypedData _data;

  /// The packed planes. Callable once, on the receiving side.
  Uint8List materialize() => _data.materialize().asUint8List();
}
