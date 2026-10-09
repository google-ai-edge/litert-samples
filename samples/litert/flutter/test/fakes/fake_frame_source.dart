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

import 'dart:async';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:litert_edge_demos/data/services/frames/frame_source.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/models/preview_source.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../support/frames.dart';

/// A [FrameSource] the test drives with [emit].
class FakeFrameSource implements FrameSource {
  FakeFrameSource({this.label = 'fake'});

  final String label;
  Result<FrameSourceInfo>? startResult;

  /// When set, start reports this through onError before returning Ok.
  Exception? failDuringStart;

  /// When true, start delivers one frame before it resolves, as a camera
  /// can.
  bool emitDuringStart = false;

  /// With [emitDuringStart]: start resolves only after this completes.
  Completer<void>? resolveGate;
  Completer<void>? startGate;

  /// Thrown by [start] (after [startGate]) instead of returning a result.
  StateError? startThrows;

  /// Thrown by [stop] after the source has stopped.
  StateError? stopThrows;
  void Function(FrameView)? _onFrame;
  void Function(Exception)? _onError;
  int startCalls = 0;
  bool stopped = false;
  final ValueNotifier<ui.Image?> image = ValueNotifier(null);

  @override
  late final PreviewSource preview = ImagePreviewSource(image);

  @override
  Future<Result<FrameSourceInfo>> start(
    void Function(FrameView) onFrame, {
    void Function(Exception error)? onError,
  }) async {
    startCalls++;
    await startGate?.future;
    if (startThrows case final error?) throw error;
    final result =
        startResult ??
        Result.ok(
          FrameSourceInfo(
            label: label,
            width: 640,
            height: 480,
            format: FramePixelFormat.rgba8888,
            mirrored: false,
          ),
        );
    if (result is Ok<FrameSourceInfo>) {
      _onFrame = onFrame;
      _onError = onError;
      if (failDuringStart case final error?) onError?.call(error);
      if (emitDuringStart) {
        onFrame(TestFrame.rgba());
        // One event-loop turn before resolving, like CameraFrameSource still
        // awaiting startImageStream when the detector's reply arrives.
        await Future<void>.delayed(Duration.zero);
        await resolveGate?.future;
      }
    }
    return result;
  }

  @override
  Future<void> stop() async {
    stopped = true;
    _onFrame = null;
    _onError = null;
    if (stopThrows case final error?) throw error;
  }

  bool get running => _onFrame != null && !stopped;

  /// Plays a runtime failure (camera unplugged).
  void failAtRuntime(Exception error) => _onError?.call(error);

  /// Delivers one frame (640×480 RGBA unless [frame] is given).
  void emit([FrameView? frame]) => _onFrame?.call(frame ?? TestFrame.rgba());
}
