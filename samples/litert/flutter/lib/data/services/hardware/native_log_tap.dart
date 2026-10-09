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

import 'package:flutter/foundation.dart';

import '../../../domain/models/hardware_profile.dart' show HostPlatform;
import 'libc.dart';

/// Captures the native log (LiteRT's stderr) so each model load can be read
/// back on its own: [mark] before the load, [since] after it. Loads run one
/// after another, so a window holds one model's lines.
abstract interface class NativeLogTap {
  /// For the report: where the log goes, or why there is none.
  String get description;

  /// A position in the log.
  int mark();

  /// The lines written after [mark].
  List<String> since(int mark);
}

/// No capture; [description] says why. Evidence then stays "inferred".
final class NoNativeLogTap implements NativeLogTap {
  const NoNativeLogTap([this.description = 'off']);

  @override
  final String description;

  @override
  int mark() => 0;

  @override
  List<String> since(int mark) => const [];
}

/// Native stderr (fd 2) redirected into [path] for the rest of the process.
///
/// Reads go through a handle opened at install time, so they stay on this
/// process's file even if a second app instance renames it to `.prev` and
/// starts its own.
final class StderrFileTap implements NativeLogTap {
  StderrFileTap._(this.path, this._reader);

  final String path;
  final RandomAccessFile _reader;

  @override
  String get description => 'native stderr → $path';

  /// Redirects fd 2 into [path]. The previous run's file is kept as
  /// `<path>.prev` (a crash's last lines survive one restart). Throws
  /// [FileSystemException] when the redirect fails: the caller decides
  /// whether to go on without a tap, and says so.
  static StderrFileTap install(String path) {
    final file = File(path);
    file.parent.createSync(recursive: true);
    if (file.existsSync()) file.renameSync('$path.prev');
    final failure = redirectStderrTo(path);
    if (failure != null) {
      throw FileSystemException('native stderr redirect: $failure', path);
    }
    // Kept open for the process's lifetime, like fd 2 itself.
    return StderrFileTap._(path, file.openSync());
  }

  @override
  int mark() {
    try {
      return _reader.lengthSync();
    } on FileSystemException {
      return 0;
    }
  }

  @override
  List<String> since(int mark) {
    try {
      final end = _reader.lengthSync();
      if (end <= mark) return const [];
      _reader.setPositionSync(mark);
      final bytes = _reader.readSync(end - mark);
      return const LineSplitter().convert(
        utf8.decode(bytes, allowMalformed: true),
      );
    } on FileSystemException catch (e) {
      debugPrint('[NativeLogTap] cannot read $path: $e');
      return const [];
    }
  }
}

/// When [tap] redirects fd 2, Dart's own report of an uncaught error (the
/// engine prints it to stderr) would land only in the native log: echo it to
/// stdout as well. Chains any handler already set and still lets the engine
/// log it (into the file). No-op for the other taps.
void echoUnhandledErrorsToStdout(NativeLogTap tap) {
  if (tap is! StderrFileTap) return;
  final previous = PlatformDispatcher.instance.onError;
  PlatformDispatcher.instance.onError = (error, stack) {
    debugPrint(
      'Unhandled exception (native stderr → ${tap.path}): $error\n$stack',
    );
    return previous?.call(error, stack) ?? false;
  };
}

/// The tap this build can have. Release builds on Linux and macOS redirect
/// native stderr into [path] (nothing else captures it there). Debug builds do
/// not: flutter_edge_ai redirects stderr to its own file at the first engine
/// load and truncates it after every dump (litert_lm_client.dart), so a tap
/// would go silent mid-run. Android, iOS and Windows have no tap.
///
/// Prints where the log goes before redirecting, so a terminal user knows.
NativeLogTap nativeLogTapFor({
  required String path,
  required HostPlatform platform,
  bool release = kReleaseMode,
}) {
  if (platform != HostPlatform.linux && platform != HostPlatform.macos) {
    return NoNativeLogTap('off: not available on ${platform.name}');
  }
  if (!release) {
    return const NoNativeLogTap(
      'off: debug build (flutter_edge_ai redirects native stderr to its own '
      'file and truncates it)',
    );
  }
  debugPrint('[Hardware] native stderr → $path');
  try {
    return StderrFileTap.install(path);
  } on FileSystemException catch (e) {
    debugPrint('[Hardware] native log tap failed: $e');
    return NoNativeLogTap('off: $e');
  }
}
