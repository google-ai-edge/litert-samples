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
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

import 'package:crypto/crypto.dart';
import 'package:ffi/ffi.dart';
import 'package:flutter/foundation.dart';

import '../../../domain/models/provisioning.dart';

/// Bytes done of a long file operation, and the total.
typedef ByteProgress = void Function(int done, int total);

/// The worker failed: an error it reported, an uncaught error, or an exit
/// without a result. An [Exception], so the store's `on Exception` handlers
/// turn it into a failed row instead of an uncaught zone error.
final class FileOpFailedException implements Exception {
  const FileOpFailedException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// Hashing and copying of multi-gigabyte model files. Each operation runs in
/// its own worker isolate (a 2.5 GB SHA-256 would block the UI isolate for
/// many seconds) and reports progress about four times a second. The store
/// takes an instance so tests can count calls or inject failures.
class ModelFileOps {
  const ModelFileOps({
    this.progressInterval = const Duration(milliseconds: 250),
    @visibleForTesting this.worker = _fileWorker,
  });

  /// How often a worker reports progress.
  final Duration progressInterval;

  /// The worker isolate's entry point; tests replace it to play a worker
  /// that crashes, reports an error or exits without a result.
  final void Function((Object, SendPort)) worker;

  /// The SHA-256 of [path] as lowercase hex. Completing [cancel] kills the
  /// worker and throws [OperationCancelledException]; a failed worker throws
  /// [FileOpFailedException], an I/O error [FileSystemException].
  Future<String> sha256OfFile(
    String path, {
    required ByteProgress onProgress,
    Future<void>? cancel,
  }) => _run(
    worker,
    _HashJob(path, progressInterval.inMicroseconds),
    onProgress,
    cancel,
  );

  /// Copies [source] to [target] (created or truncated) and returns the
  /// SHA-256 of the bytes copied: one read pass for both. [target] gets
  /// [source]'s modification time, so LiteRT-LM's GPU program cache (keyed by
  /// file name, mtime and size) built for the source is reused for the copy.
  Future<String> copyAndHash(
    String source,
    String target, {
    required ByteProgress onProgress,
    Future<void>? cancel,
  }) => _run(
    worker,
    _CopyJob(source, target, progressInterval.inMicroseconds),
    onProgress,
    cancel,
  );

  /// An APFS copy-on-write clone of [source] at [target] (macOS/iOS): instant
  /// and without extra disk. Returns false where it is not possible (another
  /// volume or file system, another OS); the caller then copies. [target]
  /// gets [source]'s modification time (see [copyAndHash]).
  bool tryClone(String source, String target) {
    if (!Platform.isMacOS && !Platform.isIOS) return false;
    final ok = using((arena) {
      final clone = DynamicLibrary.process()
          .lookupFunction<
            Int32 Function(Pointer<Utf8>, Pointer<Utf8>, Uint32),
            int Function(Pointer<Utf8>, Pointer<Utf8>, int)
          >('clonefile');
      return clone(
            source.toNativeUtf8(allocator: arena),
            target.toNativeUtf8(allocator: arena),
            0,
          ) ==
          0;
    });
    if (ok) _copyModified(source, target);
    return ok;
  }

  static Future<String> _run(
    void Function((Object, SendPort)) worker,
    _FileJob job,
    ByteProgress onProgress,
    Future<void>? cancel,
  ) async {
    final port = ReceivePort('model-file-op');
    final done = Completer<String>();
    Isolate? isolate;
    // One port for progress, the result, errors and exit: messages from one
    // isolate to one port arrive in order, so the result is never overtaken
    // by the exit notice.
    port.listen((message) {
      if (done.isCompleted) return;
      switch (message) {
        case (final int processed, final int total):
          onProgress(processed, total);
        case ('done', final String hex):
          done.complete(hex);
        case (
          'fs-error',
          final String text,
          final String? path,
          final int? code,
          final String? os,
        ):
          done.completeError(
            FileSystemException(
              text,
              path,
              code == null ? null : OSError(os ?? '', code),
            ),
          );
        case ('error', final String text):
          done.completeError(FileOpFailedException(text));
        case [final Object? error, final Object? stack]:
          debugPrint('[ModelFileOps] file worker crashed: $error\n$stack');
          done.completeError(
            FileOpFailedException('file worker crashed: $error'),
          );
        case null:
          done.completeError(
            const FileOpFailedException('file worker exited without a result'),
          );
      }
    });
    // A cancel while the isolate still spawns completes [done] before anyone
    // awaits it: marked handled here, the await below still throws it.
    done.future.ignore();
    unawaited(
      cancel?.then((_) {
        if (done.isCompleted) return;
        isolate?.kill(priority: Isolate.immediate);
        done.completeError(const OperationCancelledException());
      }),
    );
    try {
      isolate = await Isolate.spawn(
        worker,
        (job, port.sendPort),
        onExit: port.sendPort,
        onError: port.sendPort,
        debugName: 'model-file-op',
      );
      if (done.isCompleted) isolate.kill(priority: Isolate.immediate);
      return await done.future;
    } finally {
      port.close();
    }
  }
}

sealed class _FileJob {
  const _FileJob(this.intervalMicros);

  final int intervalMicros;
}

final class _HashJob extends _FileJob {
  const _HashJob(this.path, super.intervalMicros);

  final String path;
}

final class _CopyJob extends _FileJob {
  const _CopyJob(this.source, this.target, super.intervalMicros);

  final String source;
  final String target;
}

/// Receives the single [Digest] of a chunked hash.
final class _DigestCapture implements Sink<Digest> {
  Digest? value;

  @override
  void add(Digest data) => value = data;

  @override
  void close() {}
}

const _chunkBytes = 4 << 20;

/// Gives [target] the modification time of [source]. Best effort: a failure
/// costs only a GPU program cache rebuild, so it is logged, not thrown.
void _copyModified(String source, String target) {
  try {
    File(target).setLastModifiedSync(File(source).lastModifiedSync());
  } on FileSystemException catch (e) {
    debugPrint('[ModelFileOps] could not copy the mtime to $target: $e');
  }
}

/// Worker isolate entry: sync I/O in 4 MiB chunks, progress at most every
/// `intervalMicros`, then `('done', hex)` or an error tuple.
void _fileWorker((Object, SendPort) args) {
  final (message, port) = args;
  final job = message as _FileJob;
  final clock = Stopwatch()..start();
  var lastReport = -job.intervalMicros;
  void report(int processed, int total, {bool force = false}) {
    final now = clock.elapsedMicroseconds;
    if (!force && now - lastReport < job.intervalMicros) return;
    lastReport = now;
    port.send((processed, total));
  }

  RandomAccessFile? input;
  RandomAccessFile? output;
  try {
    final sourcePath = switch (job) {
      _HashJob(:final path) => path,
      _CopyJob(:final source) => source,
    };
    input = File(sourcePath).openSync();
    final total = input.lengthSync();
    if (job case _CopyJob(:final target)) {
      output = File(target).openSync(mode: FileMode.write);
    }
    final capture = _DigestCapture();
    final hasher = sha256.startChunkedConversion(capture);
    final buffer = Uint8List(_chunkBytes);
    var processed = 0;
    report(0, total, force: true);
    while (true) {
      final n = input.readIntoSync(buffer);
      if (n == 0) break;
      final chunk = n == buffer.length
          ? buffer
          : Uint8List.sublistView(buffer, 0, n);
      output?.writeFromSync(chunk);
      // The hash sink copies what it is given, so the buffer can be reused.
      hasher.add(chunk);
      processed += n;
      report(processed, total);
    }
    hasher.close();
    if (output != null) {
      output.flushSync();
      output.closeSync();
      output = null;
      final copy = job as _CopyJob;
      _copyModified(copy.source, copy.target);
    }
    report(processed, total, force: true);
    port.send(('done', capture.value.toString()));
  } on FileSystemException catch (e) {
    port.send((
      'fs-error',
      e.message,
      e.path,
      e.osError?.errorCode,
      e.osError?.message,
    ));
  } catch (e) {
    port.send(('error', '$e'));
  } finally {
    try {
      input?.closeSync();
      output?.closeSync();
    } on FileSystemException catch (e) {
      debugPrint('[ModelFileOps] closing failed: $e');
    }
  }
}
