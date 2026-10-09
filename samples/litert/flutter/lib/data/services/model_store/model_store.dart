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
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

import '../../../config/model_catalog.dart' show kRetiredGemmaFolder;
import '../../../domain/models/chat_model.dart';
import '../../../domain/models/model_id.dart';
import '../../../domain/models/provisioning.dart';
import '../../../utils/redact_url.dart';
import '../../../utils/result.dart';
import 'checksum_record.dart';
import 'http_file_downloader.dart';
import 'model_file_ops.dart';
import 'model_importer.dart';
import 'store_operation.dart';
import 'verified_commit.dart';

/// The one running operation: its cancel token, and when it has returned.
final class _Job {
  final CancelToken token = CancelToken();
  final Completer<void> _ended = Completer<void>();

  /// Completes once the operation has returned: its file sink closed, its
  /// worker isolate gone (what [ModelStore.close] waits for).
  Future<void> get ended => _ended.future;

  void end() {
    if (!_ended.isCompleted) _ended.complete();
  }
}

/// The suffix every custom chat model file must have.
const kCustomModelExtension = '.litertlm';

/// The store's folder for the custom chat model (beside the `<ModelId.name>`
/// folders earlier builds downloaded into, none of which is called this).
const _customDir = 'custom';

/// The user's own chat model on disk, in `<app support>/models/custom/`;
/// Android also extracts the built-in models into `bundled/` here
/// (`BundledModelFiles`).
///
/// A coordinator: downloads go to `<name>.<url tag>.part` through
/// [HttpFileDownloader] (HTTP Range resume, automatic reconnects, a stall
/// watchdog), imports to `<name>.import` through [ModelImporter] (moved,
/// cloned or copied). [VerifiedCommit] hashes each file (in a worker
/// isolate), checks it against the size and SHA-256 the user gave when
/// there are any, and only then renames it into place, followed by a
/// `<name>.sha256` record ([ChecksumRecord]) so the next launch need not
/// hash 2.5 GB again. The store itself owns the slot's state, the
/// one-operation lock (one download or import at a time), pruning and
/// [close].
///
/// State: [customFile], the file's [StoreFileState]. Transfer progress
/// updates it at most every [progressInterval] (about 4 Hz), never per
/// chunk.
class ModelStore {
  ModelStore({
    Future<Directory> Function()? root,
    this._ops = const ModelFileOps(),
    HttpClient Function()? httpClient,
    Duration stallTimeout = HttpFileDownloader.defaultStallTimeout,
    int maxAttempts = HttpFileDownloader.defaultMaxAttempts,
    Duration Function(int attempt) retryDelay = defaultRetryDelay,
    this.progressInterval = const Duration(milliseconds: 250),
    PartOpener openPart = openRandomAccessPart,
    this._closeWait = const Duration(seconds: 5),
  }) : _rootOf = root ?? _defaultRoot,
       _downloader = HttpFileDownloader(
         httpClient: httpClient,
         stallTimeout: stallTimeout,
         maxAttempts: maxAttempts,
         retryDelay: retryDelay,
         openPart: openPart,
       );

  static Future<Directory> _defaultRoot() async =>
      Directory('${(await getApplicationSupportDirectory()).path}/models');

  final Future<Directory> Function() _rootOf;
  final HttpFileDownloader _downloader;
  final ModelFileOps _ops;

  /// Commits downloads, and imports through [_importer].
  late final VerifiedCommit _verified = VerifiedCommit(_ops);
  late final ModelImporter _importer = ModelImporter(_ops, _verified);

  /// How long [close] waits for the cancelled operation to end.
  final Duration _closeWait;

  /// The longest a transfer goes without a state update.
  final Duration progressInterval;

  final ValueNotifier<StoreFileState> _custom = ValueNotifier(
    const StoreFileMissing(),
  );
  final ValueNotifier<bool> _busy = ValueNotifier(false);
  final Stopwatch _clock = Stopwatch()..start();

  /// When [customFile] last changed ([_clock] µs); null before the first
  /// change.
  int? _lastEmit;

  /// What the operations report to: [customFile], throttled for progress.
  late final FileStateReporter _reporter = _SlotReporter(this);

  Directory? _root;
  _Job? _job;

  /// A custom import or download runs (the rescan waits for it).
  bool _customBusy = false;
  bool _closed = false;
  Future<void>? _closing;

  /// The custom chat model's file: [StoreFileReady] once [useCustomFile]
  /// found it verified, or an import or download stored it.
  ValueListenable<StoreFileState> get customFile => _custom;

  /// True while a download or import runs.
  ValueListenable<bool> get busy => _busy;

  /// Where the store keeps its files.
  Future<Directory> root() async =>
      _root ??= await (await _rootOf()).create(recursive: true);

  /// Stops the running download or import.
  void cancel() => _job?.token.cancel();

  Future<Directory> _customFolder() async =>
      Directory('${(await root()).path}/$_customDir').create(recursive: true);

  /// Where the custom model file [name] lives in the store.
  Future<String> customPathOf(String name) async =>
      '${(await _customFolder()).path}/$name';

  /// The store file name for a custom model downloaded from [url]: the URL's
  /// last path segment when it is a plain `.litertlm` name, else
  /// `custom-model-<8 hex of the URL's SHA-256>.litertlm` (a Google Drive
  /// link names no file): another link is another file, so a new download
  /// never lands on the file the running engine uses.
  static String customFileNameFor(Uri url) {
    final segments = url.pathSegments.where((s) => s.isNotEmpty);
    final last = segments.isEmpty ? '' : segments.last;
    return last.toLowerCase().endsWith(kCustomModelExtension) &&
            isPlainFileName(last)
        ? last
        : 'custom-model-${urlTag(url)}$kCustomModelExtension';
  }

  /// Looks for the custom model [file] on disk: [StoreFileReady] (its path is
  /// returned) when its size and checksum record match, else
  /// [StoreFileUnverified] or [StoreFileMissing]. Hashes nothing. Null
  /// clears the slot.
  Future<Result<String?>> useCustomFile(CustomModelFile? file) async {
    if (_closed) return Result.error(asException(StateError('store closed')));
    if (_customBusy) return const Result.error(StoreBusyException());
    if (file == null) {
      _put(const StoreFileMissing());
      return const Result.ok(null);
    }
    try {
      final target = File(await customPathOf(file.name));
      if (!await target.exists()) {
        _put(const StoreFileMissing());
        return const Result.ok(null);
      }
      final size = await target.length();
      final record = await ChecksumRecord.read(target);
      if (size == file.sizeBytes && record == file.sha256) {
        _put(StoreFileReady(target.path));
        return Result.ok(target.path);
      }
      debugPrint(
        '[ModelStore] custom ${file.name}: $size bytes, record $record; '
        'expected ${file.sizeBytes} bytes, ${file.sha256}',
      );
      _put(const StoreFileUnverified());
      return const Result.ok(null);
    } catch (e, st) {
      debugPrint('[ModelStore] custom scan failed: $e\n$st');
      return Result.error(asException(e));
    }
  }

  /// Imports the `.litertlm` at [path] as the custom chat model: moved when
  /// it is the picker's temporary copy inside [moveFrom], else APFS-cloned
  /// (macOS/iOS) or copied, into `custom/<name>`; hashed, checked against
  /// [expectedSha256] when one is given, and its checksum recorded. A
  /// failure leaves the previous custom file as it was.
  Future<Result<CustomModelFile>> importCustomFile(
    String path, {
    String? expectedSha256,
    String? moveFrom,
  }) => _customOp((cancel) async {
    final name = _baseName(path);
    if (!name.toLowerCase().endsWith(kCustomModelExtension) ||
        !isPlainFileName(name)) {
      throw CustomModelFileException(
        '"$name" is not a $kCustomModelExtension file.',
      );
    }
    final size = await File(path).length();
    final target = File(await customPathOf(name));
    final (_, hex) = await _importer.importFile(
      path,
      target,
      name: name,
      sizeBytes: size,
      expectedSha256: expectedSha256,
      moveFrom: moveFrom,
      cancel: cancel,
      reporter: _reporter,
    );
    return CustomModelFile(
      name: name,
      sizeBytes: size,
      sha256: hex,
      checksumMatched: expectedSha256 != null,
    );
  });

  /// Downloads the custom chat model from [url] into `custom/<name>`
  /// ([customFileNameFor]), with Range resume (a `.part` per URL), the
  /// reconnects, the stall watchdog and the HTML check of every download.
  /// The size is [sizeBytes], else what the server reports; the bytes must
  /// hash to [sha256] when one is given, otherwise their checksum is
  /// recorded. A failure keeps the partial file (Retry resumes it) and
  /// leaves the previous custom file as it was.
  Future<Result<CustomModelFile>> downloadCustomFile(
    Uri url, {
    String? sha256,
    int? sizeBytes,
  }) => _customOp((cancel) async {
    final name = customFileNameFor(url);
    final target = File(await customPathOf(name));
    final part = HttpFileDownloader.partFileFor(target, url);
    final size = sizeBytes ?? await _downloader.probeSize(name, url, cancel);
    await _downloader.download(
      RemoteFile(name: name, url: url, sizeBytes: size),
      part,
      cancel,
      _reporter,
    );
    final hex = await _verified.verifyAndCommit(
      ExpectedFile(
        name: name,
        sizeBytes: size,
        sha256: sha256,
        source: redactUrl(url),
      ),
      part,
      target,
      cancel,
      _reporter,
    );
    return CustomModelFile(
      name: name,
      sizeBytes: size,
      sha256: hex,
      checksumMatched: sha256 != null,
    );
  });

  /// Runs one custom-model operation under the store's lock; any failure
  /// restores the slot's previous state and comes back as the error.
  Future<Result<CustomModelFile>> _customOp(
    Future<CustomModelFile> Function(CancelToken cancel) body,
  ) => _locked((cancel) async {
    final previous = _custom.value;
    _customBusy = true;
    try {
      return Result.ok(await body(cancel));
    } on Exception catch (e, st) {
      final error = _visible(e, 'the custom model', 0);
      debugPrint('[ModelStore] custom model: $error\n$st');
      _put(previous);
      return Result.error(error);
    } finally {
      _customBusy = false;
    }
  });

  /// Deletes the folders earlier builds downloaded models into, none of
  /// which is used any more: one per model (`<ModelId.name>/`, from the
  /// retired model manifest; every model but the chat model is built in now)
  /// and the retired Gemma 4 E2B download ([kRetiredGemmaFolder]); a folder
  /// holding one of [keepPaths] (a file used in place) is kept. `custom/`,
  /// `bundled/` and files at the root are never touched; nothing runs while
  /// an operation does. Returns the folders deleted.
  Future<List<String>> pruneOldModelFolders({
    Set<String> keepPaths = const {},
  }) async {
    if (_closed || _job != null) return const [];
    final removed = <String>[];
    try {
      final base = await root();
      for (final name in {
        for (final id in ModelId.values) id.name,
        kRetiredGemmaFolder,
      }) {
        final dir = Directory('${base.path}/$name');
        if (!await dir.exists()) continue;
        if (keepPaths.any((p) => p.startsWith('${dir.path}/'))) continue;
        await dir.delete(recursive: true);
        removed.add(dir.path);
        debugPrint(
          '[ModelStore] removed ${dir.path} (not downloaded any more)',
        );
      }
    } on FileSystemException catch (e) {
      debugPrint('[ModelStore] prune of old model folders failed: $e');
    }
    return removed;
  }

  /// Deletes every file in the custom folder but those named in [keep] and
  /// their checksum records (earlier custom models, partial downloads). Not
  /// while a custom operation runs.
  Future<void> pruneCustom({required Set<String> keep}) async {
    if (_closed || _customBusy) return;
    final kept = {
      for (final k in keep) ...[k, '$k.sha256'],
    };
    try {
      final dir = await _customFolder();
      await for (final entry in dir.list()) {
        final name = _baseName(entry.path);
        if (kept.contains(name) || entry is! File) continue;
        debugPrint('[ModelStore] custom: removing ${entry.path}');
        await deleteQuietly(entry);
      }
    } on FileSystemException catch (e) {
      debugPrint('[ModelStore] custom prune failed: $e');
    }
  }

  /// [e] as the error the Chat model card shows.
  static Exception _visible(Exception e, String name, int sizeBytes) =>
      switch (e) {
        ProvisioningException() => e,
        FileSystemException(:final osError?)
            when _isDiskFull(osError.errorCode) =>
          DiskFullException(name, sizeBytes),
        FileSystemException() => StoreWriteException(name, e),
        _ => e,
      };

  /// ENOSPC on macOS, iOS, Linux and Android; ERROR_DISK_FULL /
  /// ERROR_HANDLE_DISK_FULL on Windows.
  static bool _isDiskFull(int code) =>
      Platform.isWindows ? code == 112 || code == 39 : code == 28;

  static String _baseName(String path) => path.split(RegExp(r'[/\\]')).last;

  /// Runs [body] as the one operation of the store (busy while it runs).
  Future<Result<T>> _locked<T>(
    Future<Result<T>> Function(CancelToken cancel) body,
  ) async {
    if (_closed) return Result.error(asException(StateError('store closed')));
    if (_job != null) return Result.error(const StoreBusyException());
    final job = _job = _Job();
    _busy.value = true;
    try {
      return await body(job.token);
    } on Exception catch (e, st) {
      debugPrint('[ModelStore] operation failed: $e\n$st');
      return Result.error(e);
    } finally {
      _job = null;
      if (!_closed) _busy.value = false;
      job.end();
    }
  }

  void _put(StoreFileState state) {
    if (_closed) return;
    _lastEmit = _clock.elapsedMicroseconds;
    _custom.value = state;
  }

  /// [_put] for a transfer update: at most once per [progressInterval].
  void _progressAt(StoreFileState state) {
    final last = _lastEmit ?? -1 << 62;
    if (_clock.elapsedMicroseconds - last < progressInterval.inMicroseconds) {
      return;
    }
    _put(state);
  }

  /// Cancels the running operation, releases the HTTP client and waits
  /// (bounded, logged when it runs out) for the operation to end, so its
  /// file sink and worker isolate are gone before the notifiers are
  /// disposed. Safe to call twice; a second call waits for the first.
  Future<void> close() => _closing ??= _close();

  Future<void> _close() async {
    final job = _job;
    job?.token.cancel();
    _closed = true;
    _downloader.close();
    if (job != null) {
      await job.ended.timeout(
        _closeWait,
        onTimeout: () => debugPrint(
          '[ModelStore] close: the cancelled operation did not end within '
          '${_closeWait.inMilliseconds}ms; closing anyway',
        ),
      );
    }
    _custom.dispose();
    _busy.dispose();
  }
}

/// [ModelStore]'s [FileStateReporter]: phases at once, progress throttled.
final class _SlotReporter implements FileStateReporter {
  _SlotReporter(this._store);

  final ModelStore _store;

  @override
  void report(StoreFileState state) => _store._put(state);

  @override
  void progress(StoreFileState state) => _store._progressAt(state);
}
