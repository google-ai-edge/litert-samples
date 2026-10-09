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

import 'dart:io';

import 'package:flutter/foundation.dart';

import '../../../domain/models/provisioning.dart';
import 'model_file_ops.dart';
import 'store_operation.dart';
import 'verified_commit.dart';

/// Takes a picked file into the model store as the model store does it:
/// into `<target>.import` by [ImportMethod.moved] (the picker's own
/// temporary copy, inside `moveFrom`), [ImportMethod.cloned] (APFS
/// copy-on-write on macOS/iOS) or [ImportMethod.copied] (hashed on the way),
/// then committed by [VerifiedCommit]. A link is followed to the file it
/// points to (cloned or copied, never moved: moving would take the link,
/// not the file); the picker's temporary copy is not kept twice. Whether a
/// file is the picker's copy is decided on canonical paths (links resolved,
/// `..` normalized) of both the file and `moveFrom`, so a path that only
/// starts with `moveFrom` is never moved or deleted. A failure leaves the
/// target as it was and no `.import` behind.
///
/// State goes to a [FileStateReporter]: [StoreFileCopying] or
/// [StoreFileVerifying] at once, then as progress; [StoreFileReady] once
/// committed. Logs as `[ModelStore]`.
final class ModelImporter {
  /// [_commit] is the store's own, so imports and downloads commit alike.
  const ModelImporter(this._ops, this._commit);

  final ModelFileOps _ops;
  final VerifiedCommit _commit;

  /// Imports [source] ([sizeBytes] long, stored as [name]) onto [target]:
  /// checked against [expectedSha256] when one is given, and recorded.
  /// Returns how it came in and its SHA-256. Completing [cancel] stops a
  /// hash or a copy ([OperationCancelledException]).
  Future<(ImportMethod, String)> importFile(
    String source,
    File target, {
    required String name,
    required int sizeBytes,
    required String? expectedSha256,
    required String? moveFrom,
    required CancelToken cancel,
    required FileStateReporter reporter,
  }) async {
    final temp = File('${target.path}.import');
    try {
      final watch = Stopwatch()..start();
      final picked = await _PickedFile.resolve(source, moveFrom);
      final (method, hex) = await _takeIn(
        name,
        picked,
        sizeBytes,
        temp,
        cancel,
        reporter,
      );
      await _commit.commit(
        ExpectedFile(
          name: name,
          sizeBytes: sizeBytes,
          sha256: expectedSha256,
          source: source,
        ),
        temp,
        target,
        hex,
        reporter,
      );
      await _dropPickerCopy(method, picked);
      debugPrint(
        '[ModelStore] custom $name: imported (${method.name}) from $source in '
        '${watch.elapsedMilliseconds} ms, sha256 $hex',
      );
      return (method, hex);
    } finally {
      await deleteQuietly(temp);
    }
  }

  /// Brings [picked] into [temp] (moved from the picker's own temporary
  /// copy, APFS-cloned, or copied while hashing) and returns how and the
  /// SHA-256 of what landed in [temp].
  Future<(ImportMethod, String)> _takeIn(
    String name,
    _PickedFile picked,
    int size,
    File temp,
    CancelToken cancel,
    FileStateReporter reporter,
  ) async {
    await deleteQuietly(temp);
    ImportMethod? method;
    // Only the picker's own copy, and never through a link.
    if (picked.pickerCopy && !picked.isLink) {
      try {
        await File(picked.real).rename(temp.path);
        method = ImportMethod.moved;
      } on FileSystemException catch (e) {
        debugPrint('[ModelStore] $name: move failed ($e); copying');
      }
    }
    // A link: clone or copy the file it points to.
    if (method == null && _ops.tryClone(picked.real, temp.path)) {
      method = ImportMethod.cloned;
    }
    if (method != null) {
      return (method, await _commit.hash(name, size, temp, cancel, reporter));
    }
    reporter.report(StoreFileCopying(copied: 0, total: size));
    final hex = await _ops.copyAndHash(
      picked.real,
      temp.path,
      onProgress: (done, total) =>
          reporter.progress(StoreFileCopying(copied: done, total: total)),
      cancel: cancel.onCancel,
    );
    return (ImportMethod.copied, hex);
  }

  /// The picker's own temporary copy, when it was copied rather than moved:
  /// not kept twice (the file itself, by its canonical path).
  Future<void> _dropPickerCopy(ImportMethod method, _PickedFile picked) async {
    if (method != ImportMethod.moved && picked.pickerCopy) {
      await deleteQuietly(File(picked.real));
    }
  }
}

/// The file to import, by its canonical path: [real] has every link
/// resolved and `..` normalized. [isLink]: the picked path itself is a
/// link. [pickerCopy]: [real] lies inside the canonical `moveFrom`, the
/// picker's own temporary folder.
final class const _PickedFile({
  required final String real,
  required final bool isLink,
  required final bool pickerCopy,
}) {
  /// Throws [FileSystemException] when [source] cannot be resolved (it is
  /// gone, or a link points nowhere). A `moveFrom` that cannot be resolved
  /// holds nothing: no file is the picker's copy then.
  static Future<_PickedFile> resolve(String source, String? moveFrom) async {
    final real = await File(source).resolveSymbolicLinks();
    final isLink = await FileSystemEntity.isLink(source);
    String? folder;
    if (moveFrom != null) {
      try {
        folder = await Directory(moveFrom).resolveSymbolicLinks();
      } on FileSystemException catch (e) {
        debugPrint(
          '[ModelStore] the picker folder $moveFrom cannot be resolved ($e): '
          'the file is copied, not moved',
        );
      }
    }
    return _PickedFile(
      real: real,
      isLink: isLink,
      pickerCopy: folder != null && _isInside(real, folder),
    );
  }

  static bool _isInside(String path, String directory) {
    final dir = directory.endsWith(Platform.pathSeparator)
        ? directory
        : '$directory${Platform.pathSeparator}';
    return path.startsWith(dir);
  }
}
