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
import 'checksum_record.dart';
import 'model_file_ops.dart';
import 'store_operation.dart';

/// Where a verified file whose SHA-256 is the argument goes, instead of the
/// target it was written for (`VerifiedCommit.commit`).
typedef PlaceFor = Future<File> Function(String hex);

/// What a stored file must be: [name], [sizeBytes], and [sha256] when the
/// user gave one; [source] is where it came from (a URL or a path), for
/// the messages.
final class const ExpectedFile({
  required final String name,
  required final int sizeBytes,

  /// Null: nothing published to compare with; the hash is computed and
  /// recorded (the custom chat model without a checksum).
  final String? sha256,
  required final String source,
});

/// Puts a finished temporary file (a download's `.part`, an import's
/// `.import`) in place as the model store does it: hashed in a worker
/// isolate ([ModelFileOps]), checked against the [ExpectedFile], renamed
/// onto the target, then recorded in a [ChecksumRecord]. The old record goes
/// before the rename and the new one after it, so a crash in between leaves
/// an unverified file (hashed again), never a record that vouches for the
/// wrong bytes. An old record that cannot be deleted stops the commit
/// before the rename ([StoreWriteException]; the temporary file deleted,
/// the target and its record as they were), as the Android extraction
/// (`BundledModelFiles`) does: renaming anyway could leave the old record
/// beside the new bytes, and a file of the same size would pass the next
/// launch's check unhashed.
///
/// State goes to a [FileStateReporter]: [StoreFileVerifying] at once, then
/// as progress; [StoreFileReady] once committed. Logs as `[ModelStore]`.
///
/// Where the file goes can depend on its hash ([PlaceFor]): the model store
/// keeps another file away from the one its chat model uses.
final class VerifiedCommit {
  const VerifiedCommit(this._ops);

  final ModelFileOps _ops;

  /// The SHA-256 of [file] ([sizeBytes] long, called [name] in the log).
  /// Completing [cancel] kills the worker ([OperationCancelledException]).
  Future<String> hash(
    String name,
    int sizeBytes,
    File file,
    CancelToken cancel,
    FileStateReporter reporter,
  ) async {
    final watch = Stopwatch()..start();
    reporter.report(StoreFileVerifying(processed: 0, total: sizeBytes));
    final hex = await _ops.sha256OfFile(
      file.path,
      onProgress: (done, total) =>
          reporter.progress(StoreFileVerifying(processed: done, total: total)),
      cancel: cancel.onCancel,
    );
    debugPrint(
      '[ModelStore] $name: SHA-256 in ${watch.elapsedMilliseconds} ms',
    );
    return hex;
  }

  /// Checks [temp]'s size (a wrong one deletes it), hashes it and [commit]s
  /// it onto [target] (or where [placeFor] says); returns the hash.
  Future<String> verifyAndCommit(
    ExpectedFile expected,
    File temp,
    File target,
    CancelToken cancel,
    FileStateReporter reporter, {
    PlaceFor? placeFor,
  }) async {
    final length = await temp.length();
    if (length != expected.sizeBytes) {
      await deleteQuietly(temp);
      throw SizeMismatchException(
        expected.name,
        expected: expected.sizeBytes,
        actual: length,
        source: expected.source,
      );
    }
    final hex = await hash(
      expected.name,
      expected.sizeBytes,
      temp,
      cancel,
      reporter,
    );
    await commit(expected, temp, target, hex, reporter, placeFor: placeFor);
    return hex;
  }

  /// Puts [temp], whose SHA-256 is [hex], in place as [target], or as the
  /// file [placeFor] names for that hash. A checksum in [expected] must match
  /// (a mismatch deletes [temp]); without one, [hex] is taken as it is (and
  /// recorded). An old record that cannot be deleted deletes [temp] and
  /// throws [StoreWriteException] before the rename.
  Future<void> commit(
    ExpectedFile expected,
    File temp,
    File target,
    String hex,
    FileStateReporter reporter, {
    PlaceFor? placeFor,
  }) async {
    if (expected.sha256 case final want? when hex != want) {
      await deleteQuietly(temp);
      throw ChecksumMismatchException(
        expected.name,
        expected: want,
        actual: hex,
        source: expected.source,
      );
    }
    final into = placeFor == null ? target : await placeFor(hex);
    final name = into.path == target.path
        ? expected.name
        : into.uri.pathSegments.last;
    // Its record names these bytes already (a file picked again under the
    // chat model's own name): the record stays, never deleted and written
    // again, and the bytes are swapped in one rename (a copy damaged in
    // place since, its size and record unchanged, is still repaired).
    if (await _recordedAlready(into, temp, hex)) {
      await temp.rename(into.path);
      reporter.report(StoreFileReady(into.path));
      debugPrint('[ModelStore] $name: verified and stored (its record kept)');
      return;
    }
    await _deleteOldRecord(name, temp, into);
    await temp.rename(into.path);
    await ChecksumRecord.write(into, hex, name);
    reporter.report(StoreFileReady(into.path));
    debugPrint('[ModelStore] $name: verified and stored');
  }

  /// [target]'s record names [hex] and it is as long as [temp]. False when
  /// either cannot be read.
  static Future<bool> _recordedAlready(
    File target,
    File temp,
    String hex,
  ) async {
    try {
      return await ChecksumRecord.read(target) == hex &&
          await target.length() == await temp.length();
    } on FileSystemException {
      return false;
    }
  }

  /// Deletes [target]'s checksum record before [temp] is renamed onto it.
  /// One that cannot be deleted aborts the commit: [temp] is deleted and the
  /// target keeps its bytes and its record, which still match.
  static Future<void> _deleteOldRecord(
    String name,
    File temp,
    File target,
  ) async {
    final record = ChecksumRecord.of(target);
    try {
      if (await record.exists()) await record.delete();
    } on FileSystemException catch (e) {
      debugPrint(
        '[ModelStore] $name: the old checksum record ${record.path} cannot be '
        'deleted ($e); the new file is not stored',
      );
      await deleteQuietly(temp);
      throw StoreWriteException(
        name,
        FileSystemException(
          'the old checksum record cannot be replaced, so the new file was '
          'not stored (the previous one is kept)',
          record.path,
          e.osError,
        ),
      );
    }
  }
}
