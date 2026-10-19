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
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart'
    show
        LiteRtLmBundleException,
        LiteRtLmBundleInfo,
        LiteRtLmBundleProblem,
        readLiteRtLmBundleInfo;
import 'package:path_provider/path_provider.dart';

import '../../../domain/models/chat_model.dart';
import '../../../utils/result.dart';

/// Reads what a `.litertlm` says about itself (a seam for tests; the app's
/// is flutter_edge_ai_litertlm's [readLiteRtLmBundleInfo]).
typedef BundleReader = Future<LiteRtLmBundleInfo> Function(String path);

/// What the plugin makes of the `.litertlm` at [path]
/// ([readLiteRtLmBundleInfo]: the header and a few hundred bytes per
/// compiled section, never the weights), refused only where a load of it is
/// refused, so the app is never stricter than the plugin:
/// - a file that cannot be opened (missing, unreadable): a
///   [LocalModelException] naming it;
/// - an incomplete file: the reader's `incomplete` (it ends inside its own
///   header), with the plugin's message, or a file it reads whose
///   [LiteRtLmBundleInfo.isComplete] is false, with the lengths it reports.
///   flutter_edge_ai_litertlm 1.14.0 refuses both at the load
///   (`LiteRtLmEngine._readBundle` rethrows the first,
///   `liteRtLmTextCandidates` throws `incomplete` for the second).
///
/// Any other [LiteRtLmBundleException] (not a container the plugin knows, a
/// container version it does not read, a header or stamp out of range) is
/// [BundleNotRead] with the plugin's exception: the engine loads such a file
/// "without reading the file", so the pick goes on and the card says so.
///
/// No `exists()` first: it says false for a file in a folder the app may not
/// search (an Android push made before the app created its folder), which
/// would read as "does not exist". The open itself tells ENOENT (missing)
/// from EACCES/EPERM ([LocalModelException.permissionDenied]).
Future<Result<BundleReading>> inspectLitertlm(
  String path, {
  BundleReader read = readLiteRtLmBundleInfo,
}) async {
  try {
    final info = await read(path);
    if (!info.isComplete) {
      return Result.error(
        LocalModelException(
          '$path is incomplete: it is ${info.fileLength} bytes and its '
          'sections end at ${info.expectedLength} (a copy or download that '
          'stopped early). Copy or download it again.',
        ),
      );
    }
    return Result.ok(BundleRead(info));
  } on FileSystemException catch (e) {
    if (isNotFound(e)) {
      return Result.error(LocalModelException('$path does not exist.'));
    }
    return Result.error(
      LocalModelException(
        '$path cannot be read (${e.osError?.message ?? e.message}).',
        permissionDenied: isPermissionDenied(e),
      ),
    );
  } on LiteRtLmBundleException catch (e) {
    if (e.problem == LiteRtLmBundleProblem.incomplete) {
      return Result.error(LocalModelException(e.message));
    }
    debugPrint('[ModelsFolder] $path: the plugin cannot read its header: $e');
    return Result.ok(BundleNotRead(e));
  }
}

/// The folder users copy their `.litertlm` into, used in place (no copy):
/// - Android: the app's external files dir `…/files/models`
///   (`/sdcard/Android/data/<app id>/files/models`): `adb push` writes
///   there and the app reads it, neither needs a permission;
/// - Linux: `~/litert-demos/models`;
/// - macOS and iOS: the app's Documents/models (iOS: visible in Files).
class ModelsFolder {
  ModelsFolder({
    Future<Directory> Function()? directory,
    this.create = true,
    this.label = 'Models folder',
    this.permissionAdvice,
  }) : _directory = directory ?? defaultModelsDirectory;

  /// `/data/local/tmp/litert-models` on Android: `adb push` puts files there
  /// readable by the app whatever the order (shell-owned, world-readable),
  /// where a push into the app's own folder before the app created it is
  /// not ([kAndroidDataPushAdvice]). Never created by the app (it cannot):
  /// missing is an empty list.
  ModelsFolder.androidTmp()
    : _directory = (() async => Directory(kAndroidTmpModelsDir)),
      create = false,
      label = 'Or, readable in any order',
      permissionAdvice = null;

  final Future<Directory> Function() _directory;

  /// The app creates the folder when it is missing.
  final bool create;

  /// How the card names it.
  final String label;

  /// Appended when the folder or a file in it cannot be read (permission
  /// denied): what to do instead.
  final String? permissionAdvice;

  /// The folder's path once [directory] resolved it.
  String? _resolvedPath;

  /// [path] is in this folder, as far as known without I/O (after
  /// [directory] resolved it).
  bool containsResolved(String path) =>
      _resolvedPath != null && path.startsWith('$_resolvedPath/');

  /// The folder, created when missing (when [create]).
  Future<Result<Directory>> directory() async {
    try {
      final dir = await _directory();
      _resolvedPath = dir.path;
      return Result.ok(create ? await dir.create(recursive: true) : dir);
    } catch (e) {
      debugPrint('[ModelsFolder] $e');
      return Result.error(
        LocalModelException('The models folder cannot be created: $e'),
      );
    }
  }

  /// [path] is in this folder.
  Future<bool> contains(String path) async => switch (await directory()) {
    Ok(value: final dir) => path.startsWith('${dir.path}/'),
    Error() => false,
  };

  /// The `*.litertlm` files in the folder, by name. A folder the app does
  /// not create and that is not there is empty.
  Future<Result<List<LocalModelEntry>>> list() async {
    switch (await directory()) {
      case Error(:final error):
        return Result.error(error);
      case Ok(value: final dir):
        try {
          if (!create && !await dir.exists()) return const Result.ok([]);
          final entries = <LocalModelEntry>[];
          await for (final entity in dir.list()) {
            final name = entity.uri.pathSegments.lastWhere((s) => s.isNotEmpty);
            if (!name.toLowerCase().endsWith('.litertlm') ||
                name.startsWith('.')) {
              continue;
            }
            final stat = await File(entity.path).stat();
            if (stat.type != FileSystemEntityType.file) continue;
            entries.add(
              LocalModelEntry(
                path: entity.path,
                name: name,
                sizeBytes: stat.size,
                modified: stat.modified,
              ),
            );
          }
          entries.sort((a, b) => a.name.compareTo(b.name));
          return Result.ok(entries);
        } on FileSystemException catch (e) {
          final denied = isPermissionDenied(e);
          return Result.error(
            LocalModelException(
              'The models folder ${dir.path} cannot be read '
              '(${e.osError?.message ?? e.message}).'
              '${denied && permissionAdvice != null ? ' $permissionAdvice' : ''}',
              permissionDenied: denied,
            ),
          );
        } catch (e) {
          // Anything else (a platform error): shown on the card, never an
          // exception that leaves it on "Looking for the folders…".
          return Result.error(
            LocalModelException('The models folder ${dir.path}: $e'),
          );
        }
    }
  }
}

/// The second models folder on Android ([ModelsFolder.androidTmp]).
const kAndroidTmpModelsDir = '/data/local/tmp/litert-models';

/// Why a file in the app's own Android folder can be unreadable, and the
/// two ways out (a push made before the app has created its folder leaves
/// it owned by the shell: "Permission denied" for the app).
const kAndroidDataPushAdvice =
    'A folder or file pushed before the app created this folder belongs to '
    'the shell and the app cannot read it: launch the app first, then '
    'adb push into it; or push to $kAndroidTmpModelsDir/ instead '
    '(adb shell mkdir -p $kAndroidTmpModelsDir).';

/// [e] is ENOENT ("No such file or directory").
bool isNotFound(FileSystemException e) =>
    e.osError?.errorCode == 2 ||
    (e.osError?.message ?? e.message).toLowerCase().contains('no such file');

/// [e] is EACCES / EPERM ("Permission denied").
bool isPermissionDenied(FileSystemException e) =>
    e.osError?.errorCode == 13 ||
    e.osError?.errorCode == 1 ||
    (e.osError?.message ?? e.message).toLowerCase().contains('permission');

/// The models folders of this platform: on Android the app's own folder and
/// [kAndroidTmpModelsDir]; elsewhere one folder.
List<ModelsFolder> defaultModelsFolders() => Platform.isAndroid
    ? [
        ModelsFolder(permissionAdvice: kAndroidDataPushAdvice),
        ModelsFolder.androidTmp(),
      ]
    : [ModelsFolder()];

/// The default [ModelsFolder] per platform.
Future<Directory> defaultModelsDirectory() async {
  if (Platform.isAndroid) {
    final external = await getExternalStorageDirectory();
    if (external == null) {
      throw const LocalModelException(
        'This device has no app-specific external storage.',
      );
    }
    return Directory('${external.path}/models');
  }
  if (Platform.isLinux) {
    final home = Platform.environment['HOME'];
    if (home == null || home.isEmpty) {
      throw const LocalModelException('HOME is not set.');
    }
    return Directory('$home/litert-demos/models');
  }
  return Directory('${(await getApplicationDocumentsDirectory()).path}/models');
}
