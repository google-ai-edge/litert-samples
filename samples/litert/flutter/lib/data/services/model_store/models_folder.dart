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
import 'package:path_provider/path_provider.dart';

import '../../../domain/models/chat_model.dart';
import '../../../utils/result.dart';

/// What a `.litertlm` header says about the file (LiteRT-LM's container:
/// `litert_lm_builder` `litertlm_core.py` / `litertlm_peek.py`): the format
/// version and the `model_type` of each TFLite section
/// (`tf_lite_prefill_decode`, `tf_lite_vision_encoder`, …).
final class const LitertlmInfo({
  required final int major,
  required final int minor,
  required final int patch,
  required final List<String> modelTypes,

  /// `backend_constraint` items of the sections (`npu` for an NPU build).
  final List<String> backendConstraints = const [],
}) {
  /// A section says it runs only on the NPU (`backend_constraint: npu`).
  bool get npuOnly => backendConstraints.contains('npu');

  /// The file can read images: it has a vision encoder section.
  bool get hasVision => modelTypes.contains('tf_lite_vision_encoder');

  /// The file has an audio encoder section.
  bool get hasAudio => modelTypes.any((t) => t.startsWith('tf_lite_audio'));

  String get version => '$major.$minor.$patch';
}

/// Where the `.litertlm` header ends (u64 at byte 24) and starts (32).
const _headerEndOffset = 24;
const _headerBegin = 32;

/// A header larger than this is not one: LiteRT-LM writes a few KB.
const _maxHeaderBytes = 4 << 20;

/// The magic bytes every `.litertlm` starts with.
const kLitertlmMagic = 'LITERTLM';

/// Parses a `.litertlm` header: [prefix] is the file's first 32 bytes,
/// [header] the bytes from 32 to the header end (a `LiteRTLMMetaData`
/// flatbuffer: `section_metadata.objects[].items[]` key `model_type`, a
/// `StringValue`). Throws [FormatException] when it is not one.
LitertlmInfo parseLitertlmHeader(Uint8List prefix, Uint8List header) {
  if (prefix.length < _headerBegin ||
      ascii.decode(prefix.sublist(0, 8), allowInvalid: true) !=
          kLitertlmMagic) {
    throw const FormatException('no LITERTLM magic');
  }
  final p = ByteData.sublistView(prefix);
  final b = ByteData.sublistView(header);
  // Every read is bounds-checked: a header cut short (or an offset past its
  // end) is a FormatException.
  void need(int offset, int bytes) {
    if (offset < 0 || offset + bytes > header.length) {
      throw const FormatException('the header is cut short');
    }
  }

  int u32(int o) {
    need(o, 4);
    return b.getUint32(o, Endian.little);
  }

  int i32(int o) {
    need(o, 4);
    return b.getInt32(o, Endian.little);
  }

  int u16(int o) {
    need(o, 2);
    return b.getUint16(o, Endian.little);
  }

  int deref(int pos) => pos + u32(pos);
  // The position of a table's field [slot], or null when absent.
  int? field(int table, int slot) {
    final vtable = table - i32(table);
    final vtableLength = u16(vtable);
    final entry = 4 + 2 * slot;
    if (entry >= vtableLength) return null;
    final offset = u16(vtable + entry);
    return offset == 0 ? null : table + offset;
  }

  // Offsets may point anywhere, also at what another one already points at:
  // every object at one object, every item at one table, every key at one
  // long string. A header that holds its entries and strings once visits at
  // most as many entries as it has words and decodes at most as many bytes
  // as it has; one that repeats them (crafted: (H/4)² visits, tens of
  // minutes for 4 MB) runs out of these budgets instead. Seven real headers
  // used 4–23 entries of 88–356 and 32–362 string bytes of 352–1424.
  var entriesLeft = header.length ~/ 4;
  var bytesLeft = header.length;

  // The positions of a vector's entries (u32 offsets, 4 bytes each). A count
  // the bytes left cannot hold is corrupt: checked before the list is built,
  // or a count near 2^32 builds a list that long.
  List<int> vector(int pos) {
    final v = deref(pos);
    final count = u32(v);
    need(v + 4, 4 * count);
    if ((entriesLeft -= count) < 0) {
      throw const FormatException('the header repeats its entries');
    }
    return [for (var i = 0; i < count; i++) v + 4 + 4 * i];
  }

  String string(int pos) {
    final s = deref(pos);
    final length = u32(s);
    need(s + 4, length);
    if ((bytesLeft -= length) < 0) {
      throw const FormatException('the header repeats its strings');
    }
    return utf8.decode(header.sublist(s + 4, s + 4 + length));
  }

  final types = <String>[];
  final constraints = <String>[];
  final root = deref(0);
  final sectionMetadata = field(root, 1);
  if (sectionMetadata != null) {
    final objects = field(deref(sectionMetadata), 0);
    for (final o in objects == null ? const <int>[] : vector(objects)) {
      final items = field(deref(o), 0);
      for (final item in items == null ? const <int>[] : vector(items)) {
        final kv = deref(item);
        final key = field(kv, 0);
        if (key == null) continue;
        final name = string(key);
        if (name != 'model_type' && name != 'backend_constraint') continue;
        final value = field(kv, 2);
        if (value == null) continue;
        final text = field(deref(value), 0);
        if (text == null) continue;
        (name == 'model_type' ? types : constraints).add(string(text));
      }
    }
  }
  return LitertlmInfo(
    major: p.getUint32(8, Endian.little),
    minor: p.getUint32(12, Endian.little),
    patch: p.getUint32(16, Endian.little),
    modelTypes: types,
    backendConstraints: constraints,
  );
}

/// Opens a file for reading (a seam for tests).
typedef FileOpener = Future<RandomAccessFile> Function(String path);

Future<RandomAccessFile> _open(String path) => File(path).open();

/// Reads [path]'s header: the file exists, is readable and is a
/// `.litertlm`. Reads only the header (a few KB), never the model. No
/// `exists()` first: it says false for a file in a folder the app may not
/// search (an Android push made before the app created its folder), which
/// would read as "does not exist". The open itself tells ENOENT (missing)
/// from EACCES/EPERM ([LocalModelException.permissionDenied]).
Future<Result<LitertlmInfo>> inspectLitertlm(
  String path, {
  FileOpener open = _open,
}) async {
  RandomAccessFile? raf;
  try {
    raf = await open(path);
    final prefix = await raf.read(_headerBegin);
    if (prefix.length < _headerBegin ||
        ascii.decode(prefix.sublist(0, 8), allowInvalid: true) !=
            kLitertlmMagic) {
      return Result.error(
        LocalModelException(
          '$path is not a .litertlm file (it does not start with '
          '"$kLitertlmMagic").',
        ),
      );
    }
    final end = ByteData.sublistView(prefix)
        .getUint64(_headerEndOffset, Endian.little);
    if (end <= _headerBegin || end > _maxHeaderBytes + _headerBegin) {
      return Result.error(
        LocalModelException('$path has an unreadable .litertlm header.'),
      );
    }
    await raf.setPosition(_headerBegin);
    final header = await raf.read(end - _headerBegin);
    return Result.ok(parseLitertlmHeader(prefix, header));
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
  } on FormatException catch (e) {
    return Result.error(
      LocalModelException(
        '$path has an unreadable .litertlm header (${e.message}).',
      ),
    );
  } finally {
    await raf?.close();
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
/// two ways out (measured on a Galaxy S26: a push made before the app had
/// created its folder left it owned by the shell, "Permission denied" for
/// the app).
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
