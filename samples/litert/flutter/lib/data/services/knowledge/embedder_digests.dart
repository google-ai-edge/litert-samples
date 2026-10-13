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

import '../model_store/bundled_model_files.dart';
import '../model_store/model_file_ops.dart';

/// SHA-256 of the files the active embedder was installed from: part of the
/// knowledge-base index key.
final class const EmbedderDigest({
  required final String model,
  required final String tokenizer,
});

/// Finds an [EmbedderDigest]. The app uses [FileEmbedderDigests]; tests pass
/// fixed digests.
abstract interface class EmbedderDigests {
  Future<EmbedderDigest> of({
    required String modelPath,
    required String tokenizerPath,
  });
}

/// The digests without hashing 184 MB on every launch:
///
/// 1. **Built-in files** take their SHA-256 from the build
///    ([kBundledEmbedderModel], [kBundledEmbedderTokenizer]) when the path is
///    one the app verified: Android's extracted copy (`<file>.sha256` beside
///    it names that hash and the size matches, which is how
///    `BundledModelFiles` decides "extracted before, verified": the hash was
///    computed once, at extraction), or the asset in place in the app bundle
///    (desktop and iOS; the size matches; the bundle is signed). No I/O
///    beyond a stat and a 100-byte read, and nothing is extracted.
/// 2. **Any other file** (one the build cannot vouch for, e.g. an extracted
///    copy without its record) is hashed in a worker isolate once and
///    cached in `<cacheDir>/embedder_digests.json` by path, size and
///    modification time.
class FileEmbedderDigests implements EmbedderDigests {
  FileEmbedderDigests({
    required this._cacheDir,
    String? operatingSystem,
    String? executable,
    this._ops = const ModelFileOps(),
    this._known = const [kBundledEmbedderModel, kBundledEmbedderTokenizer],
  }) : _os = operatingSystem ?? Platform.operatingSystem,
       _executable = executable ?? Platform.resolvedExecutable;

  final Future<Directory> Function() _cacheDir;
  final String _os;
  final String _executable;
  final ModelFileOps _ops;
  final List<BundledFile> _known;

  @override
  Future<EmbedderDigest> of({
    required String modelPath,
    required String tokenizerPath,
  }) async => EmbedderDigest(
    model: await digestOf(modelPath),
    tokenizer: await digestOf(tokenizerPath),
  );

  /// The SHA-256 of [path] as lowercase hex.
  Future<String> digestOf(String path) async {
    final file = File(path);
    final size = await file.length();
    if (await _verifiedBuiltIn(path, size) case final BundledFile known) {
      return known.sha256;
    }
    return _hashedOrCached(file, size);
  }

  Future<BundledFile?> _verifiedBuiltIn(String path, int size) async {
    final assets = flutterAssetsDirFor(
      operatingSystem: _os,
      executable: _executable,
    );
    final record = File('$path.sha256');
    final recorded = await record.exists()
        ? (await record.readAsString()).trim().split(RegExp(r'\s+')).first
        : null;
    for (final known in _known) {
      if (size != known.sizeBytes) continue;
      if (recorded == known.sha256) return known;
      if (assets != null && path == '$assets/${known.asset}') return known;
    }
    return null;
  }

  Future<String> _hashedOrCached(File file, int size) async {
    final modified = (await file.lastModified()).millisecondsSinceEpoch;
    final cacheFile = File('${(await _cacheDir()).path}/embedder_digests.json');
    var cache = <String, Object?>{};
    if (await cacheFile.exists()) {
      try {
        if (jsonDecode(await cacheFile.readAsString())
            case final Map<String, Object?> json) {
          cache = json;
        }
      } on FormatException catch (e) {
        debugPrint('[EmbedderDigests] unreadable cache ($e): hashing again');
      }
    }
    if (cache[file.path]
        case {
          'size': final int cachedSize,
          'modified': final int cachedModified,
          'sha256': final String hex,
        }
        when cachedSize == size && cachedModified == modified) {
      return hex;
    }
    final watch = Stopwatch()..start();
    final hex = await _ops.sha256OfFile(file.path, onProgress: (_, _) {});
    debugPrint(
      '[EmbedderDigests] hashed ${file.path} ($size B) in '
      '${watch.elapsedMilliseconds} ms',
    );
    cache[file.path] = {'size': size, 'modified': modified, 'sha256': hex};
    await cacheFile.parent.create(recursive: true);
    final temp = File('${cacheFile.path}.tmp');
    await temp.writeAsString(jsonEncode(cache), flush: true);
    await temp.rename(cacheFile.path);
    return hex;
  }
}
