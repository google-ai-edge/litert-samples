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

import 'package:flutter/services.dart';

import '../../../config/knowledge_config.dart';

/// One knowledge-base document as shipped: its file name and raw bytes.
final class const KbDocument({
  /// The asset's file name, e.g. `litert-overview.md`; becomes the chunk ids'
  /// prefix and the citation's `doc`.
  required final String name,

  /// The asset path, e.g. `assets/kb/litert-overview.md`; part of the hash.
  required final String path,
  required final Uint8List bytes,
});

/// Where the knowledge base's documents come from. The app reads the
/// bundled assets; tests pass a fixed list.
abstract interface class KbDocumentSource {
  /// Every document, sorted by path.
  Future<List<KbDocument>> load();
}

/// `assets/kb/*.md`, listed through the [AssetManifest], so a document
/// dropped into the folder is indexed without a code change.
final class AssetKbDocumentSource implements KbDocumentSource {
  AssetKbDocumentSource({AssetBundle? bundle, this._prefix = kKbAssetPrefix})
    : _bundle = bundle ?? rootBundle;

  final AssetBundle _bundle;
  final String _prefix;

  @override
  Future<List<KbDocument>> load() async {
    final manifest = await AssetManifest.loadFromAssetBundle(_bundle);
    final paths = [
      for (final path in manifest.listAssets())
        if (path.startsWith(_prefix) && path.endsWith('.md')) path,
    ]..sort();
    if (paths.isEmpty) {
      throw StateError('No knowledge-base documents under $_prefix');
    }
    return [
      for (final path in paths)
        KbDocument(
          name: path.substring(path.lastIndexOf('/') + 1),
          path: path,
          bytes: _bytesOf(await _bundle.load(path)),
        ),
    ];
  }

  /// Exactly the asset's bytes: the documents hash must match the build
  /// machine's byte for byte.
  static Uint8List _bytesOf(ByteData data) =>
      data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
}
