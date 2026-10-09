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

import 'package:flutter/services.dart';

import '../../../config/knowledge_config.dart';
import 'kb_index_key.dart';

/// What reading the prebuilt index's manifest found.
sealed class const PrebuiltManifestRead();

final class const PrebuiltManifestFound(final PrebuiltKbManifest manifest)
    extends PrebuiltManifestRead;

/// There is no prebuilt index to use; [reason] says why (for the log and
/// the status line).
final class const PrebuiltManifestMissing(final String reason)
    extends PrebuiltManifestRead;

/// Where the prebuilt knowledge-base index comes from. The app reads its
/// assets; tests pass their own.
abstract interface class PrebuiltKbIndexSource {
  /// The manifest, or why there is none. Throws [FormatException] for a
  /// manifest that does not parse.
  Future<PrebuiltManifestRead> manifest();

  /// The shipped `kb.db`, whole (a few MB).
  Future<Uint8List> database();
}

/// `assets/kb_index/manifest.json` and `kb.db` (built by
/// `tool/build_kb_index.sh`). A build without them says so; it is never an
/// error, the app then indexes on the device.
final class AssetPrebuiltKbIndex implements PrebuiltKbIndexSource {
  AssetPrebuiltKbIndex({
    AssetBundle? bundle,
    this._manifestAsset = kKbPrebuiltManifestAsset,
    this._databaseAsset = kKbPrebuiltDatabaseAsset,
  }) : _bundle = bundle ?? rootBundle;

  final AssetBundle _bundle;
  final String _manifestAsset;
  final String _databaseAsset;

  @override
  Future<PrebuiltManifestRead> manifest() async {
    final assets = (await AssetManifest.loadFromAssetBundle(_bundle))
        .listAssets()
        .toSet();
    for (final asset in [_manifestAsset, _databaseAsset]) {
      if (!assets.contains(asset)) {
        return PrebuiltManifestMissing('this build has no $asset');
      }
    }
    return PrebuiltManifestFound(
      PrebuiltKbManifest.fromJson(
        jsonDecode(await _bundle.loadString(_manifestAsset, cache: false)),
      ),
    );
  }

  @override
  Future<Uint8List> database() async {
    final data = await _bundle.load(_databaseAsset);
    return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
  }
}

/// No prebuilt index: the build tool itself, and tests that must embed on
/// the device.
final class const NoPrebuiltKbIndex(final String reason)
    implements PrebuiltKbIndexSource {
  @override
  Future<PrebuiltManifestRead> manifest() async =>
      PrebuiltManifestMissing(reason);

  @override
  Future<Uint8List> database() =>
      throw StateError('NoPrebuiltKbIndex has no database');
}
