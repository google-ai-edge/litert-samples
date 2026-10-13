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
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/knowledge_config.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_prebuilt_index.dart';

/// An asset bundle over [files] (asset path → bytes), with the binary asset
/// manifest a build writes. Every load is a view with a non-zero offset into
/// a larger buffer, as an engine-backed load can be.
class _MapBundle extends AssetBundle {
  _MapBundle(this.files);

  final Map<String, List<int>> files;
  final List<String> loads = [];

  @override
  Future<ByteData> load(String key) async {
    loads.add(key);
    if (key == 'AssetManifest.bin') {
      return const StandardMessageCodec().encodeMessage({
        for (final asset in files.keys)
          asset: [
            {'asset': asset},
          ],
      })!;
    }
    final bytes = files[key];
    if (bytes == null) throw FlutterError('Unable to load asset: "$key".');
    final padded = Uint8List(bytes.length + 16)..setAll(8, bytes);
    return ByteData.sublistView(padded, 8, 8 + bytes.length);
  }

  @override
  void evict(String key) {}
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final shippedManifest = File(kKbPrebuiltManifestAsset).readAsStringSync();
  final database = utf8.encode('SQLite format 3\u0000 not really');

  group('AssetPrebuiltKbIndex', () {
    test('the shipped assets: the manifest parses and kb.db is the shipped '
        'file, byte for byte', () async {
      final index = AssetPrebuiltKbIndex();

      final read = await index.manifest();
      final bytes = await index.database();

      final manifest = (read as PrebuiltManifestFound).manifest;
      final json = jsonDecode(shippedManifest) as Map<String, Object?>;
      expect(manifest.chunks, json['chunks']);
      expect(manifest.dbBytes, json['dbBytes']);
      expect(manifest.dbSha256, json['dbSha256']);
      expect(bytes, File(kKbPrebuiltDatabaseAsset).readAsBytesSync());
      expect(bytes.length, manifest.dbBytes);
    });

    test('found: the manifest parsed from the bundle, read uncached; the '
        'database is exactly its asset\'s bytes (an offset view)', () async {
      final bundle = _MapBundle({
        kKbPrebuiltManifestAsset: utf8.encode(shippedManifest),
        kKbPrebuiltDatabaseAsset: database,
      });
      final index = AssetPrebuiltKbIndex(bundle: bundle);

      final read = await index.manifest();
      final bytes = await index.database();

      expect(read, isA<PrebuiltManifestFound>());
      expect(bytes, database);
      expect(bytes.length, database.length, reason: 'no padding leaks in');
      expect(bundle.loads, [
        'AssetManifest.bin',
        kKbPrebuiltManifestAsset,
        kKbPrebuiltDatabaseAsset,
      ]);
    });

    test('a build without the manifest says so, without reading '
        'anything else', () async {
      final bundle = _MapBundle({kKbPrebuiltDatabaseAsset: database});

      final read = await AssetPrebuiltKbIndex(bundle: bundle).manifest();

      expect(
        read,
        isA<PrebuiltManifestMissing>().having(
          (m) => m.reason,
          'reason',
          'this build has no $kKbPrebuiltManifestAsset',
        ),
      );
      expect(bundle.loads, ['AssetManifest.bin']);
    });

    test('a build with the manifest but without kb.db is missing too (never '
        'an error)', () async {
      final bundle = _MapBundle({
        kKbPrebuiltManifestAsset: utf8.encode(shippedManifest),
      });

      final read = await AssetPrebuiltKbIndex(bundle: bundle).manifest();

      expect(
        (read as PrebuiltManifestMissing).reason,
        'this build has no $kKbPrebuiltDatabaseAsset',
      );
      expect(bundle.loads, ['AssetManifest.bin']);
    });

    test('other asset names can be given', () async {
      final bundle = _MapBundle({
        'x/manifest.json': utf8.encode(shippedManifest),
        'x/kb.db': database,
      });
      final index = AssetPrebuiltKbIndex(
        bundle: bundle,
        manifestAsset: 'x/manifest.json',
        databaseAsset: 'x/kb.db',
      );

      expect(await index.manifest(), isA<PrebuiltManifestFound>());
      expect(await index.database(), database);
    });

    test('a manifest that does not parse throws FormatException', () async {
      for (final text in [
        'not json',
        '{"format": "kb-prebuilt/1"}',
        shippedManifest.replaceFirst('kb-prebuilt/1', 'kb-prebuilt/0'),
      ]) {
        final bundle = _MapBundle({
          kKbPrebuiltManifestAsset: utf8.encode(text),
          kKbPrebuiltDatabaseAsset: database,
        });

        await expectLater(
          AssetPrebuiltKbIndex(bundle: bundle).manifest(),
          throwsFormatException,
          reason: text,
        );
      }
    });
  });

  group('NoPrebuiltKbIndex', () {
    test('is always missing, with its reason, and has no database', () async {
      const index = NoPrebuiltKbIndex('building the prebuilt index');

      final read = await index.manifest();

      expect(
        (read as PrebuiltManifestMissing).reason,
        'building the prebuilt index',
      );
      expect(index.database, throwsStateError);
    });
  });
}
