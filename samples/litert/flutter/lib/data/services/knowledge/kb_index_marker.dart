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

import 'package:path_provider/path_provider.dart';

import '../../../domain/models/knowledge.dart';
import 'kb_index_key.dart';

/// `<application support>/kb`: the sqlite-vec database and its marker. Not
/// Documents, which is user-visible.
Future<Directory> defaultKbIndexDir() async =>
    Directory('${(await getApplicationSupportDirectory()).path}/kb');

/// What [KbIndexMarker.read] found, held against the key the index must
/// have.
sealed class const KbIndexMarkerRead();

/// No marker: nothing was indexed here, or a run did not finish.
final class const KbIndexMarkerMissing() extends KbIndexMarkerRead;

/// The marker is not JSON.
final class const KbIndexMarkerUnreadable(final FormatException error)
    extends KbIndexMarkerRead;

/// JSON without a string `hash`, an int `chunks` or an int `dim`.
final class const KbIndexMarkerIncomplete() extends KbIndexMarkerRead;

/// Written for another key. [builtFrom] says how, in words: "another
/// knowledge-base documents (1a2b3c4d ≠ 5e6f7a8b)", "a marker of an earlier
/// build" (no key in it), "a key this build cannot read", or "an equal key
/// with another digest".
final class const KbIndexMarkerStale(final String builtFrom)
    extends KbIndexMarkerRead;

/// Written for the key: what the index beside it holds, by the marker.
final class const KbIndexMarkerMatch({
  required final int chunks,
  required final int dim,
  required final KnowledgeOrigin origin,

  /// Why the prebuilt index was not used when this index was embedded on
  /// the device; null otherwise.
  final String? prebuiltSkipped,
}) extends KbIndexMarkerRead;

/// `<dir>/index.json`, the knowledge-base index's marker:
/// `{hash, chunks, dim, origin, prebuiltSkipped?, key}`. `hash` is the
/// [KbIndexKey]'s digest, `key` the key itself (to say what changed). An
/// indexing run deletes it first and writes it last, so an interrupted run is
/// never mistaken for a finished one.
final class KbIndexMarker {
  KbIndexMarker(Directory dir) : _file = File('${dir.path}/index.json');

  final File _file;

  /// The marker held against [key]. Throws when the file exists but cannot
  /// be read.
  Future<KbIndexMarkerRead> read(KbIndexKey key) async {
    if (!await _file.exists()) return const KbIndexMarkerMissing();
    final Object? json;
    try {
      json = jsonDecode(await _file.readAsString());
    } on FormatException catch (e) {
      return KbIndexMarkerUnreadable(e);
    }
    if (json case {
      'hash': final String markedHash,
      'chunks': final int chunks,
      'dim': final int dim,
    }) {
      if (markedHash != key.digest) {
        return KbIndexMarkerStale(switch (json) {
          {'key': final Object marked} => _differences(marked, key),
          _ => 'a marker of an earlier build',
        });
      }
      return KbIndexMarkerMatch(
        chunks: chunks,
        dim: dim,
        origin: switch (json) {
          {'origin': 'prebuilt'} => KnowledgeOrigin.prebuilt,
          _ => KnowledgeOrigin.device,
        },
        prebuiltSkipped: switch (json) {
          {'prebuiltSkipped': final String reason} => reason,
          _ => null,
        },
      );
    }
    return const KbIndexMarkerIncomplete();
  }

  /// Writes the marker for an index of [chunks] chunks built for [key]:
  /// into a temp file beside it, flushed, then renamed over it.
  Future<void> write({
    required KbIndexKey key,
    required int chunks,
    required KnowledgeOrigin origin,
    String? prebuiltSkipped,
  }) async {
    final temp = File('${_file.path}.tmp');
    await temp.writeAsString(
      jsonEncode({
        'hash': key.digest,
        'chunks': chunks,
        'dim': key.dim,
        'origin': origin.name,
        'prebuiltSkipped': ?prebuiltSkipped,
        'key': key.toJson(),
      }),
      flush: true,
    );
    await temp.rename(_file.path);
  }

  /// Deletes the marker, if there is one.
  Future<void> delete() async {
    if (await _file.exists()) await _file.delete();
  }

  /// What differs between the marker's key and [key], in words.
  static String _differences(Object marked, KbIndexKey key) {
    try {
      final differences = KbIndexKey.fromJson(marked).differencesFrom(key);
      return differences.isEmpty
          ? 'an equal key with another digest'
          : 'another ${differences.join(', ')}';
    } on FormatException {
      return 'a key this build cannot read';
    }
  }
}
