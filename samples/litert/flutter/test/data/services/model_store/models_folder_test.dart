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
import 'dart:isolate';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/model_store/models_folder.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';
import 'package:litert_edge_demos/utils/result.dart';

/// The headers of two real `.litertlm` files (the first bytes up to the
/// header end; no weights): Gemma 4 E2B for the GPU (vision + audio) and the
/// official Gemma 4 2B Qualcomm SM8850 NPU build (text only, NPU-only).
final gpuHeader = File('test_assets/litertlm/gemma4_e2b_gpu.header.bin')
    .readAsBytesSync();
final npuHeader = File('test_assets/litertlm/gemma4_2b_sm8850_npu.header.bin')
    .readAsBytesSync();

/// A file that starts with [header] and continues with some bytes (the
/// sections a real file would have).
File litertlm(Directory dir, String name, Uint8List header) =>
    File('${dir.path}/$name')
      ..writeAsBytesSync([...header, ...List.filled(4096, 7)]);

/// A file's first bytes: the 32-byte prefix (the magic, version 1.5.0, the
/// header end), then [header].
Uint8List withPrefix(Uint8List header) {
  final b = ByteData(32 + header.length);
  for (final (i, unit) in kLitertlmMagic.codeUnits.indexed) {
    b.setUint8(i, unit);
  }
  b
    ..setUint32(8, 1, Endian.little)
    ..setUint32(12, 5, Endian.little)
    ..setUint64(24, 32 + header.length, Endian.little);
  return b.buffer.asUint8List()..setAll(32, header);
}

/// A minimal `LiteRTLMMetaData` header (the bytes after the prefix): the
/// root table, its `section_metadata` table and that table's `objects`
/// vector, which says it holds [objects] entries. With [items], `objects`
/// holds one object whose `items` vector says it holds [items] entries.
/// Only the counts are written, never the entries they announce.
Uint8List metadataHeader({required int objects, int? items}) {
  final b = ByteData(items == null ? 40 : 64);
  void u16(int at, int value) => b.setUint16(at, value, Endian.little);
  void u32(int at, int value) => b.setUint32(at, value, Endian.little);
  // Each table is its vtable offset (i32) then one u32 field; its vtable
  // sits right before it.
  u32(0, 12); // the root table at 12
  u16(4, 8); // root vtable: 8 bytes, table 8 bytes,
  u16(6, 8);
  u16(8, 0); // field 0 absent,
  u16(10, 4); // field 1 (section_metadata) at +4
  b.setInt32(12, 8, Endian.little); // its vtable at 12 - 8
  u32(16, 12); // section_metadata at 16 + 12 = 28
  u16(20, 6); // its vtable: 6 bytes, table 8 bytes, field 0 (objects) at +4
  u16(22, 8);
  u16(24, 4);
  b.setInt32(28, 8, Endian.little);
  u32(32, 4); // objects at 32 + 4 = 36
  u32(36, objects);
  if (items != null) {
    u32(40, 12); // objects[0]: the object at 40 + 12 = 52
    u16(44, 6); // its vtable: field 0 (items) at +4
    u16(46, 8);
    u16(48, 4);
    b.setInt32(52, 8, Endian.little);
    u32(56, 4); // items at 56 + 4 = 60
    u32(60, items);
  }
  return b.buffer.asUint8List();
}

/// A header that repeats what it holds instead of holding it: like
/// [metadataHeader], but all [objects] entries of `objects` point at one
/// object, and all [items] entries of its `items` point at one key-value
/// table. That table has no fields or, with [keyLength], a key of that many
/// bytes (not one the parser looks for).
Uint8List aliasingHeader({
  required int objects,
  required int items,
  int? keyLength,
}) {
  final objectVtable = 40 + 4 * objects;
  final object = objectVtable + 8;
  final itemsAt = object + 8;
  final kvVtable = itemsAt + 4 + 4 * items;
  final kv = kvVtable + (keyLength == null ? 4 : 8);
  final keyAt = kv + 8;
  final b = ByteData(keyLength == null ? kv + 4 : keyAt + 4 + keyLength);
  void u16(int at, int value) => b.setUint16(at, value, Endian.little);
  void u32(int at, int value) => b.setUint32(at, value, Endian.little);
  // The root and section_metadata of metadataHeader; objects at 36.
  b.buffer.asUint8List().setAll(0, metadataHeader(objects: objects));
  for (var i = 0; i < objects; i++) {
    u32(40 + 4 * i, object - (40 + 4 * i));
  }
  u16(objectVtable, 6); // field 0 (items) at +4
  u16(objectVtable + 2, 8);
  u16(objectVtable + 4, 4);
  b.setInt32(object, object - objectVtable, Endian.little);
  u32(object + 4, itemsAt - (object + 4));
  u32(itemsAt, items);
  for (var i = 0; i < items; i++) {
    final at = itemsAt + 4 + 4 * i;
    u32(at, kv - at);
  }
  if (keyLength == null) {
    u16(kvVtable, 4); // no fields
    u16(kvVtable + 2, 4);
  } else {
    u16(kvVtable, 6); // field 0 (key) at +4
    u16(kvVtable + 2, 8);
    u16(kvVtable + 4, 4);
    u32(kv + 4, keyAt - (kv + 4));
    u32(keyAt, keyLength);
    b.buffer.asUint8List().fillRange(keyAt + 4, keyAt + 4 + keyLength, 0x78);
  }
  b.setInt32(kv, kv - kvVtable, Endian.little);
  return b.buffer.asUint8List();
}

/// What [parseLitertlmHeader] makes of a file's first bytes: `ok …` with
/// the result, `FormatException: …`, or anything else it threw.
String parseOutcome(Uint8List file) {
  try {
    final info = parseLitertlmHeader(
      file.sublist(0, file.length < 32 ? file.length : 32),
      file.length < 32 ? Uint8List(0) : file.sublist(32),
    );
    return 'ok ${info.version} ${info.modelTypes} ${info.backendConstraints}';
  } on FormatException catch (e) {
    return 'FormatException: ${e.message}';
  } catch (e) {
    return 'threw ${e.runtimeType}: $e';
  }
}

void _parseAll((SendPort, List<Uint8List>) message) {
  final (reply, files) = message;
  for (final file in files) {
    reply.send(parseOutcome(file));
  }
}

/// [parseOutcome] of each of [files], in its own isolate, killed when one
/// file takes longer than [limit]: a parser that runs away (a vector count
/// of 2^32) fails the test, naming the file, instead of hanging the suite.
Future<List<String>> parseBounded(
  List<Uint8List> files, {
  List<String>? names,
  Duration limit = const Duration(seconds: 2),
}) async {
  final port = ReceivePort();
  final outcomes = StreamIterator<Object?>(port);
  final isolate = await Isolate.spawn(_parseAll, (port.sendPort, files));
  try {
    final results = <String>[];
    for (var i = 0; i < files.length; i++) {
      final name = names?[i] ?? 'file $i';
      final next = await outcomes.moveNext().timeout(
        limit,
        onTimeout: () => fail('parsing $name took more than $limit'),
      );
      if (!next) fail('the parsing isolate ended before $name');
      results.add(outcomes.current as String);
    }
    return results;
  } finally {
    isolate.kill(priority: Isolate.immediate);
    await outcomes.cancel();
    port.close();
  }
}

/// [count] corruptions of [fixtures], from [seed]: random bytes overwritten,
/// a random aligned u32 set to a random or a huge value, or the file cut
/// short.
List<({String name, Uint8List file})> corruptions(
  Map<String, Uint8List> fixtures, {
  required int seed,
  required int count,
}) {
  const huge = [0xFFFFFFFF, 0x7FFFFFFF, 0x40000000, 0x01000000];
  final random = Random(seed);
  final names = fixtures.keys.toList();
  final cases = <({String name, Uint8List file})>[];
  for (var i = 0; i < count; i++) {
    final source = names[i % names.length];
    final file = Uint8List.fromList(fixtures[source]!);
    // A u32 of the header (4-aligned from its start at 32).
    final word = 32 + 4 * random.nextInt((file.length - 32) ~/ 4);
    final kind = random.nextInt(4);
    switch (kind) {
      case 0:
        final flips = 1 + random.nextInt(4);
        for (var f = 0; f < flips; f++) {
          file[random.nextInt(file.length)] = random.nextInt(256);
        }
        cases.add((name: '#$i $source: $flips bytes', file: file));
      case 1 || 2:
        final value = kind == 1
            ? random.nextInt(1 << 32)
            : huge[random.nextInt(huge.length)];
        ByteData.sublistView(file).setUint32(word, value, Endian.little);
        cases.add((name: '#$i $source: u32 at $word = $value', file: file));
      default:
        final length = random.nextInt(file.length);
        cases.add((
          name: '#$i $source: cut at $length',
          file: file.sublist(0, length),
        ));
    }
  }
  return cases;
}

void main() {
  late Directory tmp;

  setUp(() => tmp = Directory.systemTemp.createTempSync('models_folder'));
  tearDown(() => tmp.deleteSync(recursive: true));

  group('the .litertlm header', () {
    test('Gemma 4 E2B: version, sections, a vision encoder (images on)', () {
      final info = parseLitertlmHeader(
        gpuHeader.sublist(0, 32),
        gpuHeader.sublist(32),
      );

      expect(info.version, '1.5.0');
      expect(info.modelTypes, contains('tf_lite_prefill_decode'));
      expect(info.modelTypes, contains('tf_lite_vision_encoder'));
      expect(info.hasVision, isTrue);
      expect(info.hasAudio, isTrue);
      expect(info.npuOnly, isFalse);
    });

    test('the SM8850 NPU build: no vision (images off), NPU only', () {
      final info = parseLitertlmHeader(
        npuHeader.sublist(0, 32),
        npuHeader.sublist(32),
      );

      expect(info.modelTypes, [
        'tf_lite_aux',
        'tf_lite_embedder',
        'tf_lite_per_layer_embedder',
        'tf_lite_prefill_decode',
      ]);
      expect(info.hasVision, isFalse);
      expect(info.hasAudio, isFalse);
      expect(info.npuOnly, isTrue);
      expect(info.backendConstraints, ['npu']);
    });

    test('not the magic, or a header cut short, is a FormatException', () {
      expect(
        () => parseLitertlmHeader(Uint8List(32), Uint8List(16)),
        throwsFormatException,
      );
      expect(
        () => parseLitertlmHeader(
          gpuHeader.sublist(0, 32),
          gpuHeader.sublist(32, 200),
        ),
        throwsFormatException,
      );
    });

    test('every real header parses to exactly its sections, in order', () {
      expect(
        parseOutcome(gpuHeader),
        'ok 1.5.0 [tf_lite_embedder, tf_lite_per_layer_embedder, '
        'tf_lite_audio_encoder_hw, tf_lite_audio_adapter, '
        'tf_lite_end_of_audio, tf_lite_vision_encoder, '
        'tf_lite_vision_adapter, tf_lite_end_of_vision, '
        'tf_lite_prefill_decode] [cpu, cpu, cpu]',
      );
      expect(
        parseOutcome(npuHeader),
        'ok 1.5.0 [tf_lite_aux, tf_lite_embedder, '
        'tf_lite_per_layer_embedder, tf_lite_prefill_decode] [npu]',
      );
    });

    test('empty vectors parse: no sections', () {
      for (final header in [
        metadataHeader(objects: 0),
        metadataHeader(objects: 1, items: 0),
      ]) {
        expect(parseOutcome(withPrefix(header)), 'ok 1.5.0 [] []');
      }
    });

    test(
      'a vector count larger than the header holds fails fast: the '
      'header is cut short (it built a list of up to 2^32 entries)',
      () async {
        final headers = {
          'objects 2^32 - 1': metadataHeader(objects: 0xFFFFFFFF),
          'objects 1, its entry past the end': metadataHeader(objects: 1),
          'items 2^32 - 1': metadataHeader(objects: 1, items: 0xFFFFFFFF),
          'items 2^30': metadataHeader(objects: 1, items: 1 << 30),
        };
        final outcomes = await parseBounded([
          for (final header in headers.values) withPrefix(header),
        ], names: headers.keys.toList());

        expect(outcomes, [
          for (final _ in headers.keys)
            'FormatException: the header is cut short',
        ]);
      },
    );

    test('200 seeded corruptions of the real headers each end quickly in a '
        'result or a FormatException', () async {
      final cases = corruptions(
        {'gpu': gpuHeader, 'npu': npuHeader},
        seed: 20261007,
        count: 200,
      );

      final outcomes = await parseBounded(
        [for (final c in cases) c.file],
        names: [for (final c in cases) c.name],
      );

      for (final (i, outcome) in outcomes.indexed) {
        expect(
          outcome,
          anyOf(startsWith('ok '), startsWith('FormatException: ')),
          reason: cases[i].name,
        );
      }
    });

    test('a few repeated entries still parse (the work stays linear)', () {
      expect(
        parseOutcome(withPrefix(aliasingHeader(objects: 2, items: 2))),
        'ok 1.5.0 [] []',
      );
    });

    test(
      'a header that points every object at one object and every item at '
      'one table fails fast: it repeats its entries (2^30 visits before)',
      () async {
        final header = aliasingHeader(objects: 1 << 15, items: 1 << 15);

        final outcomes = await parseBounded([withPrefix(header)]);

        expect(outcomes, ['FormatException: the header repeats its entries']);
      },
    );

    test('a header that points every item at one long key fails fast: it '
        'repeats its strings (it decoded 512 KB per item)', () async {
      final header = aliasingHeader(
        objects: 1,
        items: 1 << 17,
        keyLength: 1 << 19,
      );

      final outcomes = await parseBounded([withPrefix(header)]);

      expect(outcomes, ['FormatException: the header repeats its strings']);
    });
  });

  group('inspectLitertlm (reads only the header)', () {
    test('a real header: ok, with its sections', () async {
      final file = litertlm(tmp, 'model.litertlm', npuHeader);

      final result = await inspectLitertlm(file.path);

      expect((result as Ok<LitertlmInfo>).value.npuOnly, isTrue);
    });

    test('missing, not a .litertlm, unreadable: an error naming the path', () {
      final missing = '${tmp.path}/gone.litertlm';
      final notLitertlm = File('${tmp.path}/x.litertlm')
        ..writeAsBytesSync(List.filled(64, 1));
      final locked = litertlm(tmp, 'locked.litertlm', gpuHeader);
      Process.runSync('chmod', ['000', locked.path]);
      addTearDown(() => Process.runSync('chmod', ['644', locked.path]));

      return Future.wait([
        inspectLitertlm(missing).then((r) {
          expect('${(r as Error).error}', '$missing does not exist.');
        }),
        inspectLitertlm(notLitertlm.path).then((r) {
          final message = '${(r as Error).error}';
          expect(message, contains('not a .litertlm'));
          expect(message, contains(notLitertlm.path));
        }),
        inspectLitertlm(locked.path).then((r) {
          final message = '${(r as Error).error}';
          expect(message, contains('cannot be read'));
          expect(message, contains(locked.path));
        }),
      ]);
    });
  });

  group('missing or denied: the open tells them apart (no exists() first)', () {
    FileOpener throwing(int errno, String message) =>
        (path) => throw FileSystemException(
          'Cannot open file',
          path,
          OSError(message, errno),
        );

    test('ENOENT: does not exist', () async {
      final r = await inspectLitertlm(
        '/x/gone.litertlm',
        open: throwing(2, 'No such file or directory'),
      );
      final error = (r as Error).error as LocalModelException;
      expect(error.message, '/x/gone.litertlm does not exist.');
      expect(error.permissionDenied, isFalse);
    });

    test(
      'EACCES: cannot be read, permission denied (the advice can fire)',
      () async {
        final r = await inspectLitertlm(
          '/x/locked.litertlm',
          open: throwing(13, 'Permission denied'),
        );
        final error = (r as Error).error as LocalModelException;
        expect(error.message, contains('cannot be read (Permission denied)'));
        expect(error.permissionDenied, isTrue);
      },
    );

    test('a real file in a folder the app may not search: denied, not '
        '"does not exist" (File.exists says false there)', () async {
      final dir = Directory('${tmp.path}/unsearchable')..createSync();
      final file = litertlm(dir, 'model.litertlm', npuHeader);
      Process.runSync('chmod', ['000', dir.path]);
      addTearDown(() => Process.runSync('chmod', ['755', dir.path]));
      expect(file.existsSync(), isFalse, reason: 'the trap');

      final r = await inspectLitertlm(file.path);

      final error = (r as Error).error as LocalModelException;
      expect(error.permissionDenied, isTrue);
      expect(error.message, isNot(contains('does not exist')));
    });
  });

  group('ModelsFolder', () {
    test('created on first use; lists only .litertlm files, by name, with '
        'size and date', () async {
      final dir = Directory('${tmp.path}/models');
      final folder = ModelsFolder(directory: () async => dir);

      expect((await folder.list() as Ok).value, isEmpty);
      expect(dir.existsSync(), isTrue);

      litertlm(dir, 'b.litertlm', npuHeader);
      litertlm(dir, 'a.LITERTLM', gpuHeader);
      File('${dir.path}/notes.txt').writeAsStringSync('x');
      File('${dir.path}/.hidden.litertlm').writeAsStringSync('x');
      Directory('${dir.path}/sub.litertlm').createSync();

      final entries = (await folder.list() as Ok<List<LocalModelEntry>>).value;

      expect(entries.map((e) => e.name), ['a.LITERTLM', 'b.litertlm']);
      expect(entries.first.sizeBytes, gpuHeader.length + 4096);
      expect(entries.first.path, '${dir.path}/a.LITERTLM');
      expect(
        entries.first.modified.difference(DateTime.now()).inMinutes.abs(),
        lessThan(5),
      );
    });

    test('a folder the app does not create and may not read is an error with '
        'the reason, not an exception', () async {
      final parent = Directory('${tmp.path}/locked')..createSync();
      final dir = Directory('${parent.path}/models')..createSync();
      Process.runSync('chmod', ['000', dir.path]);
      addTearDown(() => Process.runSync('chmod', ['755', dir.path]));
      final folder = ModelsFolder(directory: () async => dir, create: false);

      final r = await folder.list();

      expect('${(r as Error).error}', contains('cannot be read'));
    });

    test('a folder that cannot be created is an error, not a crash', () async {
      final blocker = File('${tmp.path}/file')..writeAsStringSync('x');
      final folder = ModelsFolder(
        directory: () async => Directory('${blocker.path}/models'),
      );

      expect(await folder.list(), isA<Error<List<LocalModelEntry>>>());
    });
  });
}
