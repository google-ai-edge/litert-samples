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
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show Skill, SkillMdParseException, SkillType, parseSkillMd;
import 'package:path_provider/path_provider.dart';

import '../../../config/model_catalog.dart';
import '../../../domain/models/skill_catalog.dart';
import '../../../domain/skills/app_intents.dart';

/// The skills that ship with the app: folder name → SKILL.md text.
abstract interface class BundledSkillSource {
  Future<Map<String, String>> load();
}

/// `assets/skills/<name>/SKILL.md`, found through the asset manifest (each
/// folder is listed in `pubspec.yaml`).
final class AssetBundledSkills implements BundledSkillSource {
  AssetBundledSkills([AssetBundle? bundle]) : _bundle = bundle ?? rootBundle;

  final AssetBundle _bundle;
  static final _key = RegExp(r'^assets/skills/([^/]+)/SKILL\.md$');

  @override
  Future<Map<String, String>> load() async {
    final manifest = await AssetManifest.loadFromAssetBundle(_bundle);
    return {
      for (final key in manifest.listAssets())
        if (_key.firstMatch(key) case final match?)
          match[1]!: await _bundle.loadString(key, cache: false),
    };
  }
}

/// The skills directory users can write to:
/// `Documents/skills` on Apple platforms (Finder or the Files app; macOS:
/// `~/Library/Containers/com.google.ai.edge.examples.litertEdgeDemos/Data/Documents/
/// skills`), the app's external files directory on Android (`adb push`; the
/// internal one is not writable by adb). No fallback: when Android has no
/// external storage the directory is unavailable and the Skills sheet says
/// so.
Future<Directory> defaultSkillsDirectory() async {
  if (Platform.isAndroid) {
    final external = await getExternalStorageDirectory();
    if (external == null) {
      throw const SkillStoreException(
        'External storage is not available, so there is no skills folder',
      );
    }
    return Directory('${external.path}/skills');
  }
  final documents = await getApplicationDocumentsDirectory();
  return Directory('${documents.path}/skills');
}

/// A conservative token estimate for a SKILL.md: Gemma averages about four
/// characters per token on English prose and fewer on JSON and markdown, so
/// three per token overestimates; a skill accepted here fits [kMaxSkillTokens]
/// for real. The budget guard measures the accepted ones with the model's
/// tokenizer.
int estimateSkillTokens(String text) =>
    (utf8.encode(text).length / _charsPerToken).ceil();

const _charsPerToken = 3;

/// The skills directory cannot be used.
final class SkillStoreException implements Exception {
  const SkillStoreException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// What [SkillStoreService.seed] did.
final class const SeedReport({
  /// Folders created from the bundle (absent before).
  required final List<String> created,

  /// Unedited bundled skills replaced by a newer bundled version.
  required final List<String> updated,

  /// Bundled skills the user edited: left alone.
  required final List<String> keptEdits,

  /// Bundled skills seeded before that the user deleted: not re-created.
  final List<String> stayedDeleted = const [],

  /// Skills an earlier version bundled and this one does not (camera-watch
  /// and timer), still as seeded: deleted, so they do
  /// not linger as broken files naming intents the app no longer has. An
  /// edited one is kept (and listed with its errors).
  final List<String> retired = const [],

  /// The bundle was the one seeded last time: nothing was touched.
  final bool skipped = false,
});

/// Reads and seeds the runtime skills directory. No state:
/// [SkillRepository] holds the latest scan.
class SkillStoreService {
  SkillStoreService({
    Future<Directory> Function()? directory,
    BundledSkillSource? bundled,
    this.maxBytes = 4 * 1024,
    this.maxTokens = kMaxSkillTokens,
    this._knownIntents = AppIntent.all,
  }) : _directory = directory ?? defaultSkillsDirectory,
       _bundled = bundled ?? AssetBundledSkills();

  final Future<Directory> Function() _directory;
  final BundledSkillSource _bundled;
  final Set<String> _knownIntents;

  /// Larger files are refused unread (a skill is a short page).
  final int maxBytes;

  /// A skill whose text is estimated above this many tokens is refused: it
  /// would sit in every agent turn's budget reserve.
  final int maxTokens;

  static const _seedFile = '.seed-version';

  /// Seeds the folder from the bundle once per bundle version.
  /// `.seed-version` records the bundle's hash and what was copied:
  /// - the same bundle as last time: nothing is touched, so a bundled skill
  ///   the user deleted stays deleted;
  /// - first launch or an app update with a changed bundle: a skill never
  ///   seeded before is created (a folder the user made under that name is
  ///   kept); one seeded before and since deleted stays deleted; one still
  ///   as seeded is replaced by the new version; one the user edited is
  ///   kept; one no longer bundled is deleted if still as seeded (retired).
  Future<SeedReport> seed() async {
    final dir = await _directory();
    await dir.create(recursive: true);
    final bundled = await _bundled.load();
    final record = await _readSeedRecord(dir);
    final bundleHash = _bundleHash(bundled);
    if (record.bundle == bundleHash) {
      debugPrint('[Skills] seed: bundle unchanged, nothing to do');
      return const SeedReport(
        created: [],
        updated: [],
        keptEdits: [],
        skipped: true,
      );
    }
    final created = <String>[];
    final updated = <String>[];
    final kept = <String>[];
    final stayedDeleted = <String>[];
    final retired = <String>[];
    // Names seeded before stay recorded (a deleted one is never re-created).
    final next = Map.of(record.seeded);
    for (final MapEntry(key: name, value: seededHash)
        in record.seeded.entries) {
      if (bundled.containsKey(name)) continue;
      final file = File('${dir.path}/$name/SKILL.md');
      if (!await file.exists()) {
        next.remove(name);
      } else if (_hash(await file.readAsBytes()) == seededHash) {
        await file.delete();
        // The folder too, unless the user put something else in it.
        if (await file.parent.list().isEmpty) await file.parent.delete();
        retired.add(name);
        next.remove(name);
      } else {
        kept.add(name);
      }
    }
    for (final MapEntry(key: name, value: text) in bundled.entries) {
      final bundledHash = _hash(utf8.encode(text));
      final file = File('${dir.path}/$name/SKILL.md');
      final seededHash = record.seeded[name];
      if (!await file.exists()) {
        if (seededHash != null) {
          stayedDeleted.add(name);
          continue;
        }
        await file.parent.create(recursive: true);
        await file.writeAsString(text, flush: true);
        created.add(name);
        next[name] = bundledHash;
        continue;
      }
      final onDisk = _hash(await file.readAsBytes());
      if (onDisk == bundledHash) {
        next[name] = bundledHash;
      } else if (seededHash != null && onDisk == seededHash) {
        // Still what an earlier version seeded: not an edit, so update.
        await file.writeAsString(text, flush: true);
        updated.add(name);
        next[name] = bundledHash;
      } else {
        kept.add(name);
        next[name] = seededHash ?? bundledHash;
      }
    }
    await File('${dir.path}/$_seedFile').writeAsString(
      jsonEncode({'version': 2, 'bundle': bundleHash, 'seeded': next}),
      flush: true,
    );
    debugPrint(
      '[Skills] seeded ${dir.path}: created=$created updated=$updated '
      'kept_edits=$kept stayed_deleted=$stayedDeleted retired=$retired',
    );
    return SeedReport(
      created: created,
      updated: updated,
      keptEdits: kept,
      stayedDeleted: stayedDeleted,
      retired: retired,
    );
  }

  /// Reads `skills/*/SKILL.md` and `skills/*.md`. Every file that is not a
  /// usable skill becomes a [SkillLoadError]: a parse error (also a
  /// non-text front-matter value), bad UTF-8, over [maxBytes], a
  /// duplicate name, a JS or MCP skill (no executor here), or an intent
  /// skill naming no intent or one outside [AppIntent.all]. Never throws.
  Future<SkillCatalog> scan() async {
    final Directory dir;
    try {
      dir = await _directory();
      await dir.create(recursive: true);
    } catch (e) {
      debugPrint('[Skills] no skills directory: $e');
      return SkillCatalog(
        directory: null,
        fingerprint: 'unavailable',
        storeError: e.toString(),
      );
    }
    final skills = <LoadedSkill>[];
    final errors = <SkillLoadError>[];
    final hashInput = BytesBuilder(copy: false);
    final byName = <String, String>{};
    for (final candidate in await _candidates(dir)) {
      final path = candidate.relative;
      final file = candidate.file;
      if (file == null) {
        errors.add(
          SkillLoadError(path: path, message: 'This folder has no SKILL.md'),
        );
        hashInput.add(utf8.encode('$path\u0000-\u0000'));
        continue;
      }
      // The size first: an oversized file is refused unread. Its
      // size and modification time stand in for its bytes in the
      // fingerprint, so replacing it still counts as a change.
      final int size;
      final Uint8List bytes;
      try {
        size = await file.length();
        if (size > maxBytes) {
          final modified = await file.lastModified();
          hashInput.add(utf8.encode('$path\u0000$size\u0000$modified'));
          errors.add(
            SkillLoadError(
              path: path,
              message: 'Larger than ${maxBytes ~/ 1024} KB ($size bytes)',
            ),
          );
          continue;
        }
        bytes = await file.readAsBytes();
      } catch (e) {
        errors.add(SkillLoadError(path: path, message: 'Unreadable: $e'));
        continue;
      }
      hashInput
        ..add(utf8.encode('$path\u0000${bytes.length}\u0000'))
        ..add(bytes);
      switch (_check(bytes, byName)) {
        case (final Skill skill, null):
          byName[_normalized(skill.name)] = path;
          skills.add(LoadedSkill(skill: skill, path: path));
        case (_, final String message):
          errors.add(SkillLoadError(path: path, message: message));
      }
    }
    final catalog = SkillCatalog(
      directory: dir.path,
      skills: skills,
      errors: errors,
      fingerprint: _hash(hashInput.takeBytes()),
    );
    debugPrint(
      '[Skills] scan ${dir.path}: ${skills.length} ok '
      '(${skills.map((s) => s.skill.name).join(', ')}), '
      '${errors.length} error(s)'
      '${errors.isEmpty ? '' : ': ${errors.map((e) => '${e.path}: ${e.message}').join(' | ')}'}',
    );
    return catalog;
  }

  /// The skill in [bytes], or why it is not one.
  (Skill?, String?) _check(Uint8List bytes, Map<String, String> byName) {
    // The file may have grown between the length check and the read.
    if (bytes.length > maxBytes) {
      return (
        null,
        'Larger than ${maxBytes ~/ 1024} KB (${bytes.length} bytes)',
      );
    }
    final String text;
    try {
      text = utf8.decode(bytes);
    } on FormatException {
      return (null, 'Not valid UTF-8 text');
    }
    final tokens = estimateSkillTokens(text);
    if (tokens > maxTokens) {
      return (
        null,
        'Too long for a skill: about $tokens tokens, at most $maxTokens '
            '(about ${maxTokens * _charsPerToken} characters); the whole file '
            'goes into the conversation when the skill loads',
      );
    }
    final Skill skill;
    try {
      skill = parseSkillMd(text);
    } on SkillMdParseException catch (e) {
      return (null, e.errors.join(' '));
    } catch (e) {
      // `name: 123` is a TypeError inside the parser, not a parse error.
      return (
        null,
        'Unreadable front matter: quote name and description as text ($e)',
      );
    }
    if (byName[_normalized(skill.name)] case final other?) {
      return (null, 'Duplicate name "${skill.name}" (also in $other)');
    }
    switch (skill.type) {
      case SkillType.js || SkillType.mcp:
        return (
          null,
          'Calls run_${skill.type.name}; this app runs only run_intent and '
              'instruction-only skills',
        );
      case SkillType.intent:
        final named = intentsNamedIn(skill.instructions);
        if (named.isEmpty) {
          return (
            null,
            'Calls run_intent but names no intent: write intent `current_time` '
                '(one of ${_knownIntents.join(', ')})',
          );
        }
        final unknown = named.difference(_knownIntents);
        if (unknown.isNotEmpty) {
          return (
            null,
            'Unknown intent ${unknown.map((i) => '`$i`').join(', ')}; this app '
                'has ${_knownIntents.join(', ')}',
          );
        }
      case SkillType.textOnly:
        break;
    }
    return (skill, null);
  }

  /// Every skill file in [dir], sorted by path: `<folder>/SKILL.md` (case
  /// insensitive; a folder without one is listed too) and top-level `*.md`.
  /// Hidden entries are skipped.
  Future<List<_Candidate>> _candidates(Directory dir) async {
    final out = <_Candidate>[];
    await for (final entity in dir.list(followLinks: false)) {
      final name = entity.uri.pathSegments.where((s) => s.isNotEmpty).last;
      if (name.startsWith('.')) continue;
      if (entity is Directory) {
        File? skillFile;
        await for (final child in entity.list(followLinks: false)) {
          if (child is File &&
              child.uri.pathSegments.last.toLowerCase() == 'skill.md') {
            skillFile = child;
            break;
          }
        }
        out.add(
          _Candidate(
            relative: skillFile == null
                ? '$name/'
                : '$name/${skillFile.uri.pathSegments.last}',
            file: skillFile,
          ),
        );
      } else if (entity is File && name.toLowerCase().endsWith('.md')) {
        out.add(_Candidate(relative: name, file: entity));
      }
    }
    return out..sort((a, b) => a.relative.compareTo(b.relative));
  }

  Future<_SeedRecord> _readSeedRecord(Directory dir) async {
    const none = _SeedRecord(bundle: null, seeded: {});
    final file = File('${dir.path}/$_seedFile');
    if (!await file.exists()) return none;
    try {
      final Object? json = jsonDecode(await file.readAsString());
      if (json case {'seeded': final Map<String, Object?> seeded}) {
        return _SeedRecord(
          // Version 1 records had no bundle hash: a changed bundle.
          bundle: switch (json) {
            {'bundle': final String bundle} => bundle,
            _ => null,
          },
          seeded: {
            for (final MapEntry(:key, :value) in seeded.entries)
              if (value is String) key: value,
          },
        );
      }
    } on FormatException catch (e) {
      debugPrint('[Skills] unreadable $_seedFile ($e): treating as absent');
    }
    return none;
  }

  /// The bundle's version: a hash over every bundled skill's name and text.
  static String _bundleHash(Map<String, String> bundled) {
    final names = bundled.keys.toList()..sort();
    final input = BytesBuilder(copy: false);
    for (final name in names) {
      input
        ..add(utf8.encode('$name\u0000'))
        ..add(utf8.encode('${bundled[name]}\u0000'));
    }
    return _hash(input.takeBytes());
  }

  /// Like `SkillRegistry.get`: case and `_` vs `-` do not make a new name.
  static String _normalized(String name) =>
      name.trim().toLowerCase().replaceAll('_', '-');

  static String _hash(List<int> bytes) => sha256.convert(bytes).toString();
}

final class const _SeedRecord({
  /// The bundle hash seeded last; null on first launch or an old record.
  required final String? bundle,
  required final Map<String, String> seeded,
});

final class const _Candidate({
  required final String relative,
  required final File? file,
});
