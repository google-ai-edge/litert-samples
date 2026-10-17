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

import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show ModelType, PreferredBackend;

import '../../../domain/models/chat_model.dart';
import '../../../utils/redact_url.dart';

/// The persisted form of a [CustomChatModel] (`Settings.customChatModel`):
/// versioned JSON, read strictly.
///
/// Version 1 had `supportImage`; version 2 added `supportAudio`, version 3
/// `encoderBackend` (`cpu` or `npu`). Version 4 dropped the three and had
/// `visionChoice` and `audioChoice` (optional, `cpu` or `npu`): a CPU | NPU
/// choice for a tower the file's header said nothing about, on the NPU.
///
/// Nothing about the towers is saved now: which ones load follows from the
/// file (`LiteRtLmBundleInfo.visionEncoder` / `audioEncoder`), and the engine
/// places each one (flutter_edge_ai_litertlm 1.14.0; a tower backend asked
/// for that the file does not allow is refused). So every tower setting an
/// earlier build saved — version 1 to 3's `supportImage`, `supportAudio`,
/// `encoderBackend`, version 4's `visionChoice` and `audioChoice` — is
/// accepted whatever its value and ignored: never applied again, never a
/// reason to refuse the record. Records are still written as version 4 (an
/// earlier build reads them, without choices).
abstract final class CustomChatModelCodec {
  static const version = 4;

  /// The tower settings of earlier records, read and ignored.
  static const _retiredTowerKeys = {
    'supportImage',
    'supportAudio',
    'encoderBackend',
    'visionChoice',
    'audioChoice',
  };

  static String encode(CustomChatModel model) => jsonEncode({
    'v': version,
    'displayName': model.displayName,
    'modelType': model.modelType.name,
    'backend': model.backend.name,
    'maxTokens': model.maxTokens,
    'tools': model.tools,
    'source': switch (model.source) {
      ImportedModelSource(:final pickedPath) => {
        'kind': 'import',
        'pickedPath': pickedPath,
      },
      // Never the user-info, query or fragment: nothing reads the link
      // back to download (a `.part` resumes when the same link is entered
      // again), and a signed link's token must not sit in the settings.
      UrlModelSource(:final url, :final sha256, :final sizeBytes) => {
        'kind': 'url',
        'url': redactUrl(url),
        'sha256': ?sha256,
        'sizeBytes': ?sizeBytes,
      },
      LocalModelSource(:final path) => {'kind': 'local', 'path': path},
    },
    if (model.file case final f?)
      'file': {
        'name': f.name,
        'sizeBytes': f.sizeBytes,
        'sha256': ?f.sha256,
        'checksumMatched': f.checksumMatched,
      },
  });

  /// Strict: anything missing, mistyped or unknown is a [FormatException]
  /// naming it, never a silent default (the Models screen shows it).
  static CustomChatModel decode(String text) {
    final Object? json;
    try {
      json = jsonDecode(text);
    } on FormatException catch (e) {
      throw FormatException('not JSON: ${e.message}');
    }
    final root = _object(json, 'the saved model');
    final savedVersion = root['v'];
    if (savedVersion is! int || savedVersion < 1 || savedVersion > version) {
      throw FormatException(
        'version $savedVersion, this build reads 1 to $version',
      );
    }
    _onlyKeys(root, {
      'v',
      'displayName',
      'modelType',
      'backend',
      'maxTokens',
      'tools',
      'source',
      'file',
      ..._retiredTowerKeys,
    }, 'the saved model');
    final name = _string(root, 'displayName');
    final type = _enum(root, 'modelType', ModelType.values);
    final backend = _enum(root, 'backend', PreferredBackend.values);
    final maxTokens = _int(root, 'maxTokens');
    if (CustomChatModel.contextProblem(maxTokens, backend)
        case final problem?) {
      throw FormatException('maxTokens $maxTokens: $problem');
    }
    final source = _object(root['source'], 'source');
    final CustomModelSource parsedSource;
    switch (source['kind']) {
      case 'import':
        _onlyKeys(source, const {'kind', 'pickedPath'}, 'source');
        parsedSource = ImportedModelSource(_string(source, 'pickedPath'));
      case 'url':
        _onlyKeys(source, const {
          'kind',
          'url',
          'sha256',
          'sizeBytes',
        }, 'source');
        final url = Uri.tryParse(_string(source, 'url'));
        if (url == null || !url.hasAuthority) {
          throw const FormatException('source.url is not an absolute URL');
        }
        final sha = source['sha256'];
        if (sha != null && (sha is! String || !isSha256Hex(sha))) {
          throw const FormatException('source.sha256 is not 64 hex digits');
        }
        final size = source['sizeBytes'];
        if (size != null && (size is! int || size <= 0)) {
          throw const FormatException('source.sizeBytes is not positive');
        }
        parsedSource = UrlModelSource(
          // An earlier build saved the whole link: its secrets are dropped
          // here and with the next save.
          Uri.parse(redactUrl(url)),
          sha256: sha as String?,
          sizeBytes: size as int?,
        );
      case 'local':
        _onlyKeys(source, const {'kind', 'path'}, 'source');
        final path = _string(source, 'path');
        if (!File(path).isAbsolute) {
          throw FormatException('source.path "$path" is not absolute');
        }
        parsedSource = LocalModelSource(path);
      case final other:
        throw FormatException(
          'source.kind "$other" is not import, url or local',
        );
    }
    CustomModelFile? file;
    if (root['file'] != null) {
      final f = _object(root['file'], 'file');
      _onlyKeys(f, const {
        'name',
        'sizeBytes',
        'sha256',
        'checksumMatched',
      }, 'file');
      // A file in place may still be hashing; a stored one always has it.
      final sha = parsedSource is LocalModelSource && f['sha256'] == null
          ? null
          : _string(f, 'sha256');
      if (sha != null && !isSha256Hex(sha)) {
        throw const FormatException('file.sha256 is not 64 hex digits');
      }
      final fileName = _string(f, 'name');
      if (!isPlainFileName(fileName)) {
        throw FormatException('file.name "$fileName" is not a file name');
      }
      final size = _int(f, 'sizeBytes');
      if (size <= 0) throw const FormatException('file.sizeBytes <= 0');
      final matched = f['checksumMatched'];
      if (matched is! bool) {
        throw const FormatException('file.checksumMatched is not a bool');
      }
      file = CustomModelFile(
        name: fileName,
        sizeBytes: size,
        sha256: sha,
        checksumMatched: matched,
      );
    }
    final tools = root['tools'];
    if (tools is! bool) {
      throw const FormatException('tools must be a bool');
    }
    return CustomChatModel(
      displayName: name,
      source: parsedSource,
      file: file,
      modelType: type,
      backend: backend,
      maxTokens: maxTokens,
      tools: tools,
    );
  }

  static Map<String, Object?> _object(Object? value, String where) =>
      value is Map<String, Object?>
      ? value
      : throw FormatException('$where is not an object');

  static void _onlyKeys(
    Map<String, Object?> json,
    Set<String> allowed,
    String where,
  ) {
    for (final key in json.keys) {
      if (!allowed.contains(key)) {
        throw FormatException('$where: unknown key "$key"');
      }
    }
  }

  static String _string(Map<String, Object?> json, String key) =>
      switch (json[key]) {
        final String s when s.trim().isNotEmpty => s,
        _ => throw FormatException('$key is not a non-empty string'),
      };

  static int _int(Map<String, Object?> json, String key) => switch (json[key]) {
    final int i => i,
    _ => throw FormatException('$key is not an integer'),
  };

  static T _enum<T extends Enum>(
    Map<String, Object?> json,
    String key,
    List<T> values,
  ) {
    final name = json[key];
    for (final value in values) {
      if (value.name == name) return value;
    }
    throw FormatException(
      '$key "$name" is not one of '
      '${values.map((v) => v.name).join(', ')}',
    );
  }
}
