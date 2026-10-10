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

import 'package:flutter/foundation.dart';
import 'package:large_file_handler/large_file_handler.dart';
import 'package:path_provider/path_provider.dart';

import '../../../domain/models/detector_spec.dart' show kDetModelBytes;
import '../../../domain/models/model_id.dart';
import '../../../utils/result.dart';
import 'checksum_record.dart';
import 'model_file_ops.dart';

/// A file of a model built into the app: its asset key, size and SHA-256.
final class const BundledFile({
  required final String asset,
  required final int sizeBytes,
  required final String sha256,
}) {
  /// The file name (the asset key's last segment).
  String get name => asset.split('/').last;
}

/// EmbeddingGemma-300M, built in (Gemma Terms of Use; NOTICE.md).
const kBundledEmbedderModel = BundledFile(
  asset: 'assets/models/embeddinggemma-300M_seq512_mixed-precision.tflite',
  sizeBytes: 179132472,
  sha256: 'ad09e81557203cb0e177abf9bf8727dfe138a7d394aa0f70f0b2ed16432e121a',
);
const kBundledEmbedderTokenizer = BundledFile(
  asset: 'assets/models/sentencepiece.model',
  sizeBytes: 4683319,
  sha256: 'd6daa52d93d7aad10e8388bd526c4e501d914b47177398d1d9621f1fe48438c7',
);

/// Whisper base int8 (litert-community/whisper-base, Apache-2.0) and its
/// tokenizer (openai/whisper-base, Apache-2.0), built in (NOTICE.md).
const kBundledWhisperModel = BundledFile(
  asset: 'assets/models/whisper_base_30s_i8.tflite',
  sizeBytes: 77012960,
  sha256: 'f6943d9d293138850b729e074057956c664891c57837692b1bac4608c4506cd1',
);
const kBundledWhisperTokenizer = BundledFile(
  asset: 'assets/models/whisper_base_tokenizer.json',
  sizeBytes: 2480466,
  sha256: '27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566',
);

/// moonshine-tiny f32 (litert-community/moonshine-tiny, MIT) and its
/// tokenizer (moonshine-ai/moonshine, MIT), built in (NOTICE.md).
const kBundledMoonshineModel = BundledFile(
  asset: 'assets/models/moonshine_tiny_5s_f32.tflite',
  sizeBytes: 109373140,
  sha256: '16f281f1d3d23124e6adbdead8730d46f97cd105c299a9b94608d033c1151b12',
);
const kBundledMoonshineTokenizer = BundledFile(
  asset: 'assets/models/moonshine_tiny_tokenizer.json',
  sizeBytes: 1985534,
  sha256: 'ed2324b3f699d8ba18a4030f33ba205d50b30ff46ff73f2b7d22661cc850efb4',
);

/// Each built-in recognizer's model and tokenizer.
const kBundledSttFiles =
    <ModelId, ({BundledFile model, BundledFile tokenizer})>{
      ModelId.whisperBase: (
        model: kBundledWhisperModel,
        tokenizer: kBundledWhisperTokenizer,
      ),
      ModelId.moonshineTiny: (
        model: kBundledMoonshineModel,
        tokenizer: kBundledMoonshineTokenizer,
      ),
    };

/// Inflect-nano-v2, the TTS bundle laid out as `installTts().fromFile`
/// reads it: its two models (litert-community/Inflect-Nano-v2 @5c184d02,
/// Apache-2.0) and the four Matcha G2P files it reuses
/// (litert-community/Matcha-TTS @ee321481, MIT), built in (NOTICE.md).
const kBundledInflectFiles = [
  BundledFile(
    asset: 'assets/models/inflect/inflect_text_encoder_fp16.tflite',
    sizeBytes: 1782268,
    sha256: 'c778f516b8557c69734f4d4308084c411b8d3a4575566385d163209bf504f083',
  ),
  BundledFile(
    asset: 'assets/models/inflect/inflect_decoder_fp16.tflite',
    sizeBytes: 6375948,
    sha256: '155d5072f104248983015e82d1b90968ab2e87bd7f96c4016928fc83231efbd0',
  ),
  BundledFile(
    asset: 'assets/models/inflect/config.json',
    sizeBytes: 2071,
    sha256: '7363e9e4dda1613aebff7005f2e8c0c76d9b0a1cac31de0f3483bef3089c6906',
  ),
  BundledFile(
    asset: 'assets/models/inflect/g2p_dict.txt.gz',
    sizeBytes: 1762038,
    sha256: '5b3493a8cd4d20b72c7b91415afaf3f32335ebd81f349698e1cedc898c59f979',
  ),
  BundledFile(
    asset: 'assets/models/inflect/dp_g2p_matcha_fp16.tflite',
    sizeBytes: 25785872,
    sha256: '6e4b481f6874dfabc32ce73bf6f0ea1ba6ab5986ee6f76a27779364be8a53c73',
  ),
  BundledFile(
    asset: 'assets/models/inflect/g2p_meta.json',
    sizeBytes: 1904,
    sha256: '7b87bfeaaa072be236e8491d771b0cb97cc92c3e5d83e3558fff8849868810f5',
  ),
];

/// Every model file built into this build: what an Android extraction keeps
/// in `bundled/`.
const kBundledModelFiles = [
  kBundledEmbedderModel,
  kBundledEmbedderTokenizer,
  kBundledWhisperModel,
  kBundledWhisperTokenizer,
  kBundledMoonshineModel,
  kBundledMoonshineTokenizer,
  ...kBundledInflectFiles,
];

/// The bytes [files] take in the app.
int bundledBytesOf(Iterable<BundledFile> files) =>
    files.fold(0, (sum, f) => sum + f.sizeBytes);

/// The bytes model [id]'s built-in files take in the app (0 for the chat
/// model: none ships).
int bundledModelBytes(ModelId id) => switch (id) {
  ModelId.yolo26n => kDetModelBytes,
  ModelId.embeddingGemma => bundledBytesOf([
    kBundledEmbedderModel,
    kBundledEmbedderTokenizer,
  ]),
  ModelId.whisperBase || ModelId.moonshineTiny => bundledBytesOf([
    kBundledSttFiles[id]!.model,
    kBundledSttFiles[id]!.tokenizer,
  ]),
  ModelId.inflectNano => bundledBytesOf(kBundledInflectFiles),
  ModelId.chat => 0,
};

/// A built-in model file is not where the build put it, or its copy does not
/// verify. Never replaced by a download behind the user's back.
final class BundledFileException implements Exception {
  const BundledFileException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// Where this platform keeps the Flutter assets as plain files, from the
/// running executable ([executable]: `Platform.resolvedExecutable`); null
/// where they live inside a package (Android: the APK).
///
/// - macOS: `X.app/Contents/MacOS/X` → `X.app/Contents/Frameworks/
///   App.framework/Resources/flutter_assets`;
/// - iOS: `Runner.app/Runner` → `Runner.app/Frameworks/App.framework/
///   flutter_assets`;
/// - Linux and Windows: the bundle's `data/flutter_assets` next to the
///   executable.
///
/// The engine knows this folder (`FlutterDartProject.assetsPath`) but does
/// not hand it to Dart, so the layout of each platform's bundle decides.
String? flutterAssetsDirFor({
  required String operatingSystem,
  required String executable,
}) {
  final sep = operatingSystem == 'windows' ? r'\' : '/';
  final dir = executable.substring(0, executable.lastIndexOf(sep));
  String parent(String path) => path.substring(0, path.lastIndexOf(sep));
  return switch (operatingSystem) {
    'macos' =>
      '${parent(dir)}/Frameworks/App.framework/Resources/flutter_assets',
    'ios' => '$dir/Frameworks/App.framework/flutter_assets',
    'linux' => '$dir/data/flutter_assets',
    'windows' => '$dir\\data\\flutter_assets',
    _ => null,
  };
}

/// Copies the Flutter asset [asset] (its full key, `assets/…`) to the
/// absolute path [target] without loading it into memory.
typedef AssetCopier = Future<void> Function(String asset, String target);

/// [AssetCopier] on Android: large_file_handler 0.5.2 (already in the app
/// through flutter_edge_ai) opens the asset with `AssetManager.open` and copies
/// it on `Dispatchers.IO`. Two quirks of its API, both from its source:
/// - it prepends `assets/` to the name itself
///   (`MethodChannelLargeFileHandler.copyAssetToLocalStorage` and
///   `…WithProgress`) before Kotlin's
///   `getLookupKeyForAsset`, so the key goes in without that prefix (as
///   flutter_edge_ai's `AssetSource.pathForLookupKey` does); the full key would
///   look up `flutter_assets/assets/assets/…` and fail;
/// - it joins the target onto the documents directory
///   (`MethodChannelLargeFileHandler._getLocalFilePath`), so
///   [target] goes in relative to it (`../files/models/…`).
///
/// The copy without progress: the progress variant posts one event per 1 KB
/// read to the main thread (~175 000 for the embedder).
Future<void> largeFileHandlerCopier(String asset, String target) async {
  const prefix = 'assets/';
  if (!asset.startsWith(prefix)) {
    throw ArgumentError.value(asset, 'asset', 'not under $prefix');
  }
  final documents = (await getApplicationDocumentsDirectory()).path;
  await LargeFileHandler().copyAssetToLocalStorage(
    assetName: asset.substring(prefix.length),
    targetPath: relativePath(target, from: documents),
  );
}

/// [path] relative to the directory [from] (both absolute, `/`-separated).
@visibleForTesting
String relativePath(String path, {required String from}) {
  final a = from.split('/').where((s) => s.isNotEmpty).toList();
  final b = path.split('/').where((s) => s.isNotEmpty).toList();
  var common = 0;
  while (common < a.length && common < b.length && a[common] == b[common]) {
    common++;
  }
  return [
    for (var i = common; i < a.length; i++) '..',
    ...b.sublist(common),
  ].join('/');
}

/// Real paths for the built-in model files (flutter_edge_ai's embedder installs
/// from files). Desktop and iOS use the asset files in place, checked by
/// size (the app bundle is signed). Android extracts each one once into
/// `<model store>/bundled/` (a streaming copy to `.extract`, the SHA-256
/// checked in a worker isolate, renamed into place, recorded in `.sha256`),
/// so later launches skip the copy; a copy interrupted before its rename is
/// started again, never used, and a failed one (a full disk) leaves no
/// `.extract` behind. After an extraction, files of earlier builds (names
/// not in [known]) are deleted from `bundled/`.
///
/// flutter_edge_ai's own asset install would copy on every platform, into
/// memory first on desktop, and verify nothing (flutter_edge_ai 2.1.0
/// `AssetSourceHandler.install` in
/// `lib/core/handlers/asset_source_handler.dart`).
class BundledModelFiles {
  BundledModelFiles({
    String? operatingSystem,
    String? executable,
    Future<Directory> Function()? storeRoot,
    AssetCopier? copier,
    this._ops = const ModelFileOps(),
    this.known = kBundledModelFiles,
  }) : _os = operatingSystem ?? Platform.operatingSystem,
       _executable = executable ?? Platform.resolvedExecutable,
       _storeRoot = storeRoot ?? _defaultStoreRoot,
       _copy = copier ?? largeFileHandlerCopier;

  static Future<Directory> _defaultStoreRoot() async =>
      Directory('${(await getApplicationSupportDirectory()).path}/models');

  final String _os;
  final String _executable;
  final Future<Directory> Function() _storeRoot;
  final AssetCopier _copy;
  final ModelFileOps _ops;

  /// This build's bundled files: what an extraction keeps in `bundled/`.
  final List<BundledFile> known;

  /// The file on disk for [file].
  Future<Result<String>> pathOf(BundledFile file) async {
    try {
      final assets = flutterAssetsDirFor(
        operatingSystem: _os,
        executable: _executable,
      );
      if (assets != null) return Result.ok(await _inPlace(assets, file));
      return Result.ok(await _extracted(file));
    } on BundledFileException catch (e) {
      debugPrint('[Bundled] $e');
      return Result.error(e);
    } catch (e, st) {
      debugPrint('[Bundled] ${file.name}: $e\n$st');
      return Result.error(asException(e));
    }
  }

  /// The one directory holding every one of [files] on disk (a bundle a
  /// package installs by directory, Inflect TTS): their asset folder in
  /// place, `bundled/` once extracted on Android.
  Future<Result<String>> directoryOf(List<BundledFile> files) async {
    final dirs = <String>{};
    for (final file in files) {
      switch (await pathOf(file)) {
        case Ok(:final value):
          dirs.add(value.substring(0, value.lastIndexOf('/')));
        case Error(:final error):
          return Result.error(error);
      }
    }
    if (dirs.length != 1) {
      return Result.error(
        BundledFileException(
          'The bundle files are not in one directory: ${dirs.join(', ')}',
        ),
      );
    }
    return Result.ok(dirs.single);
  }

  Future<String> _inPlace(String assets, BundledFile file) async {
    final path = '$assets/${file.asset}';
    final f = File(path);
    if (!await f.exists()) {
      throw BundledFileException(
        '${file.name} is not in this app bundle (looked at $path): the build '
        'is incomplete.',
      );
    }
    final size = await f.length();
    if (size != file.sizeBytes) {
      throw BundledFileException(
        '${file.name} in the app bundle is $size bytes; this build expects '
        '${file.sizeBytes}.',
      );
    }
    debugPrint('[Bundled] ${file.name}: in place at $path');
    return path;
  }

  Future<String> _extracted(BundledFile file) async {
    final dir = await Directory('${(await _storeRoot()).path}/bundled')
        .create(recursive: true);
    final target = File('${dir.path}/${file.name}');
    final record = ChecksumRecord.of(target);
    if (await target.exists() &&
        await target.length() == file.sizeBytes &&
        await ChecksumRecord.read(target) == file.sha256) {
      debugPrint('[Bundled] ${file.name}: extracted before, verified');
      return target.path;
    }
    final temp = File('${target.path}.extract');
    if (await temp.exists()) await temp.delete();
    final watch = Stopwatch()..start();
    try {
      await _copy(file.asset, temp.path);
      final size = await temp.length();
      if (size != file.sizeBytes) {
        throw BundledFileException(
          'Extracting ${file.name} from the app produced $size bytes; this '
          'build expects ${file.sizeBytes}.',
        );
      }
      final hex = await _ops.sha256OfFile(temp.path, onProgress: (_, _) {});
      if (hex != file.sha256) {
        throw BundledFileException(
          'The ${file.name} extracted from the app has SHA-256 $hex; this '
          'build expects ${file.sha256}.',
        );
      }
      if (await record.exists()) await record.delete();
      await temp.rename(target.path);
      await ChecksumRecord.write(target, hex, file.name);
    } finally {
      if (await temp.exists()) await temp.delete();
    }
    debugPrint(
      '[Bundled] ${file.name}: extracted and verified in '
      '${watch.elapsedMilliseconds} ms',
    );
    await _pruneStale(dir, keeping: file);
    return target.path;
  }

  /// Deletes what earlier builds extracted (a renamed or replaced model):
  /// every file in [dir] that is not one of [known] (or [keeping], the file
  /// just extracted), its `.sha256` record or its `.extract` in progress. A
  /// failure to delete is logged, not fatal.
  Future<void> _pruneStale(
    Directory dir, {
    required BundledFile keeping,
  }) async {
    final keep = {
      for (final f in {...known, keeping}) ...[
        f.name,
        '${f.name}.sha256',
        '${f.name}.extract',
      ],
    };
    await for (final entry in dir.list()) {
      final name = entry.uri.pathSegments.lastWhere((s) => s.isNotEmpty);
      if (keep.contains(name)) continue;
      try {
        await entry.delete(recursive: true);
        debugPrint('[Bundled] removed $name (not in this build)');
      } catch (e) {
        debugPrint('[Bundled] could not remove $name: $e');
      }
    }
  }
}
