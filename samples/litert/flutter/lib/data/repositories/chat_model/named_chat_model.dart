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

import 'dart:io';

import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart'
    show readLiteRtLmBundleInfo;
import 'package:path/path.dart' show basename;

import '../../../domain/models/chat_model.dart';
import '../../../domain/models/chat_model_config.dart';
import '../../../utils/result.dart';
import '../../services/llm/npu_availability.dart'
    show currentAbi, npuBuildRefusal;
import '../../services/model_store/models_folder.dart'
    show BundleReader, inspectLitertlm;
import '../chat_model_repository.dart' show ChatModelRepository;

/// What a file named outside the Models screen (`GEMMA_MODEL_PATH`, the
/// self-test's `--gemma`) loads with, and what its file says about itself.
typedef NamedChatModel = ({
  ChatModelConfig config,

  /// What `readLiteRtLmBundleInfo` made of the file; null when it could not
  /// be read (the load says why, or loads it without its header).
  LiteRtLmBundleInfo? bundle,

  /// [config] is the app's defaults for a file compiled for an NPU, not
  /// the given settings.
  bool npuDefaults,
});

/// A file compiled for an NPU vendor this build does not run here
/// ([npuBuildRefusal]): no backend can run it.
final class ChatModelNotHereException implements Exception {
  const ChatModelNotHereException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// What the file at [path] loads with when it is named outside the Models
/// screen: [given] (Gemma 4 E2B's settings, which the integration tests rely
/// on) for any file not compiled for an NPU, or whose header could not be
/// read; for a build compiled for one, the settings the app starts such a
/// file with when it is picked (`ChatModelRepository.defaultsFor`: the NPU,
/// the context its name gives or 4096, tools off, the towers it has), since
/// the plugin runs it nowhere else. A file compiled for an NPU vendor this
/// build does not run on [abi] is refused here, as the Models screen refuses
/// it: [ChatModelNotHereException]. [read] is flutter_edge_ai_litertlm's
/// `readLiteRtLmBundleInfo` (a fake in tests).
Future<Result<NamedChatModel>> namedChatModelFor(
  String path,
  ChatModelConfig given, {
  BundleReader read = readLiteRtLmBundleInfo,
  String? abi,
}) async {
  // A path with no file behind it has nothing to read: the load reports it
  // as it reports any missing file.
  final bundle = !File(path).existsSync()
      ? null
      : switch (await inspectLitertlm(path, read: read)) {
          Ok(:final value) => value.info,
          Error() => null,
        };
  if (bundle?.npuTarget case final target?) {
    if (npuBuildRefusal(target, abi ?? currentAbi()) case final why?) {
      return Result.error(
        ChatModelNotHereException('$path cannot run here: it is $why.'),
      );
    }
  }
  if (bundle == null || !bundle.isCompiledForNpu) {
    return Result.ok((config: given, bundle: bundle, npuDefaults: false));
  }
  final reading = BundleRead(bundle);
  final defaults = ChatModelRepository.defaultsFor(
    CustomModelFile(
      name: basename(path),
      sizeBytes: bundle.fileLength,
      checksumMatched: false,
    ),
    LocalModelSource(path),
    bundle: reading,
  );
  return Result.ok((
    config: CustomChatPlan(path: path, model: defaults, bundle: reading).config,
    bundle: bundle,
    npuDefaults: true,
  ));
}
