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

// The chat model choice saved in the app's shared preferences, for tests that
// change it (a manual run on the same machine, same bundle id, shares it).
// A file picked on the Models screen is saved only by Apply
// (`ChatModelRepository.apply`): a test that picks one applies it.

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/chat_model_repository.dart';
import 'package:litert_edge_demos/data/services/settings/settings_store.dart';
import 'package:litert_edge_demos/data/services/settings/typed_settings.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';
import 'package:litert_edge_demos/utils/result.dart';

/// The saved chat model choice and custom model, as they were.
typedef SavedChatModel = ({String? choice, String? custom});

TypedSettings _settings() =>
    TypedSettings(store: SharedPreferencesSettingsStore());

/// Reads the saved choice and custom model, then clears both: the next
/// launch loads `GEMMA_MODEL_PATH` with the app's settings for it, and a
/// local file used in place starts from its own defaults. Put them back
/// with [restoreChatModelSettings].
Future<SavedChatModel> takeChatModelSettings() async {
  final settings = _settings();
  String? read(Result<String?> result) => switch (result) {
    Ok(:final value) => value,
    Error(:final error) => fail(
      'Could not read the chat model setting: $error',
    ),
  };
  final saved = (
    choice: read(await settings.read(Settings.chatModel)),
    custom: read(await settings.read(Settings.customChatModel)),
  );
  for (final setting in [Settings.chatModel, Settings.customChatModel]) {
    if (await settings.clear(setting) case Error(:final error)) {
      fail('Could not clear ${setting.key}: $error');
    }
  }
  debugPrint(
    '[test] chat model settings cleared (were: choice=${saved.choice}, '
    'custom ${saved.custom == null ? 'none' : 'saved'})',
  );
  return saved;
}

/// Picks the `.litertlm` at [path] in place, as the Models screen's models
/// folder and Path… do: it becomes the candidate, which nothing saves or
/// loads until `ChatModelRepository.apply`. Returns it with what the plugin
/// read of it (`readLiteRtLmBundleInfo`; the saved model's own file picked
/// again is no candidate: its saved settings and file then). Fails the test
/// when it cannot be used.
Future<({CustomChatModel model, LiteRtLmBundleInfo bundle})> pickInPlace(
  ChatModelRepository chatModels,
  String path,
) async {
  switch (await chatModels.useLocalFile(path)) {
    case Ok(:final value):
      final state = chatModels.state.value;
      final bundle = (state.candidate?.bundle ?? state.bundle)?.info;
      if (bundle == null) fail('$path: the plugin did not read the file');
      return (model: value, bundle: bundle);
    case Error(:final error):
      fail('$path as a local file: $error');
  }
}

/// Writes back what [takeChatModelSettings] read (clearing what was unset).
Future<void> restoreChatModelSettings(SavedChatModel saved) async {
  final settings = _settings();
  for (final (setting, value) in [
    (Settings.chatModel, saved.choice),
    (Settings.customChatModel, saved.custom),
  ]) {
    final result = value == null
        ? await settings.clear(setting)
        : await settings.write(setting, value);
    if (result case Error(:final error)) {
      debugPrint('[test] could not restore ${setting.key}: $error');
    }
  }
  debugPrint('[test] chat model settings restored');
}
