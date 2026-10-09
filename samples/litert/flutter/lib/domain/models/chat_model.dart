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

import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show ModelType, PreferredBackend;

import '../../utils/redact_url.dart';
import 'chat_model_config.dart';

/// Why a model link with a user name or password in it is refused: it
/// would be saved, logged and shown with them.
const kUrlCredentialsMessage =
    'The link has a user name or password in it (user:token@…). Use a link '
    'that needs no login: credentials in it would be saved and shown with '
    'it.';

/// Which LLM the two demos chat with. The app ships no chat model: it comes
/// from a file the user chooses.
enum ChatModelKind {
  /// None chosen yet: the demos stay off until one is.
  none,

  /// The user's own `.litertlm` ([CustomChatModel]).
  custom,
}

/// What an earlier build saved for "Gemma 4 E2B, downloaded by the app"; read
/// as [ChatModelKind.none] with [kBundledChoiceRetiredNote].
const kRetiredBundledChoice = 'bundled';

/// Shown once when the Gemma 4 E2B an earlier build downloaded is adopted
/// in place after an upgrade.
const kEarlierGemmaAdoptedNote =
    'Using the Gemma 4 E2B you downloaded earlier, where it is: the app no '
    'longer downloads it.';

/// Shown once after a saved "Gemma 4 E2B" choice of an earlier build is read.
const kBundledChoiceRetiredNote =
    'Gemma 4 E2B is no longer downloaded by the app: choose a .litertlm (for '
    'example gemma-4-E2B-it.litertlm) from the models folder, a path or a '
    'link.';

/// Where the user's `.litertlm` came from.
sealed class const CustomModelSource();

/// Picked on this device and copied (APFS-cloned on macOS/iOS) into the
/// model store; [pickedPath] is where it was picked from.
final class const ImportedModelSource(final String pickedPath)
    extends CustomModelSource;

/// Downloaded from [url]: the link as it is shown and saved, without its
/// query and fragment ([redactUrl]; a signed link's token), never with a
/// user name or password ([kUrlCredentialsMessage]). Only the download
/// itself used the whole link, and nothing downloads from this one again (a
/// `.part` resumes when the same link is entered again). [sha256] and
/// [sizeBytes] are what the user entered (optional): when given, the
/// download must match them.
final class const UrlModelSource(
  final Uri url, {
  final String? sha256,
  final int? sizeBytes,
}) extends CustomModelSource;

/// A `.litertlm` used where it is (the models folder, or any path the user
/// typed): no copy. [path] is absolute.
final class const LocalModelSource(final String path) extends CustomModelSource;

/// The custom model's file: the verified file in the model store
/// (`<store>/custom/<name>`), or the file in place ([LocalModelSource]).
final class const CustomModelFile({
  required final String name,
  required final int sizeBytes,

  /// Lowercase hex SHA-256 of the whole file: computed when it was stored
  /// and kept next to it in `<name>.sha256`; for a file used in place
  /// computed in the background (null until then, never blocking a load).
  final String? sha256,

  /// True when [sha256] matched a checksum the user entered; false when
  /// it was only computed and recorded (nothing published to compare with).
  required final bool checksumMatched,
});

/// Context lengths the Models screen accepts (`maxTokens`, the whole
/// context window). flutter_edge_ai_litertlm passes an NPU value through
/// untouched (NPU bundles are compiled for one `cache_length`) but raises a
/// GPU/CPU value below 1024 to 1024 (1.9.0 `clampLitertlmContextTokens`,
/// `lib/src/litert_lm_engine.dart`), so GPU/CPU
/// start at 1024 here instead of being silently raised.
const kMinNpuContextTokens = 128;
const kMinGpuCpuContextTokens = 1024;
const kMaxContextTokens = 131072;

/// The model types offered for a custom `.litertlm`, in menu order. On
/// `.litertlm` LiteRT-LM applies the file's own chat template whatever the
/// type (flutter_edge_ai `core/extensions.dart`); the type picks the
/// tool-call format and the thinking-channel parsing. `functionGemma` is
/// left out: its chat wipes the history after every reply.
const kCustomModelTypes = [
  ModelType.gemma4,
  ModelType.gemmaIt,
  ModelType.general,
  ModelType.qwen35,
  ModelType.qwen3,
  ModelType.qwen,
  ModelType.llama,
  ModelType.phi,
  ModelType.deepSeek,
  ModelType.hammer,
];

/// What choosing [type] changes, for the Models screen.
String modelTypeNote(ModelType type) => switch (type) {
  ModelType.gemma4 =>
    'Gemma 4: native tool calls (the runtime gets the tool list).',
  ModelType.gemmaIt =>
    'Gemma 3 / 3n: tools as a JSON prompt; the model answers in text.',
  ModelType.general => 'Plain text; tools as a JSON prompt.',
  ModelType.qwen35 =>
    'Qwen 3.5 / 3.6: thinking off through the template (enable_thinking).',
  ModelType.qwen3 => 'Qwen 3: adds /no_think to every turn (thinking off).',
  ModelType.qwen => 'Qwen: its own text tool format.',
  ModelType.llama => 'Llama: its own text tool format.',
  ModelType.phi => 'Phi: plain text; tools as a JSON prompt.',
  ModelType.deepSeek =>
    'DeepSeek: everything before </think> is dropped from the reply.',
  ModelType.hammer => 'Hammer: tools as a JSON prompt.',
  ModelType.functionGemma =>
    'FunctionGemma: single-turn (history wiped after every reply).',
};

/// A `.litertlm` found in the models folder.
final class const LocalModelEntry({
  required final String path,
  required final String name,
  required final int sizeBytes,
  required final DateTime modified,
});

/// One models folder as listed (`ChatModelRepository.listFolders`): its
/// label, its path once resolved, why it has none or cannot be read, and
/// the `.litertlm` files in it.
final class const ModelsFolderListing({
  required final String label,

  /// Null when the folder could not be resolved or created ([error] says
  /// why).
  final String? path,
  final String? error,
  final List<LocalModelEntry> files = const [],
});

/// The file is not a usable `.litertlm` (missing, unreadable, not the
/// format); [message] names the path.
final class LocalModelException implements Exception {
  const LocalModelException(this.message, {this.permissionDenied = false});

  final String message;

  /// The OS refused to read it (EACCES/EPERM): the folder that holds it may
  /// say what to do (`ModelsFolder.permissionAdvice`).
  final bool permissionDenied;

  @override
  String toString() => message;
}

/// The user's own chat model: the file and how to run it. Persisted as
/// JSON (`CustomChatModelCodec`, in the data layer) next to the
/// [ChatModelKind] choice.
final class const CustomChatModel({
  required final String displayName,
  required final CustomModelSource source,

  /// Null until the file is downloaded or imported and verified.
  final CustomModelFile? file,
  final ModelType modelType = ModelType.gemma4,
  required final PreferredBackend backend,

  /// The context window (`maxTokens`). NPU builds usually have a fixed
  /// compiled length: this must be that length.
  required final int maxTokens,

  /// Load the vision encoder (`supportImage`): off for files without one.
  final bool supportImage = false,

  /// Send tool declarations (skills); off for models whose template has no
  /// tool calls.
  final bool tools = false,
}) {
  /// Why [maxTokens] is not usable with [backend]; null when it is.
  static String? contextProblem(int maxTokens, PreferredBackend backend) {
    final min = backend == PreferredBackend.npu
        ? kMinNpuContextTokens
        : kMinGpuCpuContextTokens;
    if (maxTokens < min) {
      return backend == PreferredBackend.npu
          ? 'The context must be at least $min tokens.'
          : 'On ${backend.name.toUpperCase()} the context must be at least '
                '$min tokens (the engine raises anything smaller to $min).';
    }
    if (maxTokens > kMaxContextTokens) {
      return 'The context must be at most $kMaxContextTokens tokens.';
    }
    return null;
  }

  CustomChatModel copyWith({
    String? displayName,
    CustomModelSource? source,
    CustomModelFile? file,
    ModelType? modelType,
    PreferredBackend? backend,
    int? maxTokens,
    bool? supportImage,
    bool? tools,
  }) => CustomChatModel(
    displayName: displayName ?? this.displayName,
    source: source ?? this.source,
    file: file ?? this.file,
    modelType: modelType ?? this.modelType,
    backend: backend ?? this.backend,
    maxTokens: maxTokens ?? this.maxTokens,
    supportImage: supportImage ?? this.supportImage,
    tools: tools ?? this.tools,
  );

  /// One line: `npu · ctx 1280 · images off · tools off · gemma4`.
  String get settingsLine => [
    backend.name,
    'ctx $maxTokens',
    'images ${supportImage ? 'on' : 'off'}',
    'tools ${tools ? 'on' : 'off'}',
    modelType.name,
  ].join(' · ');

  /// Where the file came from, for the Models screen, the log and the
  /// report (a link without its secrets: [redactUrl]).
  String get sourceLine => switch (source) {
    ImportedModelSource(:final pickedPath) => 'imported from $pickedPath',
    UrlModelSource(:final url) => 'downloaded from ${redactUrl(url)}',
    LocalModelSource(:final path) => 'in place: $path',
  };
}

final _sha256Hex = RegExp(r'^[0-9a-f]{64}$');

/// 64 lowercase hex digits.
bool isSha256Hex(String value) => _sha256Hex.hasMatch(value);

/// A name that stays inside its folder: no path, not hidden, not a
/// temporary (`.part`, `.import`) or record (`.sha256`) name.
bool isPlainFileName(String name) =>
    name.isNotEmpty &&
    !name.contains('/') &&
    !name.contains(r'\') &&
    !name.startsWith('.') &&
    !name.endsWith('.part') &&
    !name.endsWith('.import') &&
    !name.endsWith('.sha256');

/// The context length an NPU build's file name announces (`…_ekv1280_…` is
/// a 1280-token KV cache), or null.
int? contextHintFromFileName(String name) {
  final match = RegExp(r'ekv(\d{3,6})', caseSensitive: false).firstMatch(name);
  return match == null ? null : int.tryParse(match.group(1)!);
}

/// The model type a file name announces (`Qwen3-0.6B…` is Qwen 3,
/// `gemma3-1b…` Gemma 3): the tool-call format and the thinking parsing
/// follow it. A name that says nothing (or another Gemma) is Gemma 4.
ModelType modelTypeFromFileName(String name) {
  final n = name.toLowerCase();
  bool has(String pattern) => RegExp(pattern).hasMatch(n);
  return switch (n) {
    _ when has(r'qwen[-_ ]?3[._-]?[5-9]') => ModelType.qwen35,
    _ when has(r'qwen[-_ ]?3') => ModelType.qwen3,
    _ when has('qwen') => ModelType.qwen,
    _ when has('functiongemma') => ModelType.functionGemma,
    _ when has(r'gemma[-_ ]?3') => ModelType.gemmaIt,
    _ when has('gemma') => ModelType.gemma4,
    _ when has('llama') => ModelType.llama,
    _ when has(r'(^|[^a-z])phi') => ModelType.phi,
    _ when has('deepseek') => ModelType.deepSeek,
    _ when has('hammer') => ModelType.hammer,
    _ => ModelType.gemma4,
  };
}

/// The types whose chats take tool declarations well enough to default
/// tools on: Gemma 4's native tool calls and Gemma 3's JSON prompt.
const kToolsByDefaultTypes = {ModelType.gemma4, ModelType.gemmaIt};

/// What the chat model slot loads next (`ModelRepository`).
sealed class const ChatModelPlan();

/// No chat model is chosen ([note]: why, after a retired saved choice). The
/// slot loads `GEMMA_MODEL_PATH` (a `--dart-define`) with
/// `kDefineChatModel` when it is set, and is unavailable otherwise.
final class const NoChatModelPlan({final String? note}) extends ChatModelPlan;

/// The user's own model: its verified store file and its settings.
final class const CustomChatPlan({
  required final String path,
  required final CustomChatModel model,
}) extends ChatModelPlan {
  /// What `LlmService` loads.
  ChatModelConfig get config => ChatModelConfig(
    name: model.displayName,
    modelType: model.modelType,
    llm: LlmConfig(
      maxTokens: model.maxTokens,
      backend: model.backend,
      supportImage: model.supportImage,
      maxNumImages: 1,
    ),
    tools: model.tools,
  );
}

/// The user's own model is chosen but cannot be loaded ([reason]: the file
/// is missing or changed, the saved settings are unreadable). Never replaced
/// by another model behind the user's back.
final class const ChatPlanBlocked(final String reason) extends ChatModelPlan;
