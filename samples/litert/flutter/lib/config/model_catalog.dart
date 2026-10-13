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
    show ModelType, PreferredBackend, SttModelType, Tool, TtsModelType;
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show AgentToolNames;

import '../domain/models/chat_model_config.dart';
import '../domain/models/model_id.dart';

/// The [LlmConfig] of every chat model load from the app's own settings
/// (`GEMMA_MODEL_PATH`; a user's own model keeps its saved ones).
//
// 8192, not 4096: agent turns (skills system prompt + tool rounds) plus RAG
// excerpts (~1k tokens kept in history) and images (~283) reset the chat every
// 2–3 turns at 4096. Measured on macOS: +~200 MB GPU (IOAccelerator), similar
// TTFT, one-time ~8 s GPU program cache rebuild on the first load per device.
const kLlmConfig = LlmConfig(
  maxTokens: 8192,
  backend: PreferredBackend.gpu,
  supportImage: true,
  maxNumImages: 1,
);

/// The Gemma 4 E2B file earlier builds downloaded into the model store
/// (`<store>/gemma4E2b/gemma-4-E2B-it.litertlm`, verified against the
/// manifest and recorded in `.sha256`). After an upgrade it is adopted in
/// place as the chat model ([kDefineChatModel]'s settings) instead of being
/// asked for again; its folder is kept while it is the chosen file.
const kRetiredGemmaFolder = 'gemma4E2b';
const kRetiredGemmaFile = 'gemma-4-E2B-it.litertlm';
const kRetiredGemmaBytes = 2538799104;
const kRetiredGemmaSha256 =
    '2c902a8c1c7675ec57f51020f01571d765af2ca84859e8c8fcf663a56e6e587a';

/// Gemma 4 E2B's settings, for `GEMMA_MODEL_PATH`.
const kDefineChatModel = ChatModelConfig(
  name: 'Gemma 4 E2B',
  modelType: ModelType.gemma4,
  llm: kLlmConfig,
  tools: true,
);

/// The sampler every chat and the warm-up use. On `.litertlm` the first
/// session's sampler can stay in effect for later ones, so profiles never
/// vary it.
final class const SamplerConfig({
  required final double temperature,
  required final int topK,
});

/// Speech-to-text install and load arguments, one per recognizer
/// ([kSttConfigs]). The STT model is a process-wide singleton, so only one is
/// loaded at a time: each demo makes its own active on entry
/// (`ModelRepository.activateStt`).
///
/// The files are built into the app (`kBundledSttFiles`).
final class const SttConfig({
  required final SttModelType type,

  /// Whisper's output language; moonshine must get null (any value throws).
  required final String? language,

  /// Requested and actually used: there is nothing below it to fall back to,
  /// and no getter reports it (the speech runtime does not), so it shows as
  /// "requested".
  required final PreferredBackend backend,

  /// Input format: 16 kHz mono PCM16.
  required final int sampleRate,

  /// The audio the model reads at most; longer audio is cut silently, so a
  /// capture longer than this is ended at it.
  required final Duration window,
});

const kWhisperSttConfig = SttConfig(
  type: SttModelType.whisper,
  language: 'en',
  backend: PreferredBackend.cpu,
  sampleRate: 16000,
  window: Duration(seconds: 30),
);

/// moonshine on the CPU: its GPU build loads but transcribes garbage.
const kMoonshineSttConfig = SttConfig(
  type: SttModelType.moonshine,
  language: null, // any value throws for moonshine
  backend: PreferredBackend.cpu,
  sampleRate: 16000,
  window: Duration(seconds: 5),
);

/// Every recognizer the app installs, by model.
const kSttConfigs = <ModelId, SttConfig>{
  ModelId.whisperBase: kWhisperSttConfig,
  ModelId.moonshineTiny: kMoonshineSttConfig,
};

/// A recognizer switch on demo entry warms the model up too. Off, measured
/// in profile on an M4 Pro: back to Whisper without a warm-up switches in
/// 83–176 ms and its first question takes 1794–1813 ms (steady state
/// 1774–1816); with one the switch takes 1706 ms and the first question is
/// no faster. Each model still warms up
/// once at setup.
const kSttWarmUpOnSwitch = false;

/// Text-to-speech install and load arguments. The bundle is built into the
/// app (`kBundledInflectFiles`).
final class const TtsConfig({
  required final TtsModelType type,
  required final PreferredBackend backend,

  /// What the loaded synthesizer must report; playback runs at it.
  required final int sampleRate,
});

const kTtsConfig = TtsConfig(
  type: TtsModelType.inflect,
  backend: PreferredBackend.cpu,
  sampleRate: 24000,
);

/// Embedder install and load arguments. The only argument set
/// ever passed to `installEmbedder` / `getActiveEmbedder`.
final class const EmbedderConfig({
  /// File names (the built-in files' asset keys are
  /// `kBundledEmbedderModel`/`Tokenizer`).
  required final String modelFile,
  required final String tokenizerFile,

  /// What the loaded model must report; setup fails otherwise.
  required final int dimension,

  /// LiteRT embeddings always run on the CPU (the GPU delegate returns zero
  /// vectors); the loaded model must report it.
  required final PreferredBackend backend,
});

const kEmbedderModelFile = 'embeddinggemma-300M_seq512_mixed-precision.tflite';

const kEmbedderConfig = EmbedderConfig(
  modelFile: kEmbedderModelFile,
  tokenizerFile: 'sentencepiece.model',
  dimension: 768,
  backend: PreferredBackend.cpu,
);

const kSampler = SamplerConfig(temperature: 0.6, topK: 40);

/// Long side of every image sent to Gemma, in pixels: picked photos and
/// camera snapshots. Larger inputs cost the same ~270 tokens (the engine
/// resizes into a fixed patch budget) but take longer to encode and to hand
/// over.
const kLlmImageMaxSide = 1024;

/// Context budget guard. Tokens reserved for an image sent this turn:
/// Gemma 4 measured ~270; flutter_edge_ai's `InferenceChat` counts 257.
const kImageTokenAllowance = 288;

/// Chat-template tokens around one turn, reserved by the budget guard.
const kTurnOverheadTokens = 32;

/// Kept free below `maxTokens` by the budget guard: InferenceChat's own trim
/// fires at `maxTokens - tokenBuffer` (256 by default), and it replays history
/// badly on FFI (all history glued onto one message), so the guard must act
/// first.
const kContextHeadroomTokens = 256;

/// Thinking is off in every chat.
const kThinking = false;

/// A chat setting, never a model argument: profiles vary only the system
/// instruction, the reply cap and the tools.
final class const ConversationProfile({
  required final String name,
  required final String systemInstruction,
  required final int maxOutputTokens,

  /// The agent system prompt, with `__SKILLS__` where the skill list
  /// goes. A profile with a template opened with skills runs on
  /// `AgentSession` (tools `loadSkill` + `runIntent`); without one, or with
  /// no skills, it is a plain chat with [systemInstruction].
  final String? skillsTemplate,
});

/// Demo 1: multi-turn voice chat, with skills.
const kVoiceChatProfile = ConversationProfile(
  name: 'voice-chat',
  systemInstruction:
      'You are a helpful on-device assistant. Answer briefly and clearly, '
      'in a few short sentences, because replies are read aloud. Keep each '
      'sentence short and use plain text without markdown.',
  maxOutputTokens: 384,
  skillsTemplate: kSkillsTemplate,
);

/// Demo 1's agent system prompt. The skill list goes where
/// `__SKILLS__` is (name: description per skill).
///
/// Shorter (−29 tokens of prefill on every fresh chat with the
/// tool wording below) and explicit about when NOT to call a tool. Measured
/// over 6 rounds (a fresh chat per turn, macOS): spurious tool calls on
/// plain and photo questions fell from 7/18 to 3/18 (text questions
/// 2/6 → 0/6), each one an extra generation; skill requests still call
/// their intents (12/12). The last
/// sentence is the retrieval policy; live skill questions skip
/// retrieval altogether (`SkillQuestionRouter`).
const kSkillsTemplate =
    'You are a helpful on-device voice assistant. Replies are read aloud: a '
    'few short sentences, plain text.\n'
    'Skills:\n'
    '__SKILLS__\n'
    'Only when a request needs one of these skills, call loadSkill with its '
    'name and follow its instructions. Answer everything else directly with '
    'no tool call: general questions, questions about a picture, chat.\n'
    'Live facts about this device and app (models, backends, accelerators, '
    'memory) and the time come only from skills, never from knowledge-base '
    'excerpts.';

/// The tool declarations of Demo 1's agent chat: `loadSkill` and
/// `runIntent` only, with the agent package's names and parameters
/// (the loop dispatches on the name) but our own descriptions: the
/// package's `runIntent` ("Run a native intent … to interact with
/// the device") invited calls with invented intents for plain questions.
const List<Tool> kAgentTools = [
  Tool(
    name: AgentToolNames.loadSkill,
    description:
        "Returns a listed skill's instructions. Call it only when the "
        'request needs that skill.',
    parameters: {
      'type': 'object',
      'properties': {
        'skillName': {
          'type': 'string',
          'description': 'The name of the skill to load.',
        },
      },
      'required': ['skillName'],
    },
  ),
  Tool(
    name: AgentToolNames.runIntent,
    description:
        "Runs an intent that a loaded skill's instructions name. Never call "
        'it for anything else.',
    parameters: {
      'type': 'object',
      'properties': {
        'intent': {'type': 'string', 'description': 'The intent to run.'},
        'parameters': {
          'type': 'string',
          'description': 'The intent parameters as a JSON string.',
        },
      },
      'required': ['intent', 'parameters'],
    },
  ),
];

/// Generations per agent turn (`maxIterations` counts generations). A skill
/// call takes three: loadSkill, runIntent, the answer.
const kAgentMaxIterations = 5;

/// Context budget for an agent turn, on top of the plain
/// guard: the tool rounds a skill call takes before the final answer.
const kAgentToolRounds = 2;

/// Per tool round: the call the model writes (`tool_calls` JSON) plus the
/// template around the call and its response.
const kToolCallTokens = 96;

/// Per tool round: a `runIntent` result (one or two sentences).
const kToolResultTokens = 96;

/// The two tool declarations (`loadSkill`, `runIntent`) the runtime renders
/// into the first prefill.
const kToolDeclarationTokens = 256;

/// The largest SKILL.md the scan accepts, in tokens: `loadSkill`
/// puts the whole file into the turn, and the budget guard reserves room for
/// the largest skill on every agent turn — a 3000-token skill would make even
/// "hello" too long. The bundled skills are under 300.
const kMaxSkillTokens = 600;

/// Demo 3: one stateless question about one camera frame (the chat is reset
/// after each detailed turn).
const kCameraProfile = ConversationProfile(
  name: 'camera',
  systemInstruction:
      'You answer one question about one camera frame. Use what the image '
      'shows; a detector list, when given, may be incomplete. Answer in one '
      'or two short sentences, because the answer is read aloud.',
  maxOutputTokens: 160,
);
