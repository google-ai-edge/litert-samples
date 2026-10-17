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
    show
        EmbeddingModality,
        ModelType,
        PreferredBackend,
        SttModelType,
        Tool,
        TtsModelType;
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show loadSkillTool, onDemandAgentSystemPrompt, runIntentTool;

import '../domain/models/chat_model_config.dart';
import '../domain/models/model_id.dart';
import 'knowledge_config.dart' show kEmbedderMaxInputTokens;

/// The [LlmConfig] of every chat model load from the app's own settings
/// (`GEMMA_MODEL_PATH`; a user's own model keeps its saved ones and loads
/// the towers its file has).
//
// 8192, not 4096: agent turns (skills system prompt + tool rounds) plus RAG
// excerpts (~1k tokens kept in history) and images (~283) reset the chat every
// 2–3 turns at 4096. Measured on macOS: +~200 MB GPU (IOAccelerator), similar
// TTFT, one-time ~8 s GPU program cache rebuild on the first load per device.
//
// Its towers are fixed, not read from the file: the vision tower on, no
// audio tower; the engine places the vision encoder (the CPU for Gemma 4 E2B,
// whose encoder is not compiled for an NPU). `GEMMA_MODEL_PATH` runs are the
// integration tests' Gemma 4 E2B, which expect images on and voice questions
// transcribed first (integration_test/voice_loop_test.dart), and the
// self-test's "the app's Gemma 4 E2B settings".
const kLlmConfig = LlmConfig(
  maxTokens: 8192,
  backend: PreferredBackend.gpu,
  maxNumImages: 1,
  supportImage: true,
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
  /// The file name: one self-contained `.litertlm` that carries its own
  /// tokenizer (the built-in file's asset key is `kBundledEmbedderModel`).
  required final String modelFile,

  /// What the loaded model must report; setup fails otherwise.
  required final int dimension,

  /// Requested and required: the embedder runs on the CPU (a native
  /// EmbeddingGemma 2 bundle runs on the CPU or, when compiled for one SoC,
  /// on its NPU); the loaded model must report it.
  required final PreferredBackend backend,

  /// The media declared at load (`getActiveEmbedder(supportImage:,
  /// supportAudio:)`), text implied: the loaded model must report each of
  /// them in its `modalities`, or setup fails. Each one loads its encoder.
  final Set<EmbeddingModality> media = const {},

  /// The input limit the loaded model must report (`maxInputTokens`): the
  /// text question cap is derived from it.
  required final int maxInputTokens,
}) {
  /// This config declaring [media] instead (the visual memory's widening).
  EmbedderConfig withMedia(Set<EmbeddingModality> media) => EmbedderConfig(
    modelFile: modelFile,
    dimension: dimension,
    backend: backend,
    media: media,
    maxInputTokens: maxInputTokens,
  );
}

/// EmbeddingGemma 2 740m (text, image and audio towers). The app embeds text
/// and spoken questions for the knowledge base, and camera frames for the
/// visual memory. Setup declares audio only ([kEmbedderConfig]); the image
/// encoder is declared the first time the camera demo starts its visual
/// memory ([kEmbedderMemoryMedia], `EmbedderService.enableImages`): one
/// reload of about 2 s, during which the knowledge base's embeddings wait.
/// It then stays declared until the app exits: flutter_edge_ai's embedder
/// media only widen (narrowing back would be another reload, and the next
/// camera visit a third). Measured on an M4 Pro Mac (2026-10-13, a test app
/// with the embedder alone): declaring images raises the RSS after load and
/// warm-up from 648 MB to 1055 MB and the load from 1.4 s to 2.2 s; the
/// first image adds 75–175 MB more (`EmbedderService.embedImage` logs the
/// RSS around it). Re-measured 2026-10-14 (same machine and test) for the
/// widening itself: the reload 1.9 s and +327 MB process RSS, then the first
/// image +227 MB more (`docs/design/eg2-multimodal.md`). A user who never
/// opens the camera demo, or turns its visual memory off, never pays that.
/// Text vectors do not depend on it, so the knowledge base's index and
/// profile are unchanged.
const kEmbedderModelFile = 'embeddinggemma-2-740m.litertlm';

/// Vision tokens per image for EmbeddingGemma 2
/// (`LiteRtEmbeddingBackend(visionTokensPerImage:)`, 140 or 70): 70 is about
/// half the time of 140 and kept caption → image top-1 at 6/6 in the spike.
/// Only the visual memory embeds images, and its vectors live in memory for
/// one session, so no stored index depends on it.
const kEmbedderVisionTokensPerImage = 70;

/// The embedder as the overlay and the logs name it.
const kEmbedderName = 'EmbeddingGemma 2 740m';

/// Setup's embedder: text and audio (the knowledge base's questions).
const kEmbedderConfig = EmbedderConfig(
  modelFile: kEmbedderModelFile,
  dimension: 768,
  backend: PreferredBackend.cpu,
  media: {EmbeddingModality.audio},
  maxInputTokens: kEmbedderMaxInputTokens,
);

/// The media once the visual memory needs images: setup's plus the image
/// encoder.
const kEmbedderMemoryMedia = {EmbeddingModality.audio, EmbeddingModality.image};

const kSampler = SamplerConfig(temperature: 0.6, topK: 40);

/// Whether the chat model's warm-up also sends a 1 s WAV when it is loaded
/// with its audio encoder, so the first spoken question does not pay for the
/// encoder's first use. Measured by integration_test/gemma_audio_test.dart
/// (M4 Pro, GPU, Gemma 4 E2B, 2026-10-13): without it the first audio turn's
/// time to first token was 355 ms and the second's 185 ms; with it the first
/// was 184 ms, and the warm-up took 1816 ms against 1945 ms without the WAV
/// (no cost above the run-to-run spread). A failed warm-up fails the load.
const kLlmAudioWarmUp = true;

/// Long side of every image sent to Gemma, in pixels: picked photos and
/// camera snapshots. Larger inputs cost the same ~270 tokens (the engine
/// resizes into a fixed patch budget) but take longer to encode and to hand
/// over.
const kLlmImageMaxSide = 1024;

/// Context budget guard. Tokens reserved for an image sent this turn:
/// Gemma 4 measured ~270; flutter_edge_ai's `InferenceChat` counts 257.
const kImageTokenAllowance = 288;

/// Context budget guard. Tokens a recording sent to the chat model counts,
/// per second of audio: Gemma 4's audio encoder makes about 25 a second
/// (LiteRT-LM's audio frames; the same rate EmbeddingGemma 2 reports).
const kAudioTokensPerSecond = 25;

/// Chat-template tokens around one turn, reserved by the budget guard.
const kTurnOverheadTokens = 32;

/// Kept free below `maxTokens` by the budget guard: InferenceChat's own trim
/// fires at `maxTokens - tokenBuffer` (256 by default), and it replays history
/// badly on FFI (all history glued onto one message), so the guard must act
/// first.
const kContextHeadroomTokens = 256;

/// A chat the context budget guard recreated replays the newest turns of the
/// conversation's text mirror that fit this share of the window: the
/// questions and replies as text, no excerpts, images or recordings. A larger
/// share would fill the fresh chat again and make the next turn reset it
/// again. A chat rebuilt for another reason (after a stop with an audio turn
/// in it, after an interrupted skill call) is not capped: it replays every
/// turn that fits beside the next one.
const kHistoryReplayMaxFraction = 0.25;

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

/// Demo 1's agent system prompt: the agent package's
/// `onDemandAgentSystemPrompt` (flutter_edge_ai_agent 0.3.0: call `loadSkill`
/// only when a request needs a listed skill, answer everything else with no
/// tool call), then the app's own two lines. The skill list goes where the
/// package's `__SKILLS__` placeholder is (name: description per skill).
///
/// The package's text is the wording this app measured and handed upstream:
/// over 6 rounds (a fresh chat per turn, macOS), spurious tool calls on plain
/// and photo questions fell from 7/18 to 3/18 (text questions 2/6 → 0/6), each
/// one an extra generation; skill requests still called their intents (12/12).
/// Differences from that measured setup, none measured yet, include these: its
/// first line is now the package's ("You are a helpful assistant.", where the
/// app had "You are a helpful on-device voice assistant." with the read-aloud
/// sentence); the read-aloud sentence comes after the package's text; the tool
/// declarations are now the package's own ([kAgentTools]; the measured run had
/// the app's shorter wording); a photo turn's hint has new wording
/// (`PromptBuilder.directAnswerHint`; the measured run's was "(A picture is
/// attached. Answer directly; no tool is needed.)"); and the measured chat also
/// listed the camera-watch and timer skills, since removed.
///
/// The last line is the retrieval policy, also not measured yet: the measured
/// one named the bundled skills' facts ("Live facts about this device and app
/// (models, backends, accelerators, memory), the time, timers and the camera
/// come only from skills, …"), so a skill added at run time had no such claim
/// on its facts. No skill, intent or topic is named here: the list at
/// `__SKILLS__` is the only place skills appear.
const kSkillsTemplate =
    '$onDemandAgentSystemPrompt\n'
    'Replies are read aloud: a few short sentences, plain text.\n'
    'Facts a skill provides come from that skill, never from knowledge-base '
    'excerpts.';

/// The tool declarations of Demo 1's agent chat: the agent package's
/// `loadSkillTool` and `runIntentTool`, as the package words them. The app
/// registers no JS or MCP executor, so it declares neither `runSkill` nor
/// `runMcp` (the package's `agentTools` has all four).
const List<Tool> kAgentTools = [loadSkillTool, runIntentTool];

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

/// The two tool declarations ([kAgentTools]) the runtime renders into the
/// first prefill. 256 is the figure set for these same package declarations
/// when Demo 1 first declared them (Inc 6), kept: on Gemma 4 E2B (macOS GPU,
/// 2026-10-17) the first prefill with them was 177 tokens longer than without
/// them, the voice-chat system prompt and skill list the same (the app's
/// former shorter wording: 160).
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
