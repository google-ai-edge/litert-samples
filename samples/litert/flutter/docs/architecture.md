# Architecture

LiteRT Demos is one Flutter app with two demos on one on-device stack. **Voice chat** is a push-to-talk assistant: it
transcribes the question, streams an answer from a chat model (about a photo, from a built-in knowledge base, or
through an agent skill) and speaks it from the first finished sentence; with a chat model that hears audio, the
recording itself is the question. **Live camera** draws live detector boxes and answers spoken questions: simple ones
from the detections, detailed ones by sending the frame to the chat model, and questions about the past from a visual
memory of the session. Both share the chat model, speech, EmbeddingGemma 2, audio and diagnostics; with the chat model
file on the device, nothing needs a network.

## Layers

The code follows the official Flutter architecture: MVVM with an optional domain layer. Data and domain code are
grouped by layer, UI code by feature.

```
lib/ui/features/<feature>/views        widgets: read view-model state, call Commands
lib/ui/features/<feature>/view_models  ChangeNotifier view models (home, setup, voice_chat, live_camera)
lib/ui/core                            shared widgets (overlay, device card, mic button, box painter), app foreground
        |  calls
lib/domain/use_cases                   turn logic: VoiceAssistant, turn responders, routers, GpuArbiter
lib/domain/models, lib/domain/ports    immutable values; interfaces implemented in data/ or selftest/
lib/domain/<area>                      pure logic per area: audio, hardware, skills (app intents), vision (COCO)
        |  Result<T>
lib/data/repositories                  state per area: models, chat model, conversation, live detection, audio, ...
        |  may throw
lib/data/services/<area>               thin wrappers over plugins and FFI: llm, speech, detector, frames, audio, ...

lib/config    the object graph, dart-defines, tunables        lib/utils  Result, Command, PCM, chunker, gates
lib/selftest  the --selftest diagnostics (not part of the demos)
```

- **Wiring.** `AppDependencies.create()` in `lib/config/dependencies.dart` builds everything once: the native log
  tap, `initEdgeAi()` (`lib/config/bootstrap.dart`), services, repositories, the skill executors, the conversation,
  diagnostics, and the app-scoped `GpuArbiter` and `ChatModelSwitcher`. Repositories reach the UI through `provider`;
  `ModelRepository` only as its read-only `ModelStates` port. Calls that change models (`prepareModels`,
  `activateStt`, `reloadDetector`, the switcher) go explicitly to the view models that need them (`lib/app.dart`).
  Each route builds its view models, each demo its own `VoiceAssistant`; leaving the route disposes them.
- **Commands and results.** User actions are `Command0` / `Command1` (`lib/utils/command.dart`). Services may throw;
  repositories and use cases return `Result` (`lib/utils/result.dart`) and never throw across layers. Streams carry
  typed failure events (such as `AssistantFailed`), except a turn responder's `respond` stream, which throws the turn's
  failure because `VoiceSession` has no other failure channel. Public exception types end in `Exception`.
- **Rates.** View models notify at turn rate. The streamed reply (`partialReply`), the mic level (`inputLevel`) and
  detection frames (`LiveDetectionRepository.frames`) are `ValueListenable`s: their updates are confined to small
  `ValueListenableBuilder` subtrees or go straight to painters through `CustomPaint(repaint:)` (`LevelRingPainter`,
  `DetectionPainter`). `DiagnosticsRepository` coalesces the overlay's snapshot to at most 4 Hz.
- **Ports and model access.** `lib/domain/ports/` holds `KnowledgeRetriever`, `VisualMemoryEmbedder`, `ModelStates`,
  `ChatModelPlanner`, `VoiceDiagnosticsSink`, `ModelFilePicker` and `SelfTestLauncher`. Only the model services
  (`lib/data/services/llm/`, `lib/data/services/speech/`, `lib/data/services/knowledge/embedder_service.dart`) call
  `FlutterEdgeAi.getActive*`
  and `install*`, driven by `ModelRepository`.
- **Lifecycle.** `AppForeground` (`lib/ui/core/app_foreground.dart`) follows the `AppLifecycleListener` in
  `lib/app.dart`. On Android and iOS, leaving the app ends a held push-to-talk press (`VoiceAssistant.cancelCapture()`;
  a turn past its capture goes on) and releases the camera; the live view restarts on return. Desktop keeps both.
  Resume rescans the skills folder; a desktop quit closes everything (`AppDependencies.dispose()`) before exiting.

## Models and where they run

| Role | Model | File(s) | Runtime package | Default backend | Source |
|---|---|---|---|---|---|
| Chat (both demos) | Gemma 4 E2B, or a Qualcomm NPU build of it | the chosen `.litertlm` | `flutter_edge_ai` + `flutter_edge_ai_litertlm` (LiteRT-LM) | NPU if the header (of an imported or downloaded file, its stored copy) marks it NPU-only and the device offers the NPU, else GPU; NPU where offered when the header cannot be read; the user can change it | chosen by the user |
| Speech to text (Voice chat) | Whisper base int8, English, 30 s window | `assets/models/whisper_base_30s_i8.tflite` + tokenizer | `flutter_edge_ai_speech` | CPU | built in |
| Speech to text (Live camera) | moonshine tiny, 5 s window | `assets/models/moonshine_tiny_5s_f32.tflite` + tokenizer | `flutter_edge_ai_speech` | CPU | built in |
| Text to speech | Inflect-nano-v2 (24 kHz) with Matcha's G2P | `assets/models/inflect/` (6 files) | `flutter_edge_ai_speech` | CPU | built in |
| Embeddings | EmbeddingGemma 2 740m, 768 dims, inputs up to 1024 tokens: loaded with its audio encoder (text and spoken questions); its image encoder (70 vision tokens per image) is added when the camera's visual memory first starts | `assets/models/embeddinggemma-2-740m.litertlm` (one file, tokenizer inside) | `flutter_edge_ai` + `flutter_edge_ai_litertlm` (LiteRT-LM's embedding engine); index in `flutter_edge_ai_rag` + `flutter_edge_ai_sqlite` (sqlite-vec) | CPU | built in |
| Detector | YOLO26n, raw head | `assets/models/yolo26n_fp16_rawhead.tflite` | `flutter_litert` (`CompiledModel`) | GPU, strict, fp32; CPU only when chosen | built in |

The built-in files are not in git: `tool/fetch_models.sh` fetches them from the pins in `tool/models.lock` and derives
the detector with `tool/prune_yolo26n_head.py`.

- **The chat model** is the only one the app does not ship. `ChatModelRepository` holds the choice: a `.litertlm` in
  the models folder or at a typed path (used in place), an imported copy (desktop, iOS), or a download. First-file
  defaults (`defaultsFor`): the header (`inspectLitertlm` in `lib/data/services/model_store/models_folder.dart`; for an
  imported or downloaded file, the store's copy) says whether a section is NPU-only (`backend_constraint: npu`), which
  towers the file has and where their sections may run (the towers are not settings: below); the name gives the model
  type (prompt and tool format) and
  the context (`_ekvN_`, else 4096 on the NPU, 8192 otherwise); tools start on only for a Gemma-type file that is not
  NPU-only. An NPU build without the NPU-only mark starts on the GPU until the user picks NPU. A header that cannot be
  read gives the NPU where offered, else the GPU, with no towers and tools off.
  A missing or changed file blocks the slot with the reason; no other model takes its place. With nothing chosen,
  `GEMMA_MODEL_PATH` fills the slot. One LiteRT-LM engine per process: `ChatModelSwitcher` serializes reloads, unloads
  and the in-app self-test.
- **Loading.** `ModelRepository.prepareAll` loads and warms up each model: the chat model first (largest, fails first),
  then Whisper and Inflect, then the optional detector, moonshine and embedder; a required failure stops setup, an
  optional one shows on its row and tile. The STT model is a singleton: each demo activates its recognizer on entry
  (`activateStt`). Built-in files are used in place on desktop and iOS, extracted once and verified on Android.
- **The embedder's media.** Setup loads EmbeddingGemma 2 with text and audio (`kEmbedderConfig`). The first time the
  camera demo starts its visual memory, `ModelRepository.enableEmbedderImages` → `EmbedderService.enableImages` asks
  `flutter_edge_ai` for the same file with audio and images (`kEmbedderMemoryMedia`), which closes the loaded
  embedder and builds one with both: one reload of about 2 s (1.9 s and +327 MB RSS on an M4 Pro Mac). Embedding calls
  made meanwhile wait for the new model and then run on it, so the knowledge base never fails for the swap. The
  images stay until the app exits: media only widen, so narrowing back would cost another reload and the next camera
  visit a third. A failed widening turns the visual memory off with the reason and reloads text and audio if the
  failed reload took them down. Text vectors do not depend on the media, so the knowledge base's index is unchanged.
- **Two LiteRT copies.** `flutter_litert` carries its own LiteRT runtime, and the LiteRT-LM native bundle of
  `flutter_edge_ai_litertlm` contains another. Whisper, moonshine and Inflect run through the LiteRT C API of that
  bundle (`LiteRtBindings` in `flutter_edge_ai_litertlm`) and EmbeddingGemma 2 through its LiteRT-LM embedding engine,
  the same copy as the chat model; only the detector uses `flutter_litert`'s. The two copies coexist; never link both
  statically.
  - Android: LiteRT-LM links LiteRT into its own library; `flutter_litert` loads its separate `libLiteRt.so`.
  - iOS and macOS: each copy is its own image with two-level symbol binding, and each ships a Metal accelerator under
    its own name (LiteRT-LM's is `LiteRtLmMetalAccelerator`), so both GPU paths work side by side.
  - Linux: both packages would install a `libLiteRt.so` of the same name into the bundle's lib folder, and the
    LiteRT-LM bundle loads it with `RTLD_GLOBAL`, so the process has one LiteRT. `tool/flutter_litert/vendor.sh` applies
    `tool/flutter_litert/flutter_litert-3.9.3.patch` into the gitignored `third_party/flutter_litert`
    (`dependency_overrides`): on Linux `flutter_litert` then bundles none and binds to LiteRT-LM's copy, choosing the
    model-loading signature from the loaded library. Unpatched, the detector crashes the process there.
  - Self-test step 5 checks coexistence: the detector's output is bit-identical before and after the chat model ran.

## Voice chat pipeline

```
record (PCM16 16 kHz mono) -> push-to-talk capture -> energy gate -> VoiceSession.custom
  Whisper path (chat model without audio, "Voice → Whisper → Gemma"):
    runTurn -> Whisper -> corrections -> ChatTurnResponder: direct intent | KB excerpts + prompt | prompt (+ photo)
  audio mode (chat model with audio, "Voice → Gemma"):
    runAudioTurn -> ChatTurnResponder.respondToAudio: KB search as audio -> spoken prompt + WAV (+ photo)
  -> chat model (agent chat with skills, or plain) -> text deltas -> screen
                                                   -> sentence splitter -> Inflect -> flutter_soloud playback
  audio mode, after the playback drained: transcript job (Whisper + corrections) -> user bubble + text history
```

1. **Capture.** `record` captures 16 kHz mono PCM16 (Android's `voiceRecognition` source) while the mic button is held
   (`PushToTalkCapture`, `DeviceAudioRepository`), up to the recognizer's window, so the STT never cuts a question.
2. **Energy gate** (`measureVoice` in `lib/utils/pcm.dart`). A press under 300 ms is a slip; all-zero audio is a
   microphone error (macOS hands a blocked app zeros); under 160 ms of voiced 20 ms frames is silence. A frame is
   voiced at the gate (−45 dBFS by default) and 10 dB above the capture's noise floor. None of these runs a model.
3. **Transcription, or none.** `SpeechRepository` builds one `VoiceSession.custom(streamAudio: true)` per turn; it owns
   no microphone and no player: PCM goes in, events come out. Typed questions take the same path. The responder
   decides the mode per turn: when the loaded chat model hears audio (its file has an audio tower), a spoken turn's
   responder has `respondToAudio` and the turn runs as `runAudioTurn` (no STT before the reply; the user bubble is
   "🎤 3.2 s" at
   once). Otherwise Whisper transcribes it first, and `correctVoiceTranscript` (`lib/domain/audio/
   transcript_corrections.dart`) fixes its measured misses of product names, as phrases only ("what is littered" →
   "what is LiteRT"; "littered" alone stays the English word). The screen and
   the overlay show the mode ("Voice → Gemma 4 E2B" / "Voice → Whisper → Gemma 4 E2B"; in audio mode also "no
   skills" when tools are off or none are loaded: there are no words for the rules, so the time and device intents
   are reached only through the agent's tools).
4. **Routing** (`ChatTurnResponder`):
   - Time and live device questions ("which accelerator are you running on?") are matched by rules
     (`SkillQuestionRouter`); the app runs the `current_time` or `device_info` intent itself and speaks the result.
   - Other questions are embedded and searched in the knowledge base: up to 3 chunks at cosine similarity ≥ 0.70
     (`kKbMinSimilarityText`) go into the prompt with numbered sources (`PromptBuilder`). The search takes a
     domain `KnowledgeQuery` (text, or a spoken question as a WAV embedded as audio, with no task prefix, and gated
     at 0.69, `kKbMinSimilarityAudio`); a question over its cap (`kKbMaxTextQueryUtf8Bytes`, `kKbMaxAudioQuery`)
     fails as `KnowledgeQueryTooLongException`, shown as the turn's failed retrieval, never cut. Skill-topic
     questions skip retrieval.
   - The prompt and the attached photo go to `ConversationRepository.ask`. With skills loaded and a chat model with
     tools on, it is an agent chat (`AgentSession` from `flutter_edge_ai_agent`) with two tools, `loadSkill` and
     `runIntent`; otherwise a plain chat.
5. **Knowledge base.** Sixteen Markdown documents in `assets/kb/` are chunked by heading
   (`lib/utils/markdown_chunker.dart`), embedded with EmbeddingGemma 2 (documents with the `retrievalDocument` prefix,
   questions with `retrievalQuery`) and stored in one sqlite-vec index bound to the embedding profile
   `kKbEmbeddingProfileId`, prebuilt by `tool/build_kb_index.sh` into `assets/kb_index/` (304 chunks take about 25 s
   on a Mac's CPU, minutes on a phone). `KnowledgeRepository` installs it on first launch when its key (`kb-index/3`:
   documents, chunker, document prefix, embedder id, the `.litertlm`'s SHA-256, dimension) matches, else indexes on the
   device and logs why. An index of an earlier embedder (another key, another profile, or a store without a profile)
   is never searched: it is deleted and replaced. Chunk metadata becomes citation chips under the reply
   (`lib/ui/features/voice_chat/views/citation_chips.dart`).
6. **Skills.** `SKILL.md` files live in a writable `skills` folder (in the app's documents folder on macOS, iOS and
   Linux, its external files folder on Android), seeded from `assets/skills/` (`current-time`, `device-info`).
   `SkillStoreService` parses them and checks the intents they name against `AppIntent.all`; errors show in the Skills
   sheet. Reload and app resume rescan; a changed set opens a new agent chat between turns (`SkillsApplier`). A skill
   combines the app's intents (run by `AppIntentExecutor`) or is instruction-only; it cannot add code.
7. **Streaming reply.** One chat slot (`createChat`) serves both demos; entering a demo opens it with that demo's
   profile (`kVoiceChatProfile`, `kCameraProfile`). Text deltas update the bubble through `partialReply`. A budget
   guard starts the chat over before the context overflows (a recording counts 25 tokens per second), and before
   `InferenceChat`'s own trim would fire (`kContextHeadroomTokens`). The repository keeps a text mirror of the
   conversation (`HistoryMirror`: the question — "[voice question]" for an audio turn until its transcript arrives —
   and the reply). A chat that must be rebuilt replays the mirror's newest turns as text: after a stop while an audio
   turn is in it (`clearHistory(replayHistory:)`, instead of LiteRT-LM's own rebuild, whose text history has no words
   for an audio question) and after an interrupted skill call every turn that fits beside the next one, after a
   budget reset at most a quarter of the window. The engine's history holds each turn's WAV (≤ ~960 KB per 30 s
   question) until the next rebuild or reset; the context budget bounds that to about six 30 s questions at 8192
   tokens.
   **Transcript job (audio mode).** Once the reply's playback has drained and the turn is idle, Whisper transcribes
   the recording in the background (`TranscriptJob`); the words fill the user bubble in place (by the turn's number)
   and the text mirror. Later presses and turns never drop a pending transcript (Whisper's worker runs them in order,
   and a cancel would save no CPU); only leaving the demo or a switch to another recognizer does. No words show as
   "🎤 3.2 s (no transcript: reason)". A photo is normalized to ≤ 1024 px with its EXIF
   orientation (`lib/data/services/images/image_normalizer.dart`) and re-sent only when the chat can no longer see it.
8. **Speech out.** `VoiceSession` splits the reply at sentence ends (`.`, `!` or `?` before a space, or a newline;
   very short fragments merge into the next sentence), and Inflect synthesizes each sentence as soon as it is
   complete, so playback starts with the first finished sentence. `SpokenSynthesizer` strips citation markers,
   Markdown and URLs from what is spoken (`lib/utils/spoken_text.dart`); `flutter_soloud` plays the PCM stream.
9. **Half-duplex by design.** The mic is closed while a reply plays, so there is no echo to cancel. Pressing the mic
   during a reply is the barge-in: playback stops at once, the partial reply is kept as interrupted, generation is
   stopped and drained in the background, and the next turn starts after the drain.
10. **One audio-session owner.** Only `AudioSessionService` configures the platform session: iOS `.playAndRecord` with
    the default mode (no voice processing), Android speech for an assistant with transient focus, none on Linux.
    `record` does not manage the session, and the camera opens with audio off.

## Live camera pipeline

```
FrameSource (camera | network MJPEG | fixture) -> one-slot gate, <= 15 fps -> detector worker isolate
  (gather + letterbox -> CompiledModel.run -> top-k) -> LiveDetectionRepository.frames -> DetectionPainter
  demo idle, sampler due: the worker's grey-thumbnail diff -> <= 320 px PNG
    -> EmbeddingGemma 2 image vector -> MemoryStore (in memory, 200 frames / 16 MB)
voice question -> moonshine -> CameraTurnResponder -> QuestionRouter
  -> memory:   phrase -> EmbeddingGemma 2 -> judgeMemory -> card + chat model with the frame | template -> Inflect
  -> fast:     template from the detection summary, no LLM                                  -> Inflect
  -> detailed: frozen snapshot -> PNG -> chat model (camera profile)                         -> Inflect
```

- **Frame sources** (`lib/data/services/frames/`, one single-use `FrameSource` interface). `CameraFrameSource` uses
  the `camera` plugin (CameraX with NV21 on Android, AVFoundation with BGRA on iOS, `camera_desktop` on macOS and
  Linux) and requests `ResolutionPreset.high` at 30 fps with audio off; the size and rate it gets depend on the device
  and plugin. `NetworkFrameSource` reads an MJPEG stream over HTTP (such as a phone's IP-camera app), decoding in a
  TurboJPEG worker isolate where `libturbojpeg` exists, else with the engine's codec. `FixtureFrameSource` plays still
  images, for tests. `LiveCameraSettingsRepository` stores the source and detector
  backend; `LiveSettingsApplier` applies one change at a time (a new backend: stop, reload the detector, restart).
- **Latest frame wins.** `LiveDetectionRepository` sends a frame to the detector only when none is in flight, the
  detector is live and the `FrameRateGate` allows it (at most 15 fps); every other frame is dropped, never queued.
  Watchdogs turn a detector that stops answering, or a source that stops sending, into a visible failure with Retry.
- **Detector.** `DetectorService` owns one long-lived worker isolate holding the `CompiledModel`. Per frame
  (`lib/data/services/detector/detector_codec.dart`), a `GatherPlan` built once per stream folds rotation and
  letterbox into lookup tables: the upright frame becomes NCHW float32 `[1,3,640,640]`, RGB / 255, pad 114/255. The
  run returns `[1,8400,84]`: per anchor, box corners in input pixels and 80 sigmoid class scores. Dart keeps every
  (anchor, class) at score ≥ 0.25, takes the top 100 and maps boxes back to frame pixels. No NMS: YOLO26 is trained
  one-to-one. `DetectionPainter` draws at most 50 boxes at score ≥ 0.35.
- **Why the raw head.** Arm's graph ends in a selection head (`TOPK_V2`, `GATHER_ND`, int64 `CAST` / `SELECT` /
  `LESS`) that the GPU accelerator cannot place, so a GPU-only `CompiledModel` fails to compile. The int64 ops also
  left the shared `ADD` opcode at version 4, which the Metal accelerator of LiteRT 2.1.5 (`flutter_litert` on macOS)
  rejects: with `{gpu, cpu}` only 54 of 461 ops ran on the GPU there, at CPU speed. `tool/prune_yolo26n_head.py` cuts
  the graph at the `[1,8400,84]` tensor and sets `ADD` to version 1 (every remaining ADD is float32); the weights are
  unchanged. The whole graph then runs on the GPU, and the selection moves to Dart (about 1 ms).
- **Load checks** (`lib/data/services/detector/detector_engine.dart`), in order, never retried elsewhere: file size;
  a build with exactly `{gpu}` (or `{cpu}` when chosen) at fp32; for the GPU, no fallback and `isFullyAccelerated`;
  input and output sizes; a ramp-input run that matches a CPU reference within 1 % of the reference output's range
  (the TFLite interpreter, else LiteRT's CPU path: `lib/data/services/detector/detector_verification.dart`); a warm-up.
- **Questions.** At release, `CameraTurnResponder` captures the next frame with its own detection (in parallel with
  STT), and `QuestionRouter` applies rules over the COCO vocabulary:
  - Memory (checked first): a cue about the past or about something of the speaker's ("where was …", "where is my
    …", "have you seen …", "when did you see …", "where did I leave …") goes to the visual memory (`MemoryRoute`,
    below) with the noun phrase after it. Not when the words after the cue mark the present ("now", "here", "in
    the picture"), name a body part ("where is my hand?"), start an embedded wh-question ("did you see how many …")
    or are only the speaker or the listener ("where was I?"): the later rules decide those. "Where is the dog?"
    stays a detailed question about the frame.
  - Fast: inventory ("what do you see?"), counts ("how many cups?") and presence ("is there a dog?") when the noun is
    a COCO class. `FastAnswerComposer` answers from the `DetectionSummary` (median count per class over the last 5
    frames at score ≥ 0.4) with a template. No LLM runs and the view stays live.
  - Detailed: everything else (describe, read, colour, position, an unknown noun). The view freezes on the snapshot;
    `SnapshotEncoder` makes a PNG of at most 1024 px, mirrored back when the source mirrors, and it goes to the chat
    model with up to 8 detections as a hint (`buildCameraPrompt`). The chat is reset after each detailed turn, so
    questions are stateless. When the chat model file has no vision tower, the detector's list is spoken instead.
- **GPU arbitration.** `GpuArbiter` follows `ConversationRepository.isGenerating`: while the chat model generates, the
  detector is paused (`kDetectorDuringGeneration`; the frame in flight finishes, the overlay names the chat model) and
  resumes after. `flutter_litert` has no GPU priority, contention would slow the answer, and the view is frozen anyway.

### Visual memory

`VisualMemoryRepository` (`lib/data/repositories/visual_memory_repository.dart`) keeps frames EmbeddingGemma 2
embedded while the demo was idle, so a spoken question can find a scene by meaning, beyond the detector's 80
classes. Tunables are in `lib/config/live_camera_config.dart` (`kMemory*`).

- **Owner and switch.** The camera screen starts it on entry and stops it on leaving, which forgets every frame.
  "Visual memory" in the camera settings (persisted, on by default) turns it off at once: no sampling, no image
  encoder load, the frames forgotten; a memory question then answers "Visual memory is off (camera settings)".
- **Image encoder.** The first start with the switch on asks for the embedder's image media (above); the chip says
  "Memory: preparing…" and nothing is sampled until it is loaded.
- **Sampler** (`MemorySampler`, every 500 ms). A frame is due when at least 3 s and 4 × the last frame embedding
  have passed and the scene changed, or 10 s passed. It is asked for only while live detection runs, the frames are
  not black, no question capture is pending and the demo is idle (no turn, no generation, no playback). A
  busy tick drops the sample and restarts the gap, and so does the first idle tick after a busy one, so the next
  frame comes a full gap after the demo turned idle. Nothing queues; question embeddings always go first.
- **Frames from the worker.** A memory capture is a second, bounded slot next to the question capture
  (`MemoryCaptureSlot`, `SceneCapture`). The detector worker, which already holds the upright frame, compares a
  32 × 32 grey thumbnail with the last stored one and only for a scene change makes a PNG of at most 320 px
  (`lib/data/services/detector/memory_shot.dart`), in its isolate. An unchanged scene once 10 s have passed refreshes
  the newest entry's time in place (no PNG, no embedding), so a still scene never fills the store and "seen N s ago"
  is its latest sighting. A frame that does not come back within 2 s is dropped.
- **Store** (`MemoryStore`). Per entry: time, PNG, detections and the image vector (70 vision tokens,
  `retrievalDocument`). At most 200 entries and 16 MB of PNG, oldest out; cleared on leaving the demo, when a new
  detection session starts (a source change, Retry, a backend switch) and by "Clear memory"; never written to disk.
  Search is a cosine scan in Dart.
- **Acceptance** (`judgeMemory`, `lib/domain/use_cases/memory_acceptance.dart`). The phrase is embedded as text
  with no task prefix. The best frame must score at least 0.62; its rival is the best frame more than 5 s away that
  shows a different scene (image vectors below 0.88 alike: frames of one long sighting agree rather than compete).
  The best beats the rival by at least 0.025 (with no rival, it must clear 0.645): a hit, answered with the newest
  frame of the same scene within 0.01 of the best. A rival within the margin: an ambiguous hit, answered with the
  newest of the competing frames and said as "I may have seen that in more than one place". Otherwise a miss: "I
  haven't seen that in the last N minutes", without the LLM.
- **Answer.** A hit shows a card over the video (the frame and "12 s ago"; tap to close) and goes to the chat model
  with the frame and the question; when the chat model file has no vision tower, or the camera chat not open, a template says
  when it was seen and that frame's detections.
- **Failures.** A failed frame embedding, or a frame the worker could not make, counts in the overlay; three in a row
  turn the memory off with the reason (a banner), and memory questions answer with it. An embedder that is not
  ready turns it off too.
- **UI.** The chip "Memory: 37 frames · 2 min", the result card, and the switch and "Clear memory" in the settings
  sheet (`lib/ui/features/live_camera/views/`).

## Accelerators and diagnostics

The rule: the app never moves a model to another backend on its own. A requested GPU or NPU that cannot be used is an
error the UI shows, with an explicit way out (Run on GPU / Run on CPU, Run detector on CPU), never a CPU fallback.

- **Chat model.** `LlmService.load` (`lib/data/services/llm/llm_service.dart`) checks the NPU gate first
  (`lib/data/services/llm/npu_availability.dart`: on Android, only when `libcdsprpc.so` opens), then loads with exactly
  the requested backend. If `activeBackend` differs, it closes the model and fails with `BackendMismatchException`,
  quoting the reason the package printed for the failed attempt. The Android NPU stack is bundled only because
  `pubspec.yaml` sets `qualcomm_npu: true` for `flutter_edge_ai_litertlm` under `hooks: user_defines`.
- **Chat model towers.** The towers (vision: `tf_lite_vision_encoder` + `tf_lite_vision_adapter`; audio: the
  `tf_lite_audio_*` sections; the `tf_lite_end_of_*` markers do not count) are not settings: every tower the file has
  is loaded (`supportImage` / `supportAudio`), and where each runs (`preferredVisionBackend` / `preferredAudioBackend`)
  is derived every time the configuration is built (`resolveTowers` in `lib/domain/models/model_towers.dart`), from
  the header the repository read last (`ChatModelState.header`; an imported or downloaded file's from the store's copy)
  and the backend, tower by tower: a section declaring `backend_constraint: cpu` → CPU; else `npu` → NPU; else `gpu`
  → GPU (no known tower declares it); else, on the NPU, the user's choice in that tower's row on the card (CPU or NPU,
  NPU by default; saved as `CustomChatModel.visionChoice` / `audioChoice`, codec v4, reset when another file is
  chosen, kept when the same file is picked again and across a backend change, ignored off the NPU); else CPU. A
  declaration always wins over a choice. A header that cannot be read loads no tower. The card's
  read-only Towers block shows the result for the draft backend. A chat-model load that fails at any step, warm-up
  included, with an undeclared tower on the NPU adds a line saying to set it to CPU on the card
  (`TowersOnNpuLoadException`); nothing retries by itself. The official `…_qualcomm_sm8750` build declares nothing and
  its towers fail on the NPU ("Failed to create conversation"); the official `…_qualcomm_qcs8275` build and the
  standard Gemma 4 E2B pin both towers to the CPU. `GEMMA_MODEL_PATH` keeps fixed towers (`kLlmConfig`: vision on the
  CPU, no audio). Measured with all-NPU builds (2026-10-13/14): a spoken question's first token in 0.33–0.44 s on a
  Galaxy S24, S25 Ultra and S26, against 0.87 s on the S24 with Gemma on the GPU and the towers on the CPU.
- **Detector:** the strict accelerator set and the load checks above. **Embedder:** requested on the CPU (the generic
  EmbeddingGemma 2 bundle runs only there; the per-SoC NPU bundles are not used); the app checks the reported
  backend, dimension, media and input cap and closes a model that differs. **Speech:** requested on the CPU; the
  package cannot report the backend, so it shows as "requested".
- **How the real backend is known.** Each model's `AcceleratorEvidence` (`lib/domain/models/accelerator_evidence.dart`)
  tags every fact with its source: `api` (`activeBackend`, or the detector's accelerator set and `isFullyAccelerated`),
  `log`, `inferred` or `requested`. No plugin API reports the GPU API or adapter. As diagnostics only, release builds
  on Linux and macOS redirect native stderr into `<app support>/logs/native.log`
  (`lib/data/services/hardware/native_log_tap.dart`); each GPU-capable load reads its own window, and `parseNativeLog`
  (`lib/domain/hardware/native_log_parser.dart`) finds the API (Metal, WebGPU/Vulkan) and the adapter. Elsewhere both
  are inferred from the platform and hardware probe (`lib/domain/hardware/accelerator_inference.dart`). A software
  rasterizer (llvmpipe, lavapipe, SwiftShader) is an error. The log only labels; it never picks a backend.
- **Where it shows.** The "This device" card on Home and the Models screen (`lib/ui/core/device_card.dart`: chip, GPU,
  RAM, requested → actual backend per model) with Copy diagnostics (`lib/domain/hardware/diagnostics_report.dart`);
  the debug overlay (`lib/ui/core/debug_overlay.dart`: backends, models, fps, latencies; the voice turn's mode, STT
  or "audio → chat model", and the background transcript; the last retrieval's top cosine against its gate; in the
  camera the memory line: frames, span, bytes, last frame embedding, best cosine, failures); the `device_info` skill.
- **URLs.** Stream and download URLs lose user-info, query and fragment (`lib/utils/redact_url.dart`) before they
  reach logs, the screen or Copy diagnostics. A model download URL with user-info is refused; a network camera's user
  and password are sent as Basic auth and saved on the device as typed.
- **Self-test.** `--selftest` (or `SELFTEST=1`; flags in `lib/selftest/self_test_options.dart`) runs instead of the
  app: 1 hardware probe; 2 the detector on exactly the requested backend; 3 a reference image against golden
  detections, ten identical runs; 4 the chat model on its requested backend, warm-up and 64 timed tokens; 5 the
  detector again, bit-identical; 6 a tone out and one second in (unless `--skip-audio`); 7a EmbeddingGemma 2 loaded
  with audio, as setup loads it (in the app, the app's own embedder); 7b a bundled spoken "What time is it?"
  (`assets/selftest/what_time_is_it.wav`) against its words, above `kSelfTestAudioTextMinCosine` (0.73); 7c the image
  encoder added as the camera demo adds it, then the bundled cats photo made into a memory frame by the memory's own
  PNG path (`memoryFramePng`) against "two cats", above `kSelfTestImageTextMinCosine` (0.63). 7b and 7c are skipped
  when 7a fails. The memory table gives each step's process RSS and peak RSS. It exits 0 only when every step
  passed. Run self-test on the Models screen runs the same steps (`lib/selftest/self_test_in_app.dart`).

## Configuration

The dart-defines (`lib/config/env.dart`) are listed in the README's [Configuration](../README.md#configuration)
section; they are compiled into the binary. A bad value is an error the app shows, never ignored: `VOICE_GATE_DBFS` at
startup, `DETECTOR_BACKEND` when the detector loads, `FRAME_SOURCE`, `NETWORK_CAMERA_URL` and `FIXTURE_DIR` when the
Live camera opens its source, and an unreadable `GEMMA_MODEL_PATH` when the chat model loads.

Tunables are constants in `lib/config/`:

| File | Holds |
|---|---|
| `lib/config/voice_config.dart` | minimum hold, silence gate, voiced minimum, playback buffering |
| `lib/config/knowledge_config.dart` | similarity gates (text, audio), question caps, excerpts per turn, prebuilt index assets, embedding profile |
| `lib/config/live_camera_config.dart` | detection fps, summary window, counting score, camera preset, watchdogs, network timeouts, the detector's standard backend, the visual memory (sampler, caps, acceptance) |
| `lib/config/model_catalog.dart` | chat model load settings, STT/TTS/embedder settings (media, vision tokens per image), sampler, prompts, profiles, agent tools, context budget |
| `lib/config/demos.dart` | the models each demo needs and the recognizer it activates |

## Where to start reading

| File | Why |
|---|---|
| `lib/main.dart` | entry point: the `--selftest` branch, then `AppDependencies.create()` and the app |
| `lib/config/dependencies.dart` | the object graph, its build order and its shutdown order |
| `lib/app.dart` | routes, and how each route builds its view model and `VoiceAssistant` |
| `lib/ui/features/voice_chat/view_models/voice_chat_view_model.dart` | Voice chat state, commands, skills and photos |
| `lib/ui/features/live_camera/view_models/live_camera_view_model.dart` | Live camera state: frozen frame, failures, settings |
| `lib/domain/use_cases/voice_assistant.dart` | the push-to-talk turn state machine both demos share |
| `lib/domain/use_cases/chat_turn_responder.dart` | Voice chat's LLM step: direct intents, retrieval, the chat |
| `lib/domain/use_cases/camera_turn_responder.dart` | Live camera's LLM step: snapshot, memory, fast or detailed route |
| `lib/data/repositories/visual_memory_repository.dart` | the visual memory: owner, sampler, store, recall |
| `lib/data/repositories/conversation_repository_edge_ai.dart` | the single chat slot, profiles, agent skills, budget guard |
| `lib/data/repositories/live_detection_repository.dart` | frame source → gate → detector worker → frames |
| `lib/data/services/detector/detector_codec.dart` | YOLO26n letterbox, gather and top-k decode |
| `lib/data/repositories/chat_model_repository.dart` | the chosen `.litertlm`, its header defaults and settings |
| `lib/data/repositories/model_repository.dart` | load order, warm-ups and fail-fast states for every model |
| `lib/selftest/self_test_runner.dart` | the self-test steps and how evidence is judged |
