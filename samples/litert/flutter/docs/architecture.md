# Architecture

LiteRT Demos is one Flutter app with two demos on one on-device stack. **Voice chat** is a push-to-talk assistant: it
transcribes the question, streams an answer from a chat model (about a photo, from a built-in knowledge base, or
through an agent skill) and speaks it from the first finished sentence; with a chat model that hears audio, the
recording itself is the question. **Live camera** draws live detector boxes and answers spoken questions: simple ones
from the detections, detailed ones by sending the frame to the chat model, and questions about the past from a visual
memory of the session. Both share the chat model, speech, EmbeddingGemma 2, audio and diagnostics; with the chat model
file on the device, the demos need no network (only the optional network camera, Download from URL and, on macOS and
Linux, the calendar intent's Google Calendar page use one).

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
  failure because `VoiceSession` has no other failure channel. Public exception types end in `Exception`, except
  `UnexpectedError` (`lib/utils/result.dart`), which wraps a throwable that is not an `Exception`.
- **Rates.** View models notify at turn rate. The streamed reply (`partialReply`), the mic level (`inputLevel`) and
  detection frames (`LiveDetectionRepository.frames`) are `ValueListenable`s: their updates are confined to small
  `ValueListenableBuilder` subtrees or go straight to painters through `CustomPaint(repaint:)` (`LevelRingPainter`,
  `DetectionPainter`). `DiagnosticsRepository` coalesces the overlay's snapshot to at most 4 Hz.
- **Ports and model access.** `lib/domain/ports/` holds `KnowledgeRetriever`, `VisualMemoryEmbedder`, `ModelStates`,
  `ChatModelPlanner`, `VoiceDiagnosticsSink`, `ModelFilePicker` and `SelfTestLauncher`. Only the model services
  (`lib/data/services/llm/`, `lib/data/services/speech/`, `lib/data/services/knowledge/embedder_service.dart`) call
  `FlutterEdgeAi.getActive*`
  and `install*`, driven by `ModelRepository` (in `--selftest`, by the self-test's adapters in
  `lib/selftest/self_test_adapters.dart`).
- **Lifecycle.** `AppForeground` (`lib/ui/core/app_foreground.dart`) follows the `AppLifecycleListener` in
  `lib/app.dart`. On Android and iOS, leaving the app ends a held push-to-talk press (`VoiceAssistant.cancelCapture()`)
  and releases the camera; the live view restarts on return. A turn past its capture goes on, except on iOS once the
  app is hidden (`AppForeground.mustStopTurns`: a background app may not use the GPU and is suspended within seconds):
  there `VoiceAssistant.interrupt` stops it like Stop, keeps the partial reply as interrupted and shows a notice.
  Desktop keeps the camera, the press and the turn. `onHide` → `AppDependencies.onHidden` →
  `AudioRepository.appHidden`: on iOS the next turn prepares the audio again (below). Resume rescans the skills
  folder; a desktop quit closes everything (`AppDependencies.dispose()`) before exiting.

## Models and where they run

| Role | Model | File(s) | Runtime package | Default backend | Source |
|---|---|---|---|---|---|
| Chat (both demos) | Gemma 4 E2B, or a Qualcomm NPU build of it | the chosen `.litertlm` | `flutter_edge_ai` + `flutter_edge_ai_litertlm` (LiteRT-LM) | NPU if the file (of an imported or downloaded file, its stored copy) is compiled for an NPU (`LiteRtLmBundleInfo.isCompiledForNpu`), else GPU; the user can change it, except for a build compiled for an NPU, which runs only there | chosen by the user |
| Speech to text (Voice chat) | Whisper base int8 (a multilingual model, run with language `en`), 30 s window | `assets/models/whisper_base_30s_i8.tflite` + tokenizer | `flutter_edge_ai_speech` | CPU | built in |
| Speech to text (Live camera) | moonshine tiny, 5 s window | `assets/models/moonshine_tiny_5s_f32.tflite` + tokenizer | `flutter_edge_ai_speech` | CPU | built in |
| Text to speech | Inflect-nano-v2 (24 kHz) with Matcha's G2P | `assets/models/inflect/` (6 files) | `flutter_edge_ai_speech` | CPU | built in |
| Embeddings | EmbeddingGemma 2 740m, 768 dims, inputs up to 1024 tokens: loaded with its audio encoder (text and spoken questions); its image encoder (70 vision tokens per image) is added when the camera's visual memory first starts | `assets/models/embeddinggemma-2-740m.litertlm` (one file, tokenizer inside) | `flutter_edge_ai` + `flutter_edge_ai_litertlm` (LiteRT-LM's embedding engine); index in `flutter_edge_ai_rag` + `flutter_edge_ai_sqlite` (sqlite-vec) | CPU | built in |
| Detector | YOLO26n, raw head | `assets/models/yolo26n_fp16_rawhead.tflite` | `flutter_litert` (`CompiledModel`) | GPU, strict, fp32; CPU only when chosen | built in |

The built-in files are not in git: `tool/fetch_models.sh` fetches them from the pins in `tool/models.lock` and derives
the detector with `tool/prune_yolo26n_head.py`.

- **The chat model** is the only one the app does not ship. `ChatModelRepository` holds the choice: a `.litertlm` in
  the models folder or at a typed path (used in place), an imported copy (desktop, iOS), or a download. The app reads
  a file only through flutter_edge_ai_litertlm's `readLiteRtLmBundleInfo` (`inspectLitertlm` in
  `lib/data/services/model_store/models_folder.dart`; for an imported or downloaded file, the store's copy) and keeps
  its `LiteRtLmBundleInfo`. First-file defaults (`defaultsFor`): a file compiled for an NPU (`isCompiledForNpu`) starts
  on the NPU with tools off (also where the device offers no NPU: the card then says why it cannot be applied), any
  other on the GPU; the name gives the model type (the tool-call format and the thinking-channel parsing; for a
  `.litertlm`, flutter_edge_ai sends the prompt raw and LiteRT-LM applies the file's own chat template, whatever the
  type) and the context (`_ekvN_`, else 4096 on the NPU, 8192 otherwise); tools start on only for a Gemma-type file
  not compiled for an NPU. The pick refuses exactly what a load refuses (`inspectLitertlm`), so the app is never
  stricter than the plugin: a file that cannot be opened,
  and an incomplete one (the reader's `incomplete`, or `isComplete` false: a copy or download that stopped early;
  flutter_edge_ai_litertlm 1.14.0 refuses both at the load). A header the plugin cannot read (`notLitertlm`,
  `unsupportedVersion`, `malformed`) is no refusal: the engine loads such a file "without reading the file"
  (`LiteRtLmEngine._readBundle`), so it is a `BundleNotRead` with the plugin's exception, shown on the card, planned
  with no tower (none is known), and read again by a recheck or an apply of the same file. A saved file that cannot be
  opened or is incomplete at launch blocks the slot. A missing or changed file blocks the slot with the reason; no
  other model takes its place. With nothing
  chosen, `GEMMA_MODEL_PATH` fills the slot. One LiteRT-LM engine per process: `ChatModelSwitcher` serializes reloads,
  unloads and the in-app self-test.
- **Candidate and Apply.** A picked file (models folder, Path…, import, download) is only a `ChatModelCandidate`, held
  in memory: the Models screen's card shows its settings and the towers its own file has, and nothing is saved. Apply
  saves it and reloads the chat model. The plan (`ChatModelRepository.plan`), and so every reload, the in-app self-test,
  Run on NPU / GPU / CPU, the app's start and both demos, use only the applied model. Another file starts from its
  own defaults; the same file picked again keeps its settings. Retry and Rescan check a
  saved file with a problem again (`recheck`), so one copied back loads without a restart. Leaving the Models screen
  drops a candidate used in place; an imported or downloaded one stays for the app session, and its bytes never go onto
  the applied model's store file (other bytes of that name are stored as `<stem>-<sha8>.litertlm`). With no choice saved
  and no `GEMMA_MODEL_PATH`, on Android and iOS the only `.litertlm` in the models folders is applied at once when the
  Models screen lists it (at launch, or a Rescan), so one pushed file just runs; not on a desktop or a
  board, whose guide picks the backend first (a Raspberry Pi runs Gemma on the CPU); a file tapped by hand stays a
  candidate until Apply.
- **Loading.** `ModelRepository.prepareAll` loads and warms up each model: the chat model first (largest, fails first),
  then Whisper and Inflect, then the optional detector, moonshine and embedder; a required failure stops setup, an
  optional one shows on its row and tile. The STT model is a singleton: each demo activates its recognizer on entry
  (`activateStt`). Built-in files are used in place on desktop and iOS, extracted once and verified on Android.
  Demo 1 is ready only after its own chat open (or New conversation, or a skills reload) succeeded: when the open
  fails because the previous visit's reply is still stopping, the screen shows the error and offers New conversation
  instead of talking on the old chat.
- **The embedder's media.** Setup loads EmbeddingGemma 2 with text and audio (`kEmbedderConfig`). The first time the
  camera demo starts its visual memory, `ModelRepository.enableEmbedderImages` → `EmbedderService.enableImages` asks
  `flutter_edge_ai` for the same file with audio and images (`kEmbedderMemoryMedia`), which closes the loaded
  embedder and builds one with both: one reload of about 2 s. Embedding calls
  made meanwhile wait for the new model and then run on it, so the knowledge base never fails for the swap. The
  images stay until the app exits: media only widen, so narrowing back would cost another reload and the next camera
  visit a third. A failed widening turns the visual memory off with the reason and reloads text and audio if the
  failed reload took them down. The knowledge base's index is kept: its key names no media, and its text goes
  through the same text encoder.
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
  Whisper path (chat model without audio, "Voice → Whisper → <model>"):
    runTurn -> Whisper -> corrections -> ChatTurnResponder: direct intent | KB excerpts + prompt | prompt (+ photo)
  audio mode (chat model with audio, "Voice → <model>"):
    runAudioTurn -> ChatTurnResponder.respondToAudio: KB search as audio -> spoken prompt + WAV (+ photo)
  -> chat model (agent chat with skills, or plain) -> text deltas -> screen
                                                   -> sentence splitter -> Inflect -> flutter_soloud playback
  audio mode, after the playback drained: transcript job (Whisper + corrections) -> user bubble + text history
```

1. **Capture.** `record` captures 16 kHz mono PCM16 (Android's `voiceRecognition` source) while the mic button is held
   (`PushToTalkCapture`, `DeviceAudioRepository`), up to the recognizer's window, so the STT never cuts a question.
2. **Energy gate** (`measureVoice` in `lib/utils/pcm.dart`). A press under 300 ms is a slip; all-zero audio is a
   microphone error; under 160 ms of voiced 20 ms frames is silence. A frame is voiced at the gate (−45 dBFS by
   default) and 10 dB above the capture's noise floor, that part capped at −30 dBFS. None of these runs a model.
3. **Transcription, or none.** `SpeechRepository` builds one `VoiceSession.custom(streamAudio: true)` per turn; it owns
   no microphone and no player: PCM goes in, events come out. Typed questions take the same path. The responder
   decides the mode per turn: when the loaded chat model hears audio (its file has an audio tower), a spoken turn's
   responder has `respondToAudio` and the turn runs as `runAudioTurn` (no STT before the reply; the user bubble is
   "🎤 3.2 s" at
   once). Otherwise Whisper transcribes it first, and `correctVoiceTranscript` (`lib/domain/audio/
   transcript_corrections.dart`) fixes its measured misses of product names, as phrases only ("what is littered" →
   "what is LiteRT"; "littered" alone stays the English word). The screen and the overlay show the mode with
   the loaded file's name ("Voice → gemma-4-E2B-it" / "Voice → Whisper → …"; in audio mode also "no skills" when
   tools are off or none are loaded: there are no words for the rules, so the time and device intents are reached
   only through the agent's tools; how often the model calls them is in the README's
   [Using the demos](../README.md#using-the-demos)).
4. **Routing** (`ChatTurnResponder`):
   - Time and live device questions ("which accelerator are you running on?") are matched by rules
     (`SkillQuestionRouter`). While an intent-type skill in the installed catalog (what the Skills sheet lists)
     names the `current_time` or `device_info` intent (`intentsOfferedBy`), the app runs that intent itself and speaks
     the result, with tools on or off, and such a question skips retrieval; deleting or replacing the skill turns both
     off, and the question goes to the model like any other. Known limitation: the shortcut speaks the intent's own
     text, not the skill's wording.
   - Other questions are embedded and searched in the knowledge base: up to 3 chunks at cosine similarity ≥ 0.70
     (`kKbMinSimilarityText`) clear the gate, and as many of them as fit the chat model's context window beside the
     turn go into the prompt with numbered sources (`PromptBuilder`), the lowest score dropped first. The fit is
     counted with the chat model's tokenizer against the budget guard's own figures
     (`ConversationRepository.firstFittingPrompt`); the overlay and a citation chip then say "2 of 3 excerpts
     (context)", and when not even the bare question fits, the turn fails as too long. The search takes a
     domain `KnowledgeQuery` (text, or a spoken question as a WAV embedded as audio, with no task prefix, and gated
     at 0.69, `kKbMinSimilarityAudio`); a question over its cap (`kKbMaxTextQueryUtf8Bytes`, `kKbMaxAudioQuery`)
     fails as `KnowledgeQueryTooLongException`, shown as the turn's failed retrieval, never cut.
   - The prompt and the attached photo go to `ConversationRepository.ask`. With skills loaded and a chat model with
     tools on, it is an agent chat (`AgentSession` from `flutter_edge_ai_agent`) with two tools, `loadSkill` and
     `runIntent`, and a system prompt built from the package's `onDemandAgentSystemPrompt` plus two app lines
     (`kSkillsTemplate`); otherwise a plain chat. When the generation right after `loadSkill` is empty, the package
     asks once more (its default `emptyAfterSkillRetries` 1; the app adds no retry of its own) and the steps show it;
     a reply still empty ends the turn as `EmptyAnswerAfterSkillException`, shown like any failed reply. A stop sets
     the flag the agent's `isCancelled` reads before it stops native generation, so a stopped generation is never
     retried.
5. **Knowledge base.** Sixteen Markdown documents in `assets/kb/` are chunked by heading
   (`lib/utils/markdown_chunker.dart`), embedded with EmbeddingGemma 2 (documents with the `retrievalDocument` prefix,
   questions with `retrievalQuery`) and stored in one sqlite-vec index bound to the embedding profile
   `kKbEmbeddingProfileId`, prebuilt by `tool/build_kb_index.sh` into `assets/kb_index/` (344 chunks take about 30 s
   on a Mac's CPU). `KnowledgeRepository` installs it
   on first launch when its key (`kb-index/3`: documents, chunker, document prefix, embedder id, the `.litertlm`'s
   SHA-256, dimension) matches, else indexes on the device and logs why. An index of an earlier embedder (another
   key, another profile, or a store without a profile) is never searched: it is deleted and replaced. Chunk metadata
   becomes citation chips under the reply (`lib/ui/features/voice_chat/views/citation_chips.dart`).
6. **Skills.** `SKILL.md` files live in a writable `skills` folder (in the app's Documents folder on macOS and iOS,
   the user's Documents folder on Linux, usually `~/Documents/skills`, and the app's external files folder on
   Android), seeded from `assets/skills/` (`current-time`, `device-info`).
   `SkillStoreService` parses them and checks the intents they name, by their exact names, against what the intent
   executor runs (`NativeIntentExecutor.isWhitelisted`); errors show in the Skills sheet. Reload and app resume
   rescan; a changed set opens a new agent chat between turns (`SkillsApplier`). A skill combines intents or is
   instruction-only; it cannot add code. The intents are run by the agent package's `NativeIntentExecutor`: the app
   adds `current_time` and `device_info` as handlers (`buildAppIntents`), next to the package's built-in ones
   (`get_current_date_and_time`, `send_email`, `send_text`, `create_calendar_event`, `read_calendar_events`,
   `schedule_notification`), which cannot be removed, so a skill written for the package (its `get-current-time`)
   loads and runs too. The direct route runs an intent through `handleIntent`. The two seeded
   skills are examples: no prompt names a skill, an intent or a topic, and the only skill names the model sees are the
   loaded catalog's, listed in the agent chat's system prompt and in the per-turn hints `PromptBuilder` builds from
   that list.
7. **Streaming reply.** One chat slot (`createChat`) serves both demos; entering a demo opens it with that demo's
   profile (`kVoiceChatProfile`, `kCameraProfile`). Text deltas update the bubble through `partialReply`. A budget
   guard starts the chat over before the context overflows (a recording counts 25 tokens per second), and before
   `InferenceChat`'s own trim would fire (`kContextHeadroomTokens`). The repository keeps a text mirror of the
   conversation (`HistoryMirror`: the question — "[voice question]" for an audio turn until its transcript arrives —
   and the reply). A chat that must be rebuilt replays the mirror's newest turns as text: after a stop while an audio
   turn is in it (`clearHistory(replayHistory:)`, instead of flutter_edge_ai_litertlm's own rebuild after a stop,
   which replays the text only, so an audio question would have no words) and after an interrupted skill call every
   turn that fits beside the next one, after a budget reset at most a quarter of the window. `InferenceChat`'s
   history (flutter_edge_ai) holds each turn's WAV (≤ ~960 KB per 30 s question) until the next rebuild or reset;
   the context budget bounds that to about six 30 s questions at 8192 tokens.
   **Transcript job (audio mode).** Once the reply's playback has drained and the turn is idle, Whisper transcribes
   the recording in the background (`TranscriptJob`); the words fill the user bubble in place (by the turn's number)
   and the text mirror. A press on the reply's playback tail holds the job until the next turn's reply has been
   generated, a press cuts that generation, or nothing runs, whichever comes first: Whisper stays off the next
   question's capture and embedding, and the words reach the text mirror before a later rebuild needs them. Turn numbers are unique in the process:
   the text mirror belongs to the shared chat, which can outlive a visit of the demo. Later presses and turns never
   drop a pending transcript (`TranscriptJob` sends them to Whisper one at a time, in order; a native transcription
   cannot be stopped, so a cancel would save no work); only leaving the demo or a switch to another recognizer does.
   No words show as
   "🎤 3.2 s (no transcript: reason)". A photo is normalized to ≤ 1024 px with its EXIF
   orientation (`lib/data/services/images/image_normalizer.dart`) and re-sent only when the chat can no longer see it.
8. **Speech out.** `VoiceSession` splits the reply at sentence ends (`.`, `!` or `?` before a space, or a newline;
   very short fragments merge into the next sentence), and Inflect synthesizes each sentence as soon as it is
   complete, so playback starts with the first finished sentence. `SpokenSynthesizer` strips citation markers,
   Markdown and URLs from what is spoken (`lib/utils/spoken_text.dart`); `flutter_soloud` plays the PCM stream.
9. **Half-duplex by design.** The mic is closed while a reply plays, so there is no echo to cancel. Pressing the mic
   during a reply is the barge-in: playback stops at once, the partial reply is kept as interrupted, generation is
   stopped and drained in the background, and the next turn starts after the drain.
10. **One audio-session owner.** Only `AudioSessionService` configures the platform session, activates it and hears
    its interruptions: iOS `.playAndRecord` with the default mode (no voice processing), Android speech for an
    assistant with transient focus, none on Linux. Android takes the focus per turn: a press, or a reply that will be
    spoken, takes it, and it goes back when the press or turn ends (after the playback drained), so other apps' audio
    pauses only meanwhile; the audio warm-up when a demo opens takes none. iOS activates the session when the audio is
    prepared and keeps it active between turns; after an interruption, or after the app was hidden, the next turn
    activates it again. soloud is the one other activator: on iOS it re-activates the session whenever it restarts its
    stopped output device, and it is itself restarted only after the app was hidden. An interruption (a call, Siri, an
    alarm, another app's audio; the headphones removed) cancels the capture, stops the playback and ends the press or
    turn like Stop, with a notice; a duck does not interrupt. `record` does not manage the session and, on Android,
    takes no focus (`audioInterruption: none`); the camera opens with audio off.

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
  (anchor, class) at score ≥ 0.25, takes the top 100 and maps boxes back to frame pixels. No NMS: Arm's file is
  YOLO26's end-to-end export, whose one-to-one head predicts without NMS, and the raw head is the tensor its
  one-to-one selection read. `DetectionPainter` draws at most 50 boxes at score ≥ 0.35.
- **Why the raw head.** Arm's graph ends in a selection head (`TOPK_V2`, `GATHER_ND`, int64 `CAST` / `SELECT` /
  `LESS`) that the GPU accelerator cannot place, so a GPU-only `CompiledModel` fails to compile (measured on macOS). The int64 ops also
  left the shared `ADD` opcode at version 4, which the Metal accelerator of LiteRT 2.1.5 (`flutter_litert` on macOS)
  rejects: with `{gpu, cpu}` only 54 of 461 ops ran on the GPU there, at CPU speed. `tool/prune_yolo26n_head.py` cuts
  the graph at the `[1,8400,84]` tensor and sets `ADD` to version 1 (every remaining ADD is float32); the weights are
  unchanged. The whole graph then runs on the GPU, and the selection moves to Dart (about 1 ms).
- **Load checks** (`lib/data/services/detector/detector_engine.dart`), in order, never retried elsewhere: file size;
  a build with exactly `{gpu}` (or `{cpu}` when chosen) at fp32; for the GPU, no fallback and `isFullyAccelerated`;
  input and output sizes; a ramp-input run that matches a CPU reference within 1 % of the reference output's range
  (the TFLite interpreter, else LiteRT's CPU path: `lib/data/services/detector/detector_verification.dart`); a warm-up.
  `ModelRepository` then refuses a GPU that is a software rasterizer (below).
- **Questions.** At release, `CameraTurnResponder` captures the next frame with its own detection (in parallel with
  STT), and `QuestionRouter` applies rules over the COCO vocabulary:
  - Memory (checked first): a cue about the past or about something of the speaker's ("where was …", "where is my
    …", "have you seen …", "when did you (last) see …", "where did you (last) see …", "where did I leave …", "what
    was/were" before a place word) goes to the visual memory (`MemoryRoute`, below) with the noun phrase after it. The
    place words are at, behind, beside, by, in, inside, near, on, over, under, underneath, beneath, above and below,
    "next" only before "to" and "to" only before left or right ("what was on the table?"; the present "what is on the
    table?" is the frame's). "Just", "ever", "already" and "recently" between a cue's words are skipped, and "'d" and
    "'ve" are expanded first by a documented rule (`normalizeQuestion`: "where'd I put …", "what've you seen?").
    "Anything", "something" and "everything", also with one word after them that is neither a colour nor a COCO noun
    ("anything unusual"), name nothing to search for, and neither does a phrase that is only a place ("on the left",
    "behind me"): those ask for the newest stored frame, as "What did you see?" does. "Anyone" and "someone" are
    searched as "person". Not a memory question when the words after the cue mark the present ("now", "currently",
    "at the moment"; or "here", "in front of me/us/you", "in the picture" and the like, unless a past marker such as
    "earlier" overrides them), name a body part
    ("where is my hand?"), start an embedded wh-question in the present ("did you see how many … are here?") or are
    only the speaker or the listener ("where was I?"): the later rules decide those. "Where is the dog?" stays a
    detailed question about the frame.
  - Fast: inventory ("what do you see?"), counts ("how many cups?") and presence ("is there a dog?") when the noun is
    a COCO class. `FastAnswerComposer` answers from the `DetectionSummary` (median count per class over the last 5
    frames at score ≥ 0.4) with a template. No LLM runs and the view stays live.
  - Detailed: everything else (describe, read, colour, position, an unknown noun). The view freezes on the snapshot;
    `SnapshotEncoder` makes a PNG of at most 1024 px, mirrored back when the source mirrors, and it goes to the chat
    model with up to 8 detections as a hint (`buildCameraPrompt`). The chat is reset after each detailed turn, so
    questions are stateless. When the chat model file has no vision tower, the detector's list is spoken instead.
- **GPU arbitration.** `GpuArbiter` follows `ConversationRepository.isGenerating`: while the chat model generates, the
  detector is paused (`kDetectorDuringGeneration`; the frame in flight finishes, the overlay names the chat model) and
  resumes after. `flutter_litert` has no GPU priority, and in a
  detailed turn the view is frozen anyway.
  The boxes are cleared while it is paused, and a frame that lands meanwhile is not drawn: still boxes would sit over
  the moving video. The first detection after the resume brings them back.

### Visual memory

`VisualMemoryRepository` (`lib/data/repositories/visual_memory_repository.dart`) keeps frames EmbeddingGemma 2
embedded while the demo was idle, so a spoken question can find a scene by meaning, beyond the detector's 80
classes. Tunables are in `lib/config/live_camera_config.dart` (`kMemory*`).

- **Owner and switch.** The camera screen starts it on entry and stops it on leaving, which forgets every frame.
  "Visual memory" in the camera settings (persisted) turns it off at once: no sampling, no image encoder load, the
  frames forgotten; a memory question then answers "Visual memory is off (camera settings)", and "What did you see?"
  is answered from the current frame. With nothing saved it is off where the OS reports less than 6.5 GiB of RAM
  (`kMemoryDefaultOnMinRamBytes`: devices sold with 6 GB or less; the image encoder adds about 300 MB), with the reason
  on the switch, and on above that or where the RAM is not measured (Android and Linux read `/proc/meminfo`, iOS and
  macOS `sysctl`).
- **Image encoder.** The first start with the switch on asks for the embedder's image media (above); the chip says
  "Memory: preparing…" and nothing is sampled until it is loaded.
- **Sampler** (`MemorySampler`, every 500 ms). An attempt is due once at least 3 s and 4 × the last frame embedding
  have passed since the previous one ended. A scene change stores a frame; an unchanged scene, once 10 s have passed
  since the last stored or refreshed frame, only refreshes the newest frame's time (below). The sampler runs on a
  monotonic clock, so a wall clock set back or forward neither stops it nor makes a frame due early; frame times and
  ages are the wall clock's. A frame is asked for only while live detection runs, the frames are not black, no
  question capture is pending and the demo is idle (no turn, no generation, no playback). A busy tick drops the sample
  and restarts the gap, and so does the first idle tick after a busy one, so the next frame comes a full gap after the
  demo turned idle. Nothing queues; a question's embedding waits at most for the one frame embedding already running.
- **Frames from the worker.** A memory capture is a second, bounded slot next to the question capture
  (`MemoryCaptureSlot`, `SceneCapture`). The detector worker, which already holds the upright frame, makes a
  128 × 128 grey thumbnail, averaged into 32 × 32 cells of 4 × 4 pixels, and compares it with the last stored one:
  the reference is re-aligned by up to ±2 pixels in half-pixel steps (a hand-held camera's shift) and the overall
  brightness is matched first. A scene change is a mean cell difference of at least 6, or at least
  `kMemoryChangedCells` (2) cells that differ by 24 or more, so one small object put down or taken away counts. Only
  for a scene change does the worker make a PNG of at most 320 px (`lib/data/services/detector/memory_shot.dart`), in
  its isolate. An unchanged scene once 10 s have passed refreshes the newest entry's time in place (no PNG, no
  embedding), so a still scene never fills the store and "seen N s ago" is its latest sighting. A frame that does not
  come back within 2 s is dropped.
- **Store** (`MemoryStore`). Per entry: time, PNG, detections and the image vector (70 vision tokens,
  `retrievalDocument`). At most 200 entries and 16 MB of PNG, oldest out; cleared on leaving the demo, when a new
  detection session starts (a source change, Retry; not a detector backend switch, nor the restart after the camera
  was released in the background) and by "Clear memory"; never written to disk. Search is a cosine scan in Dart.
- **Acceptance** (`judgeMemory`, `lib/domain/use_cases/memory_acceptance.dart`). The phrase is embedded as text
  with no task prefix. The best frame must score at least 0.62; its rival is the best frame more than 5 s away that
  shows a different scene (image vectors below 0.88 alike: frames of one long sighting agree rather than compete).
  The best beats the rival by at least 0.025 (with no rival, it must clear 0.645): a hit, answered with the newest
  frame of the same scene within 0.01 of the best whose grey thumbnail shows no change from the best's. A rival
  within the margin, with the best at least at floor + margin (0.645): an ambiguous hit, answered with the newest of
  the competing frames and said as "I may have seen that in more than one place". Otherwise a miss: "I haven't seen
  that in the last N minutes", without the LLM. "Newest" is the store's order of sightings, not the wall-clock time,
  which a clock set back could reorder; a score that is not a finite number takes no part.
- **Answer.** A hit shows a card over the video (the frame and "12 s ago"; tap to close) and goes to the chat model
  with the frame and the question; when the chat model file has no vision tower, or the camera chat not open, a template says
  when it was seen and that frame's detections. A question with nothing to search for ("What did you see?") gets the
  newest stored frame; with none stored, or the memory off or switched off, it is a detailed question about the frame
  now.
- **Failures.** A failed frame embedding (a vector of zeros, NaN or infinite values included, which is never stored),
  or a frame the worker could not make, counts in the overlay; three in a row turn the memory off with the reason (a
  banner), and memory questions answer with it. An embedder that is not ready turns it off too. A question whose own
  vector is unusable fails that turn (`MemoryVectorException`).
- **UI.** The chip "Memory: 37 frames · 2 min", the result card, and the switch and "Clear memory" in the settings
  sheet (`lib/ui/features/live_camera/views/`).

## Accelerators and diagnostics

The rule: the app never moves a model to another backend on its own. A requested GPU or NPU that cannot be used is an
error the UI shows, with an explicit way out (Run on NPU / GPU / CPU, Run detector on CPU), never a CPU fallback.

- **Chat model.** The LiteRT-LM engine is registered without its backend fallback
  (`kStrictLiteRtLmEngine = LiteRtLmEngine(backendFallback: false)` in `lib/config/bootstrap.dart`, for the app and the
  self-test): a load makes one attempt, on the requested backend. `LlmService.load`
  (`lib/data/services/llm/llm_service.dart`) makes no check of its own; a failure is a `ChatModelLoadException` whose
  message is the plugin's: a `LiteRtLmBundleException`'s message and problem (refused before any engine: an NPU build
  asked for the GPU or the CPU, an incomplete file, an NPU this package has no dispatch for), or each
  `BackendInitException` attempt with its `LiteRtLmEngineCreateException.nativeMessage` (LiteRT-LM's own words) or
  `NpuDispatchUnavailableException.message` (the NPU cannot be reached here). A file compiled for an NPU
  (`isCompiledForNpu`) runs on no other backend (flutter_edge_ai_litertlm 1.14.0's README,
  [Where a model runs](https://pub.dev/packages/flutter_edge_ai_litertlm/versions/1.14.0#where-a-model-runs)): its
  load refuses the GPU and the CPU for it (`wrongBackend`). The card disables both, in the app's own words ("compiled
  for the Qualcomm SM8750 NPU: runs only on the NPU"), and `ChatModelRepository.problemWith` refuses them, so Apply and Run on GPU /
  CPU never unload the running model for a load the plugin would refuse. The card's NPU line names the SoC the OS
  reports ("available (SoC QTI SM8750)") and the Towers block the NPU the file is compiled for ("Text model:
  compiled for the Qualcomm SM8750 NPU"); neither says whether
  that pair runs: only the load tells. A refusal before any
  engine (`LiteRtLmBundleException`) is a `ModelFailed` with `retryable: false` and its
  `bundleProblem`: no Retry. For `wrongBackend` over a build compiled for an NPU (an earlier build saved it on the
  GPU), the card offers Run on NPU while the NPU is offered. For a generic file, a `wrongBackend` refusal or a
  retryable engine failure offers Run on GPU / Run on CPU, whichever it is not on; the other refusals (an incomplete
  file, an NPU this package has no dispatch for) and a failure that is not retryable offer none.
  `lib/data/services/llm/npu_availability.dart` mirrors flutter_edge_ai_litertlm 1.14.0's private NPU gate
  (`hostShipsNpuDispatch` / `npuDispatchShipsFor` in `lib/src/ffi/backend_preference.dart`: on Android only when
  `libcdsprpc.so` opens, on Linux arm64 when the Qualcomm probe passes) because the plugin has no public API for it
  (a plugin gap); it offers the NPU on the card, and Apply refuses an NPU it says is not there. The Android NPU
  stack is bundled only because `pubspec.yaml` sets `qualcomm_npu: true` for `flutter_edge_ai_litertlm` under
  `hooks: user_defines`.
- **Chat model towers.** The towers (vision, audio) are not settings, and the app never says where one will run:
  every tower the file has is loaded (`supportImage` / `supportAudio` from `LiteRtLmBundleInfo.visionEncoder` /
  `audioEncoder`, the applied file as the repository read it last: `ChatModelState.bundle`), with
  `preferredVisionBackend` / `preferredAudioBackend` null, so flutter_edge_ai_litertlm places each one from the file (a
  backend the encoder declares, the NPU for an encoder compiled for one, else the CPU; an explicit tower backend the
  file does not allow would be refused). After the load `InferenceModel.activeVisionBackend` / `activeAudioBackend`
  say where each went (`LlmInfo`, `ChatModelFacts.visionBackend` / `audioBackend`): the card's Towers block shows them
  for the loaded file, and before a load only which towers the file has (and, for a build compiled for an NPU, the
  chip). A CPU | NPU tower choice an earlier build saved (codec v4 `visionChoice` / `audioChoice`) is read and ignored.
  The official `…_qualcomm_sm8750` build and the standard Gemma 4 E2B got both towers on the CPU, Google's all-NPU
  builds both on the NPU; the `…_qualcomm_qcs8275` build's header
  declares the CPU for its towers. `GEMMA_MODEL_PATH` keeps fixed towers
  (`kLlmConfig`: vision on, no audio). On Developer Device Platform a spoken question's first token came in
  0.27–0.32 s with the all-NPU builds on a Galaxy S24, S25 Ultra and S26, against 0.86 s on the S24 with Gemma on the
  GPU and the towers on the CPU.
- **Detector:** the strict accelerator set and the load checks above. **Embedder:** requested on the CPU (the generic
  EmbeddingGemma 2 bundle runs only there; the per-SoC NPU bundles are not used), registered with
  `requireWeightCache: true` (`kStrictEmbeddingBackend`): a load that could write no weight cache at all fails with
  the plugin's `EmbeddingLoadException` (the embedder row's error) instead of running slower with more memory (one
  tower's cache missing beside the others stays a warning); the
  app checks the reported backend, dimension, media and input cap and closes a model that differs. **Speech:**
  requested on the CPU; the package cannot report the backend, so it shows as "requested".
- **How the real backend is known.** Each model's `AcceleratorEvidence` (`lib/domain/models/accelerator_evidence.dart`)
  tags every fact with its source: `api` (`activeBackend`, or the detector's accelerator set and `isFullyAccelerated`),
  `log`, `inferred` or `requested`. No plugin API reports the GPU API or adapter. As diagnostics only, release builds
  on Linux and macOS redirect native stderr into `<app support>/logs/native.log`
  (`lib/data/services/hardware/native_log_tap.dart`); each GPU-capable load reads its own window, and `parseNativeLog`
  (`lib/domain/hardware/native_log_parser.dart`) finds the API (Metal, WebGPU/Vulkan) and the adapter. Elsewhere both
  are inferred from the platform and hardware probe (`lib/domain/hardware/accelerator_inference.dart`). A software
  rasterizer (llvmpipe, lavapipe, SwiftShader, softpipe) is an error: the detector's GPU load fails when its log, or
  failing that the hardware probe, shows one (`ModelFailed` with backend `gpu`, no Retry on the setup row: the same
  GPU would be refused again), and Demo 3 offers "Run detector on CPU". The log never picks a backend.
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
| `lib/data/repositories/conversation_repository_edge_ai.dart` | the single chat slot, profiles, agent skills; its helpers (budget guard, history mirror, turns) are in `lib/data/repositories/conversation/` |
| `lib/data/repositories/live_detection_repository.dart` | frame source → gate → detector worker → frames |
| `lib/data/services/detector/detector_codec.dart` | YOLO26n letterbox, gather and top-k decode |
| `lib/data/repositories/chat_model_repository.dart` | the chosen `.litertlm` (read by `readLiteRtLmBundleInfo`), its defaults and settings |
| `lib/data/repositories/model_repository.dart` | load order, warm-ups and fail-fast states for every model |
| `lib/selftest/self_test_runner.dart` | the self-test's order and pass rule; the steps are in `lib/selftest/*_steps.dart`, the judging in `evidence_judge.dart` |
