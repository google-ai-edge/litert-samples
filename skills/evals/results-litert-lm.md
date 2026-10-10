# Results: litert-lm prompts

One row per run of the [litert-lm prompts](litert-lm.md) ([LiteRT](https://github.com/google-ai-edge/litert), [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)); the checks are in the [README](README.md). "Skill read" lists the skill files the agent opened (a `view_file` step in the Antigravity CLI stream; not observable in Android Studio). A result is ✓ or ✗ followed by the first thing that decided it. Transcripts and the agents' projects are not included.

## L1

| condition | tool, model | date | skill read | result |
|---|---|---|---|---|
| without the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | — (5 web searches) | ✗ builds, but on com.google.mediapipe:tasks-genai 0.10.20 (the MediaPipe LLM Inference API) with .task and .bin model files named after gemma-2b; no litertlm-android, no Engine or Conversation, no manifest lines |
| without the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | — (2 web searches) | ✗ does not build, and is on com.google.mediapipe:tasks-genai 0.10.27 (the MediaPipe LLM Inference API) with a .task file; no litertlm-android, no Engine, no manifest lines |
| with the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | litert-lm SKILL.md + imports.md | ✓ builds; the engine initializes on the ViewModel's own executor under a lock, the reply is collected chunk by chunk from the Flow, Stop cancels the process, a new chat is a new conversation, release() cancels then closes the conversation and the engine; the model is found in filesDir or external files (download or adb push described); both manifest lines and INTERNET; on the Galaxy S26 with Qwen3-0.6B.litertlm the engine initialized on the GPU and a reply streamed (partial text at 3 s, complete at 12 s), no crash |
| with the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | litert-lm SKILL.md + imports.md | ✓ builds; the skill's ChatViewModel and screen as written (own executor, Mutex, initialize off the main thread, the reply collected chunk by chunk, cancel then close conversation then engine, onCleared under the lock), both manifest lines, the model path in filesDir with the litert-community file named |

## L2

| condition | tool, model | date | skill read | result |
|---|---|---|---|---|
| without the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | — (6 web searches) | ✗ names the right repo and file after 6 web searches, keeps it out of assets, describes adb push and an in-app download, but gives the size as 1.2 to 1.8 GB (the file is 2.6 GB), omits run-as, the INTERNET permission, the size check and the chip-suffix note, and mixes in GGUF runners |
| with the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | litert-lm SKILL.md + get-a-model.md | ✓ the repo, the file name and size, the resolve URL, adb push and run-as for development, filesDir or DownloadManager into external files for users with INTERNET and a size check, the chip-suffix note; nothing under assets |
| with the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | litert-lm SKILL.md | ✗ in 17 s: the file, 2.6 GB, the chip-suffix note, litert-community, nothing under assets, adb push and run-as, filesDir or DownloadManager with a size check and the absolute path into EngineConfig; but no INTERNET permission and the organization page instead of the file URL |

## L3

| condition | tool, model | date | skill read | result |
|---|---|---|---|---|
| without the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | — (10 web searches) | ✗ builds, but on com.google.mediapipe:tasks-genai (the MediaPipe LLM Inference API) with .task and .bin files named after gemma-2b; it does add the two uses-native-library lines; no litertlm-android, no Engine |
| without the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | — (4 web searches) | ✗ on com.google.mediapipe:tasks-genai 0.10.35 (the MediaPipe LLM Inference API) with a .task file; no litertlm-android, no Engine, no uses-native-library lines |
| with the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | litert-lm SKILL.md + imports.md + get-a-model.md | ✓ builds; Backend.GPU() with both uses-native-library lines in the manifest and cacheDir set; the model path defaults to filesDir/Qwen3-0.6B.litertlm with the adb push and run-as lines shown in the app; on the Galaxy S26 the engine initialized on the GPU (OpenCL loaded, no missing-library error) and a reply streamed, no crash |
| with the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | litert-lm SKILL.md | ✓ builds; Backend.GPU() with both uses-native-library lines in the manifest and cacheDir set; the engine initializes on the ViewModel's own executor |

## L4

| condition | tool, model | date | skill read | result |
|---|---|---|---|---|
| without the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | — (4 web searches) | ✗ builds, but there is no LLM: the "engine" is a class that emits canned sentences word by word with a 35 ms delay, with stop and reset on that; no runtime dependency, no model file |
| with the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | litert-lm SKILL.md + imports.md | ✓ builds; Stop calls cancelProcess, New chat cancels, waits, closes the old conversation and creates a new one on the same engine, the send button is disabled while a reply streams (stop() clears busy at once rather than when the flow ends) |
| with the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | litert-lm SKILL.md | ✓ builds; Stop calls cancelProcess, New chat cancels, closes the old conversation and opens a new one on the same engine, and both buttons are disabled while a reply streams |

## L5

| condition | tool, model | date | skill read | result |
|---|---|---|---|---|
| without the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | — | ✗ builds, but there is no LLM: a simulated engine emits tokens with a delay and a "native handle" class that throws a simulated SIGSEGV; no runtime dependency, no model file |
| with the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | litert-lm SKILL.md + imports.md | ✓ builds; a switch and the close go through one Mutex: release() cancels until the reply has stopped, closes the conversation and then the engine once each, the engine is stored only after initialize() returns, onCleared() releases under the lock on the ViewModel's own scope; two model files offered on the screen |
| with the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | litert-lm SKILL.md + imports.md | ✓ builds; two model buttons (model1 and model2 in filesDir) both go through load() under the Mutex: release() cancels until the reply has stopped and closes the conversation and then the engine, the engine is stored only after initialize(), onCleared() releases under the same lock |

## L6

| condition | tool, model | date | skill read | result |
|---|---|---|---|---|
| without the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | — (5 web searches) | ✗ builds, but on com.google.mediapipe:tasks-genai 0.10.14 (the MediaPipe LLM Inference API) with .task and .bin files; no litertlm-android, no Engine, no SamplerConfig |
| with the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | litert-lm SKILL.md + imports.md | ✓ builds (after 44 Gradle rounds of its own, 26 min); every text closes the previous conversation and opens a new one with SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0, seed = 42) and the neutral style as the system instruction; the summary streams in |
| with the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | litert-lm SKILL.md + imports.md | ✗ does not build: SamplerConfig is called with a Float temperature and without topK and topP (three compile errors), after 6 Gradle rounds of the agent; the skill names SamplerConfig(topK, topP, temperature) but not the parameter types or that all three are required; the rest is right (fresh conversation per text, system instruction) |
