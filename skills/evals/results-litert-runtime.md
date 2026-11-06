# Results: litert-runtime prompts

One row per run of the [litert-runtime prompts](litert-runtime.md) ([LiteRT](https://github.com/google-ai-edge/litert), [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)); the checks are in the [README](README.md). "Skill read" lists the skill files the agent opened (a `view_file` step in the Antigravity CLI stream; not observable in Android Studio). A result is ✓ or ✗ followed by the first thing that decided it. The 2026-10-16 rows (Claude Code and Codex CLI, [conditions in the README](README.md#across-agents-2026-10-16)) had only the prompt's skill installed and are read for LiteRT 2.3.0; their skill read is a Skill tool call or a shell command that printed the file, and a parenthesis counts the agent's web searches and the pages it read. Transcripts and the agents' projects are not included.

## R0

| condition | tool, model | date | skill read | result |
|---|---|---|---|---|
| without the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | — | ✗ names the pre-LiteRT stack: org.tensorflow:tensorflow-lite 2.16.1 with tensorflow-lite-support and the GPU delegate, loaded through the pre-LiteRT TensorFlow Lite API; no com.google.ai.edge.litert artifact, no CompiledModel, no AGP 9 line (assets and noCompress are right) |
| without the skill | Claude Code, claude-opus-5-5 | 2026-10-16 | — | ✗ no version for the dependency (`litert = "<latest>"`) and no SDK floor; the sample code is the pre-LiteRT TensorFlow Lite API (`org.tensorflow.lite`), not CompiledModel; only the model location (`app/src/main/assets/`) holds |
| without the skill | Codex CLI, gpt-6-astra | 2026-10-16 | — (1 web search) | ✗ names `com.google.ai.edge.litert:litert:1.4.2` with its pre-LiteRT API (`org.tensorflow.lite`), not a 2.x CompiledModel dependency; minSdk 24 and `app/src/main/assets/` are right |
| with the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | litert-runtime SKILL.md | ✓ names the one dependency, minSdk 24, assets or filesDir, and the AGP 9 line with its reason |
| with the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | litert-runtime SKILL.md | ✓ names the one dependency with the GPU note, minSdk 24, assets or filesDir, and the AGP 9 line with its reason, in 15 s |
| with the skill | Claude Code, claude-opus-5-5 | 2026-10-16 | litert-runtime SKILL.md | ✓ `litert` and `litert-gpu` 2.3.0 as version-catalog entries, minSdk 24 (already in the template), the model under `app/src/main/assets/` or in `filesDir`, and no `gradle.properties` line with 2.3.0 |
| with the skill | Codex CLI, gpt-6-astra | 2026-10-16 | litert-runtime SKILL.md (1 page read) | ✓ `litert` 2.3.0 with `litert-gpu` 2.3.0 for the GPU, minSdk 24 (already in the template), the model under `app/src/main/assets/`, no `gradle.properties` line |

## R1

| condition | tool, model | date | skill read | result |
|---|---|---|---|---|
| without the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | — | ✗ builds, but on org.tensorflow:tensorflow-lite 2.16.1 with its pre-LiteRT API (no com.google.ai.edge.litert artifact, no CompiledModel, no AGP 9 line); the model is under assets and a ViewModel closes it |
| without the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | — | ✗ builds, but on org.tensorflow:tensorflow-lite 2.16.1 with the support library and its pre-LiteRT API; no com.google.ai.edge.litert artifact, no CompiledModel, no AGP 9 line |
| without the skill | Claude Code, claude-opus-5-5 | 2026-10-16 | — | ✗ builds on `litert` 2.3.0, but through the pre-LiteRT TensorFlow Lite API (`org.tensorflow.lite`) that the 2.3.0 artifacts still carry, not CompiledModel; the buffers are made on every inference and the model is never closed |
| without the skill | Codex CLI, gpt-6-astra | 2026-10-16 | — (1 web search, 1 page read) | ✗ builds, on LiteRT 1.4.2 with its pre-LiteRT API, no CompiledModel; every photo loads the model again; inference runs off the main thread on an `ARGB_8888` bitmap scaled to 224x224 |
| with the skills | Android Studio, Gemini 3.6 Flash | 2026-10-07 | not observable (the code is the skill's) | ✓ builds; the skill's inference class and ViewModel as written, litert 2.2.0 through the version catalog, the AGP 9 line, the model under assets, a photo picker, a GPU load with a CPU fallback on LiteRtException; on the Galaxy S26 the model compiled on the GPU and a picked photo gave Top Class: 7 with a score, no crash |
| with the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | litert-runtime SKILL.md + references | ✓ builds; the skill's inference class and ViewModel as written, a photo picker, a GPU load with a CPU fallback on error; on the Galaxy S26 the GPU compiled the model (2 of 2 nodes) and a picked photo gave a top class with its score, no crash |
| with the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | litert-runtime SKILL.md + preprocess.md + imports.md | ✓ builds (the agent ran assembleDebug itself); the skill's inference class and ViewModel, the model under assets, a photo picker with ImageDecoder and the preprocess() of the skill; loads on the CPU |
| with the skill | Claude Code, claude-opus-5-5 | 2026-10-16 | litert-runtime SKILL.md + references (all three) | ✓ builds; `litert` + `litert-gpu` 2.3.0, buffers made once, reused and closed before the model, create, run and close on one executor thread, the photo scaled to 224 and copied to `ARGB_8888` before `getPixels`, normalized to -1..1 with mean and std as parameters |
| with the skill | Codex CLI, gpt-6-astra | 2026-10-16 | litert-runtime SKILL.md + preprocess.md + verify.md (1 page read) | ✓ builds; a CPU-only app on `litert` 2.3.0 alone, buffers made once, reused and closed before the model, one executor, the photo decoded as `ARGB_8888` and scaled to 224x224 before `getPixels`, scaled to 0..1 as a stated assumption |

## R2

| condition | tool, model | date | skill read | result |
|---|---|---|---|---|
| without the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | — (3 web searches) | ✗ builds, but on org.tensorflow:tensorflow-lite 2.16.1 with tensorflow-lite-gpu and the GPU delegate chosen through CompatibilityList, the pre-LiteRT TensorFlow Lite API; no com.google.ai.edge.litert artifact, no CompiledModel |
| without the skill | Claude Code, claude-opus-5-5 | 2026-10-16 | — | ✗ builds on `litert` + `litert-gpu` 2.3.0 with a GPU load that falls back to a CPU load and a test inference at load, but the answer claims "Any layer the GPU can't handle runs on the CPU automatically" |
| without the skill | Codex CLI, gpt-6-astra | 2026-10-16 | — (1 web search, 1 page read) | ✗ builds, on LiteRT 1.4.0 with its pre-LiteRT API and `GpuDelegate`, `litert-gpu` 1.4.0 and a `uses-native-library` line; a GPU failure goes to the CPU, but every tap loads the model and sets up the GPU again |
| with the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | litert-runtime SKILL.md + references | ✓ builds; a GPU load that throws is caught (as Exception) and followed by a CPU load with a notice, no crash; the warm-up inference is in the constructor; no manifest line, no second artifact; on the Galaxy S26 the model compiled on the GPU and Run gave a top class with the raw scores, no crash |
| with the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | litert-runtime SKILL.md | ✓ builds; the screen loads on the GPU and the ViewModel catches LiteRtException from the GPU load and loads on the CPU instead, no crash; the warm-up inference is in the constructor; no manifest line, no second artifact |
| with the skill | Claude Code, claude-opus-5-5 | 2026-10-16 | litert-runtime SKILL.md + references (all three) | ✓ builds; `litert` + `litert-gpu` 2.3.0, a GPU load that throws is caught (as `Exception`) and followed by a CPU load, no claim that operations fall back on their own, one warm-up inference in the constructor |
| with the skill | Codex CLI, gpt-6-astra | 2026-10-16 | litert-runtime SKILL.md + verify.md + preprocess.md (1 page read) | ✓ builds; `litert` + `litert-gpu` 2.3.0, a failed GPU create, buffer or warm-up (caught as `Exception`) reruns the same input on a new CPU session, no claim that operations fall back on their own, the warm-up inference at load |

## R3

| condition | tool, model | date | skill read | result |
|---|---|---|---|---|
| without the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | — | ✗ builds, with a ViewModel that closes the model in onCleared(), but on org.tensorflow:tensorflow-lite 2.16.1 and its pre-LiteRT API; no com.google.ai.edge.litert artifact, no CompiledModel |
| without the skill | Claude Code, claude-opus-5-5 | 2026-10-16 | — | ✗ builds on `litert` 2.3.0 through the pre-LiteRT TensorFlow Lite API (`org.tensorflow.lite`), no CompiledModel; the ViewModel keeps the model across a rotation and closes it through `addCloseable` after any inference in flight |
| without the skill | Codex CLI, gpt-6-astra | 2026-10-16 | — (1 web search) | ✗ builds, on LiteRT 1.4.0 with its pre-LiteRT API, no CompiledModel; for that API the lifecycle holds: created once on one thread, kept across a rotation, closed in `onCleared()` after the inference in flight |
| with the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | litert-runtime SKILL.md + references | ✓ builds; the ViewModel owns the model, a second load() returns at once, onCleared() closes the buffers and the model on the model's executor and then cancels the scope; one lazy Environment; nothing created in a composable; on the Galaxy S26 the classifier screen loaded the model once (one LiteRT environment, GPU), survived two rotations and a leave-and-return in the same process, and Run gave a top class, no crash |
| with the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | litert-runtime SKILL.md + preprocess.md | ✓ builds; the ViewModel owns the model on its own executor, a second load() returns at once, onCleared() closes the model on that executor and then cancels the scope; one Environment; nothing created in a composable |
| with the skill | Claude Code, claude-opus-5-5 | 2026-10-16 | litert-runtime SKILL.md + references (all three) | ✓ builds (the second attempt, in the sandbox); `litert` + `litert-gpu` 2.3.0, the ViewModel owns the model, a second load is a no-op, `onCleared()` closes the buffers and then the model on the model's executor after any inference in flight, one lazy `Environment` |
| with the skill | Codex CLI, gpt-6-astra | 2026-10-16 | litert-runtime SKILL.md + verify.md (1 page read) | ✓ builds; a CPU-only app on `litert` 2.3.0 alone, the ViewModel loads the model once and keeps it across a rotation, `onCleared()` queues the close on the same worker after any inference in flight, buffers closed before the model, one lazy `Environment` |

## R5

| condition | tool, model | date | skill read | result |
|---|---|---|---|---|
| without the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | — | ✗ a sound cross-feed procedure (dump the Python input, feed it on Android, then compare preprocessing: mean and std, channel order, bitmap config), but written for the pre-LiteRT TensorFlow Lite API and its support library, not for CompiledModel; the CompiledModel API and its silent short write are absent |
| without the skill | Claude Code, claude-opus-5-5 | 2026-10-16 | — | ✗ a runnable procedure (save the Python input, log and compare both sides, feed the saved input on the device), but no CPU-then-GPU step, no layout, no `HARDWARE` bitmap copy and no silent short `writeFloat` |
| without the skill | Codex CLI, gpt-6-astra | 2026-10-16 | — (1 web search) | ✗ a runnable procedure (the model's hash, both final input tensors, Python's tensor run on Android, the raw outputs), but on the CPU only, never then on the GPU; no `HARDWARE` bitmap copy, no silent short `writeFloat` |
| with the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | litert-runtime SKILL.md + verify.md + preprocess.md | ✓ reference dump, then CPU, then GPU; preprocessing (mean/std, channel order, HARDWARE bitmap) as the first suspect; the short-array write named |
| with the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | none (the skills were installed but not opened) | ✗ a generic diagnosis in 62 s without opening the skill: compare the input tensors, check normalization and channel order, compare outputs, then CPU against the GPU delegate; the HARDWARE-bitmap copy and the silent short write of a CompiledModel buffer are missing |
| with the skill | Claude Code, claude-opus-5-5 | 2026-10-16 | litert-runtime SKILL.md + verify.md + preprocess.md (printed after a file search; no Skill tool call) | ✗ the procedure is there (a saved reference input on the CPU first, then the GPU, preprocessing first, the silent short write), but not the bitmap's config: no `HARDWARE` bitmap copied to `ARGB_8888`, which preprocess.md says |
| with the skill | Codex CLI, gpt-6-astra | 2026-10-16 | litert-runtime SKILL.md + verify.md + preprocess.md (1 web search) | ✗ a runnable sequence (Python's saved input on the Android CPU first, then preprocessing, then the same input on the GPU), but no `HARDWARE` bitmap copy and no silent short `writeFloat`, both in the files it read |

## R6

| condition | tool, model | date | skill read | result |
|---|---|---|---|---|
| without the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | — (34 web searches, 8 pages read) | ✗ finds the gradle.properties line and the reason after 34 web searches (330 s against 66 s with the skill) and warns against excluding litert-api, but also offers downgrading LiteRT to 2.1.5 or AGP to 8.x, and does not mention 2.3.0 |
| without the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | — (12 web searches) | ✓ finds the gradle.properties line and the reason after 12 web searches (129 s against 37 s with the skill), no exclusion and no downgrade; it does not know that 2.3.0 removes the need |
| without the skill | Claude Code, claude-opus-5-5 | 2026-10-16 | — | ✓ the fix is 2.3.0, put into the project and built; why: `litert` and `litert-api` 2.2.0 declare the same namespace, which it checked in the AAR manifests of the machine's Gradle cache; no `litert-api` exclusion, no AGP downgrade |
| without the skill | Codex CLI, gpt-6-astra | 2026-10-16 | — (2 web searches) | ✓ the 2.2.0 fix, `android.uniquePackageNames=false`, with the reason (AGP 9 rejects the shared namespace) from the AGP 9 release notes and LiteRT issue 8474; no `litert-api` exclusion; 2.3.0 not mentioned |
| with the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | litert-runtime + litert-lm SKILL.md | ✓ the gradle.properties line, why (shared namespace on AGP 9), no litert-api exclusion, removed at 2.3.0 |
| with the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | litert-runtime + litert-lm SKILL.md | ✓ the gradle.properties line, why (the two 2.2.0 libraries share a namespace and AGP 9 stops), and that 2.3.0 removes the need, in 37 s; no exclusion, no downgrade |
| with the skill | Claude Code, claude-opus-5-5 | 2026-10-16 | litert-runtime SKILL.md | ✓ 2.3.0 with `litert-gpu`, or the 2.2.0 line to stay on 2.2.0 (removed on 2.3.0), with the reason (the shared namespace, AGP 9); warns against excluding `litert-api` |
| with the skill | Codex CLI, gpt-6-astra | 2026-10-16 | litert-runtime SKILL.md (1 page read) | ✓ both fixes (the 2.2.0 line, or 2.3.0 with `litert-gpu`), the reason with LiteRT issue 8474, the line removed after the upgrade, no `litert-api` exclusion |

## R7

| condition | tool, model | date | skill read | result |
|---|---|---|---|---|
| without the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | — (21 web searches) | ✗ builds, with a 16000-sample input, but on org.tensorflow:tensorflow-lite 2.16.1 with the support library and its pre-LiteRT API; no com.google.ai.edge.litert artifact, no CompiledModel |
| without the skill | Claude Code, claude-opus-5-5 | 2026-10-16 | — | ✗ no build: it read the model, found the input `[1, 224, 224, 3]` against the prompt's 16,000 samples and asked instead of writing code (the fixture is the image model); it also says a 16,000-value buffer would be rejected, where `writeFloat` takes a shorter array without an error |
| without the skill | Codex CLI, gpt-6-astra | 2026-10-16 | — (1 web search) | ✗ no build: it read the model's tensors, found `[1, 224, 224, 3]` against the 16,000 samples and asked for the waveform model (the fixture is the image model) |
| with the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | litert-runtime SKILL.md + references | ✓ builds; the same inference class with INPUT_SIZE = 16000, the clip written as it is (padded or cut to 16000), one score per event read back; no bitmap step |
| with the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | litert-runtime SKILL.md | ✓ builds; the same inference class with an input of 16000 samples checked before the write, writeFloat of the clip, one score per event read back; no bitmap step |
| with the skill | Claude Code, claude-opus-5-5 | 2026-10-16 | litert-runtime SKILL.md (references listed, none opened) | ✗ no build: it read the model and stopped at the same mismatch (150,528 values against 16,000), with the skill's reason: `writeFloat` takes a short array without an error, so every prediction would be silently wrong |
| with the skill | Codex CLI, gpt-6-astra | 2026-10-16 | litert-runtime SKILL.md + verify.md (1 page read) | ✗ no build: it read the model's tensors and stopped at the same mismatch, asking for the raw-audio model or its preprocessing |
