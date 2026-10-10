# Results: litert-runtime prompts

One row per run of the [litert-runtime prompts](litert-runtime.md) ([LiteRT](https://github.com/google-ai-edge/litert), [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)); the checks are in the [README](README.md). "Skill read" lists the skill files the agent opened (a `view_file` step in the Antigravity CLI stream; not observable in Android Studio). A result is ✓ or ✗ followed by the first thing that decided it. Transcripts and the agents' projects are not included.

## R0

| condition | tool, model | date | skill read | result |
|---|---|---|---|---|
| without the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | — | ✗ names the pre-LiteRT stack: org.tensorflow:tensorflow-lite 2.16.1 with tensorflow-lite-support and the GPU delegate, loaded through the pre-LiteRT TensorFlow Lite API; no com.google.ai.edge.litert artifact, no CompiledModel, no AGP 9 line (assets and noCompress are right) |
| with the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | litert-runtime SKILL.md | ✓ names the one dependency, minSdk 24, assets or filesDir, and the AGP 9 line with its reason |
| with the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | litert-runtime SKILL.md | ✓ names the one dependency with the GPU note, minSdk 24, assets or filesDir, and the AGP 9 line with its reason, in 15 s |

## R1

| condition | tool, model | date | skill read | result |
|---|---|---|---|---|
| without the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | — | ✗ builds, but on org.tensorflow:tensorflow-lite 2.16.1 with its pre-LiteRT API (no com.google.ai.edge.litert artifact, no CompiledModel, no AGP 9 line); the model is under assets and a ViewModel closes it |
| without the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | — | ✗ builds, but on org.tensorflow:tensorflow-lite 2.16.1 with the support library and its pre-LiteRT API; no com.google.ai.edge.litert artifact, no CompiledModel, no AGP 9 line |
| with the skills | Android Studio, Gemini 3.6 Flash | 2026-10-07 | not observable (the code is the skill's) | ✓ builds; the skill's inference class and ViewModel as written, litert 2.2.0 through the version catalog, the AGP 9 line, the model under assets, a photo picker, a GPU load with a CPU fallback on LiteRtException; on the Galaxy S26 the model compiled on the GPU and a picked photo gave Top Class: 7 with a score, no crash |
| with the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | litert-runtime SKILL.md + references | ✓ builds; the skill's inference class and ViewModel as written, a photo picker, a GPU load with a CPU fallback on error; on the Galaxy S26 the GPU compiled the model (2 of 2 nodes) and a picked photo gave a top class with its score, no crash |
| with the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | litert-runtime SKILL.md + preprocess.md + imports.md | ✓ builds (the agent ran assembleDebug itself); the skill's inference class and ViewModel, the model under assets, a photo picker with ImageDecoder and the preprocess() of the skill; loads on the CPU |

## R2

| condition | tool, model | date | skill read | result |
|---|---|---|---|---|
| without the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | — (3 web searches) | ✗ builds, but on org.tensorflow:tensorflow-lite 2.16.1 with tensorflow-lite-gpu and the GPU delegate chosen through CompatibilityList, the pre-LiteRT TensorFlow Lite API; no com.google.ai.edge.litert artifact, no CompiledModel |
| with the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | litert-runtime SKILL.md + references | ✓ builds; a GPU load that throws is caught (as Exception) and followed by a CPU load with a notice, no crash; the warm-up inference is in the constructor; no manifest line, no second artifact; on the Galaxy S26 the model compiled on the GPU and Run gave a top class with the raw scores, no crash |
| with the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | litert-runtime SKILL.md | ✓ builds; the screen loads on the GPU and the ViewModel catches LiteRtException from the GPU load and loads on the CPU instead, no crash; the warm-up inference is in the constructor; no manifest line, no second artifact |

## R3

| condition | tool, model | date | skill read | result |
|---|---|---|---|---|
| without the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | — | ✗ builds, with a ViewModel that closes the model in onCleared(), but on org.tensorflow:tensorflow-lite 2.16.1 and its pre-LiteRT API; no com.google.ai.edge.litert artifact, no CompiledModel |
| with the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | litert-runtime SKILL.md + references | ✓ builds; the ViewModel owns the model, a second load() returns at once, onCleared() closes the buffers and the model on the model's executor and then cancels the scope; one lazy Environment; nothing created in a composable; on the Galaxy S26 the classifier screen loaded the model once (one LiteRT environment, GPU), survived two rotations and a leave-and-return in the same process, and Run gave a top class, no crash |
| with the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | litert-runtime SKILL.md + preprocess.md | ✓ builds; the ViewModel owns the model on its own executor, a second load() returns at once, onCleared() closes the model on that executor and then cancels the scope; one Environment; nothing created in a composable |

## R5

| condition | tool, model | date | skill read | result |
|---|---|---|---|---|
| without the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | — | ✗ a sound cross-feed procedure (dump the Python input, feed it on Android, then compare preprocessing: mean and std, channel order, bitmap config), but written for the pre-LiteRT TensorFlow Lite API and its support library, not for CompiledModel; the CompiledModel API and its silent short write are absent |
| with the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | litert-runtime SKILL.md + verify.md + preprocess.md | ✓ reference dump, then CPU, then GPU; preprocessing (mean/std, channel order, HARDWARE bitmap) as the first suspect; the short-array write named |
| with the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | none (the skills were installed but not opened) | ✗ a generic diagnosis in 62 s without opening the skill: compare the input tensors, check normalization and channel order, compare outputs, then CPU against the GPU delegate; the HARDWARE-bitmap copy and the silent short write of a CompiledModel buffer are missing |

## R6

| condition | tool, model | date | skill read | result |
|---|---|---|---|---|
| without the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | — (34 web searches, 8 pages read) | ✗ finds the gradle.properties line and the reason after 34 web searches (330 s against 66 s with the skill) and warns against excluding litert-api, but also offers downgrading LiteRT to 2.1.5 or AGP to 8.x, and does not mention 2.3.0 |
| without the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | — (12 web searches) | ✓ finds the gradle.properties line and the reason after 12 web searches (129 s against 37 s with the skill), no exclusion and no downgrade; it does not know that 2.3.0 removes the need |
| with the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | litert-runtime + litert-lm SKILL.md | ✓ the gradle.properties line, why (shared namespace on AGP 9), no litert-api exclusion, removed at 2.3.0 |
| with the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | litert-runtime + litert-lm SKILL.md | ✓ the gradle.properties line, why (the two 2.2.0 libraries share a namespace and AGP 9 stops), and that 2.3.0 removes the need, in 37 s; no exclusion, no downgrade |

## R7

| condition | tool, model | date | skill read | result |
|---|---|---|---|---|
| without the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | — (21 web searches) | ✗ builds, with a 16000-sample input, but on org.tensorflow:tensorflow-lite 2.16.1 with the support library and its pre-LiteRT API; no com.google.ai.edge.litert artifact, no CompiledModel |
| with the skills | Antigravity CLI, Gemini 3.8 Flash (High) | 2026-10-07 | litert-runtime SKILL.md + references | ✓ builds; the same inference class with INPUT_SIZE = 16000, the clip written as it is (padded or cut to 16000), one score per event read back; no bitmap step |
| with the skills | Antigravity CLI, Gemini 3.1 Pro (High) | 2026-10-07 | litert-runtime SKILL.md | ✓ builds; the same inference class with an input of 16000 samples checked before the write, writeFloat of the clip, one score per event read back; no bitmap step |
