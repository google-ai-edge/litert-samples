# Prompt evals for the litert-runtime and litert-lm skills

Sixteen prompts, written the way a developer asks a coding agent, run in a fresh checkout of [android/architecture-templates](https://github.com/android/architecture-templates) (branch `base`, commit `5467a8dc`) once without the skills and once with [`litert-runtime`](../litert-runtime/SKILL.md) ([LiteRT](https://github.com/google-ai-edge/litert)) and [`litert-lm`](../litert-lm/SKILL.md) ([LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)) in `.agents/skills/`. Each prompt says what the developer wants and names no API; nothing else is given to the agent: no rules file, no other skill, no earlier conversation. A `fix the app` prompt (R8, L7) starts from an app that already runs a model: the files under [`fixtures/R8/`](fixtures/R8/) or [`fixtures/L7/`](fixtures/L7/) are copied over the checkout before the prompt. The prompts and what a correct answer contains are in [litert-runtime.md](litert-runtime.md) (nine) and [litert-lm.md](litert-lm.md) (seven); every run's row is in [results-litert-runtime.md](results-litert-runtime.md) and [results-litert-lm.md](results-litert-lm.md).

## Run

The Antigravity CLI (`agy`, 1.3.1) in print mode, from the project directory, one prompt per fresh copy of the template; `--sandbox` runs the agent's commands in the CLI's restricted terminal (the project and the Gradle caches readable, the home directory not listable, the network open), and every tool is approved as a developer would approve it in the IDE:

```sh
git clone --branch base https://github.com/android/architecture-templates.git project && cd project
mkdir -p .agents/skills && cp -R <litert-samples>/skills/litert-runtime <litert-samples>/skills/litert-lm .agents/skills/
agy --print "$(cat prompt.txt)" --model gemini-3.8-flash-high --output-format stream-json --dangerously-skip-permissions --sandbox > run.jsonl
```

The run without the skills is the same command without the `.agents/skills/` directory. `--model gemini-3.1-pro-high` is the second model. The stream (`run.jsonl`) carries the model, every tool step and the final answer; a `view_file` step on a `SKILL.md` is what "skill read" means in the results. The agent can run Gradle in the sandbox and did on every app prompt (10 to 28 `./gradlew` commands per run); the project is then built once more as the agent left it:

```sh
./gradlew assembleDebug
```

In Android Studio (Rabbit 1 | 2026.2.1, which loads `.agents/skills/` of the project), the same prompt goes to Agent Mode in the same checkout, and the build is the IDE's.

On 2026-10-16, Claude Code and Codex CLI ran headless, from the project directory of a fresh copy per prompt and condition, with only the prompt's skill (`<skill>` = `litert-runtime` for the R prompts, `litert-lm` for the L prompts); the run without it is the same command without the skill directory:

```sh
mkdir -p .claude/skills && cp -R <litert-samples>/skills/<skill> .claude/skills/
CLAUDE_CODE_DISABLE_CLAUDE_MDS=1 claude -p "$(cat prompt.txt)" --model claude-opus-5-5 --setting-sources project --strict-mcp-config --dangerously-skip-permissions --output-format stream-json --verbose > run.jsonl

mkdir -p .agents/skills && cp -R <litert-samples>/skills/<skill> .agents/skills/
codex exec --json --skip-git-repo-check --ignore-rules -m gpt-6-astra --dangerously-bypass-approvals-and-sandbox --disable plugins --disable remote_plugin -o answer.txt "$(cat prompt.txt)" > run.jsonl
```

`CLAUDE_CODE_DISABLE_CLAUDE_MDS=1` keeps every CLAUDE.md out, also one in a parent directory. Codex CLI ran with a `CODEX_HOME` that holds only the model line and the sign-in, and `--disable plugins --disable remote_plugin` keeps the account's plugins and their skills out. For these runs, "skill read" means a Skill tool call (Claude Code) or a shell command that prints a skill file (both CLIs); the project is then built once more as the agent left it.

## What is checked

- Builds: `./gradlew assembleDebug` passes on the project the agent left (AGP 9.4.1, Kotlin 2.4.20); a question prompt has no build.
- Contains / does not contain: the dependency, manifest, Gradle and API facts the prompt file lists, in the files the agent changed or in its answer.
- One sentence read against the diff: the behaviour the prompt file names (buffers closed before the model, a reply stopped before the engine closes, a fresh conversation per text).
- Device: for the prompts marked so, the app runs on a Galaxy S26 (Android 16), CPU and GPU.

A row is ✓ when it builds (where it must), contains everything listed, contains nothing of the other list, and the sentence holds; otherwise ✗ with the first thing that failed. Whether a skill was read is a separate mark, so a ✓ without the skill and a ✗ with it are both visible.

## Results

| prompt | without, CLI 3.8 Flash | with, CLI 3.8 Flash | without, CLI 3.1 Pro | with, CLI 3.1 Pro | without, Android Studio | with, Android Studio |
|---|---|---|---|---|---|---|
| R0 | ✗ | ✓ | — | ✓ | — | — |
| R1 | ✗ | ✓ | ✗ | ✓ | — | ✓ |
| R2 | ✗ | ✓ | — | ✓ | — | — |
| R3 | ✗ | ✓ | — | ✓ | — | — |
| R5 | ✗ | ✓ | — | ✗ | — | — |
| R6 | ✗ | ✓ | ✓ | ✓ | — | — |
| R7 | ✗ | ✓ | — | ✓ | — | — |
| R8 | — | — | — | — | — | — |
| R9 | — | — | — | — | — | — |
| L1 | ✗ | ✓ | ✗ | ✓ | — | — |
| L2 | ✗ | ✓ | — | ✗ | — | — |
| L3 | ✗ | ✓ | ✗ | ✓ | — | — |
| L4 | ✗ | ✓ | — | ✓ | — | — |
| L5 | ✗ | ✓ | — | ✓ | — | — |
| L6 | ✗ | ✓ | — | ✗ | — | — |
| L7 | — | — | — | — | — | — |

R8, R9 and L7 were added on 2026-10-16 and have not run on Antigravity CLI yet.

With the skills, every prompt passed on Gemini 3.8 Flash (High) and ten of thirteen on Gemini 3.1 Pro (High); a skill was opened in 13 of the 13 Flash runs and 12 of the 13 Pro runs: on R5 the Pro agent answered without opening one, on L2 it left out the `INTERNET` permission, and on L6 its `SamplerConfig` call did not compile. Without the skills, every Flash answer and three of the four Pro answers went to another library or to no runtime: `org.tensorflow:tensorflow-lite` 2.16.1 with the pre-LiteRT TensorFlow Lite API for the `.tflite` prompts, `com.google.mediapipe:tasks-genai` or a class that emits canned text for the LLM prompts; R6 was the one prompt a web search could answer. A cell is — where the run was not made: the Pro model ran without the skills on four prompts only, and the Android Studio runs are the prompts pasted by hand in the IDE.

Each prompt's Builds, Contains, Does not contain and Read lines are the checks an automated case of that prompt would carry, and every ✗ names the check that failed. Two changes to the skills follow from the runs: the `SamplerConfig` sentence of `litert-lm` now shows the argument types (a separate change), and the description of `litert-runtime` did not then reach a question about an app that already runs a model, which is why the Pro agent answered R5 on its own (changed on 2026-10-16, the last section).

## Across agents (2026-10-16)

The same thirteen prompts, run on 2026-10-16 by two more agents, Claude Code and Codex CLI, each once without and once with the prompt's skill. Every run starts from a fresh copy of android/architecture-templates `base` at `5467a8dc`. With the skill, only the prompt's skill is installed (`litert-runtime` for the R prompts, `litert-lm` for the L prompts), in the agent's project skill directory: `.claude/skills/<name>/` for Claude Code, `.agents/skills/<name>/` for Codex CLI; the 2026-10-07 runs had both skills in `.agents/skills/`. The skills are those of litert-samples `2166c0f9`: `litert-runtime` last updated 2026-10-09 (LiteRT 2.3.0 with `litert-gpu`) and `litert-lm` last updated 2026-10-07 (`litertlm-android` 0.18.0). There is no rules file (no CLAUDE.md, no AGENTS.md), no user skill or plugin and no MCP server, and the effort is each CLI's default. The agents run headless (`claude -p … --output-format stream-json`, `codex exec --json`; the commands are under [Run](#run)) and get the prompt's `Prompt:` line exactly as the prompt file has it. Each run works in a project directory of its own, and its adb client talks to a private adb server with no device connected.

The checks are each prompt's Builds, Contains, Does not contain and Read lines, read for LiteRT 2.3.0 where a line names 2.2.0: `litert:2.3.0` with `litert-gpu:2.3.0` (or `litert:2.3.0` alone for an app that only uses the CPU), no `android.uniquePackageNames=false` in a 2.3.0 setup, and either fix for R6. A build prompt is ✓ when the project builds as the agent left it, contains everything listed, contains nothing of the other list and its Read sentence holds; a question prompt is judged on the same lines without a build.

| prompt | without, Claude Code | with, Claude Code | without, Codex CLI | with, Codex CLI | without, Antigravity CLI 3.8 Flash | with, Antigravity CLI 3.8 Flash |
|---|---|---|---|---|---|---|
| R0 | ✗ | ✓ | ✗ | ✓ | not run (quota) | not run (quota) |
| R1 | ✗ | ✓ | ✗ | ✓ | not run (quota) | not run (quota) |
| R2 | ✗ | ✓ | ✗ | ✓ | not run (quota) | not run (quota) |
| R3 | ✗ | ✓ | ✗ | ✓ | not run (quota) | not run (quota) |
| R5 | ✗ | ✗ | ✗ | ✗ | not run (quota) | not run (quota) |
| R6 | ✓ | ✓ | ✓ | ✓ | not run (quota) | not run (quota) |
| R7 | ✗ | ✗ | ✗ | ✗ | not run (quota) | not run (quota) |
| L1 | ✗ | ✓ | ✗ | ✓ | not run (quota) | not run (quota) |
| L2 | ✗ | ✓ | ✗ | ✓ | not run (quota) | not run (quota) |
| L3 | ✓ | ✓ | ✓ | ✓ | not run (quota) | not run (quota) |
| L4 | ✓ | ✓ | ✗ | ✓ | not run (quota) | not run (quota) |
| L5 | ✓ | ✓ | ✗ | ✗ | not run (quota) | not run (quota) |
| L6 | ✗ | ✓ | ✗ | ✓ | not run (quota) | not run (quota) |

With the skill, 21 of the 26 runs passed (Claude Code 11 of 13, Codex CLI 10 of 13), and all 26 opened the skill; without it, 6 of 26 passed (Claude Code 4 of 13, Codex CLI 2 of 13). The agents worked 3,021 s in all with the skill and 5,030 s without it (Claude Code 1,477 s against 3,054 s, Codex CLI 1,544 s against 1,976 s).

- Claude Code: with the skill installed, it opened it on its first or second tool call in all 13 runs, 12 times through its Skill tool and once by finding the files in the project and printing them (R5, the debugging question that the description did not name until 2026-10-16; on 2026-10-07 the 3.1 Pro agent did not open the skill on it), wrote nothing before that, and missed only a fact the skill gives but the answer left out (R5: the `HARDWARE` bitmap copied to `ARGB_8888`) and R7; without it, it rebuilt the API from the Maven AARs with `javap` or from the web in about twice the time, and failed where the skill holds the fact: the CompiledModel API and its version (R0, R1, R3), closing the engine (L1), `run-as` and the `INTERNET` permission (L2), LiteRT-LM itself (L6 went to the ML Kit GenAI Prompt API), a claim that the GPU hands unsupported layers to the CPU on its own (R2) and the CPU-then-GPU comparison (R5).
- Codex CLI: with the skill installed, it read `SKILL.md` on its first or second tool call in all 13 runs, before writing anything, and still read the web in every run; it opened a reference file on the five prompts with a model file (R1, R2, R3, R5, R7) and on L1 and L3 (`imports.md`), never `litert-runtime/references/imports.md` or `litert-lm/references/get-a-model.md`, and failed where it read a fact and did not apply it (R5: the `HARDWARE` bitmap copy and the silent short `writeFloat`; L5: a ViewModel that owns the engine and closes it in `onCleared`) and on R7; without it, it chose LiteRT 1.4.x with its pre-LiteRT API (R0 to R3), MediaPipe `tasks-genai` with a `.task` file (L1, L5, L6) or a demo engine with no runtime (L4), and its two passes came from a web search that found LiteRT issue 8474 (R6) and from the LiteRT-LM AAR already in the test machine's Gradle cache (L3).

Notes:

- not run (quota): Antigravity CLI 1.4.0 reached its weekly quota (`RESOURCE_EXHAUSTED`, code 429) on its second setup check, before the first prompt. Gemini CLI 0.63.0 was not run: its individual tier no longer accepts the client (`UNSUPPORTED_CLIENT`).
- R7: the file the prompt calls `audio.tflite` is the image model of the fixture (input `[1, 224, 224, 3]`), and all four R7 runs read the model, found that input against the prompt's 16,000 samples and asked for the right model instead of writing code. An agent that inspects the model stops here; the ✗ is the fixture, not the skill.
- Without the skill, the agents could still read the test machine's Gradle cache, which held LiteRT, LiteRT-LM and MediaPipe artifacts from earlier builds. Three of the six passes without the skill used it (Claude Code R6 and L5, Codex CLI L3), and Codex CLI's L3 run also bundled a `.litertlm` model it found in a shared folder of the machine; Codex CLI took MediaPipe for L6 from the same cache.
- Two runs were made again and only the second one counts. Claude Code's first R3 run with the skill created its own Android virtual device and booted an emulator; Codex CLI's first L1 run with the skill reached that emulator over adb. The runs started after 18:28 JST (Claude Code: R3 with, L4 without, L5 and L6; Codex CLI: L1 with, L3 to L6) ran in a sandbox that refuses to start an emulator, each with an adb server of its own. Codex CLI's L1 run without the skill, made earlier, listed that emulator as offline after it had written its code and is counted.

Device: the apps of the runs with the skill (R1, R2, R3, L1, L3, L4, L5 and L6, both agents) ran on 2026-10-16 on a Pixel 8a (Android 15) of the Developer Device Platform lab, and the six R apps also on a Galaxy S26 (Android 16) there, with the same results on both phones. The input of the R apps was one 224 × 224 picture of a single colour, and the reference is the model's output for the same input on the CPU of the test machine; the chats ran `Qwen3-0.6B.litertlm`.

- R1: Claude Code's app loads the model on the GPU and showed class 8 with a score of 0.9336 (reference 0.9346); it has no switch to the CPU. Codex CLI's app runs on the CPU only and showed class 8 with the reference's score, 0.8415. Both pass, each on the one accelerator its app has.
- R2: Claude Code's app classified the picture on the GPU, class 8 with a score of 0.8408 (reference 0.8415), without falling back. Codex CLI's app takes no picture: its fixed all-zero input ran on the GPU and matched the reference within 0.0002, so the Device line (the GPU run classifies the picture) does not hold for it.
- R3: both apps kept the same process, model and result across two rotations and a leave-and-return, with one LiteRT environment and one model created, and no crash.
- L1: the reply streamed on the CPU in both apps; the first words came 14 s (Claude Code) and 21 s (Codex CLI) after Send, and the replies were complete at 18 s and 25 s.
- L3: the reply streamed on the GPU in both apps, every node of the decode and prefill graphs on the OpenCL delegate; first words after 8 s and 9 s, complete at 10 s and 11 s.
- L4: Stop ended a reply within a second, and after New chat the model did not know the earlier question (Claude Code's app on the GPU, Codex CLI's on the CPU).
- L5: three model switches, one of them in the middle of a reply, ran in one process without a crash in both apps; Gemma 3 270M stood in for the 2.6 GB Gemma 4 E2B file both apps name as their second model.
- L6: both apps gave the same three neutral bullets for the same text twice, the second time after a restart; Codex CLI's app served the second from its own cache without a model call.

## After the change of 2026-10-16

Device: pending — no cell of this section ran on a phone.

This change updates the two skills for an app that already runs a model and adds three prompts for it:

1. The descriptions of `litert-runtime` and `litert-lm` also name fixing such an app, with the usual cause or fix in a few words. `litert-runtime`: a different result in the Android app than in Python (a `HARDWARE` bitmap, a `writeFloat` array shorter than the input), a screen that freezes while the model runs, the model loaded again on every rotation (the ViewModel owns the model and runs it on a thread of its own). `litert-lm`: a reply that takes seconds to start, memory that grows with each message, a reply that forgets the earlier messages (one engine and one conversation, owned by the ViewModel and closed in `onCleared()`).
2. [`litert-runtime/references/verify.md`](../litert-runtime/references/verify.md) gets one paragraph under "Device CPU, then the GPU": when the app builds the input from a picture, compare the array it writes with the reference input. It says to check the array's element count first (`writeFloat` writes a shorter array without an error), then the bitmap's config (`getPixels` cannot read a `HARDWARE` bitmap), then size, mean and std, channel order and layout.
3. [`audio.tflite`](../tests/litert-runtime/fixtures/audio.tflite) (input 1x16000, output 1x8) is a new test fixture with the input the R7 prompt names. Before it, R7 got the image model under that name, and all four R7 runs of the section above stopped there.
4. Three prompts: R8 (fix the app: its preprocessing writes one float per pixel), R9 (a question: a model created in a composable and run in a click handler) and L7 (fix the app: a chat that builds a new engine and conversation for every message). The R8 and L7 apps are the files under [`fixtures/R8/`](fixtures/R8/) and [`fixtures/L7/`](fixtures/L7/).

The runs keep the conditions of [Across agents (2026-10-16)](#across-agents-2026-10-16), with the skills as updated on 2026-10-16 (this change): both last updated 2026-10-16, still on LiteRT 2.3.0 with `litert-gpu` and `litertlm-android` 0.18.0. Each run starts from a fresh copy of the template with only the prompt's skill installed and no rules file, and gets the prompt's `Prompt:` line through the same headless commands. For R8 and L7, the fixture files are copied over the template before the project's first commit, so a run's changes are the agent's alone. All 24 runs ran in the sandbox that refuses to start an emulator, each with a private adb server and no device connected; none reached a device or an emulator. The six prompts are the three new ones and the three this change touches: R5 (the description and `verify.md`), R7 (the fixture) and L5 (the description).

| prompt | without, Claude Code | with, Claude Code | without, Codex CLI | with, Codex CLI |
|---|---|---|---|---|
| R5 | ✗ | ✓ | ✗ | ✓ |
| R7 | ✗ | ✓ | ✗ | ✓ |
| R8 | ✗ | ✓ | ✗ | ✗ |
| R9 | ✗ | ✓ | ✗ | ✗ |
| L5 | ✗ | ✓ | ✗ | ✓ |
| L7 | ✗ | ✓ | ✗ | ✓ |

With the skill, 10 of the 12 runs passed (Claude Code 6 of 6, Codex CLI 4 of 6), and all 12 opened the skill on their first or second tool call, before writing anything; without it, none of the 12 passed. The agents worked 991 s in all with the skill and 1,405 s without it (Claude Code 447 s against 790 s, Codex CLI 544 s against 615 s).

R5, R7 and L5 also ran on the skills of 2166c0f9 (the section above): with the skill, 1 of those 6 runs passed then and 6 of 6 now; without it, 1 of 6 then and 0 of 6 now. R5 with the skill went from ✗ to ✓ on both agents. Both answers now name the `HARDWARE` bitmap copy and the short `writeFloat` array, which this change adds to the description and to `verify.md`. Before, Claude Code's answer left out the copy and Codex CLI's left out both, though both runs had read them in `preprocess.md` and `SKILL.md`. L5 with the skill went from ✗ to ✓ on Codex CLI (a ViewModel owns the engine and closes it in `onCleared`) and stayed ✓ on Claude Code. Without the skill, Claude Code's L5 went from ✓ to ✗: it closes an engine whose `initialize()` threw. On R7, all four runs stopped at the image model before; with the new fixture, all four wrote code that builds, on `CompiledModel` with the skill (✓) and on the pre-LiteRT TensorFlow Lite API without it (✗). Each cell is one run.

- Claude Code: with the skill installed, it called its Skill tool before writing anything in all six runs, on the first tool call or, for L5, on the second. On R5 that is new: on 2166c0f9 it found the skill's files in a file listing and printed them instead. Without the skill it failed all six. On R8 and L7 it did fix the app's code, but the R8 answer never names `writeFloat`, and the L7 close runs on a new IO scope that does not wait for the engine's `initialize()`.
- Codex CLI: with the skill installed, it read `SKILL.md` on the first or second tool call in all six runs, before writing anything, as on 2166c0f9. R5 and L5, which it failed on 2166c0f9, now pass. R8 and R9 fail the way R5 failed there: its answer leaves out a fact that the file it read states (R8: `writeFloat` writes a shorter array without an error; R9: one `Environment` for the process). Without the skill it failed all six, on LiteRT 1.4.2 with its pre-LiteRT API for R7, on MediaPipe `tasks-genai` 0.10.27 for L5, and with no `onCleared` for L7.

Notes:

- R8: in Python (`ai-edge-litert`, on the CPU), the fixture model gives class 8 for a 224×224 test picture with the right input (RGB, each value / 255) and class 7 with the fixture app's input (one grey value per pixel in the first third of the input, zeros after it). All four R8 runs changed the preprocessing to the right input. The three ✗ are on what the answer says about `writeFloat`, and, without the skill, also on the missing element-count check.
- R8, Codex CLI with the skill: its code adds an element-count check before `writeFloat`, and its answer gives the cause as "supplying only 50,176 floats". It does not say that `writeFloat` takes the shorter array without an error, which `SKILL.md` and `verify.md` say.
- R9, Codex CLI with the skill: the answer has everything but one `Environment` for the process. `SKILL.md`, the only file it read, says "One `Environment` serves every model in the process and stays open".
- Without the skill, the agents found the API elsewhere. Claude Code downloaded the `litert` 2.3.0 AAR from Google Maven for R7, and read the `litertlm-android` 0.18.0 AAR in the test machine's Gradle cache with `javap` for L5 and L7. Codex CLI read the LiteRT-LM 0.18.0 JAR in the Gradle cache with `javap` and fetched LiteRT-LM's `Conversation.kt` from GitHub for L7. The web results of four Codex CLI runs held the `litert-runtime` skill page of developer.android.com: R7 and R9 without the skill, R5 and R9 with it.

## Tested on

- Antigravity CLI 1.3.1 on macOS, 2026-10-07, Gemini 3.8 Flash (High) and Gemini 3.1 Pro (High): the first four Flash runs with the skills through the CLI's sign-in, the rest through a Gemini API key.
- android/architecture-templates `base` at 5467a8dc (AGP 9.4.1, Kotlin 2.4.20, Gradle 9.8.0); `litert` 2.2.0 and `litertlm-android` 0.18.0 as the skills name them.
- Galaxy S26 (Android 16): the R1, R2, R3, L1 and L3 apps from the Flash runs with the skills, on the GPU; `Qwen3-0.6B.litertlm` for the two chats.
- Android Studio Rabbit 1 (2026.2.1), Agent Mode with its default model, Gemini 3.6 Flash.
- Claude Code 2.1.301 with claude-opus-5-5 and Codex CLI 0.162.1 with gpt-6-astra (the CLI's default effort, medium), headless on macOS, 2026-10-16; android/architecture-templates `base` at 5467a8dc; the skills of litert-samples 2166c0f9 (`litert` 2.3.0 with `litert-gpu` 2.3.0, `litertlm-android` 0.18.0); the apps of the runs with the skill on a Pixel 8a (Android 15) and, the R apps, also a Galaxy S26 (Android 16) of the Developer Device Platform lab, 2026-10-16, with `Qwen3-0.6B.litertlm` for the chats (the Device paragraph of that section). Antigravity CLI 1.4.0 stopped at its weekly quota and Gemini CLI 0.63.0 at `UNSUPPORTED_CLIENT`, so neither ran a prompt.
- Claude Code 2.1.301 with claude-opus-5-5 and Codex CLI 0.162.1 with gpt-6-astra (the CLI's default effort, medium), headless on macOS, 2026-10-16, for [After the change of 2026-10-16](#after-the-change-of-2026-10-16): android/architecture-templates `base` at 5467a8dc; the skills as updated on 2026-10-16 (this change: `litert` 2.3.0 with `litert-gpu` 2.3.0, `litertlm-android` 0.18.0); no device.
