# Prompt evals for the litert-runtime and litert-lm skills

Thirteen prompts, written the way a developer asks a coding agent, run in a fresh checkout of [android/architecture-templates](https://github.com/android/architecture-templates) (branch `base`, commit `5467a8dc`) once without the skills and once with [`litert-runtime`](../litert-runtime/SKILL.md) ([LiteRT](https://github.com/google-ai-edge/litert)) and [`litert-lm`](../litert-lm/SKILL.md) ([LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)) in `.agents/skills/`. Each prompt says what the developer wants and names no API; nothing else is given to the agent: no rules file, no other skill, no earlier conversation. The prompts and what a correct answer contains are in [litert-runtime.md](litert-runtime.md) (seven) and [litert-lm.md](litert-lm.md) (six); every run's row is in [results-litert-runtime.md](results-litert-runtime.md) and [results-litert-lm.md](results-litert-lm.md).

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
| L1 | ✗ | ✓ | ✗ | ✓ | — | — |
| L2 | ✗ | ✓ | — | ✗ | — | — |
| L3 | ✗ | ✓ | ✗ | ✓ | — | — |
| L4 | ✗ | ✓ | — | ✓ | — | — |
| L5 | ✗ | ✓ | — | ✓ | — | — |
| L6 | ✗ | ✓ | — | ✗ | — | — |

With the skills, every prompt passed on Gemini 3.8 Flash (High) and ten of thirteen on Gemini 3.1 Pro (High); a skill was opened in 13 of the 13 Flash runs and 12 of the 13 Pro runs: on R5 the Pro agent answered without opening one, on L2 it left out the `INTERNET` permission, and on L6 its `SamplerConfig` call did not compile. Without the skills, every Flash answer and three of the four Pro answers went to another library or to no runtime: `org.tensorflow:tensorflow-lite` 2.16.1 with the pre-LiteRT TensorFlow Lite API for the `.tflite` prompts, `com.google.mediapipe:tasks-genai` or a class that emits canned text for the LLM prompts; R6 was the one prompt a web search could answer. A cell is — where the run was not made: the Pro model ran without the skills on four prompts only, and the Android Studio runs are the prompts pasted by hand in the IDE.

Each prompt's Builds, Contains, Does not contain and Read lines are the checks an automated case of that prompt would carry, and every ✗ names the check that failed. Two changes to the skills follow from the runs: the `SamplerConfig` sentence of `litert-lm` now shows the argument types (a separate change), and the description of `litert-runtime` does not yet reach a question about an app that already runs a model, which is why the Pro agent answered R5 on its own.

## Tested on

- Antigravity CLI 1.3.1 on macOS, 2026-10-07, Gemini 3.8 Flash (High) and Gemini 3.1 Pro (High): the first four Flash runs with the skills through the CLI's sign-in, the rest through a Gemini API key.
- android/architecture-templates `base` at 5467a8dc (AGP 9.4.1, Kotlin 2.4.20, Gradle 9.8.0); `litert` 2.2.0 and `litertlm-android` 0.18.0 as the skills name them.
- Galaxy S26 (Android 16): the R1, R2, R3, L1 and L3 apps from the Flash runs with the skills, on the GPU; `Qwen3-0.6B.litertlm` for the two chats.
- Android Studio Rabbit 1 (2026.2.1), Agent Mode with its default model, Gemini 3.6 Flash.
