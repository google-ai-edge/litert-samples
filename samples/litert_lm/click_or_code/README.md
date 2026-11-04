# Click or code on Android with LiteRT-LM

## What this sample demonstrates

- The next step of a computer-use agent, on the phone: pick a screenshot (or tap a bundled example), say what you want done, tap **Decide**, and the app answers with one step: a point to click, ringed on the screenshot, or a shell command in a block you can copy. The screenshot stays on the phone. The app shows the step and runs nothing.
- [Mage-VL-Cowork](https://huggingface.co/microsoft/Mage-VL-Cowork) (Microsoft, 4B parameters: a 24-layer vision encoder and a Qwen3-4B text decoder), whose card calls it "One compact 4B policy that knows when to click — and when to code": for each turn it chooses `computer_use` or `bash` itself. It is converted to a LiteRT-LM bundle, with its text decoder on the GPU and its vision encoder on the CPU, through the LiteRT-LM Kotlin API (`com.google.ai.edge.litertlm:litertlm-android:0.19.0`).
- The requests as the model's own hybrid agent sends them ([`mage_hybrid_agent.py`](https://huggingface.co/microsoft/Mage-VL-Cowork/blob/main/osworld_eval/mm_agents/mage_hybrid_agent.py)): `MAGE_HYBRID_SYSTEM_PROMPT` word for word, the screenshot and your sentence as the instruction. The model's first reply is often a call for a screenshot; the app answers that call with the same picture and asks again, as the agent's loop does. So one step takes one or two inferences, and the screen says how many ran and the seconds of each. In the second request the first picture is replaced by the agent's own line for a dropped picture, `[This screenshot has been collapsed.]`, because the bundle has room for one screenshot (`Step.kt`).
- The bundle reads one 1920 × 1088 picture (2,040 image tokens). LiteRT-LM resizes every picture to that size without keeping its aspect ratio, so the app turns the screenshot upright by its EXIF orientation and stretches it to 1920 × 1088 with Pillow's bicubic filter (`Pictures.kt`, `PilBicubic.kt`), as in every check below; a phone screenshot is stretched the same way. The model's point comes back in coordinates from 0 to 999 of the whole picture, and the app converts it to your screenshot's pixels as the agent does: round(x / 999 × (width − 1)), round(y / 999 × (height − 1)).

## The model

| Hugging Face repo | File | Size | License |
|---|---|---|---|
| [litert-community/Mage-VL-Cowork](https://huggingface.co/litert-community/Mage-VL-Cowork) | `Mage-VL-Cowork_vision-fp16.litertlm` (int4 text decoder with fp16 activations on the GPU and a 4,096-token context, fp16 vision encoder) | 3.20 GB | Apache-2.0 |

The app downloads the file on first use into its private storage and checks its size and SHA-256 against [`app/src/main/assets/models.json`](android/app/src/main/assets/models.json). The model file is not in the APK. The bundle is a conversion of [microsoft/Mage-VL-Cowork](https://huggingface.co/microsoft/Mage-VL-Cowork); its card says what changed.

## Requirements

JDK 17 and Android SDK platform 36 to build. The app builds for arm64-v8a on Android 12+ (minSdk 31). The first start downloads 3.20 GB and needs twice that free in the app's storage. Before it loads the model, the app checks that about 6 GB of memory is free: over the steps below, free memory fell by at most 5.7 GB on a Galaxy S26 and 5.8 GB on a Pixel 11 Pro. Tested on those two phones.

From a phone that had cooled down, a step with two inferences took 103 to 116 s on a Galaxy S26 and 240 s on a Pixel 11 Pro. A step with one inference took 69 to 74 s and 172 s. Steps one after another heat the phone: on the Galaxy S26 a step with two inferences reached 334 s after 17 minutes of steps.

## What it decides, and what it does not

The app shows the model's next step and runs nothing, which matters for reading its answers:

- A command is often a look around first: `ls -la`, `find`, a short Python script that reads the file. The agent would read that output and go on at its next step.
- The model chooses the screen more often than you might expect. When the target has to be found by sight ("Copy the file with the red icon to the Desktop.") or the request is an edit inside an open app ("Replace every 'colour' with 'color' in this file and save it."), it usually clicks.

The choice can change with the wording and with the machine. These are the same file and the same two steps, except the last column, which is the original weights in PyTorch on a CPU:

| Request | Mac | Galaxy S26 | Pixel 11 Pro | Original weights |
|---|---|---|---|---|
| Turn off Bluetooth. | click | click | click | click |
| Count the .JPG files in this folder, hidden ones included. | command | command | not run | command |
| How many pages does this PDF have? Check the file itself. | command | click | command | command |
| Check the file to find out how many pages this PDF has. | command | click | not run | not run |
| Count the pages of this PDF from the file itself. | command | command | not run | click |

Checked with this file in the app's form (text decoder on the GPU, vision encoder on the CPU, the two steps above), with screens and requests written for this sample and the expected answer written down before any reply was read:

| Requests | Where | Chose `computer_use` or `bash` as expected | Click inside the expected box | Call parses as JSON | Reply ended by itself |
|---|---|---|---|---|---|
| 11 kept out of every choice | Galaxy S26, through this app | 10 of 11 | 4 of 6 | 11 of 11 | 11 of 11 |
| The same 11 | Mac | 10 of 11 | 4 of 6 | 11 of 11 | 11 of 11 |
| 12 first candidates | Mac | 11 of 12 | 4 of 6 | 12 of 12 | 12 of 12 |
| 11 used to choose the form of the request | Mac | 8 of 11 | 4 of 6 | 11 of 11 | 11 of 11 |
| 30, the first 12 and harder ones added later | Mac | 22 of 30 | 8 of 11 | 30 of 30 | 30 of 30 |

On those 11 the Galaxy S26 and the Mac chose the same tool every time, and their clicks hit and missed the same boxes. Their commands can differ: for "List the slide titles of this deck from the file." the Mac's command reads the titles with `python-pptx`, and the phone's first looks for the file.

On 7 of these requests the original weights (PyTorch, CPU) chose the same tool as the Mac on every one, and their two clicks landed in the same boxes.

## Architecture

```
a screenshot (system photo picker, or a bundled example) and a typed request
  -> turned upright by its EXIF orientation
  -> Pictures.stretch: 1920 x 1088 (Pillow's bicubic resize, no padding)
  -> Conversation with the system prompt
  -> sendMessageAsync(picture + the agent's step text)
  -> the reply, streamed: its reasoning, an Action line, a <tool_call>
  -> when that call asks for a screenshot: a new Conversation with the reply in its history,
     sendMessageAsync(<tool_response> picture </tool_response>), and the same reading
  -> computer_use: the point, ringed on the screenshot; bash: the command, in a block you can copy
one Engine for the app; one Conversation per inference
```

| File | What it holds |
|---|---|
| `StepEngine.kt` | The LiteRT-LM `Engine` (text decoder on the GPU, vision encoder on the CPU), one `Conversation` per inference, the stream, Stop, the runtime's counters |
| `Step.kt` | The agent's two steps as text, the reading of a reply (reasoning, `Action:` line, tool calls), the conversion of a point to pixels |
| `Pictures.kt`, `PilBicubic.kt` | The 1920 × 1088 stretch: Pillow 13.0.0's bicubic resize |
| `Photo.kt` | Decoding a photo as Pillow reads it (no color conversion), the EXIF turn, PNG for the runtime |
| `MemoryCheck.kt` | The free-memory check before the `Engine` is made |
| `Examples.kt` | The bundled screenshots and their example requests (`assets/examples/`) |
| `StepViewModel.kt`, `StepScreen.kt` | The screen: the screenshot with its ring, the request, Decide, the answer, the seconds of each inference and its stages |
| `AutoRun.kt` | Debug builds: a plan of steps through Decide's code, one result line each |
| `data/` | The reader of `models.json`, the free-space rule and the download store (resume, SHA-256, HTTPS only), from [`voice_assistant`](../voice_assistant/) |

## Quickstart

```sh
cd samples/litert_lm/click_or_code/android
./gradlew :app:installDebug
```

Open the app and tap **Download (3.2 GB) and load**. Then tap a bundled example or **Screenshot…**, type what you want done (or tap a suggested request), and tap **Decide**. **Stop** takes effect when the vision encoder has finished: tapped 10 s into a step, it took 13 to 26 s more on a Galaxy S26 and 72 s more on a Pixel 11 Pro.

Stay in the app while the file downloads. On a Galaxy S26, pressing Home ended the download 5 to 6 s later: the screen then read `Software caused connection abort`, and **Try again** resumed from the same byte.

## Development

**Side-load (development only).** A copy of the model file in the app's external files dir is hashed and, when its size and SHA-256 match `models.json`, imported instead of downloaded; the next **Download (3.2 GB) and load** logs `side-loaded <file> ... (sha256 verified)` under `adb logcat -s ClickOrCodeDownload`:

```sh
P=com.google.ai.edge.examples.click_or_code
adb shell am start -n $P/.MainActivity
adb push Mage-VL-Cowork_vision-fp16.litertlm /sdcard/Android/data/$P/files/
```

**Another file (debug builds).** `--es bundle <path>` runs a file the app can read instead of the catalog's, with no hash check: `adb shell am start -n $P/.MainActivity --es bundle /sdcard/Android/data/$P/files/other.litertlm --ez autoload true`.

**Runs (debug builds).** `--es plan <name>` reads `plans/<name>.json` from the external files dir and drives the same code as the buttons, one item per step, and writes `runs/<name>.jsonl`: each line has the request, each inference's reply and the seconds of its stages, the point or the command and the free memory. An item is `{"tag", "example": "<id>" | "image": "<path>", "request", "render", "cancel_after_ms", "need_bytes"}`.

**The example screenshots** are drawn by [`tools/make_scenes.py`](android/tools/make_scenes.py) (HTML, DejaVu fonts, Chrome headless); every window, icon and picture is written for this sample. `tools/make_picture_rows.py` writes the Pillow rows the unit tests compare the stretch with.

## Licenses

The sample code is Apache-2.0. Mage-VL-Cowork's weights are Apache-2.0 (Microsoft); the converted file keeps that license, and the system prompt in `app/src/main/assets/` is the model repository's, unchanged (see [`android/NOTICE`](android/NOTICE)). The Pillow-derived resize is under Pillow's license ([`android/licenses/`](android/licenses/)).
