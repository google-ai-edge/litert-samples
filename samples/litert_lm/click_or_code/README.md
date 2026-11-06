# Click or code on Android and macOS with LiteRT-LM

## What this sample demonstrates

- The next step of a computer-use agent, on the phone: pick a screenshot (or tap a bundled example), say what you want done, tap **Decide**, and the app answers with one step: a point to click, ringed on the screenshot, or a shell command in a block you can copy. The screenshot stays on the phone. The app shows the step and runs nothing.
- [Mage-VL-Cowork](https://huggingface.co/microsoft/Mage-VL-Cowork) (Microsoft, 4B parameters: a 24-layer vision encoder and a Qwen3-4B text decoder), whose card calls it "One compact 4B policy that knows when to click — and when to code": for each turn it chooses `computer_use` or `bash` itself. It is converted to a LiteRT-LM bundle, with its text decoder on the GPU and its vision encoder on the CPU, through the LiteRT-LM Kotlin API (`com.google.ai.edge.litertlm:litertlm-android:0.19.0`).
- The requests as the model's own hybrid agent sends them ([`mage_hybrid_agent.py`](https://huggingface.co/microsoft/Mage-VL-Cowork/blob/main/osworld_eval/mm_agents/mage_hybrid_agent.py)): `MAGE_HYBRID_SYSTEM_PROMPT` word for word, the screenshot and your sentence as the instruction. The model's first reply is often a call for a screenshot; the app answers that call with the same picture and asks again, as the agent's loop does. So one step takes one or two inferences, and the screen says how many ran and the seconds of each. In the second request the first picture is replaced by the agent's own line for a dropped picture, `[This screenshot has been collapsed.]`, because the bundle has room for one screenshot (`Step.kt`).
- The bundle reads one 1920 × 1088 picture (2,040 image tokens). LiteRT-LM resizes every picture to that size without keeping its aspect ratio, so the app turns the screenshot upright by its EXIF orientation and stretches it to 1920 × 1088 with Pillow's bicubic filter (`Pictures.kt`, `PilBicubic.kt`), as in every check below; a phone screenshot is stretched the same way. The model's point comes back in coordinates from 0 to 999 of the whole picture, and the app converts it to your screenshot's pixels as the agent does: round(x / 999 × (width − 1)), round(y / 999 × (height − 1)).

The sections below describe the Android app. The macOS app, on a screenshot of your own Mac, is in [macOS](#macos).

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

## macOS

The app in [`macos/`](macos/) works on your own Mac's screen. Capture the screen or one window, or open a PNG, type what you want done, and click **Decide**. The answer is a point, ringed on the screenshot, or a command with a **Copy** button. The app shows the step and runs nothing: to run a command, paste it into Terminal yourself. The screenshot stays on the Mac.

It sends the same requests as the Android app: the same system prompt file, the same two steps, the same 1920 × 1088 stretch. It calls LiteRT-LM's C API (the `CLiteRTLM_mac.xcframework` of the v0.19.0 release, pinned by URL and SHA-256 in `macos/LiteRTLMRuntime/Package.swift`). The text decoder and the vision encoder both run on the GPU (WebGPU over Metal).

The model file is not the Android app's. It is [`Mage-VL-Cowork.litertlm`](https://huggingface.co/litert-community/Mage-VL-Cowork) (3.80 GB, Apache-2.0), with its vision encoder on the GPU. The app downloads it on first use into `~/Library/Application Support/ClickOrCode/models`, goes on from the same byte after a stopped download, and checks its size and SHA-256 against [`macos/Resources/models.json`](macos/Resources/models.json).

### Requirements

- A Mac with Apple silicon and 13 GB of memory or more. The app refuses to start the engine below that: it counts 7 GB for the engine and 6 GB for macOS. Everything below was measured on a Mac Studio with an M4 Max and 128 GB; a smaller Mac was not tried.
- Xcode and [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`). The app targets macOS 14 and later. It was built with Xcode 27.0 and XcodeGen 2.44.1 and run on macOS 27.0 only.
- Room for the download: the app asks for twice the file's size, 7.6 GB, free.
- For **Capture screen** and **Capture window…**: the Screen & System Audio Recording permission. macOS asks the first time. **Open picture…**, **Paste Picture** and the examples work without it.

### Quickstart

```bash
cd samples/litert_lm/click_or_code/macos
xcodegen generate
xcodebuild -project ClickOrCode.xcodeproj -scheme ClickOrCode -configuration Release -derivedDataPath build build
open build/Build/Products/Release/ClickOrCode.app
```

The first build downloads the LiteRT-LM runtime (47 MB) and checks its SHA-256. In the app, click **Download (3.8 GB) and load**, or **Use a copy on this Mac…** if you already have the file. Then:

1. Click **Capture window…** and click the window you mean (Esc cancels). **Capture screen**, **Open picture…**, **Paste Picture** (⇧⌘V) and **Examples** give the app a screenshot too.
2. Type what you want done and click **Decide**.
3. A click comes back as a ring on the screenshot, its pixels, and a close-up of that spot. A command comes back in a block: read it, click **Copy**, and paste it into Terminal yourself.

### Measured in this app

A Mac Studio with an M4 Max and 128 GB, macOS 27.0, nothing else heavy running: 76 steps and 145 inferences, the app's own numbers from the recorder's JSON files. The window shows the same numbers after each step.

| What | Measured |
|---|---|
| The engine's start until **Ready**, the first time (it writes a 2.3 GB cache to `~/Library/Caches/ClickOrCode/engine`) | 8.7 s |
| The engine's start after that | 6.9 to 9.8 s, median 7.3 s |
| One inference, the vision encoder | 3.0 to 3.2 s |
| One inference, prefill | 3.2 to 3.3 s: 3,179 to 3,470 tokens at 956 to 1,068 tok/s |
| One inference, decode | 131 to 390 tokens at 75 to 90 tok/s |
| A step with one inference | 9.5 to 10.9 s |
| A step with two inferences | 16.2 to 19.4 s |
| The engine process's highest memory footprint | 4.9 GiB with that cache, 6.4 GiB without it |

### On a real Mac

The model writes for a Linux desktop. Its system prompt names no operating system, yet on a real Finder window it answered with `/home/user/...` paths and Ctrl+Alt+T. Read a command before you run it, and expect to change its path.

Where a click lands depends on what the target looks like. These requests and their boxes were written before any reply was read, on one capture of each window:

| Window | The target | In the box |
|---|---|---|
| System Settings, Desktop & Dock, with the sidebar cut off | A switch at the far end of its row. "Hide the Dock automatically." gave a click inside the switch, and the answer named the row: "Click the “Automatically hide and show the Dock” switch to turn it on." | 5 of 5 |
| Finder's toolbar, with the sidebar cut off | An icon with no text. "Search this folder." landed; "Switch to the list view." gave "Click the folder window and press Ctrl+2 to switch it to list view." | 1 of 7 (5 clicks, 2 commands) |
| Preview's two toolbars | An icon with no text. "Rotate this photo." gave "Click the Image menu to open the image operations options.", and that window has no such menu | 0 of 10 (8 clicks, a key press, a command) |

In Finder's icon view, "Open the photo of the sailboat." gave a double-click on the right photo on all 5 captures of `~/trip` we tried.

One Finder window (`~/trip`: 9 .JPG files, 3 hidden .JPG files, a text file), one picture, 12 wordings of the same request:

| What came back | Wordings |
|---|---|
| A command that runs on this Mac and counts 12. "Count the .JPG files in ~/trip, including hidden ones." gave `find "$HOME/trip" -type f -iname '*.jpg' \| wc -l` | 2 of 12 |
| A command for `/home/user/trip`, which does not exist here. "How many .JPG files are in ~/trip, hidden ones included?" gave `find /home/user/trip -type f -iname '*.jpg' \| wc -l` | 5 of 12 |
| No command, a key press to open a terminal. "In Terminal, count the .JPG files in ~/trip, hidden ones included." gave Ctrl+Alt+T | 5 of 12 |

The picture changes the answer too. Two captures of this window that look the same, one made with the screen unlocked and one locked, differ only in green, by 1 of 255, on 82 % of the pixels (the half-transparent corners aside). On the first, "Count the .JPG files in ~/trip, including hidden ones." gave the `find "$HOME/trip"` command above. On the second it gave a listing of `/home/user/trip`, and of a second list of 12 wordings 2 ran and counted 12, 7 named `/home/user/trip` and 3 pressed a key. The same picture and the same sentence gave the same text every time: 7 pairs, each run 2 or 3 times.

The Mac rows in the tables above are the Android app's file. With this app's file, through this app, on the 11 requests kept out of every choice: 9 of 11 chose the expected tool, 3 of 6 click requests landed in the expected box, all 11 calls parsed as JSON and all 11 replies ended by themselves. One request makes the difference from the Android file on the same Mac: "Save the document." came back as a command instead of a click.

### When something goes wrong

Each case below was made to happen on the Mac. The app stayed up and said so:

| What happened | What the window shows |
|---|---|
| A capture without the Screen Recording permission (the app opened with `open`) | macOS's own dialog, and: "macOS has not allowed this app to take screenshots. Turn on Click or Code in System Settings > Privacy & Security > Screen & System Audio Recording, then quit and open the app again. Or open a saved screenshot with Open picture…" |
| Esc in the window picker | Nothing changes |
| A file that is not a picture (a `.png` name on 39 bytes of text) | "macOS could not read “not_a_picture.png” as a picture. It may be damaged, cut short, or not a picture at all." |
| **Decide** with no text | "Type what you want done on this screen." |
| A request too long for the file (a sentence repeated 700 times) | "The request is too long for the model file. Shorten it and decide again." |
| A reply that reaches its limit (tried with `-maxOutputTokens 30`) | "The reply reached its token limit before it gave a step." |
| **Stop**, 1 s into a step | "Stopped.", with **Decide** back, 3.0 s after Decide was pressed: **Stop** takes effect when the vision encoder has finished |
| The engine process killed in the middle of a reply (SIGKILL) | **The engine stopped**: signal 9, what ends a process this way, the engine log, and **Start the engine again**, which works. The step says "The engine stopped before the step was finished." |
| The engine process stopped in the middle of a reply for longer than LiteRT-LM waits for the GPU (SIGSTOP for 150 s) | **The engine took too long**, the engine log, and **Start the engine again**, which works. The step says "LiteRT-LM gave up on this step: a wait passed its time limit (DEADLINE_EXCEEDED: DEADLINE_EXCEEDED: Timed out waiting for future: 2m)." |
| A Mac with too little memory (tried with `-assumePhysicalMemoryGB 8`) | **Not enough memory**, before anything is loaded: "The engine used up to 7.0 GB of memory in this app, and macOS needs about 6.0 GB besides. This Mac has 8 GB. Nothing was loaded." |
| A download that stops | The reason ("The request timed out." on the real file) and **Try again**, or after a cancel **Go on with the download (3.7 GB of 3.8 GB is here)**. Both go on from the same byte, and the finished file's SHA-256 matched |
| A download with the wrong content (tried with small test catalogs) | "“small.litertlm” has SHA-256 …, not …. Nothing was loaded.", the same for a wrong size. The download also stops on HTTP 404, on a redirect that is not HTTPS, on a body longer than the file, and when there is not enough free space |
| **Use a copy on this Mac…** with another file | "“readme_full.bin” is 46356 bytes and this app's file is 3803948496 bytes. Nothing was loaded." |
| A file with the right hash that LiteRT-LM cannot read | "Not a model this app can run", with LiteRT-LM's error |

### The recorder

`--autoplay` runs a script of the same actions a person takes and writes each answer, its timing and the engine's record to a JSON file:

```bash
build/Build/Products/Release/ClickOrCode.app/Contents/MacOS/ClickOrCode \
  --model ~/models/Mage-VL-Cowork.litertlm \
  --autoplay checks/examples.json --run-out /tmp/run.json --snapshot-dir /tmp/shots
```

`checks/examples.json` runs the two steps of the Android example screenshots and checks them. `checks/failures.json` runs the failures above. The script format is at the top of `Sources/Autoplay.swift`.

### How it works

| File in `macos/` | What it holds |
|---|---|
| `Sources/main.swift`, `Sources/App.swift` | One binary, two modes: the window, or (with `--engine-host`) the engine process. The menu commands and the command-line options |
| `Sources/LiteRTLM.swift` | The Swift layer over the C API: the engine (GPU for the decoder and the vision encoder), a conversation per inference, the stream, Stop, the runtime's counters |
| `Sources/EngineHost.swift`, `Sources/EngineClient.swift`, `Sources/Protocol.swift` | The engine process and the window's side of it, in JSON lines. An engine that dies ends that process, not the app |
| `Sources/Step.swift` | The agent's two steps as text and the reading of a reply: `Step.kt` in Swift |
| `Sources/Pictures.swift`, `Sources/Picture.swift` | The 1920 × 1088 stretch (Pillow 13.0.0's bicubic resize), the EXIF turn, the screenshots |
| `Sources/Models.swift`, `Sources/SystemMemory.swift` | The reader of `models.json`, the download (resume, HTTPS only, size, SHA-256), the memory check |
| `Sources/AppModel.swift`, `Sources/ContentView.swift`, `Sources/Examples.swift` | The window's state and actions, its views, the bundled examples |
| `Sources/Autoplay.swift`, `checks/` | The recorder and its scripts |
| `Tests/StepTests.swift` | 15 unit tests: the reading of a reply and the stretch, against the Pillow rows the Android tests use |

## Licenses

The sample code is Apache-2.0. Mage-VL-Cowork's weights are Apache-2.0 (Microsoft); the converted file keeps that license, and the system prompt in `app/src/main/assets/` is the model repository's, unchanged (see [`android/NOTICE`](android/NOTICE)). The Pillow-derived resize is under Pillow's license ([`android/licenses/`](android/licenses/)). The macOS app uses the Android app's system prompt, example screenshots and Pillow rows by reference (see [`macos/NOTICE`](macos/NOTICE)).
