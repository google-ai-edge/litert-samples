# LiteRT Demos on an Intel Core Ultra PC (Windows)

How to build the Flutter sample in [`samples/litert/flutter`](../../litert/flutter/) on a Windows 11 PC with an Intel
Core Ultra processor and run its two demos there: a voice chat with Gemma and a live camera assistant. Gemma runs on
the PC's NPU; everything runs on the PC.

## What runs where

| Model | Runs on |
|---|---|
| Chat model, Gemma 4 E2B compiled for your processor's NPU (`gemma-4-E2B-it_intel_LNL.litertlm` or `gemma-4-E2B-it_intel_PTL.litertlm`, see below) | NPU; its vision and audio towers on the CPU |
| Chat model, the generic Gemma 4 E2B (`gemma-4-E2B-it.litertlm`) | GPU (Intel Arc, through DirectX 12) or CPU |
| YOLO26n detector (Live camera) | GPU (WebGPU on DirectX 12) |
| EmbeddingGemma 2 (knowledge base, visual memory), Whisper base and moonshine tiny (speech to text), Inflect-nano-v2 (spoken replies) | CPU |

## Which PCs

The NPU builds of Gemma 4 E2B are compiled for one processor family each, and a file for one family does not load on
the other:

| Processor | NPU file |
|---|---|
| Intel Core Ultra 200V series, "Lunar Lake" (for example Core Ultra 7 258V) | `gemma-4-E2B-it_intel_LNL.litertlm` |
| Intel Core Ultra series 3, "Panther Lake" (for example Core Ultra X7 358H) | `gemma-4-E2B-it_intel_PTL.litertlm` |

Your processor is under **Settings › System › About › Processor**; **Task Manager › Performance** shows the NPU. No NPU
build is published for other processors: on any other Windows x64 PC, use the generic file on the GPU or the CPU.
Windows on Arm (Snapdragon X) is not supported.

## What you need

| | |
|---|---|
| PC | Windows 11, 64-bit (x64), an Intel Core Ultra processor from the table above, 16 GB of RAM or more |
| Drivers | current Intel graphics and NPU drivers (Windows Update, or Intel Driver & Support Assistant) |
| Storage | about 30 GB free: Visual Studio, Flutter, the build and the models |
| Camera and audio | the laptop's own camera, microphone and speakers, or USB ones. An Android phone as a Wi-Fi camera also works ([network camera](../../litert/flutter/README.md#using-the-demos)) |
| Models | a chat model from [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm), no login needed: the NPU file for your processor (about 3 GB), or `gemma-4-E2B-it.litertlm` (2.6 GB) |

## 1. Prepare the PC

Turn on **Developer Mode** (**Settings › System › For developers**): Flutter needs it to build an app with plugins.

Install the tools from a PowerShell window (Start › Terminal), and allow the installers when Windows asks:

```powershell
winget install --id Git.Git -e --source winget
winget install --id Python.Python.3.12 -e --source winget
winget install --id Microsoft.NuGet -e --source winget
winget install --id Microsoft.VisualStudio.BuildTools -e --source winget --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.ATL --includeRecommended"
```

- Visual Studio Build Tools 2026 with **Desktop development with C++** and **C++ ATL** builds the Windows app (one of
  its plugins needs ATL). If you already have Visual Studio 2026 Community, add the **Desktop development with C++**
  workload in the Visual Studio Installer instead; ATL comes with it.
- NuGet fetches the WebView2 and WIL packages one of the app's plugins builds against.
- Python derives the detector in step 3. The `python3` that Windows offers from the Microsoft Store is only a
  shortcut to the Store; install Python as above.
- `--source winget` keeps winget from asking you to accept the Microsoft Store source's terms, which none of these
  packages need.

**Open a new terminal**: the one you used does not find Git yet. Install Flutter 3.47.3 there (keep `--branch 3.47.3`:
a shallow clone without the tag cannot tell its version) in a folder without spaces, such as `C:\flutter`, and put it
on your PATH, then **open a new terminal again**:

```powershell
git clone --depth 1 --branch 3.47.3 https://github.com/flutter/flutter.git C:\flutter
[Environment]::SetEnvironmentVariable('Path', "C:\flutter\bin;" + [Environment]::GetEnvironmentVariable('Path', 'User'), 'User')
```

```powershell
flutter --version    # the first run downloads the Dart SDK
flutter doctor
```

Warnings about Android, Chrome or the channel do not matter here; only **Visual Studio - develop Windows apps** does.

## 2. Get the sample

The set-up scripts are shell scripts: run steps 2 to 4 in **Git Bash** (installed with Git; Start › Git Bash).

Clone the samples to `C:\litert-samples`, not into your user folder. Some of the build's files lie over 200 characters
deep inside the sample; from `C:\Users\<you>\litert-samples` they pass the 260-character path limit of the Visual
Studio build tools, and every build after the first fails.

```sh
git clone --depth 1 https://github.com/google-ai-edge/litert-samples.git /c/litert-samples
cd /c/litert-samples/samples/litert/flutter
tool/flutter_litert/vendor.sh
```

## 3. The built-in models

```sh
tool/fetch_models.sh
```

- Every built-in model downloads without a Hugging Face account or token.
- The script derives the YOLO26n detector in a Python virtual environment (the `ai-edge-litert` wheel exists for
  Windows x64). If you were given `yolo26n_fp16_rawhead.tflite`, copy it into `assets/models/` first and the script
  only verifies it.
- `tool/fetch_models.sh --check` verifies every file without the network.

## 4. Build

Still in Git Bash, in `/c/litert-samples/samples/litert/flutter`:

```sh
flutter pub get
flutter build windows --release
powershell -NoProfile -ExecutionPolicy Bypass -File tool/windows/package.ps1
```

The first build downloads the LiteRT-LM runtime for Windows from GitHub, with the Intel NPU stack (OpenVINO). It takes
several minutes. `package.ps1` checks the bundle's models against the lock and writes
`build\dist\litert_edge_demos-v<version>-windows-x64\` (the app, the launcher `run.cmd` and the licences) and a `.zip`
of it, which runs on another PC from the table above that has the latest Microsoft Visual C++ Redistributable (x64);
`run.cmd` says when it is missing.

## 5. Run

Download the chat model for your PC into `litert-demos\models` in your user folder. In PowerShell, for a Lunar Lake PC
(use `gemma-4-E2B-it_intel_PTL.litertlm` on Panther Lake, or `gemma-4-E2B-it.litertlm` for the GPU):

```powershell
mkdir $env:USERPROFILE\litert-demos\models
curl.exe -L -C - -o $env:USERPROFILE\litert-demos\models\gemma-4-E2B-it_intel_LNL.litertlm https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it_intel_LNL.litertlm
```

Start the app with `run.cmd` in
`C:\litert-samples\samples\litert\flutter\build\dist\litert_edge_demos-v<version>-windows-x64\` (double-click it, or
run `.\run.cmd` in a terminal there). It checks the Visual C++ runtime, the GPU, the NPU and its driver, the microphone,
speakers and camera (and their privacy switches) and the free space in `%TEMP%`, says what is missing and how to fix
it, then starts the app. Each run is logged to `run.log` next to `run.cmd`.

On **Set up models**, the **Chat model** card lists the files in `litert-demos\models` (**Rescan** if needed,
**Path…** for a file elsewhere). Choose your file, then **Use this model**:

- An NPU file runs on the **NPU** only; the card offers nothing else for it.
- The generic file runs on the **GPU** or the **CPU**.

**This device** and the diagnostics overlay show where each model runs.

## 6. Use it and check it

- **Voice chat:** hold the mic button, speak, release; or type. Try "What is the input size of the YOLO26 nano
  detector?" (answered from the knowledge base, with sources) and "What time is it?", and type "Which accelerator are
  you running on?": the app answers it and names where each model runs.
- **Live camera:** choose the camera with the **Camera** button at the top of the demo. Hold the mic and ask "What do
  you see?", "How many people are there?". It also remembers what it saw: "Where was the yellow sign?" shows the frame
  and how long ago it was.
- **Self-test:** runs the detector on a reference picture and checks it against golden detections, loads the chat
  model and times 64 tokens, checks the speaker and microphone, and saves a report: processor, GPU and driver, NPU,
  where each model ran, speed and memory; the `report` line says where. Exit code 0 means every step passed:

  ```powershell
  .\run.cmd --selftest --gemma=$env:USERPROFILE\litert-demos\models\gemma-4-E2B-it_intel_LNL.litertlm
  ```

  An NPU file runs on the NPU; for the generic file add `--gemma-backend=gpu` or `cpu`. Add `--skip-audio` without a
  microphone and speakers.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Building with plugins requires symlink support` | Turn on Developer Mode (step 1) |
| The second build fails with `error MSB3501` … `Could not find a part of the path` | The sample's path is too long: clone it to `C:\litert-samples` (step 2) |
| `Unable to find suitable Visual Studio toolchain` | Install the **Desktop development with C++** workload (step 1); `flutter doctor` names what is missing |
| The build fails in `flutter_local_notifications_windows`: `Cannot open include file: 'atlbase.h'` | Add ATL: **Visual Studio Installer** › Build Tools › **Modify** › **Individual components**, search for "ATL", tick **C++ ATL for x64/x86 (Latest MSVC)** |
| The build fails in `flutter_inappwebview_windows` with `NUGET-NOTFOUND` | Install NuGet (step 1) and open a new terminal |
| `fetch_models.sh` finds no Python | Install Python (step 1); the Microsoft Store `python3` shortcut does not count |
| `flutter pub get` cannot find `third_party/flutter_litert` | Run `tool/flutter_litert/vendor.sh` (step 2) |
| `No file or variants found for asset: assets/models/…` | Run `tool/fetch_models.sh` (step 3) |
| `package.ps1: … exists; move it away first` | Delete `build\dist\litert_edge_demos-v*`, then run it again |
| The chat model fails on the NPU with `Failed to create engine on npu` | The file is for the other processor family: take the file from the table under [Which PCs](#which-pcs) |
| The card says the NPU is unavailable: no Intel NPU found | Update the Intel NPU driver; **Device Manager › Neural processors** must list it without a warning |
| `%TEMP%` grows by several GB (`litert_ov_bank_*.bin`) | The NPU runtime copies a model's weights there while it is loaded; the app deletes the ones left behind at the next NPU load. With the app closed, you can delete them |
| "No microphone" or no camera although one is connected | **Settings › Privacy & security › Microphone** (or **Camera**): allow desktop apps |
| No sound from replies | Choose the speakers in the taskbar's volume menu |
| Windows blocks `litert_edge_demos.exe` from a downloaded `.zip` | Right-click the `.zip` › Properties › **Unblock** before you extract it |
| `run.cmd`: Visual C++ runtime missing, or the app exits with 0xC0000005 at once | Install the latest Microsoft Visual C++ Redistributable (x64); `run.cmd` prints the link |
