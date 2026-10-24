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

If you already have Visual Studio 2026 Community, add its **Desktop development with C++** workload in the Visual
Studio Installer instead of the Build Tools line.

**Open a new terminal** (the old one does not find Git yet), install Flutter 3.47.3 into a folder without spaces and
put it on your PATH, then **open a new terminal again**:

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

Clone to `C:\litert-samples`: from a deeper folder, such as your user folder, the build's paths pass Windows'
260-character limit and every build after the first fails.

```sh
git clone --depth 1 https://github.com/google-ai-edge/litert-samples.git /c/litert-samples
cd /c/litert-samples/samples/litert/flutter
tool/flutter_litert/vendor.sh
```

## 3. The built-in models

```sh
tool/fetch_models.sh
```

No Hugging Face account is needed. The script derives the detector with Python; if you were given
`yolo26n_fp16_rawhead.tflite`, copy it into `assets/models/` first.

## 4. Build

Still in Git Bash, in `/c/litert-samples/samples/litert/flutter`:

```sh
flutter pub get
flutter build windows --release
powershell -NoProfile -ExecutionPolicy Bypass -File tool/windows/package.ps1
```

The first build downloads the LiteRT-LM runtime with the Intel NPU stack (OpenVINO) and takes several minutes.
`package.ps1` writes `build\dist\litert_edge_demos-v<version>-windows-x64\` with the launcher `run.cmd`, and a `.zip`
of it for another PC from the table above (that PC needs the latest Microsoft Visual C++ Redistributable, x64).

## 5. Run

Download the chat model for your PC into `litert-demos\models` in your user folder. In PowerShell, for a Lunar Lake PC
(use `gemma-4-E2B-it_intel_PTL.litertlm` on Panther Lake, or `gemma-4-E2B-it.litertlm` for the GPU):

```powershell
mkdir $env:USERPROFILE\litert-demos\models
curl.exe -L -C - -o $env:USERPROFILE\litert-demos\models\gemma-4-E2B-it_intel_LNL.litertlm https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it_intel_LNL.litertlm
```

Start the app with `run.cmd` in
`C:\litert-samples\samples\litert\flutter\build\dist\litert_edge_demos-v<version>-windows-x64\` (double-click it, or
run `.\run.cmd` in a terminal there). It checks the PC (runtime, GPU, NPU, microphone, speakers, camera), says what is
missing, then starts the app; each run is logged to `run.log`.

On **Set up models**, the **Chat model** card lists the files in `litert-demos\models` (**Rescan** if needed).
Choose your file, then **Use this model**:

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
- **Self-test:** checks the detector, the chat model, the speakers and the microphone, and saves a report (where each
  model ran, speed, memory); the `report` line says where. Exit code 0 means every step passed:

  ```powershell
  .\run.cmd --selftest --gemma=$env:USERPROFILE\litert-demos\models\gemma-4-E2B-it_intel_LNL.litertlm
  ```

  For the generic file add `--gemma-backend=gpu` or `cpu`; without a microphone and speakers add `--skip-audio`.

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
