# LiteRT Demos on the Arduino VENTUNO Q

How to build the Flutter sample in [`samples/litert/flutter`](../../../litert/flutter/) on an Arduino VENTUNO Q (Qualcomm Dragonwing
QCS8275) and run its two demos there, with Gemma 4 E2B on the board's Hexagon NPU: a voice chat and a live camera
assistant. Everything runs on the board.

## What runs where

| Model | Runs on |
|---|---|
| Chat model `gemma-4-E2B-it_Qualcomm_QCS8275.litertlm` | NPU (Hexagon V75), its vision and audio towers too |
| Chat model `gemma-4-E2B-it.litertlm` (alternative) | CPU or GPU (Adreno 623); its vision and audio towers on the CPU |
| YOLO26n detector (Live camera) | GPU (Adreno 623, through LiteRT's WebGPU accelerator on Vulkan); CPU when chosen in the Live camera settings |
| EmbeddingGemma 2 (knowledge base, visual memory), Whisper base and moonshine tiny (speech to text), Inflect-nano-v2 (spoken replies) | CPU |

## The board

| | |
|---|---|
| SoC | Qualcomm Dragonwing IQ8 (QCS8275): 4× Cortex-A78C + 4× Cortex-A55 |
| GPU | Adreno 623, Vulkan through Mesa (turnip) |
| NPU | Hexagon V75 |
| Memory | 16 GB (about 15 GB visible to Linux) |
| System | Ubuntu 24.04, glibc 2.39 |

## What you need

| | |
|---|---|
| Board | a VENTUNO Q with its Ubuntu 24.04 image, a screen, keyboard and mouse, and a network connection |
| Storage | about 18 GB free: Flutter, the build and the models |
| Camera | a USB camera the board sees as `/dev/video*`, or an Android phone as a Wi-Fi camera ([network camera](../../../litert/flutter/README.md#using-the-demos)) |
| Audio | a USB speakerphone, or a microphone and a speaker |
| Models | Gemma 4 E2B compiled for this chip, `gemma-4-E2B-it_Qualcomm_QCS8275.litertlm` (2.72 GB) from [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm), no login needed (step 5); the built-in models download in step 3 |

## 1. Prepare the board

```sh
dpkg --print-architecture                 # must print arm64
sudo apt update
sudo apt install --allow-downgrades libasound2t64/noble-updates libasound2-dev/noble-updates
sudo apt install -y git curl unzip xz-utils patch clang cmake ninja-build pkg-config lld \
  libgtk-3-dev liblzma-dev libstdc++-12-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  libasound2-dev libturbojpeg python3-venv \
  libvulkan1 vulkan-tools pulseaudio-utils gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  qcom-fastrpc1
sudo usermod -aG fastrpc "$USER"          # then log out and back in
vulkaninfo --summary | grep deviceName     # expect a line with Adreno623
```

The first install line takes ALSA and its headers from Ubuntu: the board's Arduino repository has a newer ALSA without
headers. If the board already has that newer one, apt asks before replacing it.

After logging in again, `ls -l /dev/fastrpc-cdsp /dev/dma_heap/system` should list both, and `groups` should include
`fastrpc`.

Install Flutter 3.47.3 (keep `--branch 3.47.3`: a shallow clone without the tag cannot tell its version):

```sh
git clone --depth 1 --branch 3.47.3 https://github.com/flutter/flutter.git ~/flutter
echo 'export PATH="$HOME/flutter/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
flutter --version                         # the first run downloads the Dart SDK
```

## 2. Get the sample

```sh
git clone --depth 1 https://github.com/google-ai-edge/litert-samples.git ~/litert-samples
cd ~/litert-samples/samples/litert/flutter
tool/flutter_litert/vendor.sh
```

## 3. The built-in models

If you were given the detector, copy it in first; the script fetches and verifies the rest:

```sh
mkdir -p assets/models
cp /path/to/yolo26n_fp16_rawhead.tflite assets/models/   # only if you were given it
tool/fetch_models.sh
```

No Hugging Face account is needed. Without a provided detector, the script derives it (see the
[sample's README](../../../litert/flutter/README.md#set-up-once-per-checkout)).

## 4. Build

```sh
flutter pub get
NO_XIPH_LIBS=1 flutter build linux --release
tool/linux/package.sh
```

The first build downloads the LiteRT-LM runtime and Qualcomm's NPU libraries (under Qualcomm's licence; the build
prints a notice). `package.sh` writes `build/dist/litert_edge_demos-v<version>-linux-arm64/` with the launcher `run.sh`,
and a `.tar.gz` of it.

## 5. Run

```sh
mkdir -p ~/litert-demos/models
wget -c -P ~/litert-demos/models https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it_Qualcomm_QCS8275.litertlm
cd build/dist/litert_edge_demos-v*-linux-arm64
./run.sh
```

`run.sh` checks the sound server, microphone, speaker, GPU (Vulkan) and camera, says what is missing and how to
install it, then starts the app. Each run is logged to `run.log` next to `run.sh`.

On **Set up models**, the **Chat model** card lists the files in `~/litert-demos/models/` (**Rescan** if needed).
Choose `gemma-4-E2B-it_Qualcomm_QCS8275.litertlm`, then **Use this model**:

- It runs on the **NPU** only: the card selects it and offers nothing else.
- If the card cannot offer the NPU, it says why ([Troubleshooting](#troubleshooting)).
- For the GPU or the CPU, use `gemma-4-E2B-it.litertlm`.

The detector runs on the GPU by default (Live camera settings › **Detector**). **This device** and the diagnostics
overlay show where each model runs.

## 6. Use it

- **Voice chat:** hold the mic button, speak, release; or type. Try "What is the input size of the YOLO26 nano
  detector?" (answered from the knowledge base, with sources) and "Which accelerator are you running on?". Tools
  are off for the NPU build: type the time and accelerator questions, which the app answers itself.
- **Live camera:** choose the camera with the **Camera** button at the top of the demo: a camera on the board, or a
  network camera such as a phone running IP Webcam. Hold the mic and ask "What do you see?", "How many people are
  there?", "Describe the scene". It also remembers what it saw: "Where was the yellow sign?" shows the frame and how
  long ago it was.

## 7. Self-test

After **Use this model** on the card (step 5):

```sh
./run.sh --selftest
```

It checks the detector, the chat model, the speaker and the microphone, and saves a report (where each model ran, speed,
memory); the `report` line says where. Exit code 0 means every step passed. Add `--skip-audio` without a microphone and
speaker; over SSH, install `xvfb` and put `xvfb-run -a` in front. In the app: **This device › Copy diagnostics**.

## Troubleshooting

| Symptom | Fix |
|---|---|
| The card does not offer the NPU: "not accessible to this user" | `sudo usermod -aG fastrpc "$USER"`, then log out and back in |
| The card does not offer the NPU: `libcdsprpc.so.1 did not open` | `sudo apt install qcom-fastrpc1` |
| The card does not offer the NPU: no `/dev/fastrpc-cdsp` | The kernel exposes no compute DSP: use the board's own Ubuntu image |
| GPU and CPU are greyed out for `…_Qualcomm_QCS8275.litertlm` | It is compiled for the NPU and runs only there: for the CPU or the GPU use `gemma-4-E2B-it.litertlm` |
| The build stops reading the QAIRT zip | No network to Qualcomm's download: download the QAIRT SDK zip once and set `qualcomm_npu_qairt_zip: /path/to/the.zip` next to `qualcomm_npu: true` in `pubspec.yaml` |
| `flutter pub get` cannot find `third_party/flutter_litert` | Run `tool/flutter_litert/vendor.sh` (step 2) |
| `No file or variants found for asset: assets/models/…` | Run `tool/fetch_models.sh` (step 3) |
| `flutter build linux` fails linking `libFLAC` / `libvorbis` | Build with `NO_XIPH_LIBS=1` (step 4) |
| After a failed build, `file INSTALL cannot copy file … to "/usr/local/…": Permission denied` | CMake kept a default install prefix from the failed run: `rm -r build/linux` and build again |
| `package.sh: … exists; move it away first` | `rm -r build/dist/litert_edge_demos-v*`, then run it again |
| `pactl did not run: install pulseaudio-utils` | `sudo apt install pulseaudio-utils` |
| An NPU build for another chip fails to load | NPU builds are compiled per chip: on this board use `gemma-4-E2B-it_Qualcomm_QCS8275.litertlm` |
| `libasound2-dev : Depends: libasound2t64 (= …) but … is to be installed` | Install both from Ubuntu first: `sudo apt install --allow-downgrades libasound2t64/noble-updates libasound2-dev/noble-updates` (step 1) |
| `flutter build linux` fails: `ALSA_LIBRARY` is `NOTFOUND`, or `alsa/asoundlib.h` is not found | `libasound2-dev` is missing: run step 1's ALSA line again |
| No camera picture | Use a phone as a network camera ([sample README](../../../litert/flutter/README.md#using-the-demos)) |
