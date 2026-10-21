# LiteRT Demos on a Raspberry Pi 5

How to build the Flutter sample in [`samples/litert/flutter`](../../litert/flutter/) on a Raspberry Pi 5 and run its two demos there: a
voice chat with Gemma and a live camera assistant. Everything runs on the Pi.

## What runs where

| Model | Runs on |
|---|---|
| Chat model, Gemma 4 E2B (`gemma-4-E2B-it.litertlm`) | CPU (select it on the card, step 5); its vision and audio towers on the CPU |
| YOLO26n detector (Live camera) | GPU on Raspberry Pi OS Trixie; on Bookworm the CPU: **Detector › CPU (slower)** in the Live camera settings, or **Run detector on CPU** when the GPU is refused (step 5) |
| EmbeddingGemma 2 (knowledge base, visual memory), Whisper base and moonshine tiny (speech to text), Inflect-nano-v2 (spoken replies) | CPU |

The app offers no NPU on the Pi.

## What you need

| | |
|---|---|
| Board | Raspberry Pi 5 with **8 GB or 16 GB**; 4 GB is not enough for Gemma 4 E2B |
| System | **Raspberry Pi OS (64-bit)** with desktop, Bookworm or newer; Trixie for the detector on the GPU. The 32-bit system does not work |
| Storage | an NVMe SSD, or a fast (A2) microSD card; about 15 GB free for Flutter, the build and the models |
| Cooling and power | the Active Cooler (the models keep all four cores busy) and the official 27 W supply |
| Camera | a USB webcam, or an Android phone as a Wi-Fi camera ([network camera](../../litert/flutter/README.md#using-the-demos)). The Camera Module (CSI) is not supported by the app |
| Audio | a USB speakerphone, or a USB microphone and a speaker. The Pi 5 has no audio jack |
| Models | a chat model: Gemma 4 E2B, `gemma-4-E2B-it.litertlm` (2.6 GB) from [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm), no login needed |

## 1. Prepare the Pi

```sh
dpkg --print-architecture                 # must print arm64
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y git curl unzip xz-utils patch clang cmake ninja-build pkg-config lld \
  libgtk-3-dev liblzma-dev libstdc++-12-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  libasound2-dev libturbojpeg0 python3-venv \
  libvulkan1 mesa-vulkan-drivers vulkan-tools \
  pulseaudio-utils gstreamer1.0-plugins-base gstreamer1.0-plugins-good
sudo reboot
```

`pulseaudio-utils` gives the app `pactl` and `parecord` for the sound check and the microphone; it works with the
Pi's PipeWire sound server.

Install Flutter 3.47.3 (the Linux arm64 SDK comes from the same Git repository; keep `--branch 3.47.3`, a shallow
clone without the tag cannot tell its version):

```sh
git clone --depth 1 --branch 3.47.3 https://github.com/flutter/flutter.git ~/flutter
echo 'export PATH="$HOME/flutter/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
flutter --version                         # the first run downloads the Dart SDK
```

`flutter doctor` warnings about Android, Chrome or the channel do not matter here; only **Linux toolchain** does.

## 2. Get the sample

```sh
git clone --depth 1 https://github.com/google-ai-edge/litert-samples.git ~/litert-samples
cd ~/litert-samples/samples/litert/flutter
tool/flutter_litert/vendor.sh
```

## 3. The built-in models

```sh
tool/fetch_models.sh
```

- Every built-in model downloads without a Hugging Face account or token.
- The script derives the YOLO26n detector in a Python venv (the `ai-edge-litert` wheel exists for Linux aarch64). If
  you were given `yolo26n_fp16_rawhead.tflite`, copy it into `assets/models/` first and the script only verifies it.
- `tool/fetch_models.sh --check` verifies every file without the network.

## 4. Build

```sh
flutter pub get
NO_XIPH_LIBS=1 flutter build linux --release
tool/linux/package.sh
```

`NO_XIPH_LIBS=1` leaves out codec libraries of the audio player that exist only for x64; the app plays raw PCM and
does not need them. The first build downloads the LiteRT-LM runtime for Linux arm64 from GitHub. `package.sh` checks
the bundle's models against the lock and writes `build/dist/litert_edge_demos-v<version>-linux-arm64/` (the app,
the launcher `run.sh` and the licences) and a `.tar.gz` of it, which runs on other arm64 boards with the same OS
release or a newer one.

## 5. Run

```sh
mkdir -p ~/litert-demos/models
wget -c -P ~/litert-demos/models https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm
cd build/dist/litert_edge_demos-v*-linux-arm64
./run.sh
```

`run.sh` checks the sound server, microphone, speaker, GPU (Vulkan) and camera, says what is missing and how to
install it, then starts the app. Each run is logged to `run.log` next to `run.sh`. It also lists `llvmpipe` with a
WARN: that is Mesa's CPU driver and expected; the Pi's GPU is the `V3D` line.

On **Set up models**, the **Chat model** card lists the files in `~/litert-demos/models/` (**Rescan** if needed,
**Path…** for a file elsewhere). Choose `gemma-4-E2B-it.litertlm`, select **CPU**, then **Use this model**.

**GPU or CPU.** Keep the chat model on the CPU: the Pi 5's GPU (VideoCore VII) is far slower for it. The detector runs
on the GPU through LiteRT's WebGPU accelerator (Dawn) on Vulkan, with the `v3dv` driver of Raspberry Pi OS Trixie
(Mesa 26.2 or newer, installed in step 1). That driver offers WebGPU the features it requires only with
`V3D_WEBGPU_OVERRIDE=1`, which `run.sh` sets on a Raspberry Pi 5. With Bookworm's driver (Mesa 24.2) Dawn picks
`llvmpipe`, Mesa's CPU driver, instead; the app refuses it as a software GPU and offers **Run detector on CPU** (or
choose **Detector › CPU (slower)** in the Live camera settings). It never switches on its own. On the GPU the detector
leaves the four CPU cores to Gemma. **This device** and the diagnostics overlay show where each model runs.

## 6. Use it and check it

- **Voice chat:** hold the mic button, speak, release; or type. Try "What is the input size of the YOLO26 nano
  detector?" (answered from the knowledge base, with sources) and "What time is it?", and type "Which accelerator are
  you running on?": the app answers it and names where each model runs.
- **Live camera:** choose the camera with the **Camera** button at the top of the demo: a USB webcam, or a network
  camera such as a phone running IP Webcam. Hold the mic and ask "What do you see?", "How many people are there?".
  It also remembers what it saw: "Where was the yellow sign?" shows the frame and how long ago it was.
- **Self-test:** runs the detector on a reference picture and checks it against golden detections, loads the chat
  model and times 64 tokens, checks the speaker and microphone, and saves a report: board, OS, GPU and Vulkan driver,
  where each model ran, speed and memory; the `report` line says where. Exit code 0 means every step passed. With the
  chat model on the CPU, as in step 5:

  ```sh
  ./run.sh --selftest --gemma=$HOME/litert-demos/models/gemma-4-E2B-it.litertlm --gemma-backend=cpu
  ```

  `--gemma` names the chat model when none is chosen in the app yet. On Bookworm add `--detector-backend=cpu`. Add
  `--skip-audio` without a microphone and speaker. Over SSH, without the desktop: install `xvfb` and put
  `xvfb-run -a` in front.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Flutter or the build fails on a 32-bit system (`dpkg --print-architecture` prints `armhf`) | Install Raspberry Pi OS (64-bit): the app needs arm64 |
| `version 'GLIBC_2.34' not found` or `'GLIBCXX_3.4.30' not found` | Raspberry Pi OS older than Bookworm: install Bookworm or newer |
| `flutter pub get` cannot find `third_party/flutter_litert` | Run `tool/flutter_litert/vendor.sh` (step 2) |
| `No file or variants found for asset: assets/models/…` | Run `tool/fetch_models.sh` (step 3) |
| `flutter build linux` fails linking `libFLAC` / `libvorbis` | Build with `NO_XIPH_LIBS=1` (step 4) |
| After a failed build, `file INSTALL cannot copy file … to "/usr/local/…": Permission denied` | CMake kept a default install prefix from the failed run: `rm -r build/linux` and build again |
| The build stops with `Killed` | Out of memory: close other apps, or add swap: `sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile` |
| `package.sh: … exists; move it away first` | `rm -r build/dist/litert_edge_demos-v*`, then run it again |
| "No sound server" / "Microphone unavailable" | `systemctl --user start pipewire pipewire-pulse`; `sudo apt install pulseaudio-utils` |
| No sound from replies | Choose the USB speaker in the taskbar's volume menu (the default is often HDMI) |
| The detector's GPU load is refused: a software GPU (`llvmpipe`) | Bookworm's `v3dv` driver (older than Mesa 26.2): **Run detector on CPU**. On Trixie: `sudo apt full-upgrade`, and start the app with `./run.sh` |
| The chat model is very slow on the GPU | Choose **CPU** on the Chat model card (step 5) |
| Very slow, the Pi gets hot | Fit the Active Cooler; close other apps |
