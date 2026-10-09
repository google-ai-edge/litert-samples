# LiteRT Demos on a Raspberry Pi 5

How to build the Flutter sample in [`../flutter`](../flutter/) on a Raspberry Pi 5 and run its two demos there: a
voice chat with Gemma and a live camera assistant. Everything runs on the Pi.

> **Status:** the steps below ran unchanged on Debian 12 arm64 (Bookworm, the base of Raspberry Pi OS) on a cloud
> arm64 machine: the build took about 6 minutes, and the self-test passed on the CPU. They have not yet run on a
> physical Raspberry Pi 5, so whether the Pi's GPU runs these models is not yet known; the CPU path is the expected
> one. Google's model card for Gemma 4 E2B gives about 7.6 tokens/s decode on a Pi 5's CPU.

## What you need

| | |
|---|---|
| Board | Raspberry Pi 5 with **8 GB or 16 GB**; 4 GB is not enough for Gemma 4 E2B |
| System | **Raspberry Pi OS (64-bit)** with desktop, Bookworm or newer. The 32-bit system does not work |
| Storage | an NVMe SSD, or a fast (A2) microSD card; about 15 GB free for Flutter, the build and the models |
| Cooling and power | the Active Cooler (the models keep all four cores busy) and the official 27 W supply |
| Camera | a USB webcam, or an Android phone as a Wi-Fi camera ([network camera](../flutter/README.md#using-the-demos)). The Camera Module (CSI) is not supported by the app |
| Audio | a USB speakerphone, or a USB microphone and a speaker. The Pi 5 has no audio jack |
| Models | a Hugging Face read token for the built-in models (step 3), and a chat model: Gemma 4 E2B, `gemma-4-E2B-it.litertlm` (2.6 GB) from [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm), no login needed |

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
cp .env.example .env                      # then set HF_TOKEN=hf_... in it
tool/fetch_models.sh
```

- `HF_TOKEN` is a Hugging Face read token from an account that accepted the Gemma terms on
  [litert-community/embeddinggemma-300m](https://huggingface.co/litert-community/embeddinggemma-300m); only that
  download uses it.
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

**GPU or CPU.** The Pi 5's GPU (VideoCore VII) would run models through LiteRT's WebGPU accelerator (Dawn) on Vulkan
(Mesa `v3dv`). If a model cannot load on it, the app says so and offers the CPU (**Run on CPU** for the chat model,
**Run detector on CPU** in Live camera); it never switches on its own. **This device** and the diagnostics overlay
show where each model runs.

## 6. Use it and check it

- **Voice chat:** hold the mic button, speak, release; or type. Try "What is the input size of the YOLO26 nano
  detector?" (answered from the knowledge base, with sources) and "Which accelerator are you running on?".
- **Live camera:** choose the camera with the **Camera** button at the top of the demo: a USB webcam, or a network
  camera such as a phone running IP Webcam. Hold the mic and ask "What do you see?", "How many people are there?".
- **Self-test:** `./run.sh --selftest` runs the detector on a reference picture and checks it against golden
  detections, loads the chat model and times 64 tokens, checks the speaker and microphone, and saves a report: board,
  OS, GPU and Vulkan driver, where each model ran, speed and memory; the `report` line says where. Exit code 0 means
  every step passed. Add `--skip-audio` without a microphone and speaker, and `--gemma-backend=cpu
  --detector-backend=cpu` for the CPU numbers if the GPU steps fail. Over SSH, without the desktop: install `xvfb`
  and run `xvfb-run -a ./run.sh --selftest --skip-audio`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `flutter --version` fails with `…/dart: No such file or directory` | A 32-bit system (`dpkg --print-architecture` prints `armhf`): install Raspberry Pi OS (64-bit) |
| `version 'GLIBC_2.34' not found` or `'GLIBCXX_3.4.30' not found` | Raspberry Pi OS older than Bookworm: install Bookworm or newer |
| `flutter pub get` cannot find `third_party/flutter_litert` | Run `tool/flutter_litert/vendor.sh` (step 2) |
| `No file or variants found for asset: assets/models/…` | Run `tool/fetch_models.sh` (step 3) |
| `flutter build linux` fails linking `libFLAC` / `libvorbis` | Build with `NO_XIPH_LIBS=1` (step 4) |
| After a failed build, `file INSTALL cannot copy file … to "/usr/local/…": Permission denied` | CMake kept a default install prefix from the failed run: `rm -r build/linux` and build again |
| The build stops with `Killed` | Out of memory: close other apps, or add swap: `sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile` |
| `package.sh: … exists; move it away first` | `rm -r build/dist/litert_edge_demos-v*`, then run it again |
| A model load fails with `Invalid argument` | The Pi 5 kernel uses 16 KB memory pages (`getconf PAGESIZE`); try the 4 KB kernel: add `kernel=kernel8.img` to `/boot/firmware/config.txt` and reboot |
| "No sound server" / "Microphone unavailable" | `systemctl --user start pipewire pipewire-pulse`; `sudo apt install pulseaudio-utils` |
| No sound from replies | Choose the USB speaker in the taskbar's volume menu (the default is often HDMI) |
| A model fails on the GPU | Choose **Run on CPU** / **Run detector on CPU** |
| Very slow, the Pi gets hot | Fit the Active Cooler; close other apps |
| The camera list shows `pispbe`, or `run.sh` lists `/dev/video19`–`35` without a webcam | Those are the Pi's video decoder and image processor, not cameras: plug in a USB webcam and choose it by name |
