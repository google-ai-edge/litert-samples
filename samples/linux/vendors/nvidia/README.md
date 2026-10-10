# LiteRT Demos on an NVIDIA Jetson Orin

How to build the Flutter sample in [`samples/litert/flutter`](../../../litert/flutter/) on an NVIDIA Jetson Orin and run its two demos there:
a voice chat with Gemma and a live camera assistant. Everything runs on the board.

> **Status:** the steps below ran on Ubuntu 22.04 arm64 (the base of JetPack 6) on a cloud arm64 machine, and the
> self-test passed on the CPU. The app's GPU path (LiteRT's WebGPU accelerator on Vulkan) ran Gemma 4 E2B on
> an NVIDIA T4 at 82.5 tokens/s (x64, warm, with the previous LiteRT-LM release). It has not yet run on a physical
> Jetson, so the Jetson's own GPU driver is not verified. Google's model card for Gemma 4 E2B gives about 24 tokens/s
> decode on a Jetson Orin Nano's GPU. The self-test (step 6) tells you in two runs.

## Which Jetsons

| Board | System | Verdict |
|---|---|---|
| Orin Nano 8 GB / Orin Nano Super, Orin NX, AGX Orin | **JetPack 6** (L4T R36, Ubuntu 22.04) or newer | expected to work |
| Orin Nano 4 GB | JetPack 6 | may need swap to build; Gemma 4 E2B took about 3.3 GB on its own (arm64 CPU, 8192-token context), so expect the GPU load to fail for memory |
| Any Orin on JetPack 5 (Ubuntu 20.04) | | does not work: the runtime needs a newer glibc. Flash JetPack 6 |
| Jetson Nano (2019), TX2, Xavier | JetPack 4/5 | does not work |

## What you need

| | |
|---|---|
| Storage | an NVMe SSD (much faster than microSD) with about 15 GB free for Flutter, the build and the models |
| Camera | a USB webcam, or an Android phone as a Wi-Fi camera ([network camera](../../../litert/flutter/README.md#using-the-demos)). The CSI camera ports are not supported by the app |
| Audio | a USB speakerphone; the Orin Nano developer kit has no audio jack |
| Models | a Hugging Face read token for the built-in models (step 3), and a chat model: Gemma 4 E2B, `gemma-4-E2B-it.litertlm` (2.6 GB) from [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm), no login needed |

## 1. Prepare the board

```sh
cat /etc/nv_tegra_release                 # R36 or newer = JetPack 6 or newer (required)
dpkg --print-architecture                 # must print arm64
sudo apt update
sudo apt install -y git curl unzip xz-utils patch clang cmake ninja-build pkg-config lld \
  libgtk-3-dev liblzma-dev libstdc++-12-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  libasound2-dev libturbojpeg libunwind-dev python3-venv \
  libvulkan1 vulkan-tools pulseaudio-utils gstreamer1.0-plugins-base gstreamer1.0-plugins-good
vulkaninfo --summary | grep deviceName     # expect the NVIDIA Tegra Orin GPU; an extra llvmpipe line is Mesa's CPU driver
```

`libunwind-dev` is listed on purpose: GStreamer's build files need it, and on Ubuntu 22.04 apt may otherwise keep an
LLVM `libunwind-*-dev` instead (installing `libunwind-dev` replaces it).

If `vulkaninfo` lists only `llvmpipe`, the NVIDIA Vulkan driver is missing: `dpkg -S nvidia_icd.json` names the L4T
package to reinstall (`sudo apt install --reinstall <package>`). `pulseaudio-utils` gives the app `pactl` and
`parecord` for the sound check and the microphone.

For the fastest numbers, choose the highest power mode: `sudo nvpmodel -q` shows the current one; on an Orin Nano
with the Super configuration MAXN SUPER is `sudo nvpmodel -m 2`, on an Orin NX or AGX Orin MAXN is
`sudo nvpmodel -m 0` (the power menu in the top bar does the same). `sudo jetson_clocks` then fixes the clocks at
their maximum until the next reboot.

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
install it, then starts the app. Each run is logged to `run.log` next to `run.sh`.

On **Set up models**, the **Chat model** card lists the files in `~/litert-demos/models/` (**Rescan** if needed,
**Path…** for a file elsewhere). Choose `gemma-4-E2B-it.litertlm`, select **GPU**, then **Use this model**.

- **The first start on the GPU is slow:** the runtime compiles its GPU programs once and caches them (minutes on an
  Orin Nano). Later starts take seconds.
- **The GPU shares the board's RAM.** On an 8 GB board, close the browser and other apps before loading Gemma.
- If a model cannot load on the GPU, the app says so and offers the CPU (**Run on CPU**, **Run detector on CPU**);
  it never switches on its own. **This device** and the diagnostics overlay show where each model runs.

## 6. Use it and check it

- **Voice chat:** hold the mic button, speak, release; or type. "Which accelerator are you running on?" names where
  each model runs.
- **Live camera:** choose the camera with the **Camera** button at the top of the demo: a USB webcam, or a network
  camera such as a phone running IP Webcam. Hold the mic and ask "What do you see?", "Describe the scene".
- **Self-test:** `./run.sh --selftest` (add `--skip-audio` without a microphone and speaker) runs the detector on a
  reference picture and checks it against golden detections, loads the chat model and times 64 tokens, checks the
  audio, and saves a report; the `report` line says where. Run it twice: the first run compiles the GPU programs.
  Read in the report:
  - `--- device ---`: the board model, L4T/JetPack release, power mode (nvpmodel) and RAM;
  - `adapter`: the GPU the runtime selected, from its own log; on an Orin it should name the Tegra GPU with
    `backend=Vulkan` (`llvmpipe` means no GPU);
  - `4b chat model generate 64 tokens`: tokens per second;
  - `--- memory ---`, column `Δ available`: how much of the shared RAM each step took.

  If the GPU steps fail, run it again with `--gemma-backend=cpu --detector-backend=cpu` for the CPU numbers. Over
  SSH, without the desktop: install `xvfb` and run `xvfb-run -a ./run.sh --selftest --skip-audio`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `version 'GLIBC_2.34' not found` or `'GLIBCXX_3.4.30' not found` | JetPack 5 or older: flash JetPack 6 |
| `flutter pub get` cannot find `third_party/flutter_litert` | Run `tool/flutter_litert/vendor.sh` (step 2) |
| `No file or variants found for asset: assets/models/…` | Run `tool/fetch_models.sh` (step 3) |
| `flutter build linux` fails linking `libFLAC` / `libvorbis` | Build with `NO_XIPH_LIBS=1` (step 4) |
| `Package 'libunwind', required by 'gstreamer-1.0', not found` | `sudo apt install libunwind-dev` (step 1), `rm -r build/linux`, build again |
| After a failed build, `file INSTALL cannot copy file … to "/usr/local/…": Permission denied` | CMake kept a default install prefix from the failed run: `rm -r build/linux` and build again |
| The build stops with `Killed` | Out of memory: close other apps, or add swap (`sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`) |
| `package.sh: … exists; move it away first` | `rm -r build/dist/litert_edge_demos-v*`, then run it again |
| The report says `llvmpipe` or `CPU / Software` | The NVIDIA Vulkan driver is missing (step 1), or use the CPU |
| The first detector check fails once with `LiteRtLockTensorBuffer … RuntimeFailure` | Start again: the first GPU run compiles programs |
| The app closes while loading Gemma on an Orin Nano | Out of memory: close other apps, run the detector on the CPU, or use the CPU for Gemma |
| "No sound server" / "Microphone unavailable" | `pulseaudio --start` (JetPack 6 uses PulseAudio); `sudo apt install pulseaudio-utils` |
| No sound from replies | Choose the USB speakerphone in **Settings › Sound** (the default may be HDMI or DisplayPort) |
| No camera listed | Plug in a USB webcam, or use a phone as a network camera |
