# LiteRT Demos on Android

How to build the Flutter sample in [`../flutter`](../flutter/) for an Android phone, install it, and run its two
demos: a voice chat with Gemma and a live camera assistant. Everything runs on the phone; the chat model can run on
the GPU, the CPU, or a Qualcomm NPU.

> **Status:** checked in scripted runs on Firebase Test Lab (no live microphone or speaker) on a Galaxy S24
> (Snapdragon 8 Gen 3; Gemma and the detector on the GPU) and a Galaxy S26 (Snapdragon 8 Elite Gen 5; Gemma on the
> NPU, with a build compiled for that chip that is not public), and by hand on a phone with a Snapdragon 8 Gen 2
> (Gemma on the GPU and on the NPU, with `flutter_edge_ai_litertlm` 1.10.0).

## What you need

| | |
|---|---|
| Phone | Android 11 (API 30) or newer, 64-bit (arm64-v8a), **8 GB of RAM or more recommended**. About 4 GB free: the app (about 590 MB), its built-in models unpacked once, and the chat model |
| Computer | macOS (Apple silicon) or Linux with Flutter 3.47.3, Android SDK 36, JDK 17 or newer (Android Studio's bundled JDK works) and Python 3.10–3.14. Run `flutter doctor --android-licenses` once: the first build installs NDK 28.2.13676358 and CMake. Set up the sample once: [Set up](../flutter/README.md#set-up-once-per-checkout) in the sample's README (`vendor.sh`, `.env` with `HF_TOKEN`, `fetch_models.sh`, `flutter pub get`) |
| Connection | USB debugging on the phone, and `adb` on the computer (it comes with the Android SDK platform tools) |
| Chat model | **The chat model for Android will be provided.** Public alternatives from [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) (no login needed): `gemma-4-E2B-it.litertlm` (2.6 GB) runs on the GPU or CPU of any recent phone; on a Snapdragon 8 Elite (SM8750) phone, `gemma-4-E2B-it_qualcomm_sm8750.litertlm` (3.0 GB) runs on the NPU. The Google Tensor, Intel, QCS8275 and web files in that repository do not run in this app on a phone |

## 1. Turn on USB debugging

1. Samsung (One UI 6 and later): turn off **Settings › Security and privacy › Auto Blocker** first. It blocks USB
   debugging and installs from a computer; turn it back on when you are done.
2. **Settings › About phone** (Samsung: **About phone › Software information**), tap **Build number** seven times.
3. **Settings › Developer options › USB debugging** on.
4. Connect the phone, accept **Allow USB debugging?**, and check that `adb devices` lists it as `device`.

## 2. Build and install

From `samples/litert/flutter`:

```sh
flutter build apk --release --target-platform android-arm64
adb install -r build/app/outputs/flutter-apk/app-release.apk
```

Or build, install and start it in one step with `flutter run -d <device-id> --release` (`flutter devices` lists the
id).

- The app is arm64-v8a only; there is no x86 emulator build.
- Release builds are signed with your debug key, with a one-line Gradle warning, unless `android/key.properties`
  points to your own keystore (`storeFile`, `storePassword`, `keyAlias`, `keyPassword`).
- The first build downloads the LiteRT-LM runtime from GitHub and, because `pubspec.yaml` sets
  `qualcomm_npu: true`, Qualcomm's QNN runtime (`com.qualcomm.qti:qnn-runtime` from Maven Central, checked against
  a pinned SHA-256). Those are Qualcomm's libraries under Qualcomm's licence; the build prints a notice. Set
  `qualcomm_npu: false` to build without them: the APK is then about 80 MB smaller and the app has no NPU.
- `adb install -r` keeps the app's data. Uninstalling deletes the unpacked models and the app's own models folder.

## 3. Copy the chat model to the phone

From the computer (before or after the app's first start):

```sh
adb shell mkdir -p /data/local/tmp/litert-models
adb push <model>.litertlm /data/local/tmp/litert-models/
```

The app also lists its own folder, `/sdcard/Android/data/com.google.ai.edge.examples.litert_edge_demos/files/models/`.
Start the app once before you push there: a file pushed before the app created that folder cannot be read by it.

Without a computer: **Download from URL…** takes a direct link to a `.litertlm` file, for example
`https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm` (stay on
Wi-Fi). Android has no file import: the system picker would load the whole multi-GB file into memory.

## 4. Choose the model and the backend

1. Open **LiteRT Demos**. **Set up models** shows the **Chat model** card with the models folders and the files in
   them (**Rescan** if yours is not listed; **Path…** takes any path the app can read).
2. Tap the file and check the settings:
   - **NPU / GPU / CPU.** Gemma 4 E2B on the **GPU**. For an NPU build, select **NPU** yourself: the official builds
     do not mark themselves NPU-only, so they start on the GPU. An NPU build runs only on the chip it was compiled
     for, and the app's NPU stack covers the Snapdragon 8 Gen 2, 8 Gen 3, 8 Elite and 8 Elite Gen 5. Where the
     phone's NPU cannot be reached, the card says why.
   - **Context length.** An NPU build has one compiled length: the app reads it from the file name (`_ekv4096_`),
     otherwise it uses 4096 on the NPU and 8192 on the GPU or CPU. Change it if your build differs.
   - **Images** come from the file's header (whether it has a vision encoder); NPU builds usually have none, and the
     demos then turn photo questions off and say so. **Tools** follow the model family the file name says. A file
     added with **Download from URL…** starts with conservative defaults (its header is not read): check all of
     these.
3. Tap **Use this model**. The app loads it on exactly the backend you chose. If that fails, it shows the reason and
   offers **Run on GPU** or **Run on CPU**; it never switches on its own. The first load on the GPU is slower than
   later ones (GPU programs are compiled once and cached), and the first start also unpacks the built-in models.

## 5. Run the demos

- **Voice chat:** allow the microphone, hold the mic button, speak, release; or type. Try "What is the input size of
  the YOLO26 nano detector?" (answered from the knowledge base, with sources), "What time is it?", and "Which
  accelerator are you running on?" (it names the chip and where each model runs). With a model that supports
  images, attach a photo. Tap the mic while it speaks to interrupt it.
- **Live camera:** allow the camera and the microphone, and point the camera: boxes appear around objects. Hold the
  mic and ask, within 5 seconds, "What do you see?" or "How many people are there?" (answered from the detector at
  once), or "Describe the scene" (the frame goes to the chat model, if it supports images).

## 6. Self-test and logs

On Home, **⋮ › Models › Run self-test** runs the detector on a reference picture, loads the chat model and times
it, and checks the speaker and microphone; **Copy** puts the report on the clipboard. The **THIS DEVICE** card on
Home (**Copy diagnostics**) shows the chip and where each model really runs. After a crash or freeze:

```sh
adb logcat -d > litert-demos-log.txt
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `INSTALL_FAILED_NO_MATCHING_ABIS` | A 32-bit phone or an x86 emulator: the app needs a 64-bit Arm device |
| `INSTALL_FAILED_OLDER_SDK` | The phone runs Android 10 or older; the app needs Android 11 (API 30) |
| `INSTALL_FAILED_UPDATE_INCOMPATIBLE` | An install signed with another key (another machine's debug key): `adb uninstall com.google.ai.edge.examples.litert_edge_demos` first, which deletes its data |
| The model file is not listed | Tap **Rescan**; the name must end in `.litertlm` |
| "cannot be read (Permission denied)" | Push it to `/data/local/tmp/litert-models/` (step 3); if it still fails: `adb shell chmod 644 /data/local/tmp/litert-models/*` |
| NPU not offered, or the NPU load fails | The file is not for this phone's chip, or the phone's NPU cannot be reached: use the right NPU build, or Gemma 4 E2B on the GPU |
| The app closes while loading the model | Not enough free memory: close other apps; 8 GB of RAM is recommended |
| No voice reply | Check the media volume; the reply also appears as text |

When you are done, `adb shell rm -r /data/local/tmp/litert-models` frees the space: uninstalling the app does not.
