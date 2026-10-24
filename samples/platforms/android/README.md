# LiteRT Demos on Android

How to build the Flutter sample in [`samples/litert/flutter`](../../litert/flutter/) for an Android phone, install it, and run its two
demos: a voice chat with Gemma and a live camera assistant. Everything runs on the phone; the chat model can run on
the GPU, the CPU, or a Qualcomm NPU.

## What runs where

| Model | Runs on |
|---|---|
| Chat model `gemma-4-E2B-it.litertlm` | GPU (or CPU); its vision and audio towers on the CPU (the engine places them) |
| Chat model `gemma-4-E2B-it_qualcomm_sm8750.litertlm` | the NPU of a Snapdragon 8 Elite (SM8750); its vision and audio towers on the CPU (the engine places them: they are not compiled for the NPU) |
| Chat model: Gemma 4 E2B with the vision and audio towers compiled for the NPU too, one file per chip (provided by Google, not yet published) | the NPU, towers included |
| YOLO26n detector (Live camera) | GPU; CPU when chosen in the Live camera settings |
| EmbeddingGemma 2 (knowledge base, visual memory), Whisper base and moonshine tiny (speech to text), Inflect-nano-v2 (spoken replies) | CPU |

## What you need

| | |
|---|---|
| Phone | Android 11 (API 30) or newer, 64-bit (arm64-v8a), **8 GB of RAM or more recommended**. About 5 GB free: the app (an APK of about 0.9 GB, most of it the built-in models), those models unpacked once (about 0.7 GB), the chat model (2.6 or 3.0 GB) and the caches the models write at their first load |
| Computer | macOS (Apple silicon) or Linux with Flutter 3.47.3, Android SDK 36, JDK 17 or newer (Android Studio's bundled JDK works) and Python 3.10–3.14. Run `flutter doctor --android-licenses` once: the first build installs NDK 28.2.13676358 and CMake. Set up the sample once: [Set up](../../litert/flutter/README.md#set-up-once-per-checkout) in the sample's README (`vendor.sh`, `fetch_models.sh`, `flutter pub get`) |
| Connection | USB debugging on the phone, and `adb` on the computer (it comes with the Android SDK platform tools) |
| Chat model | A `.litertlm` file. From [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) (no login needed): `gemma-4-E2B-it.litertlm` (2.6 GB) runs on the GPU or CPU of any recent phone; on a Snapdragon 8 Elite (SM8750) phone, `gemma-4-E2B-it_qualcomm_sm8750.litertlm` (3.0 GB) runs on the NPU. If you were given another build, it is used the same way. |

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
2. Tap the file and check the settings. The app reads the file through the plugin and starts from what it says:
   - **NPU / GPU / CPU.** A build compiled for an NPU (such as `…_qualcomm_sm8750`) starts on the **NPU**, and GPU
     and CPU are off for it: the card says "compiled for the Qualcomm SM8750 NPU: runs only on the NPU". Any other
     file, Gemma 4 E2B included, starts on the **GPU**, with the **CPU** as the other choice; the NPU is offered only
     for a build compiled for one ("not compiled for an NPU"). NPU builds are compiled per chip. The app's NPU stack
     covers the Snapdragon 8 Gen 2, 8 Gen 3, 8 Elite and 8 Elite Gen 5. Where the phone's NPU cannot be reached, the
     card says why, and an NPU build cannot be applied there. A build for another vendor's NPU (an Intel PC's, a
     Google Tensor phone's) runs on no backend in this app on Android: the card says so and the file cannot be
     applied.
   - **Context length.** An NPU build has one compiled length: the app reads it from the file name (`_ekv4096_`),
     otherwise it uses 4096 on the NPU and 8192 on the GPU or CPU. Change it if your build differs.
   - **Towers** (shown, not set): the vision and audio towers the file has; for an NPU build, a first line names the
     chip its text model is compiled for. Every tower in the file loads, and the plugin places each one from the
     file: on the NPU where the tower is compiled for it, otherwise where the file says, usually the CPU (Gemma 4 E2B
     and `…_qualcomm_sm8750` run theirs on the CPU). Once the file is loaded, each row says where its tower runs.
     With a vision tower you can ask about photos; with an audio tower a spoken question goes to the chat model as
     audio, and its words appear under the question after the reply (a file without one transcribes speech first).
   - **Tools** follow the model family the file name says (on for Gemma), and start off for a build compiled for the
     NPU. With tools off the model calls no skills; the app still answers typed time and device questions itself
     (step 5).
3. Tap **Use this model**; a tapped file changes nothing until then. The app loads it on exactly the backend you
   chose and makes one attempt; it never switches on its own. If the load fails, the card shows the plugin's reason
   and, for most failures of a file that is not an NPU build, offers **Run on GPU** or **Run on CPU** (none when the
   plugin refuses the file itself, for example an incomplete download). An NPU build has no other backend: use
   a build for this phone's chip, or Gemma 4 E2B. With no model chosen yet and only this one file in the folders, the
   app applies it by itself as soon as it lists it, on the settings it starts from (the NPU for an NPU build, else the
   GPU), so let the copy finish before you open the app or tap **Rescan**. The first load on the GPU is slower than
   later ones (GPU programs are compiled once and cached), and the first start also unpacks the built-in models.

## 5. Run the demos

- **Voice chat:** allow the microphone, hold the mic button, speak, release; or type. Try "What is the input size of
  the YOLO26 nano detector?" (answered from the knowledge base, with sources) and "What time is it?", and type
  "Which accelerator are you running on?" (the app answers a typed time or device question itself: it names the chip
  and where each model runs). With a model that supports images, attach a photo. With a model that has an audio
  tower, the line above the mic says **Voice → <model>**: the question goes to the model as audio, and the model
  decides whether to call a skill. With tools off (an NPU build) the line adds **no skills (tools are off)**: a
  spoken question then reaches no skill, so type the time question too.
  Reminders do not work in this app (the agent package's notification intent is not set up); its mail, SMS and
  calendar intents are described in the sample's [README](../../litert/flutter/README.md#using-the-demos). Tap the
  mic while it speaks to interrupt it.
- **Live camera:** allow the camera and the microphone, and point the camera: boxes appear around objects. Hold the
  mic and ask, within 5 seconds, "What do you see?" or "How many people are there?" (answered from the detector at
  once), or "Describe the scene" (the frame goes to the chat model, if it supports images). It also remembers what
  it saw in the last minutes: ask "Where was the yellow sign?" or "Have you seen my mug?" and it shows the frame and
  how long ago it was (**Clear memory** is in the camera settings).

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
| NPU not offered, or the NPU load fails | The card says why: the phone's NPU cannot be reached, or the plugin's reason for the load. Use a build compiled for this phone's chip, or Gemma 4 E2B on the GPU |
| GPU and CPU are greyed out | The file is compiled for an NPU and runs only there: for the GPU or CPU use `gemma-4-E2B-it.litertlm` |
| NPU greyed out for Gemma 4 E2B: "not compiled for an NPU" | Only a build compiled for the phone's NPU runs there: use the GPU or CPU for this file |
| "This file cannot run here: it is compiled for the … NPU" | The file is built for another device's NPU (an Intel PC, a Google Tensor phone): use `gemma-4-E2B-it.litertlm`, or a build for this phone's chip |
| The app closes while loading the model | Not enough free memory: close other apps; 8 GB of RAM is recommended |
| No voice reply | Check the media volume; the reply also appears as text |

When you are done, `adb shell rm -r /data/local/tmp/litert-models` frees the space: uninstalling the app does not.
