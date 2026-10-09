# LiteRT Demos on an iPhone

How to build the Flutter sample in [`../flutter`](../flutter/) for an iPhone, sign it with your own Apple account,
and run its two demos: a voice chat with Gemma and a live camera assistant. Everything runs on the phone. iOS
installs only apps signed for your own devices, so there is no ready-made build: you build it on a Mac (about 15
minutes the first time).

> **Status:** tested on an iPhone 17 Pro (iOS 26.5.2, release build, signed without the two memory capabilities of
> step 2): Gemma 4 E2B and the YOLO26n detector both run on the GPU (Metal); speech recognition, spoken replies and
> the knowledge base run on the CPU.

## What you need

| | |
|---|---|
| Mac | Apple silicon, **Xcode 26** with the iOS platform, **CocoaPods** (`brew install cocoapods`: one plugin has no Swift Package Manager support), Flutter 3.47.3 (`flutter doctor` shows no errors for Xcode and CocoaPods), and the sample set up once: [Set up](../flutter/README.md#set-up-once-per-checkout) in the sample's README (Python 3.10+, a Hugging Face token in `.env`, `tool/flutter_litert/vendor.sh`, `tool/fetch_models.sh`, `flutter pub get`) |
| iPhone | **iOS 26 or newer** (a dependency of the agent package links a system library that older iOS lacks), **8 GB of RAM or more** (iPhone 15 Pro or newer), about 4 GB free |
| Apple account | signed in to Xcode (**Xcode › Settings › Accounts**). A paid Apple Developer Program team keeps the two memory capabilities of step 2. A free Personal Team works too, but it cannot use them, and its builds stop launching after 7 days (run step 3 again) |
| Chat model | one `.litertlm` file for the GPU: Gemma 4 E2B, `gemma-4-E2B-it.litertlm` (2.6 GB) from [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm), no login needed. NPU builds (`…_qualcomm_…`, `…_Google_Tensor_…`, `…_intel_…`) and the text-only `-gpu` and `-web` files do not run in this app on an iPhone |

## 1. Prepare the iPhone

1. Connect it to the Mac with a cable, unlock it and tap **Trust**.
2. Open Xcode with the phone connected so that it pairs. Then turn on **Settings › Privacy & Security › Developer
   Mode** (the switch appears only after pairing), tap **Restart**, and after the restart tap **Enable** and enter
   the passcode.
3. Check that the Mac sees it: `flutter devices` lists it as `(mobile) • … • ios`. If not, unlock the phone and run
   `flutter devices --device-timeout 20`.

## 2. Sign it with your team

1. From `samples/litert/flutter`, prepare the Xcode workspace (this runs `pod install`), then open it:

   ```sh
   flutter build ios --config-only --no-codesign
   open ios/Runner.xcworkspace
   ```

2. Select the **Runner** target › **Signing & Capabilities**, with **All** selected so that Debug, Profile and
   Release all change:
   - **Team:** your team, with **Automatically manage signing** on.
   - **Bundle Identifier:** change `com.google.ai.edge.examples.litertEdgeDemos` to one of your own, such as
     `com.example.<you>.litertdemos`: an identifier belongs to one team.
3. The memory capabilities, **Increased Memory Limit** and **Extended Virtual Addressing**
   (`ios/Runner/Runner.entitlements`), let iOS give the app the memory a 2.6 GB model needs:
   - **Paid team:** keep both.
   - **Free Personal Team:** Xcode reports that personal teams do not support Extended Virtual Addressing. Remove it,
     and Increased Memory Limit if Xcode flags that too (the ✕ on each). The app still runs (the status above was
     measured without them), but iOS may stop a model run under memory pressure.
4. Close Xcode. If you contribute back, do not commit the team, bundle identifier or capability changes.

## 3. Build and install

```sh
flutter run --release -d "<your iPhone>"     # the name or id from `flutter devices`
```

Use a release build: the models run at full speed, and the app keeps working after you unplug the phone. The first
build downloads the native libraries (a few hundred MB) and takes several minutes; later builds take about two.

With a free account the first launch fails and the phone shows *Untrusted Developer*: **Settings › General › VPN &
Device Management** › your account › **Trust**, then run the command again.

## 4. Put the chat model on the phone

Once the app has opened, its models folder is visible in the **Files** app under **On My iPhone › LiteRT Demos ›
models**. Any of these works:

- **Files or AirDrop:** AirDrop the `.litertlm` to the phone or save it in iCloud Drive, then move it into **On My
  iPhone › LiteRT Demos › models**.
- **Import file… in the app** (on the Chat model card): pick the file; iOS gives the app a copy, so keep another
  2.6 GB free until you delete the original.
- **Download from URL… in the app**, for example
  `https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm` (stay on
  Wi-Fi; enter the same link again to resume an interrupted download).
- **From the Mac over the cable** (fastest):

  ```sh
  xcrun devicectl list devices              # the phone's name
  xcrun devicectl device copy to --device "<name>" \
    --domain-type appDataContainer --domain-identifier <your bundle identifier> \
    --source gemma-4-E2B-it.litertlm --destination Documents/models/gemma-4-E2B-it.litertlm
  ```

## 5. First launch

1. Open **LiteRT Demos**. On **Set up models**, the **Chat model** card lists the files in the models folder (tap
   **Rescan** if yours is missing). With exactly one file there and nothing chosen yet, it is already selected.
2. Select **GPU**, then **Use this model**. A file added with **Import file…** or **Download from URL…** starts
   with conservative settings (images and tools off): turn them on for Gemma 4 E2B. The first load on the GPU
   compiles GPU programs once (up to a minute); later starts take a few seconds.
3. Allow the microphone and the camera when a demo asks.

## 6. Use it

- **Voice chat:** hold the mic button, speak, release; or type. Try "What is the input size of the YOLO26 nano
  detector?" (answered from the knowledge base, with sources), "Which accelerator are you running on?", or attach a
  photo and ask about it. Tap the mic while it speaks to interrupt it.
- **Live camera:** boxes appear around objects. Hold the mic and ask, within 5 seconds, "What do you see?" or "How
  many people are there?" (answered from the detector at once), or "Describe the scene" (the frame goes to Gemma).
- **Self-test:** on Home, the **…** menu (top right) › **Models** › **Run self-test**, then **Copy**. The **THIS
  DEVICE** card shows where each model really runs.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CocoaPods not installed or not in valid state` | `brew install cocoapods`, then `flutter build ios --config-only --no-codesign` |
| `No Accounts` or `No profiles for '…' were found` | Sign in to Xcode, choose your team and change the bundle identifier (step 2) |
| `… cannot be registered to your development team because it is not available` | Change the bundle identifier to your own (step 2) |
| `Personal development teams … do not support the Extended Virtual Addressing capability` | Remove the memory capabilities (step 2.3) |
| The phone is not listed, or the install drops | Use the cable, unlock the phone, turn on Developer Mode; `flutter devices --device-timeout 20` |
| *Untrusted Developer* | **Settings › General › VPN & Device Management** › **Trust**, then run step 3 again |
| The app stops opening after a week | Free account: its profile expired; run step 3 again |
| The app stops at launch | iOS 18 or older: the app needs iOS 26 (Xcode installs it anyway, because the deployment target is 15.0) |
| The model is not listed | Tap **Rescan**; check that it is in **On My iPhone › LiteRT Demos › models** and ends in `.litertlm` |
| A reply stops with `LiteRtRunCompiledModel … (status=3)` after a memory warning | Seen in a build without the memory capabilities: close other apps and retry; with a paid team, keep them (step 2) |
| You need the app's log | `flutter run --release` prints it while the phone is attached |
