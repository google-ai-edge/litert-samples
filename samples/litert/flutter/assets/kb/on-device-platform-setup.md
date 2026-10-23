---
title: Platform setup and memory limits for on-device models
source: https://pub.dev/packages/flutter_edge_ai/versions/2.5.1 (skills/flutter-edge-ai-inference/references/platform-setup.md, README.md, android/src/main/AndroidManifest.xml, lib/src/mobile/smart_downloader.dart) ; https://pub.dev/packages/flutter_edge_ai_litertlm/versions/1.17.0 (README.md: Context and reply length, Platforms, Google Tensor NPU, MediaTek NPU, Qualcomm NPU, Where a model runs; CHANGELOG.md; lib/src/ffi/backend_preference.dart) ; https://pub.dev/packages/flutter_edge_ai_speech/versions/0.5.10 (README.md) ; https://pub.dev/packages/flutter_edge_ai_diagnostics/versions/0.3.0 (README.md) ; https://github.com/DenisovAV/flutter_edge_ai/blob/d35c575ab91e078c47201131800cf33dd7630833/packages/flutter_edge_ai/DESKTOP_SUPPORT.md ; https://github.com/DenisovAV/flutter_edge_ai/blob/d35c575ab91e078c47201131800cf33dd7630833/website/content/docs/installation.md ; https://github.com/DenisovAV/flutter_edge_ai/blob/d35c575ab91e078c47201131800cf33dd7630833/website/content/docs/troubleshooting.md ; https://github.com/DenisovAV/flutter_edge_ai/blob/d35c575ab91e078c47201131800cf33dd7630833/website/content/docs/models.md
license: MIT
---

# Platform setup and memory limits for on-device models

This document lists the platform entries a Flutter app needs before flutter_edge_ai (formerly flutter_gemma) can load an on-device model on Android, iOS, macOS, Windows, Linux and the web, and explains the memory limits that decide whether a large model loads or the app is killed: iOS memory entitlements, the iOS Simulator GPU cap, Android architectures, the opt-in Qualcomm, Google Tensor and MediaTek NPUs, desktop drivers, browser limits and practical memory advice. Without these entries the app builds and then fails at model load, or is killed for memory.

## Android minimum SDK and internet permission

Set `minSdk = 30` in `android/app/build.gradle.kts` (or `build.gradle`) under `defaultConfig`. `minSdk 30` covers everything built on `.litertlm`: inference, embeddings and speech. On API 29 the native library fails to load at runtime — the build does not catch it. MediaPipe `.task` models run on lower API levels.

`android/app/src/main/AndroidManifest.xml` needs the internet permission to download a model:

```xml
<uses-permission android:name="android.permission.INTERNET"/>
```

Flutter's template declares it only for debug and profile builds, so without this line the release build cannot download.

Add-to-app hosts must declare the Kotlin Gradle Plugin themselves, because Flutter auto-applies it to plugin modules only when the host provides it; a normal `flutter build` app needs nothing.

## Android architectures: arm64-v8a only for LiteRT features

Only `arm64-v8a` is shipped for `.litertlm`. Everything backed by `libLiteRtLm` — `.litertlm` inference including vision and audio input, embeddings via LiteRT FFI, and speech STT and TTS — is `arm64-v8a` only and does not run on `x86_64` or `armeabi-v7a`. MediaPipe text inference (`.task` and `.bin`) also works on `x86_64` and `armeabi-v7a`.

If an Android app uses only the arm64-only features, restrict the build to arm64 so the Play Store does not offer broken APKs to incompatible devices:

```gradle
android { defaultConfig { ndk { abiFilters 'arm64-v8a' } } }
```

## Android GPU and NPU manifest entries

GPU needs nothing added. flutter_edge_ai's own plugin manifest declares the OpenCL entries, and the manifest merger folds them into the app. If you pin or audit the merged manifest, it must contain `uses-native-library` entries (all with `android:required="false"`, so they never block an install) for `libvndksupport.so`, `libOpenCL.so`, `libOpenCL-car.so`, `libOpenCL-pixel.so`, `libcdsprpc.so`, `libedgetpu_litert.so` and `libedgetpu_util.so`, and for the MediaTek NeuroPilot adapter and runtime libraries (`libneuron_adapter_mgvi.so`, the `libneuronusdk_adapter*.mtk.so`, `libapuware*` and `libmvpu*` files and others).

`libvndksupport.so` matters above all: without it the OpenCL driver load is denied on Android 12 and later, and OpenCL is the only Android GPU path — no WebGPU library ships for Android since `native-v0.18.0` (older bundles fell back to WebGPU, which hard-froze some Mali GPUs). `libcdsprpc.so` is for the Qualcomm NPU, the two `libedgetpu` entries for the Google Tensor NPU, and the NeuroPilot entries for the MediaTek NPU (`flutter_edge_ai` 2.5.1). The entries are only namespace declarations: an app that never uses the GPU or an NPU never loads those libraries. `flutter_edge_ai_litertlm` is delivered as a Native Assets library with no MediaPipe Java classes, so it needs no ProGuard rules; `flutter_edge_ai_mediapipe` ships its own consumer ProGuard rules.

## Android Qualcomm NPU is opt-in

`PreferredBackend.npu` on Snapdragon is opt-in since `flutter_edge_ai_litertlm` 1.10.0, because Qualcomm licenses its QNN runtime for redistribution inside an app only. The app's own `pubspec.yaml` asks for it — the workspace root's, if the app is a pub workspace member:

```yaml
hooks:
  user_defines:
    flutter_edge_ai_litertlm:
      qualcomm_npu: true
```

The build hook then downloads `com.qualcomm.qti:qnn-runtime` from Maven Central once, checks its SHA-256, raises the Hexagon libraries' alignment to the 16 KB Google Play requires, and bundles them: about 22 MB more to download and 83 MB more installed. Setting the flag means accepting Qualcomm's AI Stack License. A generic bundle never runs on the NPU, with or without the flag, because it has no NPU graph: a request for `npu` falls back to GPU, then CPU, for it (with the default engine; the strict `LiteRtLmEngine(backendFallback: false)` refuses it), and the log says why. A bundle compiled for an NPU is never tried on the GPU or the CPU, which cannot run it: without the flag it is refused at once with a `BackendInitException` whose one attempt is an `NpuDispatchUnavailableException` naming the flag.

The same `qualcomm_npu: true` flag bundles the Qualcomm stack into Linux arm64 builds, for Qualcomm Linux boards with a Hexagon V68–V81 compute DSP (QCS6490, QCS8275, QCS9075 and others). Qualcomm publishes no Maven artifact for Linux, so the build hook reads the libraries out of Qualcomm's public QAIRT 2.50 SDK zip by range requests (about 32 MB of the archive), checks each against a pinned SHA-256 and caches them: about 90 MB more installed. On the board, Qualcomm's FastRPC library must be installed (`qcom-fastrpc1` on Ubuntu) and the user must be in group `fastrpc`, so that `/dev/fastrpc-cdsp` opens. If either is missing, a bundle compiled for an NPU is refused with a `BackendInitException` that names what to fix. A Linux x64 build ignores the flag.

## Google Tensor and MediaTek NPUs on Android are opt-in

Google Tensor takes its own flag in the same `flutter_edge_ai_litertlm` block, `google_tensor_npu: true` (`flutter_edge_ai_litertlm` 1.15.0): the build bundles the Tensor dispatch, built from LiteRT's source, and the runtime it drives, `libedgetpu_litert.so`, is part of the Pixel system image. A `…_Google_Tensor_G5.litertlm` or `…_Google_Tensor_G6.litertlm` bundle runs on a Pixel with that Tensor generation and Android 16 or later, not a BP2A build.

MediaTek takes `mediatek_npu: true` (1.17.0): the build fetches Google's `litert-npu-runtime-mediatek` AAR from Google Maven (`mediatek_npu_aar:` names a downloaded copy for offline builds) and bundles its dispatch, about 330 KB; the NeuroPilot runtime is part of the phone. A bundle compiled for a MediaTek SoC runs on a phone with that SoC and Android 12 or later whose vendor exposes NeuroPilot to apps. Google publishes Gemma 3, Qwen3 and EmbeddingGemma 2 bundles for MediaTek, but no Gemma 4.

For both, a device outside the rule, or an app without the flag, is refused before any engine exists, and a bundle for another Tensor generation or MediaTek SoC when the engine is created; it never moves to the GPU or the CPU. An app can set several NPU flags: the bundle's own stamp picks the vendor.

## Android foreground service for large model downloads

Android has a 9-minute background execution limit. For large models you can opt into foreground service mode with `fromNetwork(url, foreground: true)`, which shows a notification and exempts the download from battery-optimization kills. The default, `null`, configures no notification and no foreground service, so file size alone changes nothing; `false` never uses one. iOS uses native URLSession, which handles long downloads automatically.

On Android 13 and later the notification permission (`POST_NOTIFICATIONS`) must be granted at run time for the foreground service to start. flutter_edge_ai requests it before a foreground download begins; if the user denies it, the download continues in the background without the exemption and may be killed. Have the app ask for the permission up front when it relies on foreground downloads.

## iOS minimum version and static linking

The minimum is iOS 15.0 — 16.0 if the app includes `flutter_edge_ai_mediapipe`, which needs MediaPipe GenAI. Core, `flutter_edge_ai_litertlm` and embeddings build from 15.0.

With CocoaPods, declare in `ios/Podfile`, once, `platform :ios, '15.0'` and `use_frameworks! :linkage => :static`. CocoaPods rejects a second platform declaration. With Swift Package Manager — the default since Flutter 3.44 — there is no Podfile: set iOS Deployment Target on the Runner target in Xcode instead, or the build fails with `requires minimum platform version 15.0`. `flutter_edge_ai_mediapipe` has no `Package.swift`, so an app using it gets a Podfile as well; set the platform there too.

No host-side Podfile `post_install` is required on iOS: flutter_edge_ai patches the LiteRT-LM `dlopen` path so dyld resolves Metal accelerators directly through the Native Assets framework, which also keeps the app bundle App Store clean.

## iOS memory entitlements for large models

In Xcode, under Signing and Capabilities, add Extended Virtual Addressing, Increased Memory Limit and Increased Debugging Memory Limit. That writes these keys to `ios/Runner/Runner.entitlements` and links the file to the target — a file edited by hand but not linked does nothing. Without them large models are killed for memory.

```xml
<key>com.apple.developer.kernel.extended-virtual-addressing</key>
<true/>
<key>com.apple.developer.kernel.increased-memory-limit</key>
<true/>
<key>com.apple.developer.kernel.increased-debugging-memory-limit</key>
<true/>
```

Other iOS `Info.plist` entries from the setup guide are `UIFileSharingEnabled` to enable file sharing, `NSLocalNetworkUsageDescription` for local network access during development, and the optional `CADisableMinimumFrameDurationOnPhone` performance setting.

## Why the iOS Simulator is CPU-only

The iOS Simulator cannot run GPU inference; use the CPU there, or a real device. It also runs only on an Apple Silicon Mac, because there is no Intel `x86_64` simulator build. The Simulator's Metal has a 256 MB single-allocation cap that LLM weight tensors exceed — Gemma 3 1B's KV cache alone is 288 MB. On a physical iPhone, `.litertlm` runs on the FFI engine with the GPU via Metal, and vision and audio are supported on physical devices. Audio input on iOS is device-only.

## macOS entitlements for downloading and loading LiteRT-LM

macOS is supported on Apple Silicon (`arm64`) only; there is no native library for Intel Macs. Add two keys to both `macos/Runner/DebugProfile.entitlements` and `macos/Runner/Release.entitlements`: `com.apple.security.cs.disable-library-validation` and `com.apple.security.network.client`, each set to true.

`network.client` lets the sandboxed app download the model. `disable-library-validation` matters once Hardened Runtime is on, which notarization requires: the build phase signs LiteRT-LM and its companion libraries ad hoc, and library validation refuses code that is not signed by Apple or by the app's own team. Add both keys to both files — the debug and release builds read different ones.

Do not copy the iOS `com.apple.developer.kernel.*` keys into these files; they are iOS entitlements. Without a signing team the build fails with `"Runner" has entitlements that require signing with a development certificate`, and a team-signed build silently drops them. A `.litertlm` model loads on macOS without them.

## macOS build phase for LiteRT-LM companion libraries

`.litertlm` on macOS also needs a build phase that copies the LiteRT-LM companion libraries into the app, because the package deliberately keeps them out of Native Assets and nothing else puts them in the bundle. The upstream Apple companion libraries, such as `libGemmaModelConstraintProvider.dylib` and the Metal accelerator, were linked without `-Wl,-headerpad_max_install_names`, so the Native Assets bundling path cannot rewrite their install name. The phase wraps them into `.framework` bundles inside `Contents/Frameworks/` and re-signs them.

With CocoaPods, the `post_install` block from the package README replaces the one in `macos/Podfile`, keeping any lines of the app's own inside it, and adds a Run Script phase named `[flutter_gemma] Setup LiteRT-LM macOS` to the Runner target only; the next build runs `pod install` itself, and the phase locates and runs a staging script that `flutter_edge_ai_litertlm` installs from its build hook. The `flutter_gemma` phase, cache and stamp names are compatibility identifiers the current package still uses, so they must not be renamed. A Swift Package Manager app has no `macos/Podfile`: either turn SPM off for the app in `pubspec.yaml` (`flutter: config: enable-swift-package-manager: false`) and run `flutter pub get`, which writes the Podfile, then replace its `post_install` block, or add the same Run Script phase by hand in Xcode. An app that already has a `macos/Podfile`, because another plugin needs CocoaPods, keeps SPM on and only replaces its `post_install` block. A green build proves nothing: without the phase the first model load fails with `Library not loaded: @rpath/libGemmaModelConstraintProvider.dylib`, on every backend, CPU included; since `flutter_edge_ai_litertlm` 1.17.0 that error contains `flutter_edge_ai_litertlm: LiteRT-LM did not load on macOS` and names the fix. Speech-to-text and text-to-speech need the same setup and fail to load the same way without it.

## Desktop architecture: LiteRT-LM via dart:ffi without a JVM

Desktop platforms run LiteRT-LM directly via `dart:ffi`. The previous Kotlin and JVM gRPC server is gone — no Java required, no separate process, no IPC overhead — and engine startup is about 2 seconds instead of 10–15 seconds. Native libraries are fetched at build time by `hook/build.dart` from a GitHub release, SHA256-verified, and bundled by Flutter Native Assets. The Dart FFI layer is shared with mobile: Android and iOS use the same client against the same C API.

Desktop does not run MediaPipe: `.bin` and `.task` models won't load there, so use `.litertlm` files. Supported desktop targets are macOS arm64 (Apple Silicon) with Metal, Windows x86_64 with DirectX 12 via Dawn and WebGPU, and Linux x86_64 and arm64 with Vulkan via Dawn and WebGPU, all with vision and audio. Qualcomm Linux arm64 boards also run `PreferredBackend.npu` on the Hexagon DSP when the app opts in with `qualcomm_npu: true`. Intel Macs and Windows on arm64 are not supported. Requirements are Flutter 3.44.0 or later and Dart SDK 3.12.0 or later.

## Windows desktop setup

Nothing needs to be added to the project. The native libraries — including the Windows GPU shader compiler and NPU runtime — are bundled at build time. The bundle includes `LiteRtLm.dll`, `LiteRt.dll`, the WebGPU accelerator and sampler, `webgpu_dawn.dll`, the DirectX Shader Compiler runtime (`dxil.dll` and `dxcompiler.dll`), and `LiteRtDispatch.dll` with the OpenVINO runtime and TBB for the Intel NPU behind `PreferredBackend.npu` on Lunar Lake and Panther Lake.

The plugin adds no Visual C++ runtime beyond the one every Flutter Windows app needs: `msvcp140`, `vcruntime140` and `vcruntime140_1`, from the latest Microsoft Visual C++ Redistributable. `LiteRtLm.dll` is linked against the static CRT and imports no C++ runtime, and 17 of the bundle's 25 DLLs (`native-v0.18.0-g`) import none; the other eight, the Intel NPU stack, import only those three. If GPU shaders fail to compile and the app silently falls back to CPU, check that `dxcompiler.dll` and `dxil.dll` sit next to the app executable, then look at the GPU driver. An Intel NPU bundle whose NPU blobs carry no weights, such as Intel's Gemma 4 12B for Panther Lake, makes the dispatch copy them to `%TEMP%\litert_ov_bank_<hex>.bin` at every load (5.55 GB for the 12B, 74.5 MB for the EmbeddingGemma 2 270m), and closing the model deletes the file: keep that much free on the system drive and close the model before the app exits.

## Linux desktop setup and Vulkan drivers

Building needs `clang cmake ninja-build libgtk-3-dev lld`. The runtime needs glibc 2.34 or later and libstdc++ 6.0.30 or later (Ubuntu 22.04+, Debian 12+, Fedora 36+, RHEL 9+).

Linux GPU uses Dawn and WebGPU on top of Vulkan, so a working vendor Vulkan driver is required (NVIDIA, AMD or Intel). On NVIDIA install the proprietary driver; on Intel and AMD the open-source Mesa driver works on most distributions. Mesa's `llvmpipe` software fallback is not enough for Gemma 4. For headless use without a display, `Xvfb` is enough as a fake display. A `glibc 2.38 not found` error on Ubuntu 22.04 means a stale copy is being loaded from the build cache; clearing `~/.cache/flutter_gemma/native` (the cache keeps its original name) lets the hook re-fetch the release.

## Where desktop apps store downloaded models

Desktop builds store downloaded models outside the user's `Documents/` folder to avoid OneDrive, iCloud or Domain Roaming sync corrupting the memory-mapped access to large `.litertlm` files. The folder keeps the name `flutter_gemma` across the rename:

- Windows: `%LOCALAPPDATA%\flutter_gemma\` (never OneDrive-synced)
- macOS: `~/Library/Application Support/<bundle-id>/flutter_gemma/` (inside the app's container for a sandboxed app)
- Linux: `~/.local/share/<app-id>/flutter_gemma/`

On mobile, Android and iOS store models in the local file system in the app documents directory. LiteRT-LM caches compiled GPU shaders in the app's support directory as `<model>.litertlm_<mtime>_<size>_mldrift_program_cache.bin`; the name is keyed on the model file's timestamp and size, so a new model build gets a fresh cache by itself.

## Web setup: script tags and model storage helpers

All script tags go in `web/index.html` `<head>`, before Flutter boots. Every web app needs the model storage helpers: copy `cache_api.js` and `opfs_helper.js` from the flutter_edge_ai package's `web/` directory into the app's `web/`, and load them with script tags. Then add the script for each engine package used: the `.litertlm` engine loads `@litert-lm/core` from a CDN as an ES module and exposes a `window.litertLmReady` promise that resolves to the `Engine` constructor, and the same script sets `window.EmbeddingEngine`, which EmbeddingGemma 2 embedders need on the web (`flutter_edge_ai_litertlm` 1.13.1 and later); MediaPipe loads `@mediapipe/tasks-genai`. Web RAG with `flutter_edge_ai_sqlite` needs the package's custom `sqlite3.wasm` copied into the web root and the app served with cross-origin isolation headers.

The `.litertlm` web engine loads the web build of a model — `gemma-4-E2B-it-web.litertlm` (2.0 GB, so use streaming storage), not `gemma-4-E2B-it.litertlm`. It takes text input only — an image or audio message throws `UnsupportedError`, and LoRA is not supported — while function calling and Gemma 4 thinking work. A `--dart-define` token is compiled into `main.dart.js`, where every visitor can read it, so serve web users a model from a repo that needs no token.

## Web storage modes: Cache API, OPFS streaming and ephemeral

The storage mode is set in `FlutterEdgeAi.initialize(webStorageMode: ...)`.

- `WebStorageMode.cacheApi`, the default, uses the browser Cache API with Blob URLs. Models persist across browser restarts. It is best for models under about 2 GB.
- `WebStorageMode.streaming` uses OPFS with streaming and bypasses the browser's roughly 2 GB blob limit. It is required for large models and requires Chrome 86+, Edge 86+ or Safari 15.2+. Use it when shipping `.litertlm` web models: it avoids Chrome's roughly 2 GB blob-fetch limit on the Gemma 4 E2B and E4B web builds.
- `WebStorageMode.none` keeps models in memory only; they are cleared when the browser closes and download on every launch, which suits testing and demos.

`FlutterEdgeAi.isStreamingSupported()` checks whether streaming is available.

## Web memory and browser cache limits

Large models may hit browser memory limits. The maximum model size in the browser cache is about 2 GB on Chrome and Firefox, an ArrayBuffer limit, and about 50 MB on Safari, which is not suitable. Use smaller models on the web, or the streaming storage mode for the Gemma 4 web builds. Mobile `.task` models often don't work on the web; use the `-web.task` (MediaPipe) or `.litertlm` web variant. Custom model servers must enable CORS headers; Hugging Face already has CORS configured correctly.

## Memory considerations for on-device models

- Model size: larger models might be too resource-intensive for on-device inference.
- Multimodal models: Gemma 4 and Gemma 3n with vision and audio need more memory; plan for devices with 8 GB or more of RAM.
- iOS memory requirements: large models require memory entitlements in `Runner.entitlements`.
- Each open chat session holds its own context, about 100–500 MB depending on model and context size, so several concurrent sessions with Gemma 4 E2B or larger can run out of memory on phones.
- On Android the GPU shares system memory, so on a 4–6 GB phone running out of GPU memory can end the app rather than fall back to CPU.
- The `flutter_edge_ai_diagnostics` package measures what a model costs in memory the OS cannot reclaim, read from the OS, on Android, iOS and macOS (macOS since version 0.3.0).

## Fixing out-of-memory problems

- On iOS, ensure `Runner.entitlements` contains the memory entitlements and is linked to the target.
- Reduce `maxTokens` if experiencing memory issues, because it sets the context window and KV-cache size — but keep it at 1024 or higher for `.litertlm` models on the CPU and the GPU (on the NPU, use the context the bundle was compiled for), and shorten replies with `maxOutputTokens` instead.
- Use smaller models (1B–2B parameters) for devices with less than 6 GB of RAM.
- Close sessions and models when not needed.
- Monitor token usage with `sizeInTokens()`.
- Keep one loaded model and create cheap sessions on it rather than loading a model per chat; cap concurrent sessions with `maxConcurrentSessions:` on `getActiveModel`.
