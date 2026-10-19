# Guide to prepare and build the app

LiteRT NPU is available to all users: https://ai.google.dev/edge/litert/next/npu

## Build the app bundle

WARNING: Before building the app, please follow instructions above to setup NPU
runtime libraries correctly.

WARNING: The version of NPU runtime libraries has to match the runtime maven package, which is available in `gradle/libs.versions.toml`.

Please make sure your NPU runtime are being placed under the project root folder
(current folder for this gradle project).

From the app's root directory, run:

```sh
$ ./gradlew :app:bundleRelease
```

This produces the unified ARM64 and x86_64 app bundle at
`./app/build/outputs/bundle/release/app-release.aab`.

Every native feature contains a uniquely named no-op library for both ABIs.
These stubs are never loaded; they only satisfy bundletool's requirement that
all native feature modules expose the same ABI set. Google Play's ABI splits
and SoC conditions select the real runtime for each device.

## Install the app bundle to a device for local testing

Download `bundletool` from
[GitHub](https://github.com/google/bundletool/releases).

```sh
$ bundletool="java -jar /path/to/the/download/bundletool-all.jar"

$ tmp_dir=$(mktemp -d)

$ $bundletool build-apks \
  --bundle=./app/build/outputs/bundle/release/app-release.aab \
  --output="$tmp_dir/image_segmentation.apks" \
  --local-testing \
  --overwrite

$ $bundletool install-apks --apks="$tmp_dir/image_segmentation.apks" \
  --device-groups=<GROUP_FOR_YOUR_DEVICE>
```

Learn more about local testing, see
[this doc](https://developer.android.com/google/play/on-device-ai#local-testing).

### Identify the group for your device

Currently, the following devices are supported:

| Vendor   | SoC Model | Android version | Group Name                 |
|----------|-----------|-----------------|----------------------------|
| Qualcomm | SM8450    |  S+             | Qualcomm_SM8450            |
| Qualcomm | SM8550    |  S+             | Qualcomm_SM8550            |
| Qualcomm | SM8650    |  S+             | Qualcomm_SM8650            |
| Qualcomm | SM8750    |  S+             | Qualcomm_SM8750            |
| Qualcomm | SM8850    |  S+             | Qualcomm_SM8850            |
| Mediatek | MT6878    |  15             | Mediatek_MT6878_ANDROID_15 |
| Mediatek | MT6897    |  15             | Mediatek_MT6897_ANDROID_15 |
| Mediatek | MT6983    |  15             | Mediatek_MT6983_ANDROID_15 |
| Mediatek | MT6985    |  15             | Mediatek_MT6985_ANDROID_15 |
| Mediatek | MT6989    |  15             | Mediatek_MT6989_ANDROID_15 |
| Mediatek | MT6991    |  15             | Mediatek_MT6991_ANDROID_15 |
| Samsung  | E9965     |  16             | Samsung_E9965_ANDROID_16   |
| Intel    | IntelR CoreTM Ultra 7 325 | S+ | Intel_Core_Ultra_7_325 |
| Intel    | IntelR CoreTM Ultra 7 355 | S+ | Intel_Core_Ultra_7_355 |
| Intel    | IntelR CoreTM Ultra 7 365 | S+ | Intel_Core_Ultra_7_365 |

### Intel NPU runtime libraries

The LiteRT Intel plugins and their manifest declarations come from the
published `litert-npu-runtime-intel-openvino` Maven package, which Gradle
resolves from the standard repositories. Fetch the OpenVINO and Intel NPU
compiler libraries they depend on before building:

```sh
$ ./litert_npu_runtime_libraries/fetch_intel_library.sh
```

The Intel runtime is an SoC-targeted feature in the unified app bundle. The
stub libraries are built with the Android NDK and CMake 3.22.1 or newer.
