# Guide to prepare and build the app

LiteRT NPU is available to all users: https://ai.google.dev/edge/litert/next/npu

## Build the app bundle

WARNING: Before building the app, please follow instructions above to setup NPU
runtime libraries correctly.

WARNING: The version of NPU runtime libraries has to match the runtime maven package, which is available in `gradle/libs.versions.toml`.

Please make sure your NPU runtime are being placed under the project root folder
(current folder for this gradle project).

To build the ARM64 multi-vendor app from the project root, run:

```sh
$ ./gradlew :app:bundleRelease
```

The app bundle is written to
`./app/build/outputs/bundle/release/app-release.aab`.

## Install the app bundle to a device for local testing

Download `bundletool` from
[GitHub](https://github.com/google/bundletool/releases).

```sh
$ bundletool="java -jar /path/to/the/download/bundletool-all.jar"

$ tmp_dir=$(mktemp -d)

$ $bundletool build-apks \
  --bundle=./build/outputs/bundle/release/app-release.aab \
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

### Intel NPU runtime libraries

The Intel app is a separate `x86_64` application module. The LiteRT Intel
plugins and their manifest declarations come from the published
`litert-npu-runtime-intel-openvino` Maven package, which Gradle resolves from
the standard repositories. Fetch the OpenVINO and Intel NPU compiler libraries
they depend on before building:

```sh
$ ./litert_npu_runtime_libraries/fetch_intel_library.sh
```

Build the Intel app bundle with:

```sh
$ ./gradlew :app_intel:bundleRelease
```

The `x86_64`-only bundle is written to
`./app_intel/build/outputs/bundle/release/app_intel-release.aab`. The Intel app
uses the application ID `com.google.ai.edge.examples.image_segmentation.intel`.

To install the release bundle on a connected Intel device, run:

```sh
$ bundletool="java -jar /path/to/the/download/bundletool-all.jar"

$ $bundletool build-apks \
  --bundle=./app_intel/build/outputs/bundle/release/app_intel-release.aab \
  --output=/tmp/image_segmentation_intel.apks \
  --connected-device \
  --overwrite

$ $bundletool install-apks --apks=/tmp/image_segmentation_intel.apks
```
