# Guide to prepare and build the app

LiteRT NPU, previously under Early access program is available to all 
users: https://ai.google.dev/edge/litert/next/npu

## Performance numbers

*   Measured on Samsung S25 Ultra
*   Synchronized execution w/o zero copy buffer interop
*   W/O pre/post processing
  *   CPU Backend: 120 - 140 ms
  *   GPU Backend: 40 - 50 ms
  *   NPU Backend: 6 - 12 ms

## Build the app bundle

WARNING: Before building the app, please follow instructions above to setup NPU
models and runtime correctly.

WARNING: The version of NPU runtime libraries has to match the runtime maven package, which is available in `gradle/libs.versions.toml`.

Please make sure your AI Pack and NPU runtime are being placed under the project
root folder (current folder for this gradle project).

From the app's root directory, run:

```sh
$ ./gradlew bundle
```

And it will produce the app bundle under the `./app` folder
`./build/outputs/bundle/release/app-release.aab`.

## Install the app bundle to a device for local testing

Download `bundletool` from [GitHub](https://github.com/google/bundletool/releases).

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

Learn more about local testing, see [this doc](https://developer.android.com/google/play/on-device-ai#local-testing).

### Identify the group for your device

Currently, the following devices are supported:

| Vendor   | SoC Model | Android version | Group Name       |
|----------|-----------|-----------------|------------------|
| Qualcomm | SM7150    |  S+             | Qualcomm_SM7150  |
| Qualcomm | SM8250    |  S+             | Qualcomm_SM8250  |
| Qualcomm | SM8350    |  S+             | Qualcomm_SM8350  |
| Qualcomm | SM8450    |  S+             | Qualcomm_SM8450  |
| Qualcomm | SM8550    |  S+             | Qualcomm_SM8550  |
| Qualcomm | SM8650    |  S+             | Qualcomm_SM8650  |
| Qualcomm | SM8750    |  S+             | Qualcomm_SM8750  |
| Qualcomm | SM8850    |  S+             | Qualcomm_SM8850  |
| Mediatek | MT6877    |  S+             | Mediatek_MT6877  |
| Mediatek | MT6878    |  S+             | Mediatek_MT6878  |
| Mediatek | MT6879    |  S+             | Mediatek_MT6879  |
| Mediatek | MT6893    |  S+             | Mediatek_MT6893  |
| Mediatek | MT6897    |  S+             | Mediatek_MT6897  |
| Mediatek | MT6983    |  S+             | Mediatek_MT6983  |
| Mediatek | MT6985    |  S+             | Mediatek_MT6985  |
| Mediatek | MT6989    |  S+             | Mediatek_MT6989  |
| Mediatek | MT6991    |  S+             | Mediatek_MT6991  |
| Mediatek | MT6993    |  S+             | Mediatek_MT6993  |
| Google   | Tensor G3 |  16+            | Google_Tensor_G3 |
| Google   | Tensor G4 |  16+            | Google_Tensor_G4 |
| Google   | Tensor G5 |  16+            | Google_Tensor_G5 |
| Google   | Tensor G6 |  16+            | Google_Tensor_G6 |
