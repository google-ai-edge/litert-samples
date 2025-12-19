# Guide to prepare and build the app

LiteRT NPU, previously under Early access program is available to all 
users: https://ai.google.dev/edge/litert/next/npu

## Build the app bundle

WARNING: Before building the app, please follow instructions above to setup NPU
runtime libraries correctly.

Please make sure your NPU runtime are being placed under the project root folder
(current folder for this gradle project).

From the app's root directory, run:

```sh
$ ./gradlew bundle
```

And it will produce the app bundle at under the `./app` folder
`./build/outputs/bundle/release/app-release.aab`.

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
