# Guide to prepare and build the app

LiteRT NPU acceleration is only available through an Early Access Program. If
you are not already enrolled, [sign up](https://forms.gle/CoH4jpLwxiEYvDvF6).

See [NPU acceleration instruction](https://ai.google.dev/edge/litert/next/eap/npu)
for more information.

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
