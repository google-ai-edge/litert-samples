# benchmark/ios

On iOS the benchmark runs inside a small app: it builds [LiteRT](https://github.com/google-ai-edge/litert)'s
`benchmark_model` into a framework at the pinned LiteRT tag and writes the same log and `results.pb`
as the Android and desktop runs. The app takes `benchmark_model`'s own flags at launch and keeps the
log and `results.pb` in its Documents folder; `run_ios.sh` moves the model in and the results out
with `devicectl`, or, as an XCTest in the same app, from Developer Device Platform (step 4). This page covers
`.tflite` models; [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) bundles are not covered here.

## Use

You need a Mac with Xcode (this page used 27.0), `bazelisk`, an Apple developer team for signing,
and an iPhone in developer mode connected to that Mac. The framework target is the `bazel/` directory
here, not part of LiteRT: the build script copies it into the LiteRT checkout as
`litert/tools/ios_benchmark/` and builds `//litert/tools:benchmark_model` (the tool behind the Android
and desktop binaries) into `LiteRtBenchmark.framework`. It also fetches the prebuilt
`libLiteRtMetalAccelerator.dylib` for `ios_arm64` at the same version, which the app embeds for GPU runs.

1. Build the framework and fetch the accelerator. Without `LITERT_SRC` the script clones the tag into
   `litert-src/` next to itself; with it, it checks out the tag in that checkout and adds the three
   files under `litert/tools/ios_benchmark/`:

```bash
./build_framework.sh v2.2.0
```

2. Open `LiteRTBenchmark.xcodeproj`, set your team and bundle identifier under Signing & Capabilities,
   and run the app once on the phone. `xcodegen generate` rebuilds the project from `project.yml`
   after source changes.

3. Run a model once per accelerator into one session directory. Everything after the model file
   is passed to `benchmark_model`; the script sets `--graph` and `--result_file_path` itself, and
   `BUNDLE_ID` must match the identifier from step 2. `--use_gpu=true` selects the embedded Metal
   accelerator, which the runtime loads by name from the app's Frameworks folder:

```bash
export BUNDLE_ID=com.example.LiteRTBenchmark SESSION_DIR=~/.cache/litert-samples-benchmark/app/2026-09-17
./run_ios.sh "My iPhone" mobilenet_v2.tflite
./run_ios.sh "My iPhone" mobilenet_v2.tflite --use_gpu=true
```

The session directory is what the leaderboard driver in `../driver/` collects into board rows.

4. Or run it on an iPhone in [Developer Device Platform](https://docs.cloud.google.com/developer-device-platform/device-run/ios)
   (DDP), Google Cloud's device lab, which runs iOS as XCTest: `Tests/LiteRTBenchmarkTests.mm` runs the tool once
   inside the app with the flags in `Documents/benchmark_args.json` and writes `Documents/out/`. `build_xctest.sh`
   builds the archive the lab takes (the Release products and the `.xctestrun` manifest, signed with your team; the
   lab re-signs the app; `DEVELOPER_DIR` picks the Xcode, the lab's default being 26.2), and one
   `gcloud beta device-run sessions submit xctest` per accelerator pushes the model and the flags file into the app
   container and pulls `out/` back:

```bash
DEVELOPMENT_TEAM=<your Apple team id> BUNDLE_ID=<the id from step 2> DEVELOPER_DIR=/Applications/Xcode.app ./build_xctest.sh
echo '["--graph=mobilenet_v2.tflite", "--use_gpu=true"]' > benchmark_args.json
gcloud beta device-run sessions submit xctest --device=iphone16pro-18-3 --test=out/LiteRTBenchmark-xctest.zip \
  --other-files-to-push=mobilenet_v2.tflite=com.example.LiteRTBenchmark:/Documents/mobilenet_v2.tflite,benchmark_args.json=com.example.LiteRTBenchmark:/Documents/benchmark_args.json \
  --paths-to-pull=com.example.LiteRTBenchmark:/Documents/out --xctest-timeout=15m
```

   `run_ddp_ios.py` does that for one accelerator, waits, and lays the pulled files out as a session directory like
   step 3's, the phone's name and OS from the lab's catalog in `session.json`; the project it names pays for the
   session (`--dry-run` prints the `gcloud` command and submits nothing, `--help` has the defaults):

```bash
LITERT_GCP_PROJECT=<your project> python3 run_ddp_ios.py mobilenet_v2.tflite --device iphone16pro-18-3 --accelerator cpu
LITERT_GCP_PROJECT=<your project> python3 run_ddp_ios.py mobilenet_v2.tflite --device iphone16pro-18-3 --accelerator gpu
```

## What comes back

| File | Content |
|---|---|
| `<session>/<accelerator>-<device id>/stdout.txt` | the `benchmark_model` log: the result block (`Inference (avg)`, `Model initialization`, `Overall footprint`) and, from the XCTest, the tool's progress lines as well |
| `<session>/<accelerator>-<device id>/results.pb` | `tflite.tools.benchmark.BenchmarkResult`, the same message the Android runs write; decode it with `protoc` as the Android page shows |
| `<session>/<accelerator>-<device id>/runtime_info.pb` | from the XCTest only: `tflite.profiling.ModelRuntimeDetails`, the delegated node counts the board shows |
| `<session>/<accelerator>-<device id>/console.log` | from `run_ios.sh` only: what `devicectl device process launch --console` printed while the app ran |
| `<session>/session.json` | runner (`app` or `ddp`), platform, device id and name, OS, runtime version and the framework's build: from `devicectl` and `Frameworks/BUILD_INFO` (`run_ios.sh`), or from the lab's device catalog and the archive's `BUILD_INFO` (`run_ddp_ios.py`, which also keeps the lab's `ddp_session.json`, `system.log` and `junit.xml` there) |

`run_ios.sh` waits `WAIT_SECS` (default 600) for the app to finish and exits non-zero if it does not;
`run_ddp_ios.py` waits for the lab's session and exits non-zero when the job did not pass or brought no `results.pb` back.

## Tested on

Developer Device Platform, iPhone 16 Pro (`iphone16pro-18-3`, iOS 18.3), on 2026-09-30, with the same file: two
`run_ddp_ios.py` sessions of one archive (Xcode 26.1.1; lab Xcode 26.2), 5.7 min and 1.9 min from submission to
results (the test itself ran 50 s and 44 s; the rest was the lab's queue and setup), both PASSED. CPU (XNNPACK):
464 runs, avg 2.16 ms, init 5.1 ms, overall footprint 21.8 MB. GPU (Metal): 1071 runs, avg 0.93 ms, init 383 ms,
60.9 MB. Each run is one process, as with `run_ios.sh`, inside the XCTest host; the footprint is the tool's own
delta from its start, as everywhere. The same archive on an iPhone 17 Pro (iOS 27.0) through
`xcodebuild test-without-building`, the same day: CPU avg 1.92 ms, GPU avg 0.83 ms.

iPhone 17 Pro (iPhone18,1), iOS 27.0 (24A437), on 2026-09-17, with `mobilenet_v2.tflite` from
[litert-community/MobileNet-v2](https://huggingface.co/litert-community/MobileNet-v2), one session
from a fresh install through the two commands above: CPU at `benchmark_model`'s default thread count
(XNNPACK, 425 runs, avg 2.36 ms, init 15.9 ms) and GPU with `--use_gpu=true` (Metal, 1192 runs, avg
0.84 ms, init 503 ms on that first launch, 39 ms on a later one). Earlier runs that day with
`--num_threads=4` gave avg 2.06, 2.00 and 2.01 ms on the CPU. Every run wrote the result block and
`results.pb`, decoded with `protoc` on the Mac. Built with LiteRT v2.2.0 (`145c7523f`) and Xcode 27.0
(27A266a); apps built with this SDK need the scene lifecycle, which the app adopts.
