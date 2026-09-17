# benchmark/ios

On iOS the benchmark runs inside a small app: it builds [LiteRT](https://github.com/google-ai-edge/litert)'s
`benchmark_model` into a framework at the pinned LiteRT tag and writes the same log and `results.pb`
as the Android and desktop runs. The app takes `benchmark_model`'s own flags at launch and keeps the
log and `results.pb` in its Documents folder; `run_ios.sh` moves the model in and the results out
with `devicectl`. This page covers `.tflite` models; [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)
bundles are not covered here.

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

3. Run a model. Everything after the model file is passed to `benchmark_model`; the script sets
   `--graph` and `--result_file_path` itself, and `BUNDLE_ID` must match the identifier from step 2:

```bash
export BUNDLE_ID=com.example.LiteRTBenchmark
./run_ios.sh "My iPhone" mobilenet_v2.tflite --num_threads=4
```

`--use_gpu=true` selects the embedded Metal accelerator, which the runtime loads by name from the
app's Frameworks folder.

## What comes back

| File | Content |
|---|---|
| `out/<time>/benchmark.log` | the `benchmark_model` log: `Inference timings in us`, memory footprint and the `count=` line |
| `out/<time>/results.pb` | `tflite.tools.benchmark.BenchmarkResult`, the same message the Android runs write; decode it with `protoc` as the Android page shows |
| `out/<time>/console.log` | what `devicectl device process launch --console` printed while the app ran |

The script waits `WAIT_SECS` (default 600) for the app to finish and exits non-zero if it does not.

## Tested on

iPhone 17 Pro (iPhone18,1), iOS 27.0 (24A437), on 2026-09-17, with `mobilenet_v2.tflite` from
[litert-community/MobileNet-v2](https://huggingface.co/litert-community/MobileNet-v2): CPU with
`--num_threads=4` three times (XNNPACK, 486, 500 and 497 runs, avg 2.06, 2.00 and 2.01 ms) and GPU
with `--use_gpu=true` twice (Metal, 1202 and 1203 runs, avg 0.83 ms both times), every run from a
fresh launch; each wrote the result block and `results.pb`, decoded with `protoc` on the Mac. The GPU's model initialization
was 583 ms on the first launch after install and 39 ms on the next. Built with LiteRT v2.2.0
(`145c7523f`) and Xcode 27.0 (27A266a); apps built with this SDK need the scene lifecycle, which the
app adopts.
