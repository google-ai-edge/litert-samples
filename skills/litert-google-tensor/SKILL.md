---
name: litert-google-tensor
description: >-
  Compiles LiteRT (.tflite) models ahead-of-time for Google Tensor TPUs
  (Tensor G5 on Pixel 10, Tensor G6 on Pixel 11) locally on Linux x86_64 or
  remotely via Google Colab (--colab on macOS, Windows, or ARM hosts),
  benchmarks them on a USB-attached Pixel with LiteRT's benchmark_model
  (NPU latency, peak memory, TPU tile time, TPU energy, optional CPU baseline),
  and packages compiled models for an Android Gradle app as a Google Play AI
  pack with the Tensor NPU runtime module and a Kotlin loader, applies it to
  the app and verifies it on a Pixel. Use when the
  user asks to compile, AOT-compile, or prepare a .tflite model for the Pixel
  TPU/NPU, check whether a .tflite is already Tensor-compiled, measure
  on-device NPU performance of a compiled model, or package or deploy a
  compiled model into their Android app (or replace the one it ships). Don't
  use for Qualcomm, MediaTek, or Samsung NPUs, GPU-only benchmarking,
  LiteRT-LM or LLM (.litertlm) models, on-device JIT compilation, non-Gradle
  builds, or publishing to Google Play.
---

# LiteRT on Google Tensor: compile, benchmark, package for apps

Three scriptable workflows for Pixel TPUs:

1.  **Compile** a plain `.tflite` into a Google Tensor AOT-compiled `.tflite`
    with the public LiteRT Python API (`ai_edge_litert.aot`) - either locally on
    Linux `x86_64` or remotely on a Google Colab VM (`--colab` via
    `google-colab-cli`) when the user is on macOS/Windows or lacks a Linux box.
2.  **Benchmark** a compiled model on a connected Pixel with LiteRT's
    `benchmark_model` binary, NPU-only, optionally next to a CPU baseline,
    including TPU tile time (pure TPU compute time) and TPU energy per
    inference, both from the Tensor runtime's hooks.
3.  **Package** compiled builds for an Android (Gradle) app: a Google Play AI
    pack written by LiteRT's own exporter (`ai_edge_litert.aot.ai_pack`) from
    the files as they are, the Google Tensor NPU runtime as a feature module
    (from Google Maven), a Kotlin loader and a step-by-step `INTEGRATION.md`.
    For an app that already ships Google Tensor builds, it prepares only the
    files to replace, under the app's own paths, with an `UPDATE.md`. When
    the user agrees, the agent applies the plan to the project (one approval,
    separate OKs for risky changes) and checks the app on a Pixel.

Everything runs synchronously in the user's terminal; the scripts never start
background processes and always clean up what they pushed to the phone or
allocated on Colab. All paths below are relative to this skill's directory
(`SKILL_DIR`).

```text
SKILL_DIR/
  versions.json                    # LiteRT versions the scripts use (single source)
  scripts/check_compiled.py        # is this .tflite already Tensor-compiled?
  scripts/compile_model.py         # AOT compile (local Linux x86_64 or --colab)
  scripts/benchmark_on_device.py   # adb push -> benchmark -> parse -> cleanup
  scripts/package_ai_pack.py       # compiled builds -> AI pack + Gradle modules for an app, or an update of one
  scripts/install_and_verify.py    # build the app, install it like Google Play, read the NPU/CPU result
  templates/                       # Gradle, Kotlin, INTEGRATION.md and UPDATE.md templates (package_ai_pack.py)
  references/compile-flags.md      # API, compiler flags, and --colab reference
  references/benchmark-flags.md    # benchmark_model flags, output, binaries
  references/app-integration.md    # packaging for an app: contents, delivery modes, checks
  references/troubleshooting.md    # symptom -> fix tables
  references/contributing.md       # maintainers: adopting a new LiteRT release
```

Every script supports `--help` and `--dry-run` (`check_compiled.py` is
read-only, so its `--dry-run` is a no-op) and is Python 3 standard library only.
`compile_model.py` imports LiteRT only for a local compile: its `--help`,
`--dry-run`, `--print-install` (`[--colab]`), and `--colab` runs work with any
local `python3`, while a local compile must use the virtualenv's interpreter
(`<venv>/bin/python SKILL_DIR/scripts/compile_model.py ...`, or activate the
venv first). `package_ai_pack.py` likewise imports LiteRT only for a real
run, which needs a Python with `ai-edge-litert` (the compile venv works; the
Beta SDK is not needed); its `--check-project` mode and the other scripts run
with any `python3`.

## Prerequisites (tell the user up front)

| Need | Compile (local) | Compile (`--colab`) | Benchmark | Package (app) |
|---|---|---|---|---|
| **Google Tensor SDK Beta access** - sign up at https://services.google.com/fb/forms/tensor_ml_sdk_experimental_access (overview: https://developers.google.com/edge/litert/next/tensor-sdk). Google emails a download link for `litert_plugin_compiler.tar.gz`; the compiler cannot be installed without it. | required | required (`--sdk-tarball` or `--sdk-url`) | - | - |
| Host OS | Linux **x86_64**, >= 16 GB RAM | any OS (`colab` CLI on PATH; runs on a remote Colab Linux `x86_64` VM) | any OS with `adb` | any OS where `ai-edge-litert` installs (the compile venv's Linux `x86_64` host works) |
| Python environment | Local venv with `ai-edge-litert` + `ai-edge-litert-sdk-google-tensor` at the version in `versions.json` (`compile_model.py --print-install`) | Any local `python3` + `google-colab-cli` (`compile_model.py --print-install --colab`); SDK is installed on the Colab VM automatically | any `python3` (stdlib only) | a Python with `ai-edge-litert` (the compile venv works; the SDK package is not needed) |
| `adb` on PATH, Pixel 10/11 with USB debugging enabled | - | - | required | step 7 (install and check the app) |
| A JDK the app's Gradle supports (`JAVA_HOME`), the Android SDK, the project's `./gradlew` | - | - | - | steps 6-7 (build, install); bundletool is downloaded once |
| `libLiteRtDispatch_GoogleTensor.so` - downloaded automatically from the public LiteRT binaries bucket (same directory as `benchmark_model`), or supplied with `--dispatch-lib` | - | - | required | - (the app gets it from Google Maven) |
| Internet | once for `pip install` | required (Colab VM + `pip install`) | once to fetch `benchmark_model` + dispatch lib (cached in `~/.cache/litert-google-tensor/`) unless supplied locally | - (the app's Gradle build downloads the LiteRT Maven artifacts) |

Supported targets: `Tensor_G5` (Pixel 10 family) and `Tensor_G6` (Pixel 11
family) - the SoCs listed as supported by the public Tensor SDK docs.

## Step 0 - Collect parameters

Ask the user only for what is missing; never invent values:

Parameter                           | How to get it
----------------------------------- | -------------
Model path                          | Absolute path to the `.tflite`. Confirm it exists.
Goal                                | check only, compile only, benchmark only, compile + benchmark, or package for an app (step 5). Infer it from the request ("how fast is it on the TPU vs CPU" = compile if needed + benchmark with a CPU baseline; "put it in my app" / "deploy it" = package); do not add an unrequested device run.
Target SoC                          | For a **compile**, the user's stated device wins (Pixel 10 family = `Tensor_G5`, Pixel 11 family = `Tensor_G6`), even if a different Pixel is attached. For a **benchmark**, the attached phone decides: `adb shell getprop ro.soc.model` -> `Tensor G5` => `Tensor_G5`, `Tensor G6` => `Tensor_G6` (the benchmark script reads and enforces this itself).
Output path                         | Default `<output-dir>/<stem>_Google_<target>.tflite`. `compile_model.py` refuses to overwrite an existing file (exit 2): run `check_compiled.py` on the existing file and reuse it, or pass `--output-name`, or `--overwrite` after asking. When a second build is needed next to an existing one, name it `<stem>_Google_<target>.<YYYYMMDD-HHMM>.tflite` (e.g. `--output-name model_Google_Tensor_G5.20260107-0930.tflite`) or, for a flag variant, `<stem>_Google_<target>.<flag-summary>.tflite` (`...G5.half-maxsharding.tflite`); keep the `_Google_<target>` part so the file stays recognisable.
Device serial                       | Only if `adb devices` lists more than one device.
Dispatch library / benchmark binary | Optional. Both are downloaded automatically from the same public bucket directory and cached. Only ask if the user wants a specific build, or if the host is offline: then `--benchmark-bin <benchmark_model>` plus `--dispatch-lib <libLiteRtDispatch_GoogleTensor.so \| dir \| .zip \| .aar>` (also env `LITERT_GOOGLE_TENSOR_DISPATCH_LIB`) use local copies and no network is touched.
CPU baseline                        | Optional; needs the path of the **original uncompiled** model (ask for it if only a compiled file was given; the benchmark script rejects a compiled file here with exit 2).
Run counts / mode                   | Defaults 5 warmup / 50 measured, `burst` performance mode. `benchmark_model` treats counts as **minimums** and also runs at least 0.5 s warmup / 1 s measured, so fast models execute more runs than requested. If the user states exact counts, say so and ask whether to disable the floors (`--extra-arg=--min_secs=0 --extra-arg=--warmup_min_secs=0`); "steady-state" / "sustained" requests map to `--performance-mode sustained_performance`.
TPU tile time                       | On by default, nothing to ask: a second, hooked NPU pass. `--no-tile-time` turns it off; the hooked pass itself (it repeats the NPU run, so slow models take about twice as long) is skipped only with `--no-tile-time --no-power`.
TPU energy                          | On by default, nothing to ask: the power metric of the same hooked pass (the TPU rail's energy per inference). `--no-power` skips it. The script leaves it off, and says why, for a dispatch library whose power hook predates the fix.
Compile mode                        | Local Linux `x86_64` by default. Use `--colab` when the host is not Linux `x86_64` (macOS/Windows/ARM) or when the user asks to compile on Google Colab; ask for the Beta tarball path (`--sdk-tarball` / `GOOGLE_TENSOR_SDK_BETA`) or URL (`--sdk-url` / `GOOGLE_TENSOR_ML_SDK_URL`) before running.
Compiler flags                      | Default to the compiler defaults; only set flags the user asks for or that troubleshooting requires (see `references/compile-flags.md`).
App deployment level                | Packaging jobs only; nothing to ask up front. After step 5, tell the user the plan is ready (`INTEGRATION.md`, or `UPDATE.md` for an update) and ask whether you should apply it to their project (step 6). After that, offer to install the app on their Pixel and check it uses the NPU (step 7).
New integration or update           | Packaging jobs only. Ask whether the app already ships Google Tensor builds (an AI pack with folders like `model#group_Google_Tensor_G5` under its `src/main/assets`). If it does, it is an **update** (step 5b): ask for the project root (the folder with `settings.gradle.kts` or `settings.gradle`). The pack, model file, device groups, delivery mode and app module then come from the project, so do not ask for them.
Builds to package                   | One compiled build per target SoC (`Tensor_G5`, `Tensor_G6`) plus the **original uncompiled** model they were compiled from (the app's CPU path). Compile missing targets first (step 2); a build is per SoC, so a Pixel 11 needs a `Tensor_G6` build. For an update, build every SoC the app already ships.
Delivery mode                       | New integrations only (an update keeps the app's). When Google Play puts the model on the phone: `install-time` (default; installed with the app, ready at first launch), `fast-follow` (downloaded right after install) or `on-demand` (downloaded when the app asks). Explain each in one sentence (`references/app-integration.md`) and use the default when the user has no preference.
Pack / model names, app module      | New integrations only. Defaults: pack name = the original's file name made valid, model name = pack name, app module `app`, Kotlin package `com.example.litert`. Ask only when the user's project differs (another app module name, their package).
Quick test (optional)               | New integrations only. Offer it when the user wants to try the model on their own Pixel before setting up bundletool: `--quick-test` adds a copy of each build and the steps for `./gradlew installDebug -PtensorQuickTest=<SoC>`, which puts the build and the NPU runtime in the debug APK. Local testing only; release builds are unaffected.

## Step 1 - Check whether the model is already compiled

```bash
python3 SKILL_DIR/scripts/check_compiled.py /abs/path/model.tflite [--expect-target Tensor_G5]
```

Run it on every model file involved (the model to compile or benchmark, the
CPU-baseline file, and an existing output file you are about to reuse).
For a benchmark job read the phone first (`adb shell getprop ro.soc.model`,
which prints `Tensor G5` or `Tensor G6`) and pass the underscore form as
`--expect-target` (`Tensor G5` -> `--expect-target Tensor_G5`), so one check
answers both "compiled?" and "for this phone?" (the flag only matters for
compiled files; a plain model exits 1 either way); the benchmark script
enforces the match again anyway (exit 4, before anything is downloaded or
pushed).

| Exit | Meaning | Next action |
|---|---|---|
| 0 | compiled for Google Tensor (SoC printed) | skip compile; report the SoC (and the `Compile info` line if a `.compile.json` sidecar exists). Continue to step 3 only if the goal includes a benchmark. No sidecar (compiled elsewhere, flags unknown): for a plain "how fast is it" job, benchmark the file as-is and say "compile flags unknown"; when the user needs to know how it was built (or wants a reproducible comparison), compile the original with the defaults under a new `--output-name` and compare SHA-256 (`sha256sum` of both files, or `output_sha256` in the new sidecar): compilation is deterministic, so an identical hash proves the existing file is a default build (`references/compile-flags.md`). Never `--overwrite` to find out. |
| 1 | plain (uncompiled) model | go to step 2 if compiling; this is the correct file for `--cpu-baseline` |
| 2 | compiled for a different SoC than `--expect-target` | compiled models are SoC-specific (a `Tensor_G5` file never runs on a `Tensor_G6` TPU or vice versa; there is no runtime fallback, and `--allow-mismatch` only produces dispatch errors). Tell the user which SoC the phone has, then offer, in this order: an existing `*_Google_<phone target>.tflite` build of the same model in the project (verify it with this check first; do not benchmark it unless the user says so), attaching the matching phone, or recompiling the **original** uncompiled export for the phone's target. |
| 3 | not a `.tflite` / missing | fix the path |
| 4 | compiled by another toolchain: an Edge TPU custom op (`edgetpu-custom-op...`) instead of LiteRT's `DISPATCH_OP` | not usable here: LiteRT cannot recompile it or run it on the Google Tensor TPU, and the CPU cannot run it (so it is no `--cpu-baseline` or `--fallback` either; the scripts refuse it). Ask for the original model it was compiled from and compile that (step 2) |

The check looks for LiteRT's `DISPATCH_OP` custom op and `LiteRtStamp`
metadata, which the AOT compiler always writes; it needs no packages. Files
from the Edge TPU compiler or older Google Tensor tooling carry an
`edgetpu-custom-op` custom op instead (exit 4). The
stamp is authoritative, the `_Google_Tensor_G5` file suffix is only the compile
script's naming convention. The stamp records the SoC but **not** the compiler
flags - those come from the `<model>.compile.json` sidecar that
`compile_model.py` writes next to its output (printed by the check when
present); without it, say "flags unknown" rather than guessing.

## Step 2 - Compile (Linux x86_64 or Google Colab via `--colab`)

### 2a. One-time environment

The SDK pip package contains no compiler by itself: at install time it unpacks
the Beta tarball that `GOOGLE_TENSOR_SDK_BETA` points to (same flow as the
public codelab). Ask the user for the tarball path first, then print the
install commands for the LiteRT version the skill pins in `versions.json`:

```bash
python3 SKILL_DIR/scripts/compile_model.py --print-install [--colab]
```

**Local Linux `x86_64`** (`--print-install`): runs with any `python3` and prints
comment lines (the pinned version, when and on what it was last verified, the
supported Python range, and whether this host and Python qualify) followed by:

```text
python3 -m venv ~/.venvs/litert-tensor && source ~/.venvs/litert-tensor/bin/activate
pip install --upgrade pip
export GOOGLE_TENSOR_SDK_BETA=/abs/path/litert_plugin_compiler.tar.gz   # from the Beta e-mail
pip install --no-cache-dir ai-edge-litert==<pinned> ai-edge-litert-sdk-google-tensor==<pinned>
```

**Remote Google Colab (`--print-install --colab`)** - for macOS, Windows,
non-x86 Linux, or when compiling on Colab without a local Linux venv: installs
`google-colab-cli` into `~/.venvs/colab-cli` (`references/compile-flags.md`).
Never ask the user to run a terminal command for OAuth: when `--colab` needs
one-time OAuth, show its printed URL in chat, ask the user for the code, and
re-run with `--colab-auth-code <CODE>`. `--colab` provisions a Colab Linux
`x86_64` VM, installs the pinned LiteRT + SDK with the user's Beta tarball,
compiles, downloads `<output>.tflite` + `<output>.compile.json`, and stops the
VM. Run the printed commands with the real tarball path; never type a LiteRT
version yourself.

*   Any venv location works; a local compile needs only its interpreter. On
    Debian/Ubuntu `python3 -m venv` may need `sudo apt install python3-venv`. If
    `--print-install` warns about Python version, use `python3.N -m venv ...` in
    the supported range.
*   Without `GOOGLE_TENSOR_SDK_BETA` pip aborts with `SDK location is not set.
    Aborting installation.`; with a wrong path, `Google Tensor SDK beta file not
    found`. The e-mailed URL works via `GOOGLE_TENSOR_ML_SDK_URL=<url>` (or
    `--sdk-url <url>` with `--colab`) instead of a local tarball.
*   `--no-cache-dir` matters when re-installing (add `--force-reinstall` then):
    a wheel built earlier **without** the compiler - on another OS, or with
    `SKIP_SDK_DOWNLOAD` - stays in pip's cache and fails with
    `liblitert_plugin_compiler.so: cannot open shared object file` (exit 5).
*   Tell the user before installing anything. Install the exact pair
    `--print-install` prints: the runtime and SDK packages must be one version
    (`WARNING` when they differ; `NOTE` when they match each other but not the
    pin). The pin changes only via `references/contributing.md` (compatibility:
    https://developers.google.com/edge/tensor-sdk/release-notes#compatibility).
    Use `-nightly` packages only when the user explicitly asks for nightlies.

**If the user has no Beta access yet:** say so plainly - the compiler cannot be
installed until Google e-mails the tarball link (sign-up form above) - and do
not install anything or allocate a Colab VM meanwhile (`--colab` checks for the
tarball before `colab new`). Still useful now: run `check_compiled.py` on their
files, confirm the host and phone SoC, and run `--dry-run` / `--print-install`.

### 2b. Compile

```bash
# Local Linux x86_64 (using the virtualenv's interpreter):
python3 SKILL_DIR/scripts/compile_model.py /abs/path/model.tflite \
  --target Tensor_G5 --output-dir /abs/path/compiled

# Remote Google Colab VM (any OS, any local python3 + google-colab-cli):
python3 SKILL_DIR/scripts/compile_model.py /abs/path/model.tflite \
  --target Tensor_G5 --colab --sdk-tarball /abs/path/litert_plugin_compiler.tar.gz \
  --output-dir /abs/path/compiled
# -> /abs/path/compiled/model_Google_Tensor_G5.tflite
```

*   Sets `GOOGLE_TENSOR_BACKEND_ENABLED=1` (required by the stable package),
    checks the installed LiteRT package versions against each other and
    `versions.json`, calls `aot_compile(..., target=Target(SocModel.TENSOR_G5),
    keep_going=False, google_tensor_*=...)`, prints the compilation report,
    exports to `--output-dir`, re-verifies the Tensor stamp and writes
    `<output>.compile.json` (flags, `ops offloaded / total`, package versions,
    SHA-256 of input/output/compiler lib, elapsed time, plus `compiled_via:
    "colab"` with `--colab`). Never overwrites an existing output (exit 2)
    unless `--overwrite` is given.
*   Optional compiler flags (`references/compile-flags.md`): `--truncation-type
    {auto,bfloat16,half,no_truncation}`, `--sharding-intensity
    {minimal,moderate,extensive,maximum}`, `--int64-to-int32`,
    `--large-model-support`, `--enable-4bit`, `--extra-options-path FILE`.
    Script options: `--output-name`, `--overwrite`, `--keep-going`,
    `--keep-intermediates`, `--no-report-file`, `--json`, `--dry-run`. Colab
    options: `--colab`, `--sdk-tarball`, `--sdk-url`, `--colab-session`,
    `--keep-colab-session`, `--colab-high-mem`, `--colab-timeout`,
    `--colab-auth`, `--colab-auth-code`, `--colab-bin`.
* If the compiler crashes with an internal error - or the user reports that it
  did with the defaults (no need to reproduce the crash) - use
  `--truncation-type half --sharding-intensity maximum` (public FAQ guidance)
  and tell the user which flags produced the artifact. `half` changes the
  numerics (16-bit floats on the TPU), so tell the user to validate the
  model's outputs against the original before relying on it.
* Read the printed compilation report: ops left on the CPU mean the strict
  NPU benchmark will fail (see step 3, `--allow-cpu-fallback`).
* Report a compile-only job from the script's output and sidecar, nothing
  else:

  ```markdown
  **Compiled:** <output path> (<bytes from "Compiled model written to" line>)
  **Target / flags:** <cli_flags from the sidecar, or "--target X (compiler defaults)">
  **Ops offloaded:** <N / M from the "Ops offloaded to the TPU" line>
  **LiteRT:** <"packages" from the sidecar, plus the script's WARNING/NOTE line if it printed one>
  **Verified:** check_compiled.py -> <its Status line>
  ```

Exit codes: 0 ok, 2 bad input, 3 already compiled, 4 unsupported host,
5 LiteRT packages or Beta compiler library missing (message includes the
install fix), 6 compile failed (compiler log errors are printed),
7 output verification failed.

## Step 3 - Benchmark on the device

```bash
python3 SKILL_DIR/scripts/benchmark_on_device.py \
  --model /abs/path/compiled/model_Google_Tensor_G5.tflite \
  [--cpu-baseline /abs/path/model.tflite] [--serial SERIAL] \
  [--dispatch-lib /abs/path/libLiteRtDispatch_GoogleTensor.so] \
  [--output-dir /abs/path/results]
```

The script, in order:

1. Selects the device, reads `ro.soc.model`, and refuses to run a model
   compiled for another SoC (`--allow-mismatch` overrides). `--dry-run`
   skips this - it never contacts the device - so a dry run proves nothing
   about the SoC match.
2. Resolves the two host-side binaries, both cached under
   `~/.cache/litert-google-tensor/` (file names `<release>_android_arm64_<file>`,
   `<release>` being the bucket directory) after the first download:
   * `benchmark_model`: `--benchmark-bin`, else a one-time download of
     `--benchmark-url` (default
     `https://storage.googleapis.com/litert/binaries/<release>/android_arm64/benchmark_model`
     with `<release>` = `binaries_release` from `versions.json`; `--help`
     prints the resolved URL).
   * `libLiteRtDispatch_GoogleTensor.so`: `--dispatch-lib` (a `.so`, or a
     `.zip`/`.aar` it is extracted from), else env
     `LITERT_GOOGLE_TENSOR_DISPATCH_LIB`, else a one-time download of
     `--dispatch-url` (default: the same directory as `--benchmark-url`).
   The script prints where each binary came from; `downloaded from <url>`
   means this was the first time - tell the user what was fetched. Downloads
   end with `(fetched <UTC time>, md5 <hash>)`, recorded when the file was
   fetched; `fetch date not recorded` marks a copy cached by an older version
   of the script (`--refresh-binary` re-downloads it and records the fetch).
3. Pushes files to a fresh `/data/local/tmp/litert_benchmark/run-<timestamp>-<pid>/`
   (`--device-dir` changes the base). Files already on the phone are never
   overwritten or deleted. Leftovers from an interrupted run are therefore
   always `<base>/run-*`: if the user asks to clean up, `adb shell ls -la <base>`
   first and remove only `run-*` entries (never the base or anything else in it).
4. Reads the thermal status (`dumpsys thermalservice`), then runs the NPU
   benchmark in strict mode:
   `--use_npu=true --use_cpu=false --require_full_delegation=true
   --dispatch_library_path=<dir> --num_runs=50 --warmup_runs=5
   --report_peak_memory_footprint=true --use_profiler=true`, retrying once
   without the profiler if that run fails.
5.  Runs the same NPU benchmark once more, without profiler and memory monitor,
    with `LITERT_VENDOR_HOOK_ARGS` pointing at a hook config in the run
    directory that enables the TPU tile-time and power metrics, and divides the
    hooks' totals (`Total TPU tile time`, `Total Energy`) by that pass's own
    warmup + measured runs. The hooks' own counts are only checks: older
    dispatch libraries count every inference 1 + D times (D = dispatch nodes, so
    2x for a normal compiled model), which is why the hooks' `Average` lines are
    never used; an old power hook is not used at all (the script checks the
    library first). Checks and evidence: `references/benchmark-flags.md`, "TPU
    tile time" and "TPU energy".
6.  Optionally runs the CPU baseline on the **uncompiled** model
    (`--use_cpu=true`, same run counts).
7.  Prints the Markdown header + table, writes `results.json`, `report.md` (the
    same block as a file), `npu.log`, `tile_time.log` (the hooked pass, + the
    hooks' `tile_time_report.txt` and `power_report.txt`), `cpu.log`,
    `npu_profile.txt`, `cpu_profile.txt`, and deletes the per-run device
    directory (even on failure). `--output-dir` is only created once a benchmark
    actually ran - after exit 2/3/4 or `--dry-run` there are no logs to look
    for.

The script is NPU-only by design: `--extra-arg` passes extra `benchmark_model`
flags but cannot switch the accelerator, and an uncompiled model is rejected
(exit 2). For a GPU or non-Tensor run use `benchmark_model` directly
(`--use_gpu=true` etc., https://developers.google.com/edge/litert/next/benchmark);
that is outside this skill.

Useful options: `--num-runs`, `--warmup-runs`, `--signature`,
`--performance-mode
{extreme_power_saver,power_saver,balanced,high_performance,sustained_performance,burst}`
(benchmark default `burst`), `--cpu-num-threads`, `--extra-arg=--min_secs=5`,
`--device-dir`, `--benchmark-bin` + `--dispatch-lib` (local copies, offline),
`--serial`, `--allow-cpu-fallback`, `--keep-device-files`, `--refresh-binary`,
`--no-tile-time`, `--no-power`, `--dry-run`.

Offline / air-gapped host: pass both `--benchmark-bin` and `--dispatch-lib`
and run with `--dry-run` first - the plan then shows `user-provided: <path>`
for both binaries without contacting the phone or the network, which is the
proof the user wants before the real run.

Binary versions: keep `benchmark_model` and the dispatch library from the
**same** release directory of the bucket (the default does: both come from
`binaries_release` in `versions.json`); pass
`--benchmark-url .../binaries/<release>/android_arm64/benchmark_model` to pin
another one and the dispatch URL follows. A rolling directory (`latest/`,
`nightly/`) changes over time, but a cached copy stays the build that was first
downloaded until `--refresh-binary`; the md5 in its provenance line tells builds
apart. A numbered directory can be re-uploaded as well (bucket notes in
`references/benchmark-flags.md`): if a provenance md5 differs from
`verified.binaries` in `versions.json`, rerun with `--refresh-binary`. Local
files carry no version string the script could compare, so for
`--benchmark-bin`/`--dispatch-lib` the rule is "both from the same download"; if
that is uncertain, re-download both from one directory. Not every versioned
directory ships the Google Tensor dispatch library (which ones do: the bucket
table in `references/benchmark-flags.md`) - if the download 404s, use the
default URLs or supply `--dispatch-lib` from `litert_npu_runtime_libraries.zip`
on https://github.com/google-ai-edge/LiteRT/releases or a build per
https://developers.google.com/edge/tensor-sdk/tensor-faqs.

## Step 4 - Report

Nothing in the report is typed by hand. `benchmark_on_device.py` prints a
Markdown header + table (also stored as the `markdown` field of
`results.json`) in which every value is read from the device or the run:

Report field                                | Comes from
------------------------------------------- | ----------
Device name, SoC, Android version           | `ro.product.model`, `ro.soc.model`, `ro.build.version.release` on the attached phone (a Pixel 11 reports itself as Pixel 11 / Tensor G6)
Model name, size, compiled-for SoC          | the `.tflite` file and its `LiteRtStamp`
**Compile** (flags, ops offloaded / total)  | the `<model>.compile.json` sidecar next to the model; `unknown` when the model was compiled elsewhere - never fill it in from memory
**Versions**                                | the sidecar's `ai-edge-litert` version (the LiteRT that compiled the model) vs the bucket release in the binaries' URLs (`user-provided` for local files): `same release`, `DIFFERENT releases`, or why the two cannot be compared (no sidecar, a rolling `latest`/`nightly` build, local files, binaries from two sources); also stored as `versions` in `results.json`
**Delegation (runtime)**                    | the `Replacing N out of M node(s) with delegate (DispatchDelegate)` line of this run's `npu.log` - the ground truth that everything ran on the TPU
Mode, requested run counts, NPU flags       | the flags actually passed to `benchmark_model`
**Thermal before NPU run**                  | `dumpsys thermalservice`: the throttling status (0 = none) is the signal; the sensor value is one reading, informational only
**TPU tile time** (header line + table row) | the tile-time pass: the hook's `Total TPU tile time` ÷ that pass's warmup + measured runs. The header names the hook's call count and what it means (`1 call per run, one per dispatch invoke`, or `2 calls per run (... the known double count, corrected)` with older dispatch libraries); `unavailable`, `failed`, `no report` or `inconsistent` (count matches no known pattern: totals only) carry the reason instead
**TPU energy** (header line + table row)    | the same hooked pass: the power hook's `Total Energy` (the TPU rail from the first inference to the model's teardown, idle gaps included) ÷ that pass's warmup + measured runs, in mJ. The header names the hook's call count (`1 call per run, one per dispatch invoke`); `unavailable` (an old power hook, which the script does not enable, or a phone that refused the energy counter), `failed`, `no report`, `unreliable` (old hook's count pattern) or `inconsistent` carry the reason instead
Binary / dispatch library provenance        | the resolver (`downloaded from <url>`, `cached download of <url>`, `user-provided: <path>`); downloads end with `(fetched <UTC time>, md5 <hash>)` recorded at download time, which identifies the build of a rolling release
All timings, memory, actual run counts      | parsed from the `BENCHMARK RESULTS` block of this run's log
**Inference p5 / p50 / p95 (ms)**           | the measured-phase `count=N ... p5= median= p95=` stats line of the log (microseconds, converted to ms by the script); these are the only percentiles `benchmark_model` produces - there is no p99
**Log check**                               | the script's scan of `npu.log` / `cpu.log` / `tile_time.log`: WARNING/ERROR-level lines that match the benign list in `references/benchmark-flags.md` are only counted (`N known-benign`); any other such line is listed verbatim under **Unexpected log lines** after the table (de-duplicated, with a repeat count), and the line also says whether the profiler retry happened. No grepping of the logs is needed.

Paste that block verbatim (it is also saved as `report.md` in the output
directory) and add only the context the benchmark script cannot know:

```markdown
**Summary:** <NPU "Inference avg" from the table> ms on the TPU vs <CPU "Inference avg"> ms on the CPU (<the N.NNx value from the script's "NPU speed-up vs CPU" line>); peak memory <NPU> vs <CPU> MB; <strict NPU | CPU fallback allowed>. TPU tile time <"TPU tile time / inference" row> ms per inference. TPU energy <"TPU energy / inference" row> mJ per inference.

<header and table exactly as printed by benchmark_on_device.py, including any "Unexpected log lines" section>

**Notes:** <what the Log check line reports: "none" when both logs show 0 unexpected lines and no profiler retry; otherwise each unexpected line with its meaning from references/benchmark-flags.md or troubleshooting.md, plus anything the user should validate>
```

Never edit `report.md` (it is script output). If the user wants the full
answer saved, write it as `summary.md` next to `report.md`, with the block
copied from `report.md`; do not keep hand-typed numbers anywhere else.

Combined compile + benchmark job: present only this benchmark block (its
**Compile** header line already carries the sidecar's flags and op counts)
and start the Notes with one line of compile context - that the input was
uncompiled and was compiled first, the output path, and that nothing existing
was overwritten (`compile_model.py` made sure of that). The step 2b
compile-only block is for jobs that end at the compile.

Without a CPU baseline the Summary line is just `**Summary:** <NPU "Inference
avg"> ms per inference on the TPU (<strict NPU | CPU fallback allowed>, no CPU
comparison requested); peak memory <NPU> MB.` plus the same TPU tile time and
TPU energy sentences. Drop each of those two sentences when its header line says
`not measured`, `unavailable`, `failed`, `no report`, `unreliable` or
`inconsistent`, and give the header's reason in the Notes.

When nothing ran (check exit 2, benchmark exit 2/3/4/5, no device), report
the same way - from output, not from memory:

```markdown
**Not run:** <the script's ERROR line verbatim>

- Device: <the "Device: ..." line printed by benchmark_on_device.py, or the `adb devices -l` line>
- Model: <the "Status: ..." line from check_compiled.py>
- Nothing was pushed to or removed from the phone.

**Options:** <the next actions from step 1 / troubleshooting, most specific first>
```

* NEVER type, round, or "tidy up" numbers, and NEVER reuse values from an
  earlier run or from any example in this skill - copy the script output.
  Prose numbers (the Summary line) must be character-for-character the table's
  values.
* Report the **actual** run counts from the table: `benchmark_model` treats
  `--num_runs`/`--warmup_runs` as minimums and also runs for at least 1 s /
  0.5 s, so fast models show more runs than requested (details in
  `references/benchmark-flags.md`, which also shows a complete example
  output from a real run and the list of benign log lines).
* The **Mode** line says whether the run was strict NPU or allowed CPU
  fallback; call that out in prose as well, because fallback numbers mix NPU
  and CPU time.
* The **Versions** line compares the LiteRT that compiled the model with the
  release that ran it. Mention `DIFFERENT releases`, a
  `but SDK package X (DIFFERENT)` or binaries from two sources in the Notes -
  numbers from such a mix are harder to compare with other runs - unless the
  line ends with `(the pair verified in versions.json)`.
  `not comparable (...)` and `cannot compare` need no comment beyond the line.
* A speed-up below 1x is a real result, not a setup error, for tiny models
  (per-inference NPU dispatch overhead of a few ms dominates); say so instead
  of hunting for thermal or configuration problems.
* The per-op profile (`npu_profile.txt`) shows one `DELEGATE` node whose
  average is lower than the table's `Inference avg` (it excludes input/output
  handling); report the table value, not the profile value.
*   **TPU tile time** is pure TPU hardware time, so it is below `Inference avg`
    (which adds host and driver work). Quote it from the table only, never from
    the hook's own lines in `tile_time.log` / `tile_time_report.txt`: older
    dispatch libraries inflate its `Total inferences`, so their `Average` is too
    low (half for a normal compiled model), while the table value is right with
    every library.
*   **TPU energy** is the TPU rail's energy (the phone's on-device power
    monitor) for the whole hooked pass divided by its runs, so it includes the
    TPU's idle gaps between back-to-back inferences: compare it between models,
    builds and `--performance-mode`s on the same phone, and do not present it as
    a lab power measurement or as the energy of one isolated inference. Quote it
    from the table only.

## Step 5 - Package for an app (Gradle)

Two cases (Step 0, "New integration or update"): **5a** for an app without
Google Tensor builds yet, **5b** for an app that already ships them and gets
new builds (a new model version, other compile flags, a new LiteRT release).

### 5a - New integration

Turns the builds from step 2 into what an Android Gradle app needs to ship
them through Google Play (details: `references/app-integration.md`): an AI
pack written by LiteRT's own exporter from the files as they are (nothing is
recompiled; the packed models are byte-identical), the Google Tensor NPU
runtime as an install-time feature module (from Google Maven), the original
model for the app's base assets (the CPU path), a Kotlin loader, and an
`INTEGRATION.md` with the user's names and versions filled in.

```bash
<venv>/bin/python SKILL_DIR/scripts/package_ai_pack.py \
  --compiled /abs/path/compiled/model_Google_Tensor_G5.tflite \
  [--compiled /abs/path/compiled/model_Google_Tensor_G6.tflite] \
  --fallback /abs/path/model.tflite \
  [--delivery install-time|fast-follow|on-demand] [--pack-name NAME] \
  [--model-name NAME] [--app-module app] [--kotlin-package com.example.litert] \
  [--quick-test] [--output-dir /abs/path/app_package/NAME]
# -> <fallback dir>/app_package/<pack name>/INTEGRATION.md and the modules
```

*   Run `check_compiled.py` on every file first (step 1), and package builds
    that passed the strict benchmark (step 3). To benchmark with the exact
    runtime library the app will ship, pass the Maven AAR as `--dispatch-lib`
    (`references/app-integration.md`, "Runtime facts").
*   A real run needs a Python with `ai-edge-litert` (the compile venv works; no
    Beta SDK). `--help` and `--dry-run` work with any `python3`: run
    `--dry-run` first to check the inputs without writing anything.
*   Before writing, the script checks that each `--compiled` file is stamped
    for `Tensor_G5`/`Tensor_G6` with one build per SoC, that `--fallback` is
    uncompiled, and that each build matches its `.compile.json` (SHA-256,
    target) and was compiled from `--fallback` (only that last check can be
    overridden, with `--allow-source-mismatch`). It writes into a staging
    directory, verifies the result (pack folders, byte-identical models,
    stamps, the same device groups in the XML and the runtime module, the CPU
    model) and only then moves it to `--output-dir`. It never replaces a
    directory it did not write; `--overwrite` replaces only its own package.
*   Report from the script's output: paste its `**Packaged:** ...` block
    verbatim, then one line for the user: the delivery mode in plain words and
    "next: follow `<output-dir>/INTEGRATION.md`".
*   `INTEGRATION.md` lists everything left to do in the user's project: copy
    the modules, the Gradle and manifest edits, the loader call, and testing
    on a Pixel with bundletool. The app needs no LiteRT or NPU code
    beforehand: the package brings the LiteRT runtime dependency, the Google
    Tensor dispatch library (inside the runtime feature module, from Maven),
    the models and the loader. Then ask: "The integration plan is ready
    (`INTEGRATION.md`). Do you want me to apply it to your project?" If yes,
    go to step 6a; if not, the guide is theirs to follow. Publishing to Google
    Play stays with the user.
*   `--quick-test` (Step 0, optional) also writes `quick_test/<SoC>/assets/`
    and an INTEGRATION.md section: copied to `<app module>/tensor_quick_test/`
    with a small Gradle block, `./gradlew installDebug
    -PtensorQuickTest=<SoC>` builds a debug APK that carries the Tensor build
    and the NPU runtime itself, so the model runs on the TPU with no
    bundletool. Nothing changes without the property, and release builds
    never change. It is for trying the model, not a substitute for the
    bundletool test before a release.

Exit codes: 0 ok, 2 bad input (missing file, duplicate SoC, invalid name,
output exists, a build that differs from its sidecar), 3 wrong model kind
(uncompiled `--compiled`, compiled `--fallback`), 5 LiteRT not importable or
its exporter API changed (the message has the fix), 7 post-export verification
failed (nothing is left at `--output-dir`).

### 5b - Update builds the app already ships

The same script with `--update-project` prepares only the files to replace,
under the app's own paths and names, and finds out whether anything else must
change (details: `references/app-integration.md`, "Updating builds an app
already ships").

```bash
<venv>/bin/python SKILL_DIR/scripts/package_ai_pack.py \
  --update-project /abs/path/to/app-project \
  --compiled /abs/path/compiled/model_Google_Tensor_G5.tflite \
  [--compiled /abs/path/compiled/model_Google_Tensor_G6.tflite] \
  --fallback /abs/path/model.tflite \
  [--pack-name NAME] [--model-name NAME] [--cpu-asset PATH|none] [--dry-run]
# -> <fallback dir>/app_update/<pack>/UPDATE.md, update.json, files/<paths>
```

*   The project is only read. The script finds the AI pack with Google Tensor
    folders, the device-targeting XML (groups are matched by SoC, so the
    app's own group names work), the runtime (Maven artifact, version catalog
    or a bundled `.so`), a generated `TensorModelLoader`, and the app's CPU
    copy of the model (the pack's `#group_other` folder in LiteRT's default
    layout, or the base assets in 5a's layout). It runs the same input checks
    and LiteRT export as 5a and writes the new files to `files/`, at the same
    paths as in the project. `--pack-name` and `--model-name` only pick one
    when the project has several; `--cpu-asset` names the CPU copy when the
    script cannot find it (the error lists candidates). Quick-test copies in
    `<app module>/tensor_quick_test/<SoC>/` (5a's `--quick-test`) are
    replaced too.
*   Run `--dry-run` first and show the user the verdict and the file list:
    `drop-in` (only model files change), `structural` (section 2 of
    `UPDATE.md` lists the other changes: a new SoC needs a group in the XML,
    the runtime module and the loader; a SoC without a new build keeps
    shipping its old one until it is compiled or removed; changed model
    inputs or outputs need app code changes; a missing runtime or mismatched
    LiteRT versions) or `up-to-date`. The inputs/outputs comparison needs
    LiteRT, so it runs only in the real run, which can turn `drop-in` into
    `structural`. For `structural`, list the changes and ask how to proceed
    before anything else.
*   A `Runtime` check of `device` means the app's runtime is not the release
    this skill pins, or its version is unknown: the new builds must pass the
    strict benchmark (step 3) with the app's own library before release
    (`--dispatch-lib` with the app bundle, the runtime AAR or the `.so`);
    `UPDATE.md` section 4 has the exact commands.
*   Report from the script's output: paste its `**Update plan:** ...` block
    verbatim, then one line: the verdict in plain words and "next:
    `<output-dir>/UPDATE.md`". Then ask: "The update plan is ready
    (`UPDATE.md`). Do you want me to apply it to your project?" If yes, go to
    step 6b. Never copy a 5a package over an existing pack.

Exit codes as in 5a; exit 2 also covers a project the script cannot update
as is (no AI pack with Google Tensor builds - then it is a new integration -,
several packs or models without `--pack-name`/`--model-name`, a device group
that selects several SoCs, an unknown CPU copy).

## Step 6 - Apply the plan to the user's project (when they say yes)

You apply the plan; the user approves it once. Before changing anything,
show the **whole plan** and get **one approval**. Then ask **separately** for
each risky item (the check marks them): upgrading the Android Gradle plugin
or Kotlin, adding Kotlin, raising `minSdk`, limiting the app to arm64-v8a
(devices without it can no longer install the app), changing the app's
LiteRT version, replacing project files that differ from the package,
removing a SoC. A declined item is skipped; say what will not work without
it.

### 6a - New integration

1.  Ask for the project root (the folder with `settings.gradle.kts` or
    `settings.gradle`). If it is a git checkout with uncommitted changes
    (`git status --porcelain`), say so in the plan and suggest committing
    first.
2.  Get the plan (read-only, any `python3`, no LiteRT):

    ```bash
    python3 SKILL_DIR/scripts/package_ai_pack.py --check-project /abs/path/to/app-project \
      --package <output-dir of step 5a> [--app-module app] [--json]
    ```

    Every requirement of `INTEGRATION.md` comes back as `pass`, `fail` or
    `warn`, each `fail` with its fix: a copy command, or a snippet and the
    file it goes in (in the project's Kotlin or Groovy DSL), and `risky` when
    it needs a separate OK. The `fail` items are the plan; offer the `warn`
    items (for example the optional quick test) as options.
3.  Show the plan once: the files to copy, the files to edit with their
    snippets, and the risky items marked. Get the approval, then the
    separate OKs.
4.  Apply it: run the copy commands; make the edits with your file-editing
    tool, merged into the existing blocks (`android { }`, `defaultConfig`,
    `dependencies`, the version catalog when the project uses one) in the
    file's own style. Never delete or reformat unrelated code, and never
    touch signing configs.
5.  Re-run step 2 until nothing fails (apart from declined items).
6.  Build once from the project root: `./gradlew :<app module>:bundleDebug`.
    Fix errors your edits caused; report errors that were there before
    instead of fixing them.
7.  Report what changed, and how the app uses the model: `INTEGRATION.md`
    step 9 has the loader call, `model_io.json` the inputs and outputs. If
    the user tells you where the model is used, offer to write that code
    (the pre- and post-processing depend on their app, so show it first).
    Then offer step 7.

Exit codes of `--check-project`: 0 nothing to do, 1 something fails (the
report lists it), 2 bad input (no project, no package, no app module).

### 6b - Update

The plan is `UPDATE.md`: the file table and section 2. Get one approval for
the files and the non-risky edits, and separate OKs for the risky ones
(removing a SoC, changing the runtime or LiteRT version, replacing a loader
with local changes; changed model inputs or outputs need app code changes,
so show that code first). Then: copy each project file you replace to
`<output-dir>-backup/<path>` (not inside `<output-dir>`, which `--overwrite`
replaces), copy `files/<path>` to `<project>/<path>`, make the section 2
changes, and raise `versionCode`. Re-run the step 5b command with
`--dry-run`: every file must show `unchanged`. Build once (as in 6a, step
6), then offer step 7.

## Step 7 - Install and check on a Pixel (when the user agrees)

Builds the user's app, installs it on the attached Pixel the way Google Play
would (bundletool local testing, for the phone's device group), and reads
the loader's `LITERT_TENSOR_RESULT` line:

```bash
python3 SKILL_DIR/scripts/install_and_verify.py --project /abs/path/to/app-project \
  [--app-module app] [--variant debug|release|<flavor>Debug] [--serial <id>] \
  [--uninstall-first] [--dry-run]
```

*   Needs a JDK the project's Gradle supports (`JAVA_HOME`), the Android SDK
    (adb), the project's `./gradlew` (or `--gradle`, which may carry
    options such as `-Pname=value`), and a Pixel with USB debugging.
    bundletool is downloaded once (`bundletool_release`, cached) or given
    with `--bundletool <bundletool-all jar or folder of jars>`. Run
    `--dry-run` first: it shows the phone, its device group and whether the
    NPU path is expected. An app with product flavors needs `--variant`
    (for example `freeDebug`).
*   Local testing cannot update an installed app. Exit 7 means the app is
    installed: ask the user (uninstalling deletes the app's data on the
    phone), then rerun with `--uninstall-first`.
*   The app must call `TensorModelLoader.load()`. If it loads the model on a
    particular screen, tell the user to open it while the script waits (exit
    6 = no result line within `--timeout`).
*   Report the script's block verbatim. `"path":"npu"` is the pass; for
    `"path":"cpu"` on a phone the app has a build for, the `npu_error` says
    why (`references/troubleshooting.md`). The app stays installed.

Exit codes: 0 ran on the NPU (or on the CPU where no Tensor build is
expected; with `--no-launch`: installed), 1 CPU on a phone with a Tensor
build, 2 environment (project, tools, device), 3 Gradle build failed, 4
bundletool failed, 6 no result line, 7 already installed.

## Rules

* ALWAYS run `check_compiled.py` before compiling or benchmarking, on every
  model file involved - the compiler cannot recompile a compiled model, and
  the TPU cannot run an uncompiled one.
* ALWAYS verify the device SoC matches the model's stamp (the benchmark
  script does) - a `Tensor_G5` model on a `Tensor_G6` phone fails with an
  opaque dispatch error, wasting a debugging round.
* ALWAYS use the original uncompiled model for `--cpu-baseline` - compiled
  models contain `DISPATCH_OP` nodes that only the NPU can execute.
* ALWAYS clean up the per-run device directory (automatic; after an
  interrupted run list the base first, then remove only its `run-*` entries:
  `adb shell ls -la <base>` and `adb shell rm -rf <base>/run-*`, where `<base>`
  is `/data/local/tmp/litert_benchmark` or the `--device-dir` that was used).
  NEVER `rm -rf` the base directory itself - users keep their own binaries
  and models there.
* NEVER combine `--use_cpu=true` with `--require_full_delegation=true` -
  `benchmark_model` aborts on that combination.
* NEVER install packages into the system interpreter or without telling the
  user; use a virtualenv so pinned LiteRT versions do not break other work.
* NEVER type LiteRT version numbers into install commands or download URLs
  yourself - take them from `compile_model.py --print-install` and the
  scripts' default URLs, which all read `versions.json`, so every job uses the
  pair that was verified together. Use another release only when the user
  asks for one (`--benchmark-url .../binaries/<release>/...`, or the pip pair
  at that version) and say so in the report. Changing the pin is a maintainer
  task (`references/contributing.md`), not part of a user job.
*   NEVER treat macOS, Windows, or ARM hosts as a compile blocker or give manual
    browser Colab notebook steps - run `compile_model.py --colab` directly.
*   NEVER leave adb, benchmark, or Colab VM sessions running in the background -
    every command here is bounded by a timeout and returns before the next step
    (`compile_model.py --colab` stops any Colab VM it creates unless
    `--keep-colab-session` is passed).
*   NEVER guess compile flags or overwrite an existing output without asking
    (`compile_model.py` exits 2; use `--output-name` or ask before
    `--overwrite`).
*   ALWAYS package for an app with `package_ai_pack.py` (it calls LiteRT's
    exporter and verifies the result); NEVER create or edit AI pack folders,
    the device-targeting XML or the runtime modules by hand.
*   ALWAYS use `--update-project` (step 5b) when the app already ships Google
    Tensor builds; NEVER copy a new package's `ai_packs/` over an existing
    pack - the app's group names and the place of its CPU copy can differ.
*   ALWAYS ask before changing the user's project, show the whole plan, and
    get one approval; ask separately for each risky item (step 6) and before
    `install_and_verify.py --uninstall-first`. NEVER change signing configs.
*   NEVER put a compiled build where the app runs on the CPU (base assets,
    `--fallback`): only the original uncompiled model.
*   NEVER publish to Google Play, sign, or upload anything for the user; the
    skill stops at the package and the local-testing steps.

## Troubleshooting (short list)

Symptom                                                                                            | Fix
-------------------------------------------------------------------------------------------------- | ---
`compile_model.py` exit 2 `output already exists`                                                  | Reuse it (after `check_compiled.py`), or `--output-name`, or ask and `--overwrite`
`compile_model.py` exit 3 `already compiled`                                                       | The input is a compiled artifact; compile the original export (`--force` does not help)
`check_compiled.py` exit 4 / `compiled by another toolchain` (any script)                          | The file holds an Edge TPU custom op (`edgetpu-custom-op...`) from another compiler; no flag helps. Get the original model it was made from and compile that (step 2)
`compile_model.py` exit 5                                                                          | Local: LiteRT/SDK not importable or no compiler library (use the venv's interpreter, or reinstall with `GOOGLE_TENSOR_SDK_BETA` set). `--colab`: `colab` CLI missing, auth failed (`colab sessions`), or `--sdk-tarball` not supplied - see 2a
`compile_model.py` `WARNING: ai-edge-litert X and ai-edge-litert-sdk-google-tensor Y differ`       | The two packages must be one version: run the reinstall line the warning prints (pinned version, `--force-reinstall`) with `GOOGLE_TENSOR_SDK_BETA` set
`INTERNAL error at compiler_plugin.cc` / abort during compile                                      | `--truncation-type half --sharding-intensity maximum`
"does not support layers with dynamic shapes"                                                      | Re-export the model with static shapes
`compile_model.py` exit 4                                                                          | Host is not Linux `x86_64` - re-run with `--colab --sdk-tarball /abs/path/litert_plugin_compiler.tar.gz`, or compile on a Linux `x86_64` machine and benchmark here
`adb devices` lists nothing                                                                        | USB: re-plug, accept the USB-debugging prompt. Forwarded phone (serial `host:port`, e.g. `localhost:41015`): `adb connect host:port`. Prefer `adb reconnect` over `adb kill-server` (the latter drops every device on a shared host)
"more than one device connected"                                                                   | `--serial <id>` from `adb devices`
Strict NPU run fails, `--allow-cpu-fallback` works                                                 | Not fully delegated; check the compilation report
Run dies right after `Loading shared library: .../libLiteRtDispatch_GoogleTensor.so`               | Unusable dispatch `.so` (bad download/ABI, or far from the binary's release) - `--refresh-binary` with the defaults
`download failed (...)` for the dispatch library (exit 5)                                          | HTTP 404 = that bucket directory has no Google Tensor dispatch lib; otherwise the host is offline. Use the default URLs (release from `versions.json`), or pass `--dispatch-lib`
`benchmark_on_device.py` exit 6 `exited 0 but printed no results in the format this script parses` | The `benchmark_model` build prints a format the parser does not know (newer or older binary). Re-run with the default URLs, or pin a release with `--benchmark-url`; the fix in the skill is in `references/contributing.md`
`**TPU tile time:** unavailable - the dispatch library returned no telemetry hook`                 | The `--dispatch-lib` was built from the LiteRT GitHub source (hooks compiled out): use the default bucket binaries, or `--no-tile-time`
`**TPU energy:** unavailable - this dispatch library's power hook predates the fix ...`            | The library has the old power hook (e.g. a Maven runtime AAR passed as `--dispatch-lib`, or a stale cache): use the default binaries (`--refresh-binary` for a stale cache). Latency and tile time are unaffected
`**TPU energy:** unavailable - the power hook could not read the TPU energy counter: ...`          | The phone refused the shell access to its power monitor: report the quoted line. Latency and tile time are unaffected
`package_ai_pack.py` exit 5                                                                        | Run it with the Python that has `ai-edge-litert` (the compile venv; the message prints the install line). "exporter failed": install the pinned `ai-edge-litert` (`compile_model.py --print-install`)
`package_ai_pack.py` exit 2 `differs from the build its .compile.json describes`                   | The file changed after it was compiled: package the build that was compiled and benchmarked, or recompile
bundletool: `refers to device groups [...] which are not in the list [...]`                        | The app's `device_targeting_configuration.xml` lacks the package's groups: copy or merge it from the package (`INTEGRATION.md` step 2)
App log `LITERT_TENSOR_RESULT {"path":"cpu", ...}` on a packaged Pixel                             | Check `adb shell pm path` for the `google_tensor_runtime` split, `useLegacyPackaging = true`, and one release for `litert` and `litert-npu-runtime-google-tensor`
`--update-project` exit 2 `no AI pack with Google Tensor builds`                                   | The app has no Google Tensor setup yet: package without `--update-project` (step 5a)
`--update-project` exit 2 `cannot tell where the app keeps the original (CPU) model`               | `--cpu-asset <path in the project>` (the error lists the models it found), or `--cpu-asset none` if the app has no CPU copy
Quick test (`installDebug -PtensorQuickTest=<SoC>`) logs `"path":"cpu"`                            | `<app module>/tensor_quick_test/<SoC>/assets/` must exist (Gradle ignores a missing folder), `<SoC>` must be the phone's (`adb shell getprop ro.soc.model`, underscore form), and the Gradle block must be in the app module
`install_and_verify.py` exit 7 `already installed`                                                 | Ask the user (it deletes the app's data on the phone), then rerun with `--uninstall-first`
`install_and_verify.py` exit 6 (no result line)                                                    | The app must call `TensorModelLoader.load()`: open the screen that uses the model while it waits, or raise `--timeout`
`install_and_verify.py` exit 4 `cannot download .../bundletool-all-...jar`                         | Offline or GitHub blocked: download a `bundletool-all` jar elsewhere and pass `--bundletool <jar>`

Full tables: `references/troubleshooting.md`.

## Public references

* LiteRT NPU guide: https://developers.google.com/edge/litert/next/npu
* Google Tensor with LiteRT: https://developers.google.com/edge/litert/next/tensor-sdk
* Compilation flags: https://developers.google.com/edge/tensor-sdk/compilation-flags
* Tensor SDK FAQ: https://developers.google.com/edge/tensor-sdk/tensor-faqs
* Release notes / compatibility: https://developers.google.com/edge/tensor-sdk/release-notes
* benchmark_model tool: https://developers.google.com/edge/litert/next/benchmark
* Codelab: https://codelabs.developers.google.com/codelabs/google-tensor-ml-sdk
*   LiteRT AOT Compilation Tutorial (Colab):
    https://github.com/google-ai-edge/litert-samples/blob/main/samples/litert/colab/LiteRT_AOT_Compilation_Tutorial.ipynb
*   google-colab-cli: https://github.com/googlecolab/google-colab-cli
* LiteRT source and releases: https://github.com/google-ai-edge/LiteRT
* Play for On-device AI (AI packs, delivery, device targeting):
  https://developer.android.com/google/play/on-device-ai
* Play Feature Delivery: https://developer.android.com/guide/playcore/feature-delivery
* bundletool: https://github.com/google/bundletool/releases

## Contributions

To contribute or modify this skill or its references, follow the
[contribution guidelines](references/contributing.md) before making changes.
