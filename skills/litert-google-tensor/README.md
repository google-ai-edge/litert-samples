# litert-google-tensor: user guide

An agent skill that compiles LiteRT (`.tflite`) models for the TPU in Pixel 10
and Pixel 11 phones (Google Tensor G5 / G6), benchmarks them on a phone
connected to your computer, and packages them for your Android app. You ask in
plain words; the agent runs the skill's scripts and reports the results.

This page is for people using the skill. The agent's instructions are in
[SKILL.md](SKILL.md); maintainers start at
[references/contributing.md](references/contributing.md).

## What it offers

-   **Check** whether a `.tflite` is already compiled for Google Tensor, and
    for which chip.
-   **Compile** a plain `.tflite` ahead of time for `Tensor_G5` (Pixel 10
    family) or `Tensor_G6` (Pixel 11 family), with the public LiteRT Python
    API - either locally on a Linux `x86_64` machine or remotely on a Google
    Colab VM (`--colab` via `google-colab-cli`) when you are on macOS/Windows or
    do not have a Linux machine.
-   **Benchmark** a compiled model on the phone's TPU: latency (average,
    min/max, p5/p50/p95), peak memory, TPU tile time (pure TPU compute time) and
    TPU energy per inference, optionally next to a CPU run of the original
    model, with the speed-up.
-   **Report** a ready-to-paste Markdown table, plus `results.json` and the raw
    logs. Every number comes from the run itself, never typed by hand.
-   **Package** compiled models for your Android (Gradle) app: a Google Play AI
    pack built with LiteRT's own exporter from the exact files you
    benchmarked, the Google Tensor NPU runtime as a feature module (from Google
    Maven), the original model as the CPU fallback, a Kotlin loader, and an
    `INTEGRATION.md` with every remaining step for your project.
-   **Update** an app that already ships Google Tensor builds: it reads your
    project (without changing it), prepares only the files to replace under
    your app's own paths and names, and tells you whether anything else must
    change (a new chip, a changed model input or output, the runtime).
-   **Apply** the package or the update to your project, if you want: the
    agent checks your project against it (without changing anything), shows
    you one plan, and edits files only after you approve. Risky changes, such
    as upgrading the Android Gradle plugin or raising `minSdk`, each get their
    own question.
-   **Install and check** your app on the Pixel the way Google Play would
    install it, and report whether the model ran on the TPU or on the CPU
    (and why).

It is careful with your files: it never overwrites an existing compiled model
without asking, changes your app's project only after you approve the plan,
never touches signing settings, never touches files already on the phone and
removes what it pushed there, asks before uninstalling your app from the
phone, leaves no background processes, and installs nothing on your computer
without telling you (and then only into a Python virtualenv).

## What you can ask (CUJs)

You ask                                                                                | The skill
-------------------------------------------------------------------------------------- | ---------
"Is `model.tflite` compiled for the Pixel TPU?"                                        | Says compiled or not, for which chip, and with which compile flags when known.
"Compile `model.tflite` for Pixel 10" (or Pixel 11)                                    | Writes `compiled/model_Google_Tensor_G5.tflite` (or `..._G6`) plus a `.compile.json` record of flags, ops placed on the TPU, and versions.
"Compile `model.tflite` for Pixel 10 using Colab" / "I'm on a Mac, compile this model" | Uses `google-colab-cli` (`--colab`) to spin up a remote Google Colab Linux `x86_64` VM, install the SDK with your Beta tarball, compile the model, download the `.tflite` and `.compile.json` sidecar locally, and stop the Colab VM.
"Compile it with these flags"                                                          | Passes the Tensor compiler options you ask for: precision, sharding, int64 to int32, models over 2 GB, 4-bit weights.
"Benchmark this compiled model on my phone"                                            | Runs it TPU-only on the attached Pixel and reports latency, memory, TPU tile time and TPU energy.
"How long does the TPU itself take?"                                                   | Same benchmark; the TPU tile time row answers it.
"How much energy does the TPU use per inference?"                                      | Same benchmark; the TPU energy row answers it (from the phone's power monitor).
"How fast is `model.tflite` on the TPU vs the CPU?"                                    | Compiles it if needed, runs both, and reports the speed-up.
"The compile crashed" / "some ops stayed on the CPU"                                   | Retries with the documented fallback flags, or offers a run that allows CPU fallback and labels the result as such.
"This computer has no internet"                                                        | Benchmarks with local copies of the two runtime files instead of downloading them.
"I don't have Beta access yet"                                                         | Checks your files, computer and phone now, and gives you the exact install commands for when access arrives.
"Package my compiled model for my app" / "Deploy it to my Android app"                 | Writes `app_package/<name>/` with the AI pack, the NPU runtime module, the CPU fallback model, `TensorModelLoader.kt` and an `INTEGRATION.md` filled in with your names and versions. It asks when Google Play should deliver the model (default: with the app), then offers to apply the plan to your project.
"My app already runs a Tensor build; swap in the new one"                              | Reads your project and writes `app_update/<name>/` with only the files to replace (same paths and names as in your app) and an `UPDATE.md`: a drop-in swap, or the list of other changes needed. Then offers to apply it: it backs up each file it replaces and checks the result.
"Yes, apply it to my app" / "Add it to my app at `/path/to/MyApp`"                     | Checks your project against the package without changing it, shows one plan (the files to copy, the edits with their code) and applies it after you approve. Upgrading the Android Gradle plugin or Kotlin, adding Kotlin, raising `minSdk`, limiting the app to 64-bit ARM (arm64-v8a), changing your LiteRT version, replacing your own files or removing a chip each get their own question. Builds the app once at the end.
"Install my app on my Pixel and check it uses the TPU"                                 | Builds your app bundle, installs it the way Google Play would for your phone (bundletool local testing), starts the app and reports `"path":"npu"` (the TPU) or `"path":"cpu"` with the reason. Asks before uninstalling a copy that is already installed (that deletes the app's data on the phone).
"I just want to try it on my Pixel quickly"                                            | Adds a quick test to the package (`--quick-test`): with one Gradle block, `./gradlew installDebug -PtensorQuickTest=Tensor_G5` runs the model on the TPU without bundletool. Debug builds only; your release build is unaffected.

Not covered: Qualcomm, MediaTek or Samsung NPUs; GPU benchmarks; LLM /
LiteRT-LM (`.litertlm`) models; on-device (JIT) compilation; non-Gradle
builds; publishing to Google Play; Pixel 9 and older phones.

## Prerequisites

| You need | To compile (local) | To compile (`--colab`) | To benchmark | To package for an app | To install and check your app |
|---|---|---|---|---|---|
| **Google Tensor SDK Beta access** (setup step 1) | yes | yes (`--sdk-tarball` or `--sdk-url`) | no | no | no |
| Computer OS | Linux `x86_64` with at least 16 GB RAM | any OS (runs on a remote Google Colab Linux `x86_64` VM) | any OS with `adb` | any OS where `pip install ai-edge-litert` works (the compile virtualenv works) | any OS that builds your app |
| Python 3 / CLI | supported Python in a virtualenv with two LiteRT packages (setup step 2) | any local Python 3 + `google-colab-cli` (`colab` on PATH) | any Python 3, no extra packages | a Python with `ai-edge-litert` (no Beta SDK); applying the plan needs only Python 3 | any Python 3, a JDK your project's Gradle supports (`JAVA_HOME`), the Android SDK and your project's `./gradlew` |
| `adb` on your PATH, and a Pixel 10 or 11 with USB debugging on | no | no | yes | no | yes |
| Internet | once for `pip install` | yes (Colab VM + SDK install) | on first use (two runtime files are downloaded, then cached), unless you supply the files | no (your app's Gradle build downloads the LiteRT runtime from Google Maven) | your Gradle build's usual downloads, and bundletool once from GitHub (then cached) unless you supply a bundletool jar |

Compile and benchmark can run on different machines, for example compile on a
Linux workstation or cloud VM, or on a remote Google Colab VM (`--colab`), and
benchmark where the phone is plugged in (or over a forwarded adb connection).

## One-time setup (manual steps)

1.  **Request Beta access** (compile only). Sign up at
    https://services.google.com/fb/forms/tensor_ml_sdk_experimental_access
    ([overview](https://developers.google.com/edge/litert/next/tensor-sdk)).
    Google e-mails you a link to the compiler tarball
    `litert_plugin_compiler.tar.gz`; download it. Approval is on Google's side,
    and nothing can be compiled until you have the tarball. Benchmarking an
    already-compiled model does not need it.
2.  **Set up the compiler** (compile only - pick **2A** for local Linux `x86_64`
    or **2B** for remote Google Colab from any OS):

    -   **2A. Local Linux `x86_64`:** ask the agent to set it up, or print the
        commands yourself:

        ```bash
        python3 <skill dir>/scripts/compile_model.py --print-install
        ```

        Run the printed commands with your real tarball path. They create a
        virtualenv (`~/.venvs/litert-tensor`) and install `ai-edge-litert` and
        `ai-edge-litert-sdk-google-tensor` at the version the skill pins; the
        output also says whether your Python version is supported.
        `GOOGLE_TENSOR_SDK_BETA` must point at the tarball **during** `pip
        install`, because that is when the compiler is unpacked. If your system
        is missing packages (for example `ensurepip is not available` means
        `sudo apt install python3-venv`), see
        [troubleshooting](references/troubleshooting.md).
    -   **2B. Remote Google Colab (`--colab`, any OS):** if you are on macOS or
        Windows, or do not want to set up a local Linux environment, print the
        Colab CLI setup commands:

        ```bash
        python3 <skill dir>/scripts/compile_model.py --print-install --colab
        ```

        This installs
        [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli)
        (`colab`) and runs `colab sessions` once to sign in with your Google
        account in the browser. When you compile with `--colab --sdk-tarball
        /abs/path/litert_plugin_compiler.tar.gz`, the script uploads your model
        and tarball to a Colab VM, installs the pinned LiteRT SDK there,
        downloads the compiled `.tflite` and `.compile.json` sidecar, and stops
        the Colab VM. (If you prefer clicking through a notebook in the browser
        instead of the CLI, see the
        [LiteRT AOT Compilation Tutorial Colab](https://github.com/google-ai-edge/litert-samples/blob/main/samples/litert/colab/LiteRT_AOT_Compilation_Tutorial.ipynb).)
3.  **Install `adb`** (benchmark and app check only) from
    [Android SDK platform-tools](https://developer.android.com/tools/releases/platform-tools)
    and put it on your PATH.
4.  **Prepare the phone** (benchmark and app check only). Turn on Developer
    options > USB debugging, plug the phone in, and accept the "Allow USB
    debugging" prompt. `adb devices` must list the phone as `device`.
5.  **Install the skill** in your agent (next section).

## Install the skill

The skill is a folder in the `SKILL.md` format that coding agents load as a
skill: `SKILL.md` holds the instructions and the `name` / `description` the
agent matches your requests against, and `scripts/`, `templates/`,
`references/` and `versions.json` are what the agent runs and reads.

1.  Copy the **whole folder**, not just `SKILL.md`, into the skills directory
    of your agent, keeping the folder name `litert-google-tensor`. Your
    agent's documentation says where that directory is and how skills are
    discovered.
2.  Start a new agent session and check that the agent recognizes the skill:
    many agents list their loaded skills, or offer it as a
    `/litert-google-tensor` command. If it is missing, check the folder
    location and restart the agent.

## Use it

Ask in plain words: mentioning a `.tflite` and the Pixel TPU or NPU is enough
for the agent to pick the skill. If your agent can invoke skills by name, you
can also call it explicitly, for example
`/litert-google-tensor Compile /path/to/model.tflite for Pixel 10.`
Have these ready:

-   the absolute path of the `.tflite`;
-   for a compile: the target phone (Pixel 10 or Pixel 11);
-   for a CPU comparison: the original, uncompiled model;
-   if several phones are connected: the serial from `adb devices`.

Example prompts:

-   "Is /path/to/model.tflite compiled for Google Tensor?"
-   "Compile /path/to/model.tflite for Pixel 10."
-   "Compile /path/to/model.tflite for Pixel 10 on Colab using
    /path/to/litert_plugin_compiler.tar.gz."
-   "Benchmark /path/to/compiled/model_Google_Tensor_G5.tflite on my phone,
    with a CPU baseline from /path/to/model.tflite."
-   "How fast is /path/to/model.tflite on the Pixel 10 TPU compared with the
    CPU?"
-   "Package /path/to/compiled/model_Google_Tensor_G5.tflite for my Android app
    (original: /path/to/model.tflite)."
-   "Update the Tensor model in my app at /path/to/MyApp with
    /path/to/compiled/model_Google_Tensor_G5.tflite (original:
    /path/to/model.tflite)."
-   "Apply the integration plan to my app at /path/to/MyApp."
-   "Install my app at /path/to/MyApp on my Pixel and check that the model
    runs on the TPU."

Where the results go:

-   Compile: `<model dir>/compiled/<model>_Google_<target>.tflite`, with a
    `.compile.json` file next to it.
-   Benchmark: `./litert_benchmark_results/<timestamp>/` with `report.md`,
    `results.json`, logs and per-op profiles.
-   Package: `<model dir>/app_package/<name>/` with `INTEGRATION.md` (your next
    steps), the Gradle modules, the models and `pack.json` (where every file
    came from).
-   Update: `<model dir>/app_update/<name>/` with `UPDATE.md` (your next
    steps), `files/` (the new files, at your project's paths) and
    `update.json`. When the agent applies it, a copy of each project file it
    replaces goes to `<model dir>/app_update/<name>-backup/` first.
-   App check on a Pixel: `./litert_app_verify/<timestamp>/` with `report.md`,
    `results.json` and the build and install logs.

The scripts also work without an agent: each one has `--help` and `--dry-run`.
Only the real compile and packaging need a Python with LiteRT installed.

## Good to know

-   **Compiled models are chip-specific.** A `Tensor_G5` build does not run on
    a Pixel 11 (`Tensor_G6`), and the other way round. The skill checks the
    phone and stops before running a mismatched model.
-   **Models compiled by other tools are recognized.** Files from the Edge TPU
    compiler or older Google Tensor tooling (one `edgetpu-custom-op` op
    inside) can neither be recompiled nor run on the CPU; the skill says so
    and asks for the original model.
-   **The CPU baseline needs the original model.** A compiled model cannot run
    on the CPU.
-   **Benchmarks are TPU-only by default.** If any part of the model would fall
    back to the CPU, the run fails instead of reporting mixed numbers. You can
    allow fallback; the report then says so.
-   **Tiny models can be slower on the TPU** than on the CPU, because each
    inference has a fixed TPU dispatch cost of a few ms. That is a real
    result, not a setup error.
-   **TPU tile time.** The TPU's own compute time per inference, reported by the
    Tensor runtime in a second run of the model; it is lower than the latency,
    which also counts work on the CPU side. Older Tensor runtime builds count
    each inference twice, so the skill always divides by the real number of runs
    instead of using the runtime's own average. Skip it with `--no-tile-time`.
-   **TPU energy.** The energy the TPU used per inference, from the phone's own
    power monitor, measured in the same second run. It covers the whole run,
    including the short pauses between inferences, so use it to compare models,
    builds and performance modes on the same phone rather than as a lab
    measurement. Older Tensor runtime builds measure it wrongly, so the skill
    leaves it out for them and says so. Skip it with `--no-power`.
-   **Numbers vary between runs.** Keep the screen on and let a hot phone cool
    down; the report shows the phone's throttling status. The numbers come
    from a command-line tool, so timings inside an app can differ: use them to
    compare models and builds.
-   **The crash-fallback flags lower precision** (`--truncation-type half`
    uses 16-bit floats). Check the model's outputs before relying on such a
    build.
-   **Versions line in the report.** It compares the LiteRT version that
    compiled the model with the release of the runtime files on the phone.
    `DIFFERENT releases (the pair verified in versions.json)` is expected and
    fine.
-   **Downloads** (`benchmark_model` and the Tensor dispatch library, from the
    public LiteRT binaries bucket) are cached in
    `~/.cache/litert-google-tensor/`.
-   **Interrupted runs.** Ctrl-C still cleans up the phone. Only a hard kill
    can leave a `/data/local/tmp/litert_benchmark/run-*` folder behind; ask the
    agent to remove it.
-   **Packaging ships what you benchmarked.** The AI pack holds byte-identical
    copies of your compiled files (checked against their `.compile.json`),
    and nothing is recompiled. Each Pixel gets only the build for its chip;
    other phones run the original model on the CPU. Publishing the app on
    Google Play stays with you.
-   **Updating a model needs an app release.** Google Play delivers AI packs
    with the app, so a new model reaches phones with your next app version
    (higher `versionCode`). If your app's NPU runtime is not the release the
    skill pins, benchmark the new build with your app's own runtime library
    first: `UPDATE.md` gives the command.
-   **Checking your app on a Pixel needs a fresh install.** bundletool's local
    testing cannot update an installed app, so the agent asks before
    uninstalling it (that deletes the app's data on the phone). Your app must
    load the model with `TensorModelLoader.load()`; if it does so on a
    particular screen, open that screen while the agent waits.

When something fails, the script's error message names the fix, and
[references/troubleshooting.md](references/troubleshooting.md) lists symptoms
and fixes.

## Updates

**The skill.** Replace the folder with a newer release when you need a fix
or a new feature (for example support for a new Pixel), then start a new agent
session. To see what changed between two copies:

```bash
diff -r <old copy> <new copy>
```

**Your setup.** The skill pins one tested LiteRT version in
[versions.json](versions.json), so you do not need to chase new LiteRT
releases. When the pin changes:

-   Compile: your current virtualenv keeps working. Each compile prints a
    `NOTE` when your installed version differs from the pin, and records the
    version it used. To move to the new pin, run `--print-install` again and
    install into a fresh virtualenv. If a compile prints a `WARNING` that the
    two LiteRT packages differ, reinstall both at one version (a mixed pair is
    not supported). If the install rejects your tarball, you need the tarball
    that matches the new SDK version from the Beta program.
-   Benchmark: nothing to do. The new runtime files are downloaded on the next
    run, because the cache is kept per release.

Need a different LiteRT version for one job? Ask the agent: it uses that
version for the job and says so in the report. Changing the pin itself is a
maintainer task ([contributing.md](references/contributing.md)).

## Links

-   [Google Tensor with LiteRT](https://developers.google.com/edge/litert/next/tensor-sdk)
-   [Tensor SDK FAQ](https://developers.google.com/edge/tensor-sdk/tensor-faqs)
-   [Release notes and compatibility](https://developers.google.com/edge/tensor-sdk/release-notes)
-   [benchmark_model tool](https://developers.google.com/edge/litert/next/benchmark)
-   [Codelab](https://codelabs.developers.google.com/codelabs/google-tensor-ml-sdk)
-   [LiteRT AOT Compilation Tutorial (Colab)](https://github.com/google-ai-edge/litert-samples/blob/main/samples/litert/colab/LiteRT_AOT_Compilation_Tutorial.ipynb)
-   [google-colab-cli](https://github.com/googlecolab/google-colab-cli)
-   [Play for On-device AI (AI packs)](https://developer.android.com/google/play/on-device-ai)
-   [LiteRT NPU guide](https://developers.google.com/edge/litert/next/npu)

Questions or bugs in the skill: report them where you got it. Issues with
LiteRT or the Google Tensor SDK themselves:
https://github.com/google-ai-edge/LiteRT/issues.
