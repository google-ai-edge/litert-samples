# FunctionGemma 270M mobile-actions: on-device harness

`ToolRoutingProbeActivity.kt` runs the recipe's tool-routing gate on an Android
phone. The Python harness in [`../converted/`](../converted/) cannot run on a
device — `litert-lm` ships an x86-64 `liblitert-lm.so`, so Termux is out — and the
Python API is not the app-side path anyway. This file is the Kotlin equivalent,
built against the LiteRT-LM Android AAR, and it produced the SM-A145F row on the
[recipe page](../README.md#on-device-sm-a145f).

## What it has to match

A device number is only comparable to the host number if the prompt, the tool
schemas, the sampler and the conversation lifetime are the same. So:

| | Harness here | Python harness |
|---|---|---|
| 4 tool schemas | same `name` / `description` / `parameters` text | same |
| Prompt set | 26 prompts, same literal/paraphrase/terse labels | same |
| Tool calling | `automaticToolCalling = false` | `automatic_tool_calling=False` |
| Sampling | `topK=1, topP=1.0, temperature=0.0, seed=0` | greedy |
| Conversation | fresh per prompt | fresh per prompt |

One difference in the schema shape is forced by the API, and it is worth knowing
if you port anything: the AAR's `tool(OpenApiTool)` adapter reads `name` off the
**top level** of the JSON you hand it, so the tool object must be
`{"name":…,"description":…,"parameters":{…}}`. The Python harness uses the
`{"type":"function","function":{…}}` envelope, because `litert_lm`'s Python side
wants that shape. Passing the envelope here throws `ToolException: Failed to
parse field "name" as String` on **every** prompt — verified by decompiling
`ToolKt$tool$2`, which calls `JsonObject.get("name")` on the parsed root. The
descriptions and parameter schemas are identical between the two forms, so the
accuracy numbers remain comparable; only the wrapper differs.

## Build and run

The AAR coordinate, verified against `dl.google.com`'s maven2 tree:

```kotlin
// app/build.gradle.kts
android { defaultConfig { minSdk = 24 } }   // the AAR declares 24; do not override
dependencies {
    implementation("com.google.ai.edge.litertlm:litertlm-android:0.16.0")
}
```

The group is `com.google.ai.edge.litertlm`, one level below
`com.google.ai.edge` — which is where a 404 search tends to land.

```bash
adb push mobile_actions_q8_ekv1024.litertlm /data/local/tmp/
adb shell run-as <applicationId> cp /data/local/tmp/mobile_actions_q8_ekv1024.litertlm \
    /data/user/0/<applicationId>/files/
adb install -r app-debug.apk
adb logcat -c
adb shell am start -n <applicationId>/.ToolRoutingProbeActivity
adb logcat -d | grep ' FGP '
```

The model must be in the app's own `filesDir` — a 289 MB bundle does not fit the
scoped-storage path from `/data/local/tmp` directly, and `adb shell "run-as …
cp src ."` lands in the app root rather than `files/`, which is why the
destination is spelled out above.

Three rounds run per launch; each round logs `ROUND_SUMMARY`, and each prompt logs
a `ROW` line with kind, expected tool, got tool, hit flag, latency and the first
80 characters of any refusal prose.

## Reading the output

| Line | Meaning |
|---|---|
| `MODEL` / `DEVICE` | Bundle size, and the phone's model and SDK — quote these with every number |
| `ROUND n INIT_MS … rss_after_init_mb=…` | Init cost and peak RSS once weights are resident |
| `ROW round\|kind\|expected\|got\|ok\|ms\|prose\|prompt` | One prompt. `got=null` is the model's own refusal, not a runtime error |
| `ROUND_SUMMARY round=… scored=… hits=… acc=… peak_rss_mb=…` | The cell's verdict |

`peak_rss_mb` reads `VmHWM` from `/proc/self/status`, the kernel's peak-RSS
watermark. A before/after `Runtime.totalMemory()` delta does **not** work here:
the first reading of this probe was `-6`, because the GC reclaimed between the
two samples. `VmHWM` is monotonic, so it survives a collection mid-run.

## Determinism

Three rounds of 18 prompts produced **zero** verdict changes — 54 runs, identical
accuracy and identical per-prompt verdicts. Greedy decoding on this bundle is
deterministic on device, so a single round is sufficient to quote, and a
regression is a real change rather than sampling noise. Latency is the only
figure that moves (2.50–3.54 s per prompt).