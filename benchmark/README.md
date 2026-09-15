# benchmark

Run the public [LiteRT](https://github.com/google-ai-edge/litert) `benchmark_model` binary
on phones in the Developer Device Platform (DDP) lab through its Device Run REST API: one
HTTP request per model and device, results back as files in your bucket, no app and no adb.
The CPU and wrapper request bodies and the script are the DDP team's examples; the GPU body
adds one flag. This page covers `.tflite` models; [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)
bundles are not covered here.

## Use

You need a Google Cloud project with the Device Run API enabled and a bucket. Sessions
consume device time billed to that project; the platform is in Preview ([overview](https://docs.cloud.google.com/developer-device-platform/overview), [REST reference](https://docs.cloud.google.com/developer-device-platform/reference/device-run/rest/v1alpha/projects.locations.sessions)).
The binary is LiteRT's prebuilt `benchmark_model` ([benchmark page](https://developers.google.com/edge/litert/next/benchmark)), read from the
public bucket at a pinned release, `gs://litert/binaries/2.2.0/android_arm64/benchmark_model` (the page's own links point at `latest/`). Tools:
`gcloud` (to enable the API and mint the token), `gsutil`, `curl`, `python3`, `protoc`.

1. Put your model in your bucket, and the wrapper script if you use it:

```bash
gcloud services enable devicerun.googleapis.com
BUCKET=gs://your-gcs-bucket
gsutil cp model-file.tflite $BUCKET/model-file.tflite
gsutil cp multi_benchmark_harness.sh $BUCKET/multi_benchmark_harness.sh
```

2. Pick a request body and replace its `gs://your-…` placeholders with your paths.
   `session-cpu.json` runs the model on the CPU with 4 threads; `session-gpu.json` is the
   same request with `--use_gpu=true`; `session-cpu-gpu-wrapper.json` runs
   `multi_benchmark_harness.sh` as the binary: CPU and GPU in one session, a cooldown
   between them, each run's stdout kept. `GET $API/devices` lists the lab phones
   (`pa3q-35` is a Galaxy S25 Ultra, `caiman-35` a Pixel 9 Pro).

3. Submit, wait for the operation (it carries the whole session report, output paths
   included), then fetch the files; `OUT` is the request's `gcsOutputDirectory`:

```bash
PROJECT=your-project-id
BASE=https://devicerun.googleapis.com/v1alpha
API=$BASE/projects/$PROJECT/locations/global
TOKEN=$(gcloud auth print-access-token)
REQUEST=session-cpu.json
curl -s -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d @$REQUEST $API/sessions | tee operation.json
OP=$(python3 -c 'import json; print(json.load(open("operation.json"))["name"])')
until curl -s -H "Authorization: Bearer $TOKEN" $BASE/$OP | tee operation.json | grep -q '"done": true'; do sleep 30; done
python3 -c 'import json; r=json.load(open("operation.json"))["response"]["sessionReport"]; print(r["result"]["resultType"]); [print(f["gcsOutputFile"]["path"]) for j in r["jobReports"] for e in j["executionReports"] for f in e["outputFiles"]]'
OUT=gs://your-output-gcs-bucket/folder
S=$(python3 -c 'import json; print(json.load(open("operation.json"))["metadata"]["target"].split("/")[-1])')
mkdir -p out && gsutil -m -q cp -r $OUT/$S/ out/
```

## What comes back

| File | Content |
|---|---|
| `artifacts/data/local/tmp/results.pb` | `tflite.tools.benchmark.BenchmarkResult`: latency in ms (avg, min, max, std, median, p5, p95, init, first inference, warm-up), memory footprint in kB, model size, run counts and throughput |
| `artifacts/data/local/tmp/runtime_info.pb` | the runtime info written by `--model_runtime_info_output_file` |
| `logcat.txt` | the device log for the run; `benchmark_model` logs its `BENCHMARK RESULTS` block here too |

The binary's stdout is not stored on this path. The wrapper variant keeps it: each run's
stdout goes to `output/<run>.log` beside `results_<run>.pb`, and the request pulls the
whole `output` directory. To read the results files:

```bash
curl -sLO https://raw.githubusercontent.com/google-ai-edge/litert/main/tflite/tools/benchmark/proto/benchmark_result.proto
for f in $(find out -name 'results*.pb'); do echo $f; protoc --decode=tflite.tools.benchmark.BenchmarkResult benchmark_result.proto < $f; done
```

## Verification

`session-cpu.json` and `session-gpu.json` ran on `pa3q-35` on 2026-09-13 with `mobilenet_v2.tflite`
from [litert-community/MobileNet-v2](https://huggingface.co/litert-community/MobileNet-v2):
`session-e1d64576` (CPU, XNNPACK, 70/70 nodes) and `session-233c4731` (GPU, OpenCL, 70/70 nodes),
both PASSED. `session-cpu-gpu-wrapper.json` ran as shipped on `caiman-35` on 2026-09-15:
`session-bd10daf6`, PASSED, the `output` directory pulled with both runs' logs, `results_<run>.pb`
and `runtime_info_<run>.pb`, no unconsumed-flag warning in either log. Each operation was done
60–82 s after its POST. The wrapper body pushes `gs://litert/binaries/latest/…`, a different object
from `2.2.0`; pin the version for rows that must be reproducible.
