# MiniCPM5-2B

[openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B), OpenBMB's dense 2.5B model with hybrid thinking, as `.litertlm` bundles for the [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) engine, the LLM runtime of [LiteRT](https://github.com/google-ai-edge/litert). The bundles are published at [litert-community/MiniCPM5-2B](https://huggingface.co/litert-community/MiniCPM5-2B). They need litert-lm 0.16 or newer, which adds the thought channel and `ThinkingConfig`; every command in the code blocks on this page was run on 0.17.0.

## Run

```bash
pip install litert-lm
hf download litert-community/MiniCPM5-2B MiniCPM5-2B_int4.litertlm --local-dir .
litert-lm run MiniCPM5-2B_int4.litertlm --backend gpu --prompt "What is the capital of Japan?"
```

Drop `--prompt` for an interactive chat. `--backend cpu` runs the same file on the CPU.

## Which file

| File | Size | Use it for |
|---|---|---|
| `MiniCPM5-2B_int4.litertlm` | 1.55 GB | Phones and laptops: the fastest GPU decode on both devices measured below; direct answers and short reasoning |
| `MiniCPM5-2B_int8.litertlm` | 2.60 GB | Reasoning that has to run to its end: its thinking is three to four times shorter than int4's on the same questions. On iOS use int4: int8's main section (2.33 GB) is larger than a default-entitlement app maps in one piece |

Both files run on the CPU and GPU backends, carry the checkpoint's own chat template and declare the thought channel, so the thinking switch below works the same through every API. The two `minicpm_*` files in the same repository run on the CPU backend.

## Thinking

The model reasons before every answer unless told otherwise. On the command line:

```bash
litert-lm run MiniCPM5-2B_int4.litertlm --backend gpu --thinking false --prompt "What is the capital of Japan?"
litert-lm run MiniCPM5-2B_int4.litertlm --backend gpu --thinking true --thinking-budget 1024 --prompt "What is 17 + 25?"
```

`--thinking false` gives direct answers. `--thinking-budget N` caps the reasoning at N tokens; the answer still follows when the cap is reached. int4 reasons at length, so on int4 set a budget or turn thinking off, and use int8 for problems whose reasoning must finish. `--max-num-tokens` is the budget for prompt plus reply together, up to the 4096-token cache the bundles were built with.

## Python

```python
import litert_lm

with litert_lm.Engine("MiniCPM5-2B_int4.litertlm", backend=litert_lm.Backend.GPU()) as engine:
    thinking = litert_lm.ThinkingConfig(enable_thinking=True, thinking_token_budget=1024)
    with engine.create_conversation(thinking_config=thinking, max_output_tokens=2048) as conversation:
        response = conversation.send_message("What is 17 + 25?")
        print(response.get("channels", {}).get("thought"))
        print(response["content"][0]["text"])
```

`pip install litert-lm` installs the [Python API](https://ai.google.dev/edge/litert-lm/python) as well. `ThinkingConfig(enable_thinking=False)` turns thinking off, per conversation or per message.

## Android and iOS

- [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) (Android, iOS, macOS): Model manager, **+**, then import from Hugging Face with the file's link, or from a local file (`adb push MiniCPM5-2B_int4.litertlm /sdcard/Download/` on Android).
- In an app: the [Kotlin API](https://ai.google.dev/edge/litert-lm/android) on Android and the JVM, the [Swift API](https://ai.google.dev/edge/litert-lm/swift) on iOS and macOS. Both take the GPU backend and `ThinkingConfig` the same way as the Python example. Sample apps that use the engine are listed in [`models/README.md`](../../README.md#where-to-find-examples).
- Terminal on Android: `litert_lm_main` built for `android_arm64` from the [LiteRT-LM build guide](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/getting-started/build-and-run.md), pushed with the GPU `.so` files, `--backend=gpu --model_path=... --input_prompt=...`.

## Serve

```bash
litert-lm import MiniCPM5-2B_int4.litertlm minicpm5-2b-int4
litert-lm serve --port 9379
curl http://localhost:9379/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "minicpm5-2b-int4", "messages": [{"role": "user", "content": "What is the capital of Japan?"}]}'
```

The server speaks the OpenAI chat-completions API ([CLI guide](https://ai.google.dev/edge/litert-lm/cli/openai_server)); the reply counts reasoning tokens separately from the answer.

## Tested on

```bash
litert-lm benchmark MiniCPM5-2B_int4.litertlm --backend gpu -p 256 -d 256 --runs 3 --cache no
```

Decode and prefill speed in tokens per second, 256-token prompt and reply, three runs, no compile cache.

| Device | Runtime | int4 (decode / prefill) | int8 (decode / prefill) |
|---|---|---|---|
| Mac M4 Max, GPU | litert-lm 0.17.0 | 93 / 1800 | 68 / 1300 |
| Mac M4 Max, CPU | litert-lm 0.17.0 | 28 / 140 | 26 / 140 |
| Galaxy S26 (SM-S942Q), GPU (OpenCL) | `litert_lm_advanced_main` built from LiteRT-LM v0.16.0, 203-token prompt, two runs | 16–17 / 270–370 | 10–12 / 120 |
| iPhone 17 Pro, GPU (Metal) and CPU | LiteRT-LM on the device, test harness | int4 loads and answers on both backends; speed not measured | |

Every row was checked for a correct text answer before its speed was recorded. The Galaxy S26 runs delegate every node of the decoder's prefill and decode graphs to OpenCL.

## Conversion

How the two files were built, the two post-export steps that make one file run on both backends, and the full verification record: [`converted/`](converted/), and the cookbook's [worked example](../../conversion.md#11-worked-example-minicpm5-2b).

## References

- [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B), the source checkpoint; [litert-community/MiniCPM5-2B](https://huggingface.co/litert-community/MiniCPM5-2B), the bundles, with a machine-readable manifest.
- [MiniCPM 5 in LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM/tree/main/models/minicpm5): the canonical chat template and metadata for the family.
- LiteRT-LM guides: [CLI](https://ai.google.dev/edge/litert-lm/cli), [Python](https://ai.google.dev/edge/litert-lm/python), [Kotlin](https://ai.google.dev/edge/litert-lm/android), [Swift](https://ai.google.dev/edge/litert-lm/swift).
