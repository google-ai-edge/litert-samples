# Get a model

## litert-community first

https://huggingface.co/litert-community holds one repo per model, such as `litert-community/gemma-4-E2B-it-litert-lm` and `litert-community/Qwen3-0.6B`; the files to download from these two are `gemma-4-E2B-it.litertlm` and `Qwen3-0.6B.litertlm`, which run on the CPU and the GPU. The other files in a repo are other builds of the same model, such as other quantizations and context lengths (the Qwen3 card lists each file with its quantization, its context length in tokens and its size), web builds (`-web`), and builds for one chip, named after it, such as `_Google_Tensor_G5` or `.mediatek.mt6993` (NPU builds, not covered here).

The download URL is `https://huggingface.co/<repo>/resolve/main/<file>`. The file stays out of `assets/` and the APK.

## Converting a model yourself

Only when the model is missing, and on a workstation, not in the app: export with `litert-torch`, quantize with `ai-edge-quantizer` (int8 dynamic as the default; int4 blockwise), and bundle tokenizer and metadata into `.litertlm` with `litert_lm_builder` from the `litert-lm` package. The full procedure with its traps is the `litert-conversion-workflow` skill in litert-samples: https://github.com/google-ai-edge/litert-samples/tree/main/skills/litert-conversion-workflow
