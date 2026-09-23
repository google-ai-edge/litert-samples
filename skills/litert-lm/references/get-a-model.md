# Get a model

## litert-community first

https://huggingface.co/litert-community holds one repo per model (`litert-community/Qwen3-0.6B`, `litert-community/Gemma3-1B-IT`, ...). Choose a `.litertlm` file; its name carries the variant: `q8`, `q4` or `wi4b32` (int4 weights, block 32) for the quantization, `ekv1280` or `ekv4096` for the KV cache size (the longest prompt plus reply, in tokens), and a SoC suffix such as `.mediatek.mt6993` for an NPU build, which this skill does not use. The model card says which backend the file was tested on and its size. Gemma repos are gated: downloading needs an accepted license and a Hugging Face token; Qwen3 files download without one.

The download URL is `https://huggingface.co/<repo>/resolve/main/<file>`. The file stays out of `assets/` and the APK.

## Converting a model yourself

Only when the model is missing, and on a workstation, not in the app: export with `litert-torch`, quantize with `ai-edge-quantizer` (int8 dynamic as the default; int4 blockwise), and bundle tokenizer and metadata into `.litertlm` with `litert_lm_builder` from the `litert-lm` package. The full procedure with its traps is the `litert-conversion-workflow` skill in litert-samples: https://github.com/google-ai-edge/litert-samples/tree/main/skills/litert-conversion-workflow
