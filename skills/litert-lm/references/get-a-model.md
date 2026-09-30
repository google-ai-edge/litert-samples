# Get a model

## litert-community first

https://huggingface.co/litert-community holds one repo per model, such as `litert-community/gemma-4-E2B-it-litert-lm` and `litert-community/Qwen3-0.6B`; the files to download from these two are `gemma-4-E2B-it.litertlm` and `Qwen3-0.6B.litertlm`, which run on the CPU and the GPU. The other files carry their variant in the name: `wi4b32` (int4 weights, block 32), `q8` or `q4` for the quantization, `ekv1280` or `ekv4096` for the context length (prompt plus reply, in tokens), and a SoC suffix such as `_Google_Tensor_G5` or `.mediatek.mt6993` for an NPU build (not covered here). The model card says which backend each file was tested on and its size.

The download URL is `https://huggingface.co/<repo>/resolve/main/<file>`. The file stays out of `assets/` and the APK.

## Converting a model yourself

Only when the model is missing, and on a workstation, not in the app: export with `litert-torch`, quantize with `ai-edge-quantizer` (int8 dynamic as the default; int4 blockwise), and bundle tokenizer and metadata into `.litertlm` with `litert_lm_builder` from the `litert-lm` package. The full procedure with its traps is the `litert-conversion-workflow` skill in litert-samples: https://github.com/google-ai-edge/litert-samples/tree/main/skills/litert-conversion-workflow
