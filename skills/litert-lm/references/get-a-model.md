# Get a model

## litert-community first

https://huggingface.co/litert-community holds one repo per model (`litert-community/Gemma3-1B-IT`, `litert-community/Qwen3-0.6B`, ...). Choose a `.litertlm` file; its name carries the variant: `q8`, `q4` or `wi4b32` (int4 weights, block 32) for the quantization, `ekv1280` or `ekv4096` for the KV cache size (the longest prompt plus reply, in tokens), and a SoC suffix such as `.mediatek.mt6993` for an NPU build. The model card says which backend the file was tested on and its size.

Download to `context.filesDir` from `https://huggingface.co/<repo>/resolve/main/<file>` with progress, and check the size before loading. The file stays out of `assets/` and the APK.

## Convert only when the model is missing

On a workstation, not in the app:

1. Export with `litert-torch` (`pip install litert-torch`; the generative exporter; `ai-edge-torch` is the deprecated name of the same package).
2. Quantize with `ai-edge-quantizer`: int8 dynamic is the safe default; int4 must be blockwise (block 32, or 128 for larger models); channelwise int4 degrades decoders.
3. Bundle tokenizer, chat template and metadata into `.litertlm` with `litert_lm_builder` (`pip install litert-lm`).
4. Gate before shipping: an 8-prompt sanity check on CPU and on the target backend, a task-level parity check against the source model (the task the app needs), then a multi-turn conversation.

The procedure with its traps (chat templates, tokenizers, architectures that do not convert) is `skills/litert-conversion-workflow` and `models/conversion.md` in litert-samples.
