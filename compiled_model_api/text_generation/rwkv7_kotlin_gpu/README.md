# Text Generation with LiteRT — RWKV-7 World 0.1B (full forward on GPU)

An Android sample that runs an autoregressive language model with its **entire per-token forward pass on the LiteRT `CompiledModel` GPU delegate** — type a prompt and watch tokens stream in. [RWKV-7](https://github.com/BlinkDL/RWKV-LM) is an RNN-style LLM: one token per step with a small fixed-size recurrent state, so the whole model fits a **single static GPU graph** — no KV cache growth, no dynamic shapes, and no CPU fallback for any op.

## Model

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| RWKV-7 step | `x_emb[1,768]`, `att_shift[12,768]`, `ffn_shift[12,768]`, `wkv[144,64,64]` → `logits[1,65536]` + 3 updated states | GPU (1863/1863 nodes, 1 partition) |

fp16 weights, 282 MB, ~18 ms/token. Converted with [litert-torch](https://github.com/google-ai-edge/litert) from the Apache-2.0 [RWKV-x070-World-0.1B-v2.8](https://huggingface.co/BlinkDL/rwkv-7-world) checkpoint — see [`conversion/`](conversion/).

## Pipeline

```
prompt --tokenizer (host, greedy longest-match trie)--> token ids
       --host: fp16 embedding-table row lookup-->  step graph (GPU) -> logits + state'
       --host: argmax -> next token; state' recycled-->  step graph (GPU) -> ...
```

All recurrent state lives host-side and is fed back every step. Host work per token is tiny: one 768-float embedding row lookup (memory-mapped fp16 table; `GATHER` is not GPU-compatible, and the graph applies its own input LayerNorm), an argmax over 65536 logits, and copying the three states back in. Prefill runs the same step loop over the prompt tokens. Decoding is greedy.

`Rwkv7Generator.kt` does the CompiledModel step loop; `RwkvTokenizer.kt` is a Kotlin port of the official RWKV World trie tokenizer (fixture-tested against the Python reference on 50 strings).

## Verification

- Step-mode vs parallel GPT-mode logits (fp32 PyTorch): corr **1.0000000**.
- fp16 tflite vs fp32 PyTorch through the CompiledModel API: prefill corr **1.0000000**, 30-token greedy generation identical (`conversion/validate_rwkv7.py`).
- On device (Pixel 8a): prefill logits corr 0.99995; 30-token greedy generation stays inside the fp32 near-tie envelope (28/30 tokens identical; the 2 flips are fp32 top1–top2 gaps ≤ 0.04).

## Build & run

```bash
cd android
./gradlew :app:installDebug
# the step graph + embedding table are pushed to the app's filesDir (too big to bundle):
./install_to_device.sh <dir-with-the-files>
```

Get `rwkv7_step_fp16.tflite` and `rwkv7_emb_fp16.bin` from Hugging Face ([`litert-community/RWKV-7-World-0.1B-LiteRT`](https://huggingface.co/litert-community/RWKV-7-World-0.1B-LiteRT)) or build them with [`conversion/`](conversion/). The vocabulary is bundled in the app assets. The first launch shows "model files missing" until the install script has run.

## License

Model: Apache-2.0 (RWKV / BlinkDL).
