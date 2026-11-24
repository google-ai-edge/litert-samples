# LFM3-1.7B for the Google Tensor G6 NPU

Scripts that build [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) (revision `1da06bec`) into one `.litertlm` bundle for the NPU of Google Tensor G6 (Pixel 11), run by [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)'s NPU executor: int8 weights, prefill calls of 128 tokens, a 2,048-token cache with ring caches of 256 in the 5 sliding-window layers, compiled ahead of time for Tensor G6. The CPU and GPU files come from [`../converted/`](../converted/), whose `build_llm_metadata` this recipe reuses.

The recipe is scripts only: no compiled model and no file of the Tensor SDK. You compile with your own Tensor SDK access and check your bundle against the CPU ([below](#check-a-bundle-on-a-pixel-11)).

## Before you start

- **Tensor SDK.** The compiler for Tensor G6 is in Google's Tensor SDK, a Beta that each developer signs up for on Google's [Tensor SDK page](https://developers.google.com/edge/litert/next/tensor-sdk). Install it as its documentation says (Linux x86_64). The compile stage runs with the Python of that environment and needs nothing else from this recipe.
- **Gemma terms.** The compile stage hands the compiler the option file Google published with its Gemma 3 1B bundle for Tensor G6, `Gemma3-1B-IT_Google_Tensor_G6.pb` from [litert-community/Gemma3-1B-IT](https://huggingface.co/litert-community/Gemma3-1B-IT) at revision `a9bd8564`. That repository is gated: accept the Gemma terms there and `hf auth login`. The download stage fetches the file at build time; it is not part of this recipe, and the bundle does not contain it.
- **litert-torch with two changes that are not merged yet.** [#1505](https://github.com/google-ai-edge/litert-torch/pull/1505) exports models with conv cache layers, such as LFM3's 13 short convolutions, with a split cache; [#1510](https://github.com/google-ai-edge/litert-torch/pull/1510) adds `torch.ops.tfl.sum` with `keep_dims`, which the extension's single-token convolution calls. This recipe was checked with litert-torch main `046b0e5` with both applied, which [Environment](#environment) builds. The export stage stops at once, naming what is missing, on a litert-torch without them.
- **A Linux x86_64 host.** The export and the CPU references peak at about 27 GB and 28 GB of memory, and the working directory holds about 22 GB after the last stage.

## Environment

```bash
git clone https://github.com/google-ai-edge/litert-torch
git -C litert-torch checkout 046b0e5
git -C litert-torch fetch origin pull/1505/head pull/1510/head
git -C litert-torch cherry-pick -n fa1b0b9 e6a651c
uv venv --python 3.12 venv && . venv/bin/activate
uv pip install ./litert-torch \
    torch==2.14.0 transformers==5.22.2 ai-edge-litert-nightly==2.4.0.dev20261121 \
    litert-converter==0.6.0.dev20261119 litert-quantizer-nightly==0.12.0.dev20261123 \
    litert-lm-builder-nightly==0.20.0.dev20261119
```

`./litert-torch` is main `046b0e5` with the commits of #1505 (`fa1b0b9`) and #1510 (`e6a651c`) applied: the tree the hashes below come from. These are the versions the recipe was checked with, on Python 3.12.14; LFM3's remote code needs transformers 5.15.0 or later, and the export ran only with 5.22.2.

## Build

```bash
python lfm3_g6_extension_test.py                                        # 19 tests, CPU, no download
python build_lfm3_1_7b_g6.py download export embedder --work out_g6
<Tensor SDK environment>/bin/python build_lfm3_1_7b_g6.py compile --work out_g6
python build_lfm3_1_7b_g6.py bundle --work out_g6                       # out_g6/LFM3-1.7B_Google_Tensor_G6.litertlm
python verify_lfm3_1_7b_g6.py reference --work out_g6                   # out_g6/reference.json
```

| Stage | What it does | Writes |
|---|---|---|
| `download` | The checkpoint's 10 files that the export reads, and the option file | `out_g6/LFM3-1.7B/`, `out_g6/Gemma3-1B-IT_Google_Tensor_G6.pb` |
| `export` | litert-torch's stock exporter with `lfm3_g6_extension.py`: `--split_cache=True --prefill_lengths=128 --cache_length=2048 --sliding_window_ring_buffer_size=256 --quantization_recipe=weight_only_wi8_afp32 --keep_temporary_files=True --bundle_litert_lm=True`, with `PYTHONHASHSEED=0` | `out_g6/export/` |
| `embedder` | Points the embedder's `EMBEDDING_LOOKUP` at the int8 table, which it dequantizes row by row, and drops the `DEQUANTIZE` of the whole 128,000 × 2,048 table in front of it | `out_g6/export/embedder_direct.tflite` |
| `compile` | `ai_edge_litert.aot.aot_compile` with `Target(SocModel.TENSOR_G6)`, `keep_going=False`, `google_tensor_truncation_type="half"` and `google_tensor_extra_options_path` set to the option file. Stops unless every operator of both signatures goes to the TPU | `out_g6/compiled/` |
| `bundle` | litert-lm-builder: the LlmMetadata, ExecutorMetadata, tokenizer, compiled model, embedder and auxiliary model, weights inside the graphs; reads the file back and compares every section with its source | `out_g6/LFM3-1.7B_Google_Tensor_G6.litertlm` |

`--model DIR` and `--options FILE` use local copies instead of downloads. The checkpoint's folder must be named `LFM3-1.7B`: transformers names the remote code's module after it, and every tensor name carries that module's name. For the same reason the export stage imports the extension by its module name, with this folder on `PYTHONPATH`: from a file path, litert-torch names the module after a hash of the path.

In this environment the export and embedder stages give these bytes, wherever the recipe and the checkpoint are:

| File | Bytes | sha256 |
|---|---:|---|
| `model_quantized.tflite` | 1,751,238,336 | `15570661fe8a4101db7ca74a09ac96aa83aea855d0b75065ff1a30477eba91ee` |
| `auxiliary.tflite` | 115,440 | `a63ace2d6dd4a54113f6e7a999799cf2cd9455b41ad6edcc56eae00e4e2f7ed8` |
| `executor_metadata.pb` | 2,346 | `d2bf447c4a5c0e02fd6b454b2725d107059a9c678f8cc42e0b09b580a15d4637` |
| `llm_metadata.pb` | 12,029 | `6c139465267b47cf941bb774ca51c8a8a8d102f37436cf36028b452a4ca9310a` |
| `tokenizer.json` | 17,908,634 | `41927bd314d17d7d7fadb44e10cd0919f7b71528d7741596d8cbd15c6945ec44` |
| `embedder_direct.tflite` | 265,220,136 | `b4dd9608270646fffe3e819b87e5afb15b16d1f89a76f889ff6950b5d1b93157` |

Check a bundle by its replies, not by its hash.

## Check a bundle on a Pixel 11

These steps have not been run yet, neither with a bundle from this recipe nor with stock LiteRT-LM.

You need LiteRT-LM's `litert_lm_advanced_main` built for Android arm64 (`bazel build --config=android_arm64 //runtime/engine:litert_lm_advanced_main`, as LiteRT-LM's build guide describes) with `libGemmaModelConstraintProvider.so` from LiteRT-LM's `prebuilt/android_arm64/`, and LiteRT 2.4.0's dispatch library for Google Tensor, [`libLiteRtDispatch_GoogleTensor.so`](https://storage.googleapis.com/litert/binaries/2.4.0/android_arm64/libLiteRtDispatch_GoogleTensor.so). The bundle turns on the NPU executor's handwritten sampler, masks and cache update itself, so the command line needs no NPU flag.

```bash
adb shell mkdir -p /data/local/tmp/lfm3
adb push litert_lm_advanced_main libGemmaModelConstraintProvider.so libLiteRtDispatch_GoogleTensor.so \
    out_g6/LFM3-1.7B_Google_Tensor_G6.litertlm prompts /data/local/tmp/lfm3/
mkdir -p replies
for p in arithmetic passage; do
  adb shell "cd /data/local/tmp/lfm3 && LD_LIBRARY_PATH=. ./litert_lm_advanced_main --backend=npu \
      --model_path=LFM3-1.7B_Google_Tensor_G6.litertlm --litert_dispatch_lib_dir=. \
      --input_prompt_file=prompts/$p.txt --max_output_tokens=256" > replies/$p.txt
done
python verify_lfm3_1_7b_g6.py compare --work out_g6 --replies replies
```

`compare` takes the CLI's `[thought]` markers out of each reply and compares it with the CPU's text for the same prompt from `reference.json`: the uncompiled export through ai_edge_litert's interpreter, greedy, fed as the NPU executor feeds the bundle. `arithmetic` ends with the stop token after 55 tokens; `passage` asks about a 7,939-character text, so its 1,779 prompt tokens and 256 reply tokens fill the cache up to position 2,033 and every ring cache wraps around. On a difference it prints both texts there and how far apart the CPU's two best tokens were at that step: under about 0.3, the TPU's 16-bit floats can pick the other token.

## What the extension changes, and why

`lfm3_g6_extension.py` subclasses the CPU/GPU recipe's `Lfm3Extension`, keeps its `build_llm_metadata` (the EOS id 124900 as the only stop, earlier assistant turns without their thought) and refuses any export without `split_cache`. In the table, the on-device compiler is the one LiteRT runs on the phone, which this export was first compiled with. The Tensor SDK's compiler takes the same forms; it was not tried without them. The 16-bit float changes hold with `google_tensor_truncation_type="half"` as well.

| Change | Why |
|---|---|
| `get_export_defaults`: the Jinja template on, and `enable_neon_for_npu_greedy_sampling`, `use_hw_masking_for_npu` and `use_hw_cache_update_for_npu` in the ExecutorMetadata | The NPU preset turns the template off, and LFM3's thought channel needs it. LiteRT-LM's NPU executor uses its handwritten sampler, masks and cache update only when the bundle asks for them |
| Each RMSNorm as plain operators on x / max\|x\| | In 16-bit floats the squares of LFM3's small activations underflow to 0; the on-device compiler also fails to import the `odml.rms_norm` composite |
| `lm_head` with its weight times 8 and the final norm's weight over 8; each MLP's `w1` and `w3` times 4 and its `ffn_norm` over 4 | The TPU sums an int8 fully connected layer in 16-bit floats before its per-row scale, and these layers came closest to 65,504. The logits are the same |
| A floor of -3e4 on the sliding layers' decode mask | LiteRT-LM's -1e9 is -inf in 16-bit floats, which gave NaN over a 256-slot cache |
| The single-token convolution as one product with the 7 stacked taps and a `tfl.sum` that keeps the reduced axis | The on-device compiler fails on litert-torch's tap-by-tap form over six one-row slices of the state; `Tensor.sum` loses the kept axis in conversion |
| The prefill convolution through litert-torch's `short_conv` composite, with the real slots read from the mask's diagonal | Padding slots of a short prefill call stay out of the conv state |
| Prefill attention in blocks of 26 queries (one block of all 128 in the ring-cache layers) | The on-device compiler rejects LFM3's attention against 2,048 + 128 keys at 32 queries and more |
| The per-head gate spread by a 0/1 matmul, and `rotate_half` as a matmul with a signed permutation | The on-device compiler scrambles the gate's `unsqueeze(-1)` and rejects the half-width slices of a 128-wide head |

## Limits

- 2,048 tokens in all, for the prompt, the reply and every earlier turn; prefill runs in calls of 128 tokens.
- Greedy decoding only: the NPU executor ignores the sampler settings.
- LFM3 thinks before it answers. On the NPU, stock LiteRT-LM keeps each turn's thought in the KV cache, so a conversation fills the 2,048 tokens sooner and its later turns differ from a CPU run of the same conversation. `verify_lfm3_1_7b_g6.py` checks single turns only.
- Tensor G6 only: a bundle compiled for one SoC does not run on another.

## Checked

- `lfm3_g6_extension_test.py`: 19 of 19 on Linux x86_64 and on macOS arm64.
- The export and embedder stages, on Linux x86_64 in the environment above, from two copies of the recipe and of the checkpoint in different folders: the six files of the table, the same from both.

The other checks compare with a bundle made from the same changes inside litert-torch, whose replies to both prompts on a Pixel 11 Pro are the CPU's text (with a LiteRT-LM CLI that has changes for multi-turn conversations, which are not upstream):

- Its export has the same operators in the same order, the same constants and the same tensors. Only the tensor names differ: they carry the module of the extension and of the exportables, which were in litert-torch then. The five other files are byte-equal.
- The compile and bundle stages give its six sections byte for byte, the compiled model included; the two files differ only in the header's uuid, timestamp and sections' sha256 entries.
- The reference stage gives the tokens of its CPU references, and `compare` finds its phone replies equal to `reference.json`.

Not run yet: a bundle from this recipe on a phone, and stock LiteRT-LM with any of these bundles.

## Files

| File | What |
|---|---|
| `lfm3_g6_extension.py` | The out-of-tree `ModelExtension`: the split-cache patches above, `get_export_defaults`, and the CPU/GPU recipe's `build_llm_metadata` |
| `lfm3_g6_extension_test.py` | `ModelExtensionTestMixin` contract tests and 14 cases of its own on small stand-ins for the remote code: the convolution against litert-torch's composite, the rescaled weights, the plain RMSNorm, the mask diagonal, the query blocks against one block, the decode mask floor, the export plan and the ExecutorMetadata; CPU, no download |
| `build_lfm3_1_7b_g6.py` | The five stages |
| `verify_lfm3_1_7b_g6.py` | `reference` (CPU references) and `compare` (the phone's replies against them) |
| `prompts/` | The two prompts, without a trailing newline: the CLI's `--input_prompt_file` reads them as they are |
