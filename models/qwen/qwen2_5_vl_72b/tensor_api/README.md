# Qwen2.5-VL-72B-Instruct Tensor API Implementation

The text decoder of [Qwen/Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) (80 layers),
written straight from the bf16 checkpoint with the
[LiteRT Tensor API](https://github.com/google-ai-edge/LiteRT/tree/main/tensor) (C++, no converter), int4, with the
inputs and outputs that litert-torch's `export_hf` gives a decoder whose embedder is a separate graph. With the vision
tower (litert-torch, one fixed picture size of 448 × 448), the token table and the tokenizer, it makes one `.litertlm`
file that the stock LiteRT-LM 0.19.0 (PyPI wheel) runs on the CPU and on the GPU without any executor change. That file
is published as
[litert-community/Qwen2.5-VL-72B-Instruct-LiteRT](https://huggingface.co/litert-community/Qwen2.5-VL-72B-Instruct-LiteRT).

Built with Qwen.

## This project demonstrates:

1.  A 72B decoder without the fp32 graph. The converter quantizes after it has written the whole fp32 graph, and for
    72B parameters that graph alone is about 290 GB. `tapi_pack.py` quantizes the checkpoint one tensor at a time,
    and the writer maps the packed weights, so neither ever holds that graph.
2.  The stock LiteRT-LM signature layout from an authored graph: `prefill_<N>` and `decode` signatures that take the
    embeddings, `input_pos`, the runtime's mask and a K and a V cache per layer (V stored transposed), and return the
    updated caches (and the logits, for `decode`). RoPE and the cache write position are computed in the graph from
    `input_pos`; grouped-query attention runs over the whole cache, and the runtime's mask hides the unwritten slots.
3.  The converter's bytes from the Tensor API: the int4 blockwise-128 weights and the int8 per-channel output layer
    are made with ai-edge-quantizer's own functions, so they equal what `export_hf` and ai-edge-quantizer write with
    the same recipe.
4.  A vision tower made static for one picture size: windows batched by reshape and transpose (no gather), the patch
    Conv3d as a Conv2d, the 2 × 2 merge as four strided slices, the RoPE tables from Transformers' own rotary module.

## Files

| File | What it does |
|---|---|
| `tapi_pack.py` | Reads the checkpoint one tensor at a time and writes `weights.bin`, `index.tsv` and `config.tsv`: fully connected weights int4 blockwise-128 (symmetric min-max, scales floored at 2^-14), the output layer int8 per channel, norms and biases fp32. |
| `qwen2_writer.cc`, `BUILD` | The Tensor API graph: `prefill_<N>` and `decode` signatures, the KV cache in and out per layer, RoPE from `input_pos`, grouped-query attention, `ModelFactory::Save` with the weights mapped from `weights.bin`. |
| `embedder_export.py` | The token table as its own graph (`token_ids [1,1]` to `embeddings [1,1,8192]`). |
| `qwen25vl_vision.py` | The vision tower for one fixed size; subcommands `selftest`, `ref`, `export` and `check`. |
| `qwen72_bundle.py`, `qwen72_chat_template.jinja` | The `.litertlm` file: sections, stop tokens, prompt templates and the chat template, whose token ids equal the Transformers processor's on 8 conversations. |
| `lm_run.py` | One picture and one question through the LiteRT-LM Python API, with its BenchmarkInfo. |
| `NOTICE`, `QWEN_LICENSE` | The Qwen license notice and agreement for the chat template (see [License](#license)). |

## Prerequisites

1.  **Bazel**: installed via bazelisk (version pinned by `.bazelversion`).
2.  **Python 3.12** with the packages the published file was made with: litert-torch 0.10.0, litert-converter 0.5.0,
    ai-edge-quantizer 0.10.0, ai-edge-litert 2.3.0, litert-lm-builder 0.19.0, litert-lm 0.19.0, transformers 5.20.2,
    torch 2.14.1, safetensors 0.8.0, tokenizers 0.23.3, numpy 2.5.4 and pillow.
3.  **`tfl_rank4.py`** from [hf-to-litertlm/tools/gpu_graph](https://github.com/john-rocky/hf-to-litertlm/tree/main/tools/gpu_graph),
    for step 5.
4.  **Weights**: Qwen/Qwen2.5-VL-72B-Instruct at revision `89c86200` (147 GB of bf16 safetensors), not included in
    this repository:

    ```bash
    hf download Qwen/Qwen2.5-VL-72B-Instruct --revision 89c86200743eec961a297729e7990e8f2ddbc4c5 --local-dir SNAPSHOT
    ```

5.  **A Mac with 128 GB of memory** for step 7 (the GPU run's peak footprint was 46 GiB), and disk for the packed
    weights (38.6 GB), the decoder (40.7 GB) and the bundle (44.8 GB).

## Build Instructions

All commands are run from the repository root (the workspace pulls LiteRT `main` as `@litert_archive`).

```bash
bazel build -c opt //models/qwen/qwen2_5_vl_72b/tensor_api:qwen2_writer
```

The writer only writes a `.tflite` file; it runs no model. On a Mac where this workspace's macOS toolchain does not
resolve (`no such target '@@build_bazel_apple_support//crosstool:toolchain'`), build the same source in a LiteRT
checkout instead:

```bash
git clone https://github.com/google-ai-edge/LiteRT && cd LiteRT
mkdir -p tensor/examples/qwen2_writer
cp <litert-samples>/models/qwen/qwen2_5_vl_72b/tensor_api/qwen2_writer.cc tensor/examples/qwen2_writer/
sed 's|@litert_archive//tensor|//tensor|g' <litert-samples>/models/qwen/qwen2_5_vl_72b/tensor_api/BUILD \
  > tensor/examples/qwen2_writer/BUILD
bazel build -c opt //tensor/examples/qwen2_writer:qwen2_writer
```

## Run

Steps 1 to 6 make the file first published at commit
[`9ac47fea`](https://huggingface.co/litert-community/Qwen2.5-VL-72B-Instruct-LiteRT/tree/9ac47fea16b1e7816ce59e6bc6c7837f682a92e5)
of the Hugging Face repository; step 7 asks it one question. `SNAPSHOT` is the checkpoint directory; `PACK`,
`VISION` and `OUT` are output directories. The commands run in one working directory, and the Python scripts are
the ones in this directory.

1.  Pack the weights (147 GB of bf16 to 38.6 GB; about 1.6 h from a USB disk, peak memory about 11 GB):

    ```bash
    python tapi_pack.py SNAPSHOT PACK --fc int4 --block 128 --lm-head int8
    ```

2.  Build the writer (see [Build Instructions](#build-instructions)).

3.  Write the decoder. The prefill lengths and the KV cache length are flags; these are the published file's:

    ```bash
    qwen2_writer --weights_dir=PACK --out=PACK/model_p32c2048.tflite --prefill=32,128,1024 --cache=2048
    ```

4.  The token table, then int8 per channel on its `EMBEDDING_LOOKUP`:

    ```bash
    python embedder_export.py SNAPSHOT embedder_fp32.tflite --lightweight
    python - <<'EOF'
    from ai_edge_quantizer import qtyping, quantizer, recipe_manager
    from ai_edge_quantizer.algorithm_manager import AlgorithmName

    rm = recipe_manager.RecipeManager()
    rm.add_dynamic_config(regex=".*", operation_name=qtyping.TFLOperationName.EMBEDDING_LOOKUP, num_bits=8,
                          granularity=qtyping.QuantGranularity.CHANNELWISE,
                          algorithm_key=AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    quantizer.Quantizer("embedder_fp32.tflite", rm.get_quantization_recipe()).quantize(
        serialize_to_path="embedder_wi8.tflite")
    EOF
    ```

5.  The vision tower in fp32 (dynamic int8 moved the image embeddings by a relative L2 of about 0.2), then the
    rewrite that lets the GPU delegate run it:

    ```bash
    python qwen25vl_vision.py export --src SNAPSHOT --g 448 --out-dir VISION
    python hf-to-litertlm/tools/gpu_graph/tfl_rank4.py rewrite VISION/vision_encoder_g448_fp32.tflite \
      VISION/vision_encoder_g448_fp32_r4.tflite
    ```

6.  The bundle:

    ```bash
    python qwen72_bundle.py --dec PACK --decode model_p32c2048.tflite --emb-path embedder_wi8.tflite --vis VISION \
      --g 448 --venc vision_encoder_g448_fp32_r4.tflite --tok SNAPSHOT/tokenizer.json --out-dir OUT \
      --name Qwen2.5-VL-72B-Instruct.litertlm --max-tokens 2048
    ```

7.  One question about a picture, stretched to 448 × 448 first (PIL's bicubic resize, as for every published
    result), on the GPU:

    ```bash
    python -c "from PIL import Image, ImageOps; ImageOps.exif_transpose(Image.open('photo.jpg')).convert('RGB').resize((448, 448), Image.BICUBIC).save('picture_448.png')"
    python lm_run.py OUT/Qwen2.5-VL-72B-Instruct.litertlm --backend gpu --image picture_448.png \
      --prompt "How many cats are in this picture?" --greedy
    ```

    The run writes LiteRT-LM's caches next to the `.litertlm` file: 37.3 GB on 2026-11-03, 36.4 GB of it the GPU
    weight cache. `--cache-dir DIR` writes them to `DIR` instead.

The file now on the main branch of the Hugging Face repository has int8 weights in its two vision sections and the
same bytes in its other sections; that last step is in the Qwen2.5-VL-72B-Instruct entry of
[hf-to-litertlm's REPRODUCE.md](https://github.com/john-rocky/hf-to-litertlm/blob/main/REPRODUCE.md).

## Measured highlights

*   A tiny Qwen2 (2 layers, random weights): the writer's fp32 logits equal `export_hf`'s exactly, and its int4
    logits equal those of `export_hf` and ai-edge-quantizer with the same recipe exactly.
*   The real weights cut to 2 layers, fp32: the same top-1 token as Transformers at 160 of 160 positions, largest
    logit difference 4.2e-5.
*   The full int4 file in LiteRT-LM on the Mac GPU: Transformers (the bf16 weights, fp32 arithmetic) picks the same
    token at 98.8 % of the reply positions of four questions (three with a picture, one text only).
*   Apple M4 Max (128 GB), LiteRT-LM 0.19.0 from PyPI, GPU, one 448 × 448 picture and a short question: decode
    8.55 tokens/s, first text 4.74 s, a follow-up question on the same picture 0.89 s, peak footprint 46 GiB
    (2026-10-31; the model card has the method).
*   The writer built at LiteRT `dcb657e8` (the LiteRT pin of LiteRT-LM 0.19.0) and run again on 2026-10-31 on the
    packed weights gave the published decoder section byte for byte: 40,713,491,328 bytes, sha256
    `7100b1e61726f285714797ecec9e84f5f6784f34a21a4e66ae88c9c1d370464e`.
*   On 2026-11-03, on an Apple M4 Max (128 GB), steps 3 to 7 as written above, from the files of this directory and
    the packed weights of the conversion: the writer, built from `qwen2_writer.cc` and `BUILD` at LiteRT `main`
    `5aac30d4` (in a LiteRT checkout, as in Build Instructions), and steps 4 to 6 gave a file whose six sections
    equal those of the published file of commit `9ac47fea` byte for byte, and step 7 answered "There are two cats in
    the picture." for the model card's picture (COCO val2017 image 39769, stretched to 448 × 448).

## License

The code in this directory is under the Apache License 2.0 of this repository. `qwen72_chat_template.jinja` is
changed from the chat template of Qwen/Qwen2.5-VL-72B-Instruct, which is under the Qwen LICENSE AGREEMENT:
`QWEN_LICENSE` is a copy of that agreement, and `NOTICE` holds its attribution notice and says what was changed. No
weights are in this directory; the source checkpoint and the converted file are under the Qwen LICENSE AGREEMENT.
