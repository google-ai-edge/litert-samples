# GLiNER2 PII to LiteRT (`.tflite`) Conversion Guide

This directory contains a standalone Python script (`convert.py`) to export the
[`fastino/gliner2-privacy-filter-PII-multi`](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi)
model (a GLiNER2 span extraction model built on `microsoft/mdeberta-v3-base`
supporting 42 PII entity labels across 7 languages) into a `.tflite` model and
`tokenizer.json` for use with the Rust `text_classifier` binary
(`../main.rs`).

> **Note**: Run this script inside an isolated local Python virtual environment.

## 1. Prerequisites

Install [`uv`](https://docs.astral.sh/uv/) (or Python 3.12 `venv` + `pip`).

Create a Python 3.12 virtual environment and install the CPU-only dependencies
(on Linux/Windows, install the pinned CPU-only `torch` wheel first to avoid
pulling multi-GB CUDA wheels; on macOS, skip the first `uv pip install` line):

```bash
cd tflite_conversion
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt --override overrides.txt
```

## 2. Convert the Model and Export the Tokenizer

Run `convert.py` to download the HuggingFace checkpoint, export
`tokenizer.json`, convert the static-shape `EntityExtractionGraph` to float32
`.tflite`, and produce an `int8` weight-quantized `.tflite` model (pass
`--out-dir=/tmp/gliner2_pii` to write generated models outside the working
directory):

```bash
python convert.py --out-dir=/tmp/gliner2_pii
```

### Command-Line Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--model-id` | `fastino/gliner2-privacy-filter-PII-multi` | HuggingFace repository ID or local checkpoint directory. |
| `--out-dir` | `./out` | Output directory for generated artifacts. |
| `--seq-len` | `512` | Static subword sequence length (prompt + text). |
| `--max-words` | `256` | Maximum number of text words per inference call. |
| `--max-labels` | `42` | Maximum number of entity labels per inference call. |
| `--no-quantize` | `False` | Skip generating `gliner2_pii_int8.tflite`. |

> **Important**: If you customize `--seq-len`, `--max-words`, or `--max-labels`,
> update the corresponding `SEQ_LEN`, `MAX_WORDS`, and `MAX_LABELS` constants in
> `../main.rs` to match.

### Generated Artifacts (`out/`)

- `gliner2_pii.tflite` (~1.24 GiB): Float32 LiteRT model.
- `gliner2_pii_int8.tflite` (~358 MiB): Dynamic-range `int8` weight-quantized,
  `float32` activation LiteRT model.
- `tokenizer.json`: HuggingFace `tokenizers` configuration including GLiNER2's
  added special tokens (`[P]`, `[E]`, `[SEP_TEXT]`).

## 3. Graph Contract

`EntityExtractionGraph` in `convert.py` wraps only the tensor computations of
GLiNER2 (the encoder, word/marker token embedding gathers, span representation
head, count prediction head, and the initial count-embedding GRU step). Schema
prompt formatting, word pre-tokenization, and greedy non-overlapping span
decoding run on the host in Rust (`../main.rs`).

All inputs and outputs use static shapes (`batch = 1`):

### Inputs

| Name | Type & Shape | Description |
| :--- | :--- | :--- |
| `input_ids` | `int32 [1, seq_len]` | Subword token IDs (`0`-padded). |
| `attention_mask` | `int32 [1, seq_len]` | `1` for valid subwords, `0` for padding. |
| `word_positions` | `int32 [1, max_words]` | Subword index of the first piece of each text word (`0`-padded). |
| `label_positions` | `int32 [1, max_labels]` | Subword index of each `[E]` label marker (`0`-padded). |
| `prompt_position` | `int32 [1]` | Subword index of the `[P]` prompt marker. |

### Outputs

| Name | Type & Shape | Description |
| :--- | :--- | :--- |
| `count_logits` | `float32 [1, 20]` | Entity count logits (`argmax == 0` indicates no entities). |
| `span_scores` | `float32 [1, max_labels, max_words, 8]` | Sigmoid probability that the span starting at word `w` with width `m + 1` belongs to label `k`. |

## 4. Running the Rust Example

Once `out/gliner2_pii_int8.tflite` (or `out/gliner2_pii.tflite`) and
`out/tokenizer.json` are generated, run the Rust example from the parent
`samples/litert/text_classification/rust/` directory with Cargo:

```bash
cd ..
cargo run --release -- \
  --model-path=/path/to/out/gliner2_pii_int8.tflite \
  --tokenizer-path=/path/to/out/tokenizer.json \
  --classify-text="Hi, I'm Jean Dupont, email jean.dupont@acme.fr, call +33 6 12 34 56 78."
```

See the [parent README](../README.md) for prerequisites, full CLI options, and
example output.
