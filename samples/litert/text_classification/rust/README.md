# GLiNER2 PII Text Classifier (LiteRT Rust Sample)

A Rust command-line sample demonstrating on-device Personally Identifiable
Information (PII) entity extraction using the
[`google-ai-edge-litert`](https://crates.io/crates/google-ai-edge-litert)
Rust bindings (`litert` crate) and a converted
[GLiNER2](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi)
model (`fastino/gliner2-privacy-filter-PII-multi`, built on
`microsoft/mdeberta-v3-base`).

## Overview

Given an input text string, `text_classifier` extracts PII entity spans across
**42 labels** (supporting the 7 languages covered by the upstream multilingual
model) entirely on CPU via LiteRT's `LiteRTModelExecutor` and `SignatureRunner`
APIs, with subword tokenization handled by the HuggingFace
[`tokenizers`](https://crates.io/crates/tokenizers) crate.

### How It Works

The model graph (`EntityExtractionGraph` in
[`tflite_conversion/convert.py`](tflite_conversion/convert.py)) contains the
encoder, marker/word embedding gathers, span representation head, count
prediction head, and count-aware label projection. The Rust host binary
([`main.rs`](main.rs)) runs the end-to-end pipeline:

1. **Initialization (`TextClassificationModel::try_new`)**: Configures the CPU
   accelerator (`LiteRtHwAccelerator::Cpu`), loads and compiles the `.tflite`
   model with `LiteRTModelExecutor::from_file`, creates a `SignatureRunner` for
   the default signature, and loads `tokenizer.json`.
2. **Pre-tokenization (`split_words`)**: Splits input text in linear time using
   rules matching GLiNER2's `WhitespaceTokenSplitter` (URLs, email addresses,
   `@handle` tokens, words with internal `-` / `_` connectors, and single
   punctuation characters). Words are lowercased while preserving original UTF-8
   byte offset ranges (`start..end`).
3. **Prompt construction (`build_inputs`)**: Formats the GLiNER2 schema prompt
   `( [P] entities ( [E] <label_1> [E] <label_2> ... ) ) [SEP_TEXT]` followed by
   the tokenized text words, records subword indices for `[P]`, each `[E]` label
   marker, and the first subword of each word, and zero-pads all tensors to the
   model's static input shapes (`SEQ_LEN = 512`, `MAX_WORDS = 256`,
   `MAX_LABELS = 42`). Text exceeding `MAX_WORDS` words or `SEQ_LEN` subwords is
   truncated.
4. **Inference (`SignatureRunner`)**: Binds the five `int32` input tensors
   (`input_ids`, `attention_mask`, `word_positions`, `label_positions`,
   `prompt_position`), runs inference on CPU, reports latency, and reads the
   `count_logits` and `span_scores` outputs.
5. **Decoding (`decode_entities`)**: Checks `count_logits` (`argmax == 0`
   indicates no entities and exits early), filters candidate spans up to
   `MAX_WIDTH = 8` words by confidence threshold (`0.5` by default), and applies
   greedy non-maximum suppression (NMS) per label over overlapping byte spans.
   Overlapping spans across *different* labels (for example, `full_name` and
   `first_name`) are both kept.

### Directory Layout

- [`Cargo.toml`](Cargo.toml) — Package manifest (`google-ai-edge-litert`,
  `clap`, `tokenizers`).
- [`main.rs`](main.rs) — Pre-tokenizer, schema prompt builder, LiteRT runner,
  entity decoder, CLI entry point, and unit tests.
- [`tflite_conversion/`](tflite_conversion/) — Python conversion script and
  [guide](tflite_conversion/README.md) to export the HuggingFace model to
  `.tflite` and `tokenizer.json`.

## Prerequisites

- **Rust toolchain** (edition 2021, with `cargo` and `rustc`) — see
  [rustup.rs](https://rustup.rs/).
- **Clang** and **CMake** — required by `google-ai-edge-litert`'s build script
  (`bindgen` bindings and the prebuilt
  [LiteRT C++ SDK](https://ai.google.dev/edge/litert/next/cpp_sdk)).
- **Python 3.12** (with `uv` or `venv` + `pip`) — required to run the one-time
  model conversion script in [`tflite_conversion/`](tflite_conversion/README.md).

## 1. Export the Model and Tokenizer

Follow [`tflite_conversion/README.md`](tflite_conversion/README.md) to download
`fastino/gliner2-privacy-filter-PII-multi` and export the `.tflite` model files
and `tokenizer.json`:

- `gliner2_pii_int8.tflite` (~358 MiB) — Dynamic-range `int8` weight-quantized,
  `float32` activation model (recommended).
- `gliner2_pii.tflite` (~1.24 GiB) — Unquantized `float32` model.
- `tokenizer.json` — HuggingFace tokenizer including GLiNER2's special tokens
  (`[P]`, `[E]`, `[SEP_TEXT]`).

## 2. Build and Run

From this directory (`samples/litert/text_classification/rust/`):

```bash
cargo run --release -- \
  --model-path=/path/to/gliner2_pii_int8.tflite \
  --tokenizer-path=/path/to/tokenizer.json \
  --classify-text="Contact Jane Doe at jane.doe@example.com or 555-867-5309."
```

### Command-Line Options

| Flag | Short | Required | Description |
| :--- | :--- | :--- | :--- |
| `--model-path` | `-m` | Yes | Path to the converted GLiNER2 `.tflite` model file. |
| `--tokenizer-path` | `-t` | Yes | Path to the exported HuggingFace `tokenizer.json` file. |
| `--classify-text` | `-c` | Yes | Input text string to scan for PII entities. |

## Example Output

### Example 1

Input:

```text
Contact Jane Doe at jane.doe@example.com or 555-867-5309.
```

Output (`stdout`, using `gliner2_pii_int8.tflite`):

```text
Model initialized successfully!
Input name: input_ids
Input name: attention_mask
Input name: word_positions
Input name: label_positions
Input name: prompt_position
Inference completed in 5.84s (5838.87 ms)
Detected 5 entities:
  full_name              "Jane Doe" [8..16] (confidence: 1.000)
  first_name             "Jane" [8..12] (confidence: 0.997)
  last_name              "Doe" [13..16] (confidence: 0.990)
  email                  "jane.doe@example.com" [20..40] (confidence: 1.000)
  phone_number           "555-867-5309" [44..56] (confidence: 1.000)
```

### Example 2

Input:

```text
Hi, I'm Jean Dupont, email jean.dupont@acme.fr, call +33 6 12 34 56 78.
```

Output (`stdout`, using `gliner2_pii_int8.tflite`):

```text
Model initialized successfully!
Input name: input_ids
Input name: attention_mask
Input name: word_positions
Input name: label_positions
Input name: prompt_position
Inference completed in 6.06s (6064.62 ms)
Detected 6 entities:
  person                 "Jean Dupont" [8..19] (confidence: 0.613)
  full_name              "Jean Dupont" [8..19] (confidence: 1.000)
  first_name             "Jean" [8..12] (confidence: 0.998)
  last_name              "Dupont" [13..19] (confidence: 0.988)
  email                  "jean.dupont@acme.fr" [27..46] (confidence: 1.000)
  phone_number           "+33 6 12 34 56 78" [53..70] (confidence: 1.000)
```

Each detected entity line prints:

- **Label** — One of the 42 PII entity labels (see below).
- **Text** — Trimmed substring matched from the original input text.
- **Byte range (`[start..end]`)** — Half-open UTF-8 byte offsets into the input
  string.
- **Confidence** — Model sigmoid score in `[0.0, 1.0]`.

> **Note:** LiteRT also emits `INFO`/`WARNING` runtime logs (such as XNNPACK
> CPU delegate registration) to `stderr`. Actual inference latency depends on
> CPU hardware and compiler optimization flags.

## Default PII Entity Labels (42)

The application queries the following fixed set of 42 PII labels defined in
`PII_LABELS` in [`main.rs`](main.rs):

- **Personal & Identity**: `person`, `full_name`, `first_name`, `middle_name`,
  `last_name`, `date_of_birth`
- **Contact & Address**: `email`, `phone_number`, `address`, `street_address`,
  `city`, `state_or_region`, `postal_code`, `country`
- **Government & Tax IDs**: `government_id`, `national_id_number`,
  `passport_number`, `drivers_license_number`, `license_number`, `tax_id`,
  `tax_number`
- **Financial & Payment**: `bank_account`, `account_number`, `routing_number`,
  `iban`, `payment_card`, `card_number`, `card_expiry`, `card_cvv`
- **Accounts & Credentials**: `username`, `ip_address`, `account_id`,
  `sensitive_account_id`, `password`, `secret`, `api_key`, `access_token`,
  `recovery_code`
- **Dates**: `sensitive_date`, `document_date`, `expiration_date`,
  `transaction_date`

## Model Constants & Shapes

The compiled model uses static batch-1 shapes matching these constants in
[`main.rs`](main.rs):

| Constant | Value | Meaning |
| :--- | :--- | :--- |
| `SEQ_LEN` | `512` | Fixed subword sequence length (schema prompt + text). |
| `MAX_WORDS` | `256` | Maximum number of pre-tokenized text words per call. |
| `MAX_LABELS` | `42` | Maximum number of entity labels in the schema prompt. |
| `MAX_WIDTH` | `8` | Maximum entity span width in words. |
| `MAX_COUNT` | `20` | Length of the `count_logits` output tensor (`0..20`). |
| `DEFAULT_THRESHOLD` | `0.5` | Minimum confidence score required to emit an entity. |

If you re-export the model with custom `--seq-len`, `--max-words`, or
`--max-labels` flags, update the corresponding constants in [`main.rs`](main.rs)
to match. See [`tflite_conversion/README.md`](tflite_conversion/README.md) for
the full input/output tensor contract.

## License

This sample is licensed under the Apache License, Version 2.0. Refer to the
[`fastino/gliner2-privacy-filter-PII-multi`](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi)
model card for upstream model weights and license terms.
