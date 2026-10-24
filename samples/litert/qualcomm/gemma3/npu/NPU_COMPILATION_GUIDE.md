# Qualcomm NPU Compilation Guide (`compile_npu.py`)

## `.litertlm` Bundles vs. Standalone `.tflite` Graphs

Understanding the difference between LiteRT-LM bundles and standalone LiteRT FlatBuffer graphs avoids the common `Model provided has model identifier 'RTLM', should be 'TFL3'` error:

| Format | Magic Bytes | Purpose | How NPU Execution Works |
| :--- | :--- | :--- | :--- |
| **`.litertlm`** | `RTLM` (`LITERTLM`) | Multi-component archive bundling prefill/decode subgraphs, tokenizer, and prompt templates | Download the ready-to-run SM8750 bundle (`Gemma3-1B-IT_q4_ekv1280_sm8750.litertlm`) directly from Hugging Face [`litert-community/Gemma3-1B-IT`](https://huggingface.co/litert-community/Gemma3-1B-IT). |
| **`.tflite`** | `TFL3` | Single FlatBuffer computation graph | Compile ahead-of-time on Linux (x86_64 / WSL2) using `compile_npu.py` (`ai_edge_litert.aot.aot_compile`) to embed the QNN HTP context payload. |

> [!IMPORTANT]
> **Running this Android sample does NOT require running `compile_npu.py` first.**
> Download [`Gemma3-1B-IT_q4_ekv1280_sm8750.litertlm`](https://huggingface.co/litert-community/Gemma3-1B-IT/blob/main/Gemma3-1B-IT_q4_ekv1280_sm8750.litertlm) and push it to your device:
> ```bash
> ./run.sh --model /path/to/Gemma3-1B-IT_q4_ekv1280_sm8750.litertlm
> ```

---

## Compiling Standalone `.tflite` Models with `compile_npu.py`

If you have exported a standalone `.tflite` model (`TFL3`) and want to embed a pre-compiled Qualcomm Hexagon NPU context binary:

### Prerequisites
- Linux environment (`x86_64` Ubuntu 22.04+ or WSL2 on Windows)
- Python 3.10+

### 1. Install `ai-edge-litert`
```bash
pip install ai-edge-litert
```

### 2. Run `compile_npu.py`
```bash
python3 compile_npu.py \
  --model_path path/to/model.tflite \
  --soc_model SM8750 \
  --output_dir compiled
```

Optional flags:
- `--soc_model`: Target Qualcomm SoC (default: `SM8750`; also supports `SM8650`, `SM8550`, `SM8475`, etc.).
- `--qairt_root`: Optional path to a custom local QAIRT SDK root (defaults to `$QAIRT_ROOT` if set, otherwise uses `ai-edge-litert`'s bundled toolchain automatically).
