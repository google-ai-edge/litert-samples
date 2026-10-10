# Qualcomm NPU Ahead-of-Time (AOT) Compilation Guide

To utilize the Snapdragon Hexagon NPU (e.g., Snapdragon 8 Elite / SM8750, Snapdragon 8 Gen 3 / SM8650), you can Ahead-of-Time (AOT) compile compatible `.tflite` model subgraphs using the Google AI Edge LiteRT toolchain and Qualcomm AI Runtime (QAIRT / QNN) SDK.

> [!NOTE]
> For Large Language Models distributed as `.litertlm` container bundles, pre-compiled NPU bundles are available directly on Hugging Face (e.g. `gemma-*-qualcomm-*.litertlm`). To compile custom models from source, compile the constituent `.tflite` computational graphs.

> [!WARNING]
> The `ai-edge-litert` NPU compilation package is designed for Linux environments. On Windows, run compilation within **WSL (Windows Subsystem for Linux)**, Linux Docker, or a Linux VM.

---

## Prerequisites (Linux / WSL)

1. **Python 3.10+**
2. **Qualcomm AI Runtime (QAIRT / QNN) SDK** downloaded from Qualcomm Developer Network.
3. **RAM**: At least 16GB+ of memory for tracing and QNN context binary generation.

---

## Step 1: Install Dependencies

In your Linux / WSL environment:

```bash
pip install ai-edge-litert
```

Set your QAIRT SDK path:

```bash
export QAIRT_ROOT=/path/to/qairt/<version>
```

---

## Step 2: Run the Compilation Script

Run `compile_npu.py` with your input `.tflite` model file:

```bash
python3 compile_npu.py --model_path /path/to/model.tflite --soc_model SM8750 --output_dir compiled
```

### Supported CLI Arguments

| Flag | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model_path` | `-m` | *Required* | Path to input `.tflite` model file |
| `--output_dir` | `-o` | `compiled` | Output directory for the compiled artifact |
| `--soc_model` | `-s` | `SM8750` | Target Qualcomm SoC (e.g. `SM8750`, `SM8650`, `SM8550`) |
| `--qairt_root` | `-q` | `$QAIRT_ROOT` | Path to Qualcomm QAIRT / QNN SDK root directory |

---

## Step 3: Deploy the Compiled Model

The script generates an NPU-accelerated model graph containing the `TF_LITE_AUX` payload. Push the model to your Android device via ADB:

```bash
adb push compiled/model.tflite /sdcard/Android/data/<package_name>/files/model.tflite
```

When initialized on device, LiteRT will automatically detect the `TF_LITE_AUX` binary and dispatch execution to the Qualcomm Hexagon NPU.
