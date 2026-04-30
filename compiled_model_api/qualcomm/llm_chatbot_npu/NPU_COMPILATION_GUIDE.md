# NPU Compilation Guide (Gemma 4 E2B)

To utilize the Snapdragon 8 Elite Hexagon NPU, you must Ahead-of-Time (AOT) compile the `gemma-4-E2B-it.litertlm` model using the Google Edge LiteRT toolchain.

> [!WARNING]
> The `ai-edge-litert` compilation package is not currently available for native Windows Python. You must run this compilation step within **WSL (Windows Subsystem for Linux)**, a Linux VM, or a Google Colab notebook. 
> 
> **Highly Recommended**: Use the provided [LiteRT_Gemma4_NPU_AOT_Compilation.ipynb](file:///c:/Users/rawat/ModelGarden-QNN-LiteRT/google_colab/LiteRT_Gemma4_NPU_AOT_Compilation.ipynb) in Google Colab to avoid x86_64 emulation issues on ARM64 Windows.

## Prerequisites (in WSL/Linux)
1. Python 3.10+ installed in your Linux environment.
2. At least 32GB of RAM (AOT compiling a 2.58GB LLM requires significant memory to trace and generate the QNN context).

## Step 1: Install Dependencies
Open your WSL/Linux terminal and run:
```bash
pip install ai-edge-litert
```

## Step 2: Run the Compilation Script
I have created the `compile_npu.py` script for you in the project root. Copy it to your Linux environment along with your `gemma-4-E2B-it.litertlm` file.

Run it:
```bash
python3 compile_npu.py
```

*Note: This process may take anywhere from 30 minutes to an hour depending on your CPU.*

## Step 3: Deploy the Compiled Model
The script will generate an NPU-ready model (containing the `TF_LITE_AUX` payload). Push this new model to your Android device using the exact same ADB command we used previously:

```bash
adb push gemma-4-E2B-it.litertlm /sdcard/Android/data/com.example.qnn_litertlm_gemma/files/gemma-4-E2B-it.litertlm
```

Once pushed, the Android app will automatically detect the NPU payload and initialize the QNN Delegate instead of falling back to the GPU!
