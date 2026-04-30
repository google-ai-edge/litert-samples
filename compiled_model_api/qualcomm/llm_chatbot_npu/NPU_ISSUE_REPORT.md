# Issue Report: Gemma 4 NPU Initialization Failure (SM8750)

## 1. Issue Summary
The application crashes with a **SIGABRT (Signal 6)** during the initialization of the LiteRT-LM Engine when attempting to use the **NPU** backend on the Samsung S25 Ultra (Snapdragon 8 Elite / SM8750).

## 2. Root Cause Analysis
The crash is caused by a **Symbol Lookup Error** in the native dynamic linker (`dlopen`).

### Error Log:
```text
W LiteRTLMManager: Could not load native libraries explicitly: dlopen failed: 
cannot locate symbol "LiteRtGetEnvironmentOptions" referenced by 
"/data/app/.../lib/arm64-v8a/libLiteRtDispatch_Qualcomm.so"
```

### Technical Detail:
- **`libLiteRtDispatch_Qualcomm.so`**: This is the Qualcomm-specific dispatch library for SM8750.
- **Missing Symbol**: `LiteRtGetEnvironmentOptions`.
- **Dependency**: This symbol is expected to be provided by the core LiteRT runtime library (`libLiteRt.so`).
- **Version Mismatch**: The version of LiteRT currently bundled (from `litertlm-android` AAR or local `jniLibs`) is older than the version required by the SM8750 NPU dispatch library.

## 3. Impact
NPU-accelerated inference is currently impossible. The app fails to initialize the engine and terminates immediately upon attempting to load the NPU backend.

## 4. Attempted Mitigations
- **Library Replacement**: Replaced all QNN/LiteRT libraries in `jniLibs` with the latest provided files from `C:\Users\rawat\Downloads\litertlm_sm8750\litertlm`.
- **Explicit Loading**: Added `System.loadLibrary` calls in the correct order (`LiteRt` -> `GemmaModelConstraintProvider` -> `LiteRtDispatch_Qualcomm`).
- **Backend Isolation**: Switched vision and audio backends to CPU to isolate the NPU text engine initialization.

## 5. Recommended Solutions

### Solution A: Obtain Newer LiteRT Runtime (Recommended)
You need a version of `libLiteRt.so` (and associated AAR) that includes the `LiteRtGetEnvironmentOptions` symbol. This is likely available in a newer **LiteRT-LM SDK** or a specialized Qualcomm/Google early-access branch for Snapdragon 8 Elite.

### Solution B: Check for Missing Dependency
Verify if there is another library in the SM8750 release package (e.g., `libLiteRtEnvironment.so` or similar) that should be loaded alongside the dispatch library.

### Solution C: Fallback to GPU
If NPU-specific libraries cannot be reconciled, the app can be modified to prioritize the **GPU** backend, which uses standard OpenCL/Vulkan paths and does not require the specialized Qualcomm dispatch symbols.

## 6. Environment Information
- **Device**: Samsung S25 Ultra (Snapdragon 8 Elite / SM8750)
- **Model**: Gemma 4 2B (Int4)
- **Framework**: Google LiteRT-LM (latest.release)
- **Delegate**: Qualcomm QNN 2.42.0
