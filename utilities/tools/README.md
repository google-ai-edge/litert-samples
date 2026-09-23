# Developer Tools & Diagnostic Utilities

This directory contains standalone developer tools and diagnostic utilities for the Google AI Edge LiteRT ecosystem.

---

## 🛠️ Available Tools

### 1. `check_npu_compatibility.py` — NPU Pre-Flight & Graph Diagnostics Checker
Statically audits `.tflite` FlatBuffer graphs to detect quantization gaps, dynamic tensor dimensions, and unsupported vendor operators before on-device NPU deployment (Qualcomm Snapdragon HTP, Google Tensor TPU, MediaTek NeuroPilot).

#### Usage:
```bash
# Check a model for Qualcomm Snapdragon NPU (default)
python utilities/tools/check_npu_compatibility.py model.tflite --target qualcomm

# Check for Google Tensor TPU (Pixel 9 Pro / 8 / 7)
python utilities/tools/check_npu_compatibility.py model.tflite --target google_tensor

# Output machine-readable JSON for automated CI/CD gating
python utilities/tools/check_npu_compatibility.py model.tflite --json
```

#### Running Tests:
```bash
pytest utilities/tools/tests/test_check_npu_compatibility.py -v
```

---

### 2. `sync_common.py` — Common Utilities Synchronization
Checks and updates vendored copies of shared Kotlin helpers across sample modules to keep them synchronized with canonical sources in `utilities/common/kotlin/`.

#### Usage:
```bash
# Check for drift
python utilities/tools/sync_common.py --check

# Synchronize copies
python utilities/tools/sync_common.py --apply
```
