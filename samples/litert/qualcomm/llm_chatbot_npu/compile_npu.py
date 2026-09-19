# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Qualcomm NPU Ahead-of-Time (AOT) compiler utility for LiteRT models."""

import argparse
import os
import sys

try:
    from ai_edge_litert.aot import aot_compile as aot_lib
    from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target
except ImportError:
    aot_lib = None
    qnn_target = None


SOC_MODEL_MAP = {
    "SM8750": "SM8750",  # Snapdragon 8 Elite
    "SM8650": "SM8650",  # Snapdragon 8 Gen 3
    "SM8550": "SM8550",  # Snapdragon 8 Gen 2
    "SM8475": "SM8475",  # Snapdragon 8+ Gen 1
    "SM8450": "SM8450",  # Snapdragon 8 Gen 1
}


def setup_qairt_environment(qairt_root: str | None = None) -> None:
    """Configure environment variables for the Qualcomm AI Runtime (QAIRT) SDK."""
    sdk_root = qairt_root or os.environ.get("QAIRT_ROOT")
    if not sdk_root:
        return

    if not os.path.exists(sdk_root):
        print(f"Warning: Specified QAIRT_ROOT does not exist: {sdk_root}")
        return

    os.environ["QAIRT_ROOT"] = sdk_root

    # Add compiler toolchain binaries and shared libraries to PATH and LD_LIBRARY_PATH
    for arch in ("aarch64-ubuntu-gcc9.4", "x86_64-linux-clang"):
        bin_dir = os.path.join(sdk_root, "bin", arch)
        lib_dir = os.path.join(sdk_root, "lib", arch)
        if os.path.exists(bin_dir):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        if os.path.exists(lib_dir):
            os.environ["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")


def check_model_magic_bytes(model_path: str) -> bool:
    """Inspect model header to verify format compatibility (TFL3 vs RTLM)."""
    try:
        with open(model_path, "rb") as f:
            header = f.read(4)
    except OSError as e:
        print(f"Error reading model file: {e}")
        return False

    if header == b"RTLM":
        print(
            "\n[Error: Incompatible Model Container Format Detected]\n"
            f"The provided file '{model_path}' has identifier 'RTLM' (LiteRT Language Model container).\n"
            "The LiteRT AOT NPU compiler operates directly on FlatBuffer TFLite graphs ('TFL3').\n\n"
            "For LLMs distributed as .litertlm bundles:\n"
            "  1. Use pre-compiled NPU bundles from Hugging Face (e.g. gemma-*-qualcomm-*.litertlm)\n"
            "  2. Or compile the constituent .tflite subgraphs before archiving into .litertlm.\n"
        )
        return False
    elif header != b"TFL3":
        print(f"Warning: Model header '{header!r}' is not the standard 'TFL3' FlatBuffer identifier.")
    return True


def get_qualcomm_target(soc_name: str):
    """Retrieve Qualcomm SocModel target enum."""
    if not hasattr(qnn_target, "SocModel"):
        raise RuntimeError("Qualcomm vendor targets not available in ai_edge_litert.")

    soc_enum_name = SOC_MODEL_MAP.get(soc_name.upper(), soc_name.upper())
    if hasattr(qnn_target.SocModel, soc_enum_name):
        soc_model_val = getattr(qnn_target.SocModel, soc_enum_name)
        return qnn_target.Target(soc_model_val)

    available = [m for m in dir(qnn_target.SocModel) if not m.startswith("_")]
    raise ValueError(f"Unknown SoC model '{soc_name}'. Available targets: {', '.join(available)}")


def compile_for_npu(
    model_path: str,
    output_dir: str = "compiled",
    soc_model: str = "SM8750",
    qairt_root: str | None = None,
) -> bool:
    """Compile a .tflite model for Qualcomm NPU."""
    if aot_lib is None or qnn_target is None:
        print("Error: 'ai-edge-litert' is not installed.")
        print("Please install via: pip install ai-edge-litert")
        return False

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return False

    if not check_model_magic_bytes(model_path):
        return False

    setup_qairt_environment(qairt_root)

    print(f"Starting AOT compilation for: {model_path}")
    print(f"Target SoC: {soc_model}")

    try:
        target = get_qualcomm_target(soc_model)
    except Exception as e:
        print(f"Error resolving target SoC: {e}")
        return False

    try:
        os.makedirs(output_dir, exist_ok=True)
        compiled_models = aot_lib.aot_compile(
            model_path,
            output_dir=output_dir,
            target=[target],
        )
        print(f"\nCompilation successful! Output saved in '{output_dir}/'")
        print("The compiled model contains the TF_LITE_AUX payload for Qualcomm QNN NPU.")
        return True
    except Exception as e:
        print(f"\nCompilation failed: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ahead-of-Time (AOT) Qualcomm NPU compilation for LiteRT models."
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        required=True,
        help="Path to the input .tflite model file.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="compiled",
        help="Directory to save the compiled model (default: 'compiled').",
    )
    parser.add_argument(
        "--soc_model",
        "-s",
        type=str,
        default="SM8750",
        help="Target Qualcomm SoC Model (e.g. SM8750 for Snapdragon 8 Elite, SM8650 for 8 Gen 3).",
    )
    parser.add_argument(
        "--qairt_root",
        "-q",
        type=str,
        default=os.environ.get("QAIRT_ROOT"),
        help="Path to Qualcomm QAIRT / QNN SDK root directory (defaults to $QAIRT_ROOT).",
    )

    args = parser.parse_args()
    success = compile_for_npu(
        model_path=args.model_path,
        output_dir=args.output_dir,
        soc_model=args.soc_model,
        qairt_root=args.qairt_root,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
