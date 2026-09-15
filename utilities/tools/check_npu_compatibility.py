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
"""LiteRT NPU Pre-Flight & Diagnostics Checker.

Statically audits .tflite FlatBuffer graphs to detect quantization gaps, dynamic
shapes, and unsupported vendor operations before on-device NPU deployment.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import struct
import sys
from typing import Any


# Standard TFLite Tensor Types
TENSOR_TYPE_NAMES = {
    0: "FLOAT32",
    1: "FLOAT16",
    2: "INT32",
    3: "UINT8",
    4: "INT64",
    5: "STRING",
    6: "BOOL",
    7: "INT16",
    8: "COMPLEX64",
    9: "INT8",
    10: "FLOAT64",
    11: "COMPLEX128",
    12: "UINT64",
    13: "RESOURCE",
    14: "VARIANT",
    15: "UINT32",
    16: "UINT16",
    17: "INT4",
}

INT_QUANT_TYPES = {3, 7, 9, 17}  # UINT8, INT16, INT8, INT4


@dataclasses.dataclass
class TensorDiagnostic:
    index: int
    name: str
    dtype: str
    shape: list[int]
    is_quantized: bool
    scale: list[float]
    zero_point: list[int]
    issues: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class OpDiagnostic:
    index: int
    opcode: str
    inputs: list[int]
    outputs: list[int]
    issues: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class NpuAuditResult:
    model_path: str
    target: str
    is_compatible: bool
    compatibility_score: float  # 0.0 - 100.0%
    total_tensors: int
    quantized_tensors: int
    float_leak_tensors: list[TensorDiagnostic]
    dynamic_shape_tensors: list[TensorDiagnostic]
    operator_issues: list[OpDiagnostic]
    remediation_tips: list[str]


# ─── FlatBuffer Parsing Logic ──────────────────────────────────────────────────

class FlatBufferModelParser:
    """Lightweight direct FlatBuffer binary inspector for .tflite models."""

    def __init__(self, data: bytes):
        self.data = data
        if len(data) < 8:
            raise ValueError("File is too small to be a valid TFLite model.")
        
        # Verify FlatBuffer identifier
        identifier = data[4:8]
        if identifier != b"TFL3":
            if identifier == b"RTLM":
                raise ValueError(
                    "Detected 'RTLM' container bundle. The NPU checker requires a flat .tflite graph ('TFL3'). "
                    "Extract the constituent .tflite subgraphs from the bundle before auditing."
                )
            raise ValueError(f"Invalid FlatBuffer identifier: {identifier!r} (expected 'TFL3').")

    def _read_u32(self, offset: int) -> int:
        if offset + 4 > len(self.data):
            return 0
        return struct.unpack_from("<I", self.data, offset)[0]

    def _read_i32(self, offset: int) -> int:
        if offset + 4 > len(self.data):
            return 0
        return struct.unpack_from("<i", self.data, offset)[0]

    def _read_u16(self, offset: int) -> int:
        if offset + 2 > len(self.data):
            return 0
        return struct.unpack_from("<H", self.data, offset)[0]

    def _get_table_field_offset(self, table_offset: int, vtable_field_idx: int) -> int:
        soff = self._read_i32(table_offset)
        vtable_offset = table_offset - soff
        if vtable_offset < 0 or vtable_offset >= len(self.data):
            return 0
        vtable_size = self._read_u16(vtable_offset)
        field_offset_in_vtable = 4 + vtable_field_idx * 2
        if field_offset_in_vtable >= vtable_size:
            return 0
        field_offset = self._read_u16(vtable_offset + field_offset_in_vtable)
        if field_offset == 0:
            return 0
        return table_offset + field_offset

    def _read_string(self, offset: int) -> str:
        if offset == 0 or offset >= len(self.data):
            return ""
        length = self._read_u32(offset)
        str_start = offset + 4
        if str_start + length > len(self.data):
            return ""
        return self.data[str_start : str_start + length].decode("utf-8", errors="replace")

    def _read_vector(self, offset: int) -> list[int]:
        if offset == 0 or offset >= len(self.data):
            return []
        length = self._read_u32(offset)
        vec = []
        for i in range(length):
            elem_offset = offset + 4 + i * 4
            if elem_offset + 4 <= len(self.data):
                vec.append(self._read_i32(elem_offset))
        return vec

    def parse_tensors_and_ops(self) -> tuple[list[TensorDiagnostic], list[OpDiagnostic]]:
        """Extract tensors and operators from the first subgraph."""
        root_table_offset = self._read_u32(0)
        
        # In TFLite Schema: Model table field 2 (0-indexed) is subgraphs vector
        # Model: 0: version, 1: operator_codes, 2: subgraphs
        op_codes_offset = self._get_table_field_offset(root_table_offset, 1)
        subgraphs_offset = self._get_table_field_offset(root_table_offset, 2)

        op_names = []
        if op_codes_offset != 0:
            op_codes_len = self._read_u32(op_codes_offset)
            for i in range(op_codes_len):
                op_code_table_pos = op_codes_offset + 4 + i * 4
                op_code_table = op_code_table_pos + self._read_u32(op_code_table_pos)
                # OperatorCode: 0: deprecated_builtin_code, 1: custom_code, 2: version, 3: builtin_code
                builtin_code = self._read_i32(self._get_table_field_offset(op_code_table, 3))
                custom_code_off = self._get_table_field_offset(op_code_table, 1)
                custom_code = self._read_string(custom_code_off + self._read_u32(custom_code_off)) if custom_code_off else ""
                op_names.append(custom_code if custom_code else f"OP_CODE_{builtin_code}")

        tensors: list[TensorDiagnostic] = []
        ops: list[OpDiagnostic] = []

        if subgraphs_offset == 0:
            return tensors, ops

        subgraphs_len = self._read_u32(subgraphs_offset)
        if subgraphs_len == 0:
            return tensors, ops

        # Primary Subgraph (0)
        sg_pos = subgraphs_offset + 4
        sg_table = sg_pos + self._read_u32(sg_pos)

        # SubGraph: 0: tensors, 1: inputs, 2: outputs, 3: operators, 4: name
        tensors_vec_off = self._get_table_field_offset(sg_table, 0)
        operators_vec_off = self._get_table_field_offset(sg_table, 3)

        if tensors_vec_off != 0:
            tensors_len = self._read_u32(tensors_vec_off)
            for i in range(tensors_len):
                t_pos = tensors_vec_off + 4 + i * 4
                t_table = t_pos + self._read_u32(t_pos)

                # Tensor: 0: shape, 1: type, 2: buffer, 3: name, 4: quantization
                shape_off = self._get_table_field_offset(t_table, 0)
                type_off = self._get_table_field_offset(t_table, 1)
                name_off = self._get_table_field_offset(t_table, 3)
                quant_off = self._get_table_field_offset(t_table, 4)

                shape = self._read_vector(shape_off + self._read_u32(shape_off)) if shape_off else []
                dtype_id = self._read_i32(type_off) if type_off else 0
                dtype_name = TENSOR_TYPE_NAMES.get(dtype_id, f"UNKNOWN_{dtype_id}")
                name = self._read_string(name_off + self._read_u32(name_off)) if name_off else f"tensor_{i}"

                is_quant = dtype_id in INT_QUANT_TYPES
                tensors.append(
                    TensorDiagnostic(
                        index=i,
                        name=name,
                        dtype=dtype_name,
                        shape=shape,
                        is_quantized=is_quant,
                        scale=[],
                        zero_point=[],
                    )
                )

        if operators_vec_off != 0:
            ops_len = self._read_u32(operators_vec_off)
            for i in range(ops_len):
                op_pos = operators_vec_off + 4 + i * 4
                op_table = op_pos + self._read_u32(op_pos)

                # Operator: 0: opcode_index, 1: inputs, 2: outputs
                opcode_idx_off = self._get_table_field_offset(op_table, 0)
                inputs_off = self._get_table_field_offset(op_table, 1)
                outputs_off = self._get_table_field_offset(op_table, 2)

                opcode_idx = self._read_i32(opcode_idx_off) if opcode_idx_off else 0
                opcode_name = op_names[opcode_idx] if opcode_idx < len(op_names) else f"OP_{opcode_idx}"
                inputs = self._read_vector(inputs_off + self._read_u32(inputs_off)) if inputs_off else []
                outputs = self._read_vector(outputs_off + self._read_u32(outputs_off)) if outputs_off else []

                ops.append(
                    OpDiagnostic(
                        index=i,
                        opcode=opcode_name,
                        inputs=inputs,
                        outputs=outputs,
                    )
                )

        return tensors, ops


# ─── Audit Rules Engine ────────────────────────────────────────────────────────

def audit_npu_compatibility(
    model_path: str,
    target: str = "qualcomm",
    model_bytes: bytes | None = None,
) -> NpuAuditResult:
    """Audit a .tflite model for NPU execution compatibility."""
    if model_bytes is None:
        with open(model_path, "rb") as f:
            model_bytes = f.read()

    parser = FlatBufferModelParser(model_bytes)
    tensors, ops = parser.parse_tensors_and_ops()

    float_leaks: list[TensorDiagnostic] = []
    dynamic_shapes: list[TensorDiagnostic] = []
    op_issues: list[OpDiagnostic] = []
    remediation_tips: list[str] = []

    # 1. Audit Quantization
    total_quantized = 0
    for t in tensors:
        if t.is_quantized:
            total_quantized += 1
        elif t.dtype in ("FLOAT32", "FLOAT16"):
            t.issues.append(f"Unquantized {t.dtype} tensor in NPU execution graph.")
            float_leaks.append(t)

    # 2. Audit Static Shapes
    for t in tensors:
        if any(dim <= 0 for dim in t.shape):
            t.issues.append(f"Dynamic or unassigned dimension detected: {t.shape}.")
            dynamic_shapes.append(t)

    # 3. Audit Target-Specific Vendor Operator Constraints
    target_lower = target.lower()
    for op in ops:
        if "FLEX" in op.opcode or "CUSTOM" in op.opcode:
            op.issues.append(f"Custom/Flex operator '{op.opcode}' cannot run on hardware NPU.")
            op_issues.append(op)
        elif target_lower in ("qualcomm", "snapdragon", "sm8750", "sm8650", "sm8550"):
            # Qualcomm HTP strictness: check if outputs are float in mixed graph
            for out_idx in op.outputs:
                if out_idx < len(tensors) and not tensors[out_idx].is_quantized:
                    if tensors[out_idx].dtype in ("FLOAT32", "FLOAT16"):
                        op.issues.append(f"Operator output tensor #{out_idx} has {tensors[out_idx].dtype} data type.")
                        if op not in op_issues:
                            op_issues.append(op)

    # Build Remediation Tips
    if float_leaks:
        remediation_tips.append(
            f"Found {len(float_leaks)} float tensor(s). Full INT8 quantization is required by hardware NPUs "
            "(Qualcomm HTP & Google Tensor TPU). Re-run post-training quantization using AI Edge Quantizer "
            "or apply SafeLayerNorm to prevent float overflow."
        )
    if dynamic_shapes:
        remediation_tips.append(
            f"Found {len(dynamic_shapes)} tensor(s) with dynamic shapes. Hardware NPUs require compile-time static shapes. "
            "Export the model with a fixed batch size (e.g. batch_size=1) and static spatial dimensions."
        )
    if op_issues:
        remediation_tips.append(
            f"Found {len(op_issues)} operator partition issue(s). Replace custom/flex ops with standard LiteRT "
            "supported operations or pre-convert via litert_gpu_toolkit / litert-torch."
        )

    is_compatible = len(float_leaks) == 0 and len(dynamic_shapes) == 0 and len(op_issues) == 0
    
    total_tensor_count = max(len(tensors), 1)
    quant_score = (total_quantized / total_tensor_count) * 100.0
    shape_penalty = 50.0 if dynamic_shapes else 0.0
    op_penalty = 30.0 if op_issues else 0.0
    score = max(0.0, min(100.0, quant_score - shape_penalty - op_penalty))

    return NpuAuditResult(
        model_path=model_path,
        target=target,
        is_compatible=is_compatible,
        compatibility_score=score,
        total_tensors=len(tensors),
        quantized_tensors=total_quantized,
        float_leak_tensors=float_leaks,
        dynamic_shape_tensors=dynamic_shapes,
        operator_issues=op_issues,
        remediation_tips=remediation_tips,
    )


# ─── Formatting & CLI Reporting ────────────────────────────────────────────────

def print_npu_diagnostic_report(res: NpuAuditResult) -> None:
    """Print human-readable NPU diagnostic report."""
    print(f"\n{'=' * 70}")
    print("  LiteRT NPU Pre-Flight & Graph Diagnostics Report")
    print(f"{'=' * 70}")
    print(f"  Model File : {res.model_path}")
    print(f"  Target SoC : {res.target.upper()}")
    
    if res.is_compatible:
        status_str = "✅ NPU-READY (100% Hardware Compatible)"
    elif res.compatibility_score >= 70.0:
        status_str = f"⚠️ PARTIAL COMPATIBILITY ({res.compatibility_score:.1f}% Score - CPU Fallback Risk)"
    else:
        status_str = f"❌ INCOMPATIBLE ({res.compatibility_score:.1f}% Score - Will Fail On Device)"

    print(f"  Status     : {status_str}")
    print(f"  Quantized  : {res.quantized_tensors} / {res.total_tensors} tensors ({res.compatibility_score:.1f}%)")

    if res.float_leak_tensors:
        print(f"\n  🔴 Float32/Float16 Leaks ({len(res.float_leak_tensors)} tensors):")
        for t in res.float_leak_tensors[:8]:
            print(f"     - Tensor #{t.index:03d} [{t.dtype:7s}]: {t.name} (shape: {t.shape})")
        if len(res.float_leak_tensors) > 8:
            print(f"     ... and {len(res.float_leak_tensors) - 8} more.")

    if res.dynamic_shape_tensors:
        print(f"\n  🔴 Dynamic Shapes ({len(res.dynamic_shape_tensors)} tensors):")
        for t in res.dynamic_shape_tensors[:8]:
            print(f"     - Tensor #{t.index:03d} {t.name}: shape {t.shape}")
        if len(res.dynamic_shape_tensors) > 8:
            print(f"     ... and {len(res.dynamic_shape_tensors) - 8} more.")

    if res.operator_issues:
        print(f"\n  🔴 Operator & Partition Warnings ({len(res.operator_issues)} nodes):")
        for op in res.operator_issues[:8]:
            print(f"     - Op #{op.index:03d} [{op.opcode}]: {', '.join(op.issues)}")

    if res.remediation_tips:
        print(f"\n  💡 Actionable Remediation Tips:")
        for idx, tip in enumerate(res.remediation_tips, 1):
            print(f"     {idx}. {tip}")

    print(f"{'=' * 70}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LiteRT NPU Pre-Flight & Diagnostics Checker (Qualcomm Snapdragon, Google Tensor TPU, MediaTek)."
    )
    parser.add_argument("model_path", type=str, help="Path to the .tflite model file to audit.")
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        default="qualcomm",
        choices=["qualcomm", "google_tensor", "mtk", "all"],
        help="Target hardware NPU accelerator vendor (default: qualcomm).",
    )
    parser.add_argument("--json", action="store_true", help="Output results in machine-readable JSON format.")

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found.", file=sys.stderr)
        return 1

    try:
        result = audit_npu_compatibility(args.model_path, target=args.target)
    except Exception as e:
        print(f"Audit Error: {e}", file=sys.stderr)
        return 2

    if args.json:
        data = {
            "model_path": result.model_path,
            "target": result.target,
            "is_compatible": result.is_compatible,
            "compatibility_score": result.compatibility_score,
            "total_tensors": result.total_tensors,
            "quantized_tensors": result.quantized_tensors,
            "float_leaks_count": len(result.float_leak_tensors),
            "dynamic_shapes_count": len(result.dynamic_shape_tensors),
            "operator_issues_count": len(result.operator_issues),
            "remediation_tips": result.remediation_tips,
        }
        print(json.dumps(data, indent=2))
    else:
        print_npu_diagnostic_report(result)

    return 0 if result.is_compatible else 1


if __name__ == "__main__":
    sys.exit(main())
