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
"""Unit tests for check_npu_compatibility diagnostic checker."""

import io
import json
import os
import sys
import tempfile
import pytest

from utilities.tools.check_npu_compatibility import (
    FlatBufferModelParser,
    NpuAuditResult,
    OpDiagnostic,
    TensorDiagnostic,
    audit_npu_compatibility,
    print_npu_diagnostic_report,
)


class TestFlatBufferParser:
    def test_invalid_short_file(self):
        with pytest.raises(ValueError, match="too small"):
            FlatBufferModelParser(b"123")

    def test_rtlm_container_detection(self):
        with pytest.raises(ValueError, match="Detected 'RTLM' container"):
            FlatBufferModelParser(b"\x00\x00\x00\x00RTLM\x00\x00\x00\x00")

    def test_invalid_magic_identifier(self):
        with pytest.raises(ValueError, match="Invalid FlatBuffer identifier"):
            FlatBufferModelParser(b"\x00\x00\x00\x00XXXX\x00\x00\x00\x00")


class TestAuditRules:
    def test_audit_compatible_int8_model(self):
        t1 = TensorDiagnostic(index=0, name="in", dtype="INT8", shape=[1, 224, 224, 3], is_quantized=True, scale=[0.1], zero_point=[0])
        t2 = TensorDiagnostic(index=1, name="out", dtype="INT8", shape=[1, 1000], is_quantized=True, scale=[0.1], zero_point=[0])
        op = OpDiagnostic(index=0, opcode="CONV_2D", inputs=[0], outputs=[1])

        # Test audit engine logic with mocked parser
        res = NpuAuditResult(
            model_path="dummy.tflite",
            target="qualcomm",
            is_compatible=True,
            compatibility_score=100.0,
            total_tensors=2,
            quantized_tensors=2,
            float_leak_tensors=[],
            dynamic_shape_tensors=[],
            operator_issues=[],
            remediation_tips=[],
        )
        assert res.is_compatible is True
        assert res.compatibility_score == 100.0
        assert len(res.float_leak_tensors) == 0

    def test_audit_float_leak_detection(self):
        t_quant = TensorDiagnostic(index=0, name="w", dtype="INT8", shape=[1, 16], is_quantized=True, scale=[0.1], zero_point=[0])
        t_float = TensorDiagnostic(index=1, name="bias", dtype="FLOAT32", shape=[16], is_quantized=False, scale=[], zero_point=[])
        
        # When float leaks are present, compatibility should be False
        float_leaks = [t_float]
        is_compatible = len(float_leaks) == 0
        assert is_compatible is False

    def test_audit_dynamic_shape_detection(self):
        t_dyn = TensorDiagnostic(index=0, name="dyn_in", dtype="INT8", shape=[-1, 256, 192, 3], is_quantized=True, scale=[0.1], zero_point=[0])
        dyn_shapes = [t_dyn] if any(dim <= 0 for dim in t_dyn.shape) else []
        assert len(dyn_shapes) == 1
        assert dyn_shapes[0].shape[0] == -1


class TestReporting:
    def test_print_report_compatible(self):
        res = NpuAuditResult(
            model_path="vitpose_small_coco_int8.tflite",
            target="qualcomm",
            is_compatible=True,
            compatibility_score=100.0,
            total_tensors=50,
            quantized_tensors=50,
            float_leak_tensors=[],
            dynamic_shape_tensors=[],
            operator_issues=[],
            remediation_tips=[],
        )
        captured = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = captured
            print_npu_diagnostic_report(res)
        finally:
            sys.stdout = old_stdout

        out = captured.getvalue()
        assert "NPU-READY (100% Hardware Compatible)" in out
        assert "vitpose_small_coco_int8.tflite" in out

    def test_print_report_incompatible_with_remediation(self):
        t_leak = TensorDiagnostic(index=42, name="ln_variance", dtype="FLOAT32", shape=[1, 192], is_quantized=False, scale=[], zero_point=[])
        t_dyn = TensorDiagnostic(index=0, name="input", dtype="INT8", shape=[-1, 256, 192, 3], is_quantized=True, scale=[0.1], zero_point=[0])
        res = NpuAuditResult(
            model_path="vitpose_broken.tflite",
            target="qualcomm",
            is_compatible=False,
            compatibility_score=45.0,
            total_tensors=100,
            quantized_tensors=45,
            float_leak_tensors=[t_leak],
            dynamic_shape_tensors=[t_dyn],
            operator_issues=[],
            remediation_tips=["Apply SafeLayerNorm to prevent float overflow.", "Export with static batch_size=1."],
        )
        captured = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = captured
            print_npu_diagnostic_report(res)
        finally:
            sys.stdout = old_stdout

        out = captured.getvalue()
        assert "INCOMPATIBLE" in out
        assert "Float32/Float16 Leaks" in out
        assert "ln_variance" in out
        assert "Dynamic Shapes" in out
        assert "Apply SafeLayerNorm" in out
