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
"""Unit tests for litert_gpu_toolkit verification checker."""

import io
import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from litert_gpu_toolkit.checker import (
    _random_inputs,
    _run_signature,
    _static_shape,
    check_gpu_compatibility,
    print_report,
)


class TestCheckerHelpers:
    def test_static_shape(self):
        assert _static_shape([1, 3, 224, 224]) == [1, 3, 224, 224]
        assert _static_shape([-1, 3, 0, 100]) == [1, 3, 1, 100]

    def test_random_inputs(self):
        rng = np.random.default_rng(42)
        details = {
            "float_in": {"shape": [1, 4], "dtype": "float32"},
            "int_in": {"shape": [1, 8], "dtype": "int32"},
            "bool_in": {"shape": [2, 2], "dtype": "bool"},
        }
        inputs = _random_inputs(details, rng)
        assert inputs["float_in"].shape == (1, 4)
        assert inputs["float_in"].dtype == np.float32
        assert inputs["int_in"].shape == (1, 8)
        assert inputs["int_in"].dtype == np.int32
        assert np.all(inputs["int_in"] == 0)
        assert inputs["bool_in"].shape == (2, 2)
        assert inputs["bool_in"].dtype == bool
        assert np.all(inputs["bool_in"] == False)


class TestRunSignature:
    def test_run_signature(self):
        mock_model = MagicMock()
        mock_model.get_output_tensor_details.return_value = {
            "output_0": {"shape": [1, 2], "dtype": "float32"}
        }
        in_buf = MagicMock()
        out_buf = MagicMock()
        mock_model.create_input_buffer_by_name.return_value = in_buf
        mock_model.create_output_buffer_by_name.return_value = out_buf

        # Return flat float32 array
        out_buf.read.return_value = np.array([1.0, 2.0], dtype=np.float32)

        inputs = {"input_0": np.zeros((1, 2), dtype=np.float32)}
        outputs = _run_signature(mock_model, "serving_default", inputs)

        assert "output_0" in outputs
        assert outputs["output_0"].shape == (1, 2)
        assert np.allclose(outputs["output_0"], [[1.0, 2.0]])


class TestCheckGPUCompatibility:
    def _create_mock_compiled_model(self, out_val=1.0):
        m = MagicMock()
        m.get_signature_list.return_value = ["serving_default"]
        m.get_input_tensor_details.return_value = {
            "input_0": {"shape": [1, 4], "dtype": "float32"}
        }
        m.get_output_tensor_details.return_value = {
            "output_0": {"shape": [1, 4], "dtype": "float32"}
        }
        in_buf = MagicMock()
        out_buf = MagicMock()
        m.create_input_buffer_by_name.return_value = in_buf
        m.create_output_buffer_by_name.return_value = out_buf
        out_buf.read.return_value = np.full((1, 4), out_val, dtype=np.float32)
        return m

    def _mock_litert_modules(self, from_file_side_effect):
        mock_compiled_model_class = MagicMock()
        mock_compiled_model_class.from_file.side_effect = from_file_side_effect

        class MockHardwareAccelerator:
            CPU = 1
            GPU = 2

        mock_compiled_model_mod = MagicMock(CompiledModel=mock_compiled_model_class)
        mock_hw_accel_mod = MagicMock(HardwareAccelerator=MockHardwareAccelerator)

        return {
            "ai_edge_litert": MagicMock(),
            "ai_edge_litert.compiled_model": mock_compiled_model_mod,
            "ai_edge_litert.hardware_accelerator": mock_hw_accel_mod,
        }

    def test_check_gpu_success(self):
        mock_cpu = self._create_mock_compiled_model(1.0)
        mock_gpu = self._create_mock_compiled_model(1.0)

        modules = self._mock_litert_modules([mock_cpu, mock_gpu])
        with patch.dict("sys.modules", modules):
            res = check_gpu_compatibility("dummy.tflite")

            assert res["compatible"] is True
            assert res["gpu_compile_ok"] is True
            assert res["gpu_cpu_fallback"] is False
            assert res["numerics_ok"] is True
            assert res["max_abs_diff"] == 0.0

    def test_check_gpu_fallback(self):
        mock_cpu = self._create_mock_compiled_model(1.0)
        mock_gpu_fallback = self._create_mock_compiled_model(1.0)

        modules = self._mock_litert_modules([mock_cpu, RuntimeError("GPU-only fail"), mock_gpu_fallback])
        with patch.dict("sys.modules", modules):
            res = check_gpu_compatibility("dummy.tflite")

            assert res["compatible"] is True
            assert res["gpu_compile_ok"] is True
            assert res["gpu_cpu_fallback"] is True
            assert len(res["warnings"]) > 0

    def test_check_cpu_fail(self):
        modules = self._mock_litert_modules(RuntimeError("CPU invalid graph"))
        with patch.dict("sys.modules", modules):
            res = check_gpu_compatibility("dummy.tflite")

            assert res["compatible"] is False
            assert len(res["errors"]) == 1
            assert "CPU compile failed" in res["errors"][0]

    def test_check_numerical_divergence(self):
        mock_cpu = self._create_mock_compiled_model(1.0)
        mock_gpu = self._create_mock_compiled_model(5.0)  # Diverged output

        modules = self._mock_litert_modules([mock_cpu, mock_gpu])
        with patch.dict("sys.modules", modules):
            res = check_gpu_compatibility("dummy.tflite")

            assert res["compatible"] is False
            assert res["numerics_ok"] is False
            assert res["max_abs_diff"] == 4.0
            assert any("diverge" in err for err in res["errors"])


class TestPrintReport:
    def test_print_report_compatible(self):
        res = {
            "compatible": True,
            "gpu_cpu_fallback": False,
            "max_abs_diff": 1e-4,
            "errors": [],
            "warnings": [],
            "signatures": {"serving_default": {"ran": True, "max_abs_diff": 1e-4, "error": None}},
        }
        captured = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = captured
            print_report(res)
        finally:
            sys.stdout = old_stdout

        out_str = captured.getvalue()
        assert "LiteRT CompiledModel GPU Verification Report" in out_str
        assert "Status: VERIFIED (fully on GPU)" in out_str

    def test_print_report_failed(self):
        res = {
            "compatible": False,
            "gpu_cpu_fallback": False,
            "max_abs_diff": None,
            "errors": ["Signature 'serving_default' failed"],
            "warnings": ["Warning 1"],
            "signatures": {"serving_default": {"ran": False, "max_abs_diff": None, "error": "Crash"}},
        }
        captured = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = captured
            print_report(res)
        finally:
            sys.stdout = old_stdout

        out_str = captured.getvalue()
        assert "Status: FAILED" in out_str
        assert "Signature 'serving_default' failed" in out_str
