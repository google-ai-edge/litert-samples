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
"""Unit tests for litert_gpu_toolkit convert_for_gpu pipeline."""

import os
import tempfile
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
import pytest

from litert_gpu_toolkit.converter import convert_for_gpu


class TestConverter:
    def test_convert_for_gpu_flow(self):
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2)

            def forward(self, x):
                return self.fc(x)

        model = SimpleNet()
        dummy_input = torch.randn(1, 4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "test_model.tflite")

            mock_convert_result = MagicMock()

            def fake_export(path):
                with open(path, "wb") as f:
                    f.write(b"dummy_tflite_bytes")

            mock_convert_result.export.side_effect = fake_export

            with patch.dict("sys.modules", {"litert_torch": MagicMock(convert=MagicMock(return_value=mock_convert_result))}):
                with patch("litert_gpu_toolkit.converter.check_gpu_compatibility") as mock_check:
                    mock_check.return_value = {
                        "compatible": True,
                        "gpu_compile_ok": True,
                        "gpu_cpu_fallback": False,
                        "numerics_ok": True,
                        "max_abs_diff": 0.0,
                        "signatures": {},
                        "errors": [],
                        "warnings": [],
                    }

                    res_path = convert_for_gpu(
                        model=model,
                        dummy_input=dummy_input,
                        output_path=out_file,
                        check=True,
                        verbose=False,
                    )

                    assert res_path == out_file
                    assert os.path.exists(out_file)
                    assert not model.training
                    mock_check.assert_called_once_with(out_file)

    def test_convert_for_gpu_no_check(self):
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2)

            def forward(self, x):
                return self.fc(x)

        model = SimpleNet()
        dummy_input = torch.randn(1, 4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "test_model_nocheck.tflite")

            mock_convert_result = MagicMock()

            def fake_export(path):
                with open(path, "wb") as f:
                    f.write(b"dummy_tflite_bytes")

            mock_convert_result.export.side_effect = fake_export

            with patch.dict("sys.modules", {"litert_torch": MagicMock(convert=MagicMock(return_value=mock_convert_result))}):
                with patch("litert_gpu_toolkit.converter.check_gpu_compatibility") as mock_check:
                    res_path = convert_for_gpu(
                        model=model,
                        dummy_input=dummy_input,
                        output_path=out_file,
                        check=False,
                        verbose=False,
                    )

                    assert res_path == out_file
                    assert os.path.exists(out_file)
                    mock_check.assert_not_called()
