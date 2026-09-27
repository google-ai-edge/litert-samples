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
"""Pytest configuration and global isolation fixtures for litert_gpu_toolkit tests."""

import pathlib
import sys
import pytest
import torch.nn as nn
import torch.nn.functional as F

# Ensure utilities is in python path
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
_UTILITIES_DIR = _REPO_ROOT / "utilities"
if str(_UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILITIES_DIR))

from litert_gpu_toolkit.patches import (
    restore_gelu,
    restore_grid_sample,
    restore_interpolate,
    restore_normalize,
)

_ORIGINAL_LAYERNORM_FORWARD = nn.LayerNorm.forward


@pytest.fixture(autouse=True)
def clean_functional_state():
    """Guarantee all global monkey-patches are reverted after each test."""
    yield
    restore_gelu()
    restore_interpolate()
    restore_normalize()
    restore_grid_sample()
    nn.LayerNorm.forward = _ORIGINAL_LAYERNORM_FORWARD
