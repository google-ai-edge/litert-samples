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

"""Pure helpers of litert_runtime: parsing tensor metadata and reading a buffer.

No real hardware and no model loading — these are pure-function unit tests
that complement live CompiledRunner runs on the board (the detector/ASR are
tested live, not here).
"""

import numpy as np

from emulator.litert_runtime import _meta, _read


# --- _meta: normalizing shape/dtype from tensor details ---

def test_meta_normalizes_shape_to_tuple_of_int():
    detail = {"shape": [1, 3, 4], "dtype": np.float32}
    meta = _meta(detail)
    assert meta["shape"] == (1, 3, 4)
    assert all(isinstance(x, int) for x in meta["shape"])


def test_meta_accepts_numpy_int_shape_entries():
    # LiteRT returns shape as a numpy array/numpy int, not a plain int —
    # without an explicit cast, np.prod(shape) can later misbehave with
    # incompatible types, so _meta casts them to int right away.
    detail = {"shape": np.array([1, 64], dtype=np.int64), "dtype": np.int32}
    meta = _meta(detail)
    assert meta["shape"] == (1, 64)
    assert all(isinstance(x, int) for x in meta["shape"])


def test_meta_wraps_dtype_as_numpy_dtype():
    detail = {"shape": [2], "dtype": np.float32}
    meta = _meta(detail)
    assert meta["dtype"] == np.dtype(np.float32)
    assert isinstance(meta["dtype"], np.dtype)


# --- _read: flat buffer -> ndarray of the target shape ---

class FakeBuffer:
    """Stub for a LiteRT buffer: read(count, dtype) -> flat ndarray."""

    def __init__(self, flat: np.ndarray):
        self._flat = flat

    def read(self, count, dtype):
        assert count == self._flat.size, "requested the wrong element count"
        return self._flat.astype(dtype)


def test_read_reshapes_flat_buffer_to_meta_shape():
    meta = {"shape": (2, 3), "dtype": np.dtype(np.float32)}
    flat = np.arange(6, dtype=np.float32)
    out = _read(FakeBuffer(flat), meta)
    assert out.shape == (2, 3)
    assert (out == flat.reshape(2, 3)).all()


def test_read_computes_count_as_product_of_shape():
    # yolox: [1, 3549, 85] -> 301665 elements in a single flat read.
    meta = {"shape": (1, 3549, 85), "dtype": np.dtype(np.float32)}
    flat = np.zeros(1 * 3549 * 85, dtype=np.float32)
    out = _read(FakeBuffer(flat), meta)
    assert out.shape == (1, 3549, 85)


def test_read_passes_meta_dtype_type_to_buffer_read():
    calls = {}

    class RecordingBuffer:
        def read(self, count, dtype):
            calls["count"] = count
            calls["dtype"] = dtype
            return np.zeros(count, dtype=dtype)

    meta = {"shape": (1, 4), "dtype": np.dtype(np.int32)}
    _read(RecordingBuffer(), meta)
    assert calls["count"] == 4
    assert calls["dtype"] == np.int32
