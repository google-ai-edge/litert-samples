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

"""Thin wrapper over the LiteRT CompiledModel API (the current API; Interpreter
is deprecated).

A model is called through one of two signature adapters, both exposing the same
tiny interface — `get_input_details() -> {name: {"shape","dtype"}}` and
`__call__(**inputs) -> {name: ndarray}` — so a stage swaps `Interpreter` for this
with almost no change to its call site, and callers that classify inputs by
shape/dtype (moonshine's `classify_decode_inputs`) keep working unchanged.

Two adapters, because exported `.tflite` files come in two flavours:
- **named signatures** (moonshine: `encode`, `decode`) — driven by `run_by_name`;
- **a single default subgraph, no named signatures** — driven by
  `run_by_index(0, ...)`, the same path `demo/gpu_detect.py` uses for the detector.

Only fixed-shape models belong here. The Inflect CPU synthesizer is dynamic-shape
(it resizes its input per sentence), which is what CompiledModel is *not* built
for, so it stays on Interpreter — that dynamic shape is exactly why a fixed-chunk
re-export is needed to move it onto CompiledModel.

Buffers are created once and reused; `write()` overwrites them, which matters for
the moonshine decoder that runs its `decode` signature ~5–10× per utterance.
"""

from __future__ import annotations

import numpy as np
from ai_edge_litert.compiled_model import CompiledModel, HardwareAccelerator
from ai_edge_litert.options import CpuOptions, Options


def _meta(detail) -> dict:
    return {"shape": tuple(int(x) for x in detail["shape"]),
            "dtype": np.dtype(detail["dtype"])}


def _read(buffer, meta) -> np.ndarray:
    count = int(np.prod(meta["shape"]))
    flat = np.asarray(buffer.read(count, meta["dtype"].type))
    return flat.reshape(meta["shape"])


class _NamedSignature:
    """One named signature (moonshine encode/decode), via run_by_name."""

    def __init__(self, model: CompiledModel, key: str) -> None:
        self._model = model
        self._key = key
        sig = model.get_signature_list()[key]
        self._in_names = list(sig["inputs"])
        self._out_names = list(sig["outputs"])
        # get_*_tensor_details(key) returns {signature_name: detail}.
        in_det = model.get_input_tensor_details(key)
        out_det = model.get_output_tensor_details(key)
        self._in = {n: _meta(in_det[n]) for n in self._in_names}
        self._out = {n: _meta(out_det[n]) for n in self._out_names}
        self._in_bufs = {n: model.create_input_buffer_by_name(key, n)
                         for n in self._in_names}
        self._out_bufs = {n: model.create_output_buffer_by_name(key, n)
                          for n in self._out_names}

    def get_input_details(self) -> dict[str, dict]:
        return self._in

    def __call__(self, **inputs) -> dict[str, np.ndarray]:
        for name, array in inputs.items():
            self._in_bufs[name].write(np.ascontiguousarray(array))
        self._model.run_by_name(self._key, self._in_bufs, self._out_bufs)
        return {n: _read(self._out_bufs[n], self._out[n]) for n in self._out_names}


class _IndexSignature:
    """The single default subgraph, no named signature, via run_by_index(0).

    Inputs and outputs are addressed positionally; buffer.get_tensor_details()
    carries the shape and dtype. Names are synthesized (`in0`, `out0`, …) so the
    interface matches _NamedSignature — a single-input model reads back one key.

    FACTUAL NOTE: the real yolo26n export used by `Detector` in this repo
    DOES have a named signature (`serving_default`), so `CompiledRunner.only()`
    returns `_NamedSignature` for it (the `len(self._names) == 1` branch), not
    this class. This path exists for the general no-named-signature case and is
    currently exercised only by tests that construct a model with no
    signature list, not by the real detector model.
    """

    def __init__(self, model: CompiledModel, index: int = 0) -> None:
        self._model = model
        self._index = index
        self._in_bufs = model.create_input_buffers(index)
        self._out_bufs = model.create_output_buffers(index)
        self._in_names = [f"in{i}" for i in range(len(self._in_bufs))]
        self._out_names = [f"out{i}" for i in range(len(self._out_bufs))]
        self._in = {n: _meta(b.get_tensor_details())
                    for n, b in zip(self._in_names, self._in_bufs)}
        self._out = {n: _meta(b.get_tensor_details())
                     for n, b in zip(self._out_names, self._out_bufs)}
        self._in_by_name = dict(zip(self._in_names, self._in_bufs))

    def get_input_details(self) -> dict[str, dict]:
        return self._in

    def __call__(self, **inputs) -> dict[str, np.ndarray]:
        for name, array in inputs.items():
            self._in_by_name[name].write(np.ascontiguousarray(array))
        self._model.run_by_index(self._index, self._in_bufs, self._out_bufs)
        return {n: _read(b, self._out[n])
                for n, b in zip(self._out_names, self._out_bufs)}


class CompiledRunner:
    """A CompiledModel plus its signatures.

    `signature(key)` for a named signature (moonshine); `only()` for a
    single-signature or single-default-subgraph model (the YOLO detector),
    which spares the caller from knowing whether the export named its signature.
    """

    def __init__(self, model_path,
                 accel: HardwareAccelerator = HardwareAccelerator.CPU,
                 threads: int = 4) -> None:
        # CompiledModel defaults to CpuOptions(num_threads=1); the old
        # Interpreter path ran 4 threads. Without this the CPU stages regress
        # ~1.75–2.6× (detector 181→316 ms). Pass the thread count explicitly.
        if accel == HardwareAccelerator.CPU:
            options = Options(hardware_accelerators=HardwareAccelerator.CPU,
                              cpu_options=CpuOptions(num_threads=threads))
            self._model = CompiledModel.from_file(str(model_path), options=options)
        else:
            self._model = CompiledModel.from_file(str(model_path),
                                                  hardware_accel=accel)
        self._names = list(self._model.get_signature_list())

    def signature(self, key: str) -> _NamedSignature:
        return _NamedSignature(self._model, key)

    def only(self):
        if len(self._names) == 1:
            return _NamedSignature(self._model, self._names[0])
        if not self._names:
            return _IndexSignature(self._model, 0)     # default subgraph, no names
        raise ValueError(
            f"only() needs one signature, model has {self._names}")

    def is_fully_accelerated(self) -> bool:
        return self._model.is_fully_accelerated()
