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
"""
Post-conversion GPU verification for TFLite models via LiteRT CompiledModel.

The model is compiled for the GPU accelerator, every signature is run on
random inputs, and the outputs are compared against a CPU-compiled
reference. When GPU compilation fails, the runtime's error message names
the offending op — the patches table in the README maps the common ones
to the rewrite that clears them.

When compilation succeeds but the outputs disagree, nothing in the runtime
names an op: the graph reports full residency and returns wrong numbers.
`bisect_gpu_divergence` finds the op. It cuts the graph after op k (the
prefix keeps ops 0..k and every tensor still live at the cut becomes a
graph output), compiles the prefix on CPU and GPU, and binary-searches k
for the first cut whose outputs disagree. The op at that cut is the first
op whose result the GPU gets wrong.

Command line:

    python checker.py model.tflite            # verify (all signatures)
    python checker.py model.tflite --bisect   # ... and bisect any divergence
    python checker.py model.tflite --bisect --enforce-f32 --uniform --json out.json

Only `numpy` and `ai-edge-litert` are needed to run this file directly.
"""

import argparse
import copy
import json
import logging
import os
import re
import shutil
import sys
import tempfile

import numpy as np

log = logging.getLogger("litert_gpu_toolkit")


# --------------------------------------------------------------------------
# Inputs, execution, comparison
# --------------------------------------------------------------------------

def _static_shape(shape) -> list:
    """Replace dynamic (-1/0) dims with 1 so buffers can be sized."""
    return [int(s) if int(s) > 0 else 1 for s in shape]


def _random_inputs(input_details: dict, rng, distribution: str = "normal",
                   int_high: int = 0) -> dict:
    """Build random input arrays keyed by input name.

    Floats get standard-normal noise (or uniform [0, 1) with
    `distribution="uniform"` — image models expect non-negative pixels, and
    the fp16 reduction overflow of LiteRT #9249 only shows on such inputs).
    Integer/bool inputs get zeros (random ints could be out of range for
    index-like inputs); with `int_high > 0` integer inputs get uniform ids in
    [0, int_high) instead, which a token-input model needs for its output to
    depend on the computation at all.
    """
    inputs = {}
    for name, detail in input_details.items():
        shape = _static_shape(detail['shape'])
        dtype = np.dtype(detail['dtype'])
        if dtype.kind == 'f':
            if distribution == "uniform":
                arr = rng.random(shape).astype(dtype)
            else:
                arr = rng.standard_normal(shape).astype(dtype)
        elif dtype.kind in 'iu' and int_high > 0:
            arr = rng.integers(0, int_high, size=shape).astype(dtype)
        else:
            arr = np.zeros(shape, dtype=dtype)
        inputs[name] = arr
    return inputs


def _resolve_inputs(signature_key, input_details, inputs, rng, distribution,
                    int_high: int = 0):
    """Pick the caller's arrays for this signature, or generate random ones.

    `inputs` may be `{signature_key: {name: array}}` or, for convenience,
    a plain `{name: array}` that is used for every signature whose input
    names it covers.
    """
    if inputs:
        chosen = inputs.get(signature_key, inputs)
        if all(name in chosen for name in input_details):
            return {name: np.asarray(chosen[name]) for name in input_details}
    return _random_inputs(input_details, rng, distribution, int_high)


def _run_signature(model, signature_key: str, inputs: dict) -> dict:
    """Run one signature through a CompiledModel and return output arrays."""
    output_details = model.get_output_tensor_details(signature_key)
    input_buffers = {
        name: model.create_input_buffer_by_name(signature_key, name)
        for name in inputs
    }
    output_buffers = {
        name: model.create_output_buffer_by_name(signature_key, name)
        for name in output_details
    }
    for name, arr in inputs.items():
        input_buffers[name].write(np.ascontiguousarray(arr))
    model.run_by_name(signature_key, input_buffers, output_buffers)
    outputs = {}
    for name, detail in output_details.items():
        shape = _static_shape(detail['shape'])
        outputs[name] = output_buffers[name].read(
            int(np.prod(shape)), np.dtype(detail['dtype'])).reshape(shape)
    return outputs


def _compile(tflite_path: str, accel, enforce_f32: bool = False):
    """Compile for an accelerator (GPU precision defaults to fp16)."""
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.gpu_options import GpuOptions
    from ai_edge_litert.options import Options
    return CompiledModel.from_file(
        tflite_path,
        options=Options(hardware_accelerators=accel,
                        gpu_options=GpuOptions(enforce_f32=enforce_f32)))


_RUNTIME_REASON = re.compile(
    r"(Failed to create DelegateKernel[^\n]*|Following operations are not supported[^\n]*"
    r"|^(?!WARNING|INFO|ERROR|VERBOSE)[A-Z][A-Z_0-9]+: [^\n]*"
    r"(not supported|Can't|Only support|Invalid|Expected|Unsupported|mismatch)[^\n]*"
    r"|\d+ operations will run on the GPU[^\n]*)", re.M)


class _CaptureStderr:
    """Redirect fd 2 to a file for the block; the runtime logs there, not
    to `sys.stderr`, and its compile-failure reason never reaches the Python
    exception. `.reasons()` returns the lines that name ops or errors."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryFile(mode="w+b")
        sys.stderr.flush()
        self.saved = os.dup(2)
        os.dup2(self.tmp.fileno(), 2)
        return self

    def _read(self) -> str:
        fd = self.tmp.fileno()
        os.lseek(fd, 0, os.SEEK_SET)
        chunks = []
        while True:
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                break
            chunks.append(chunk)
        os.lseek(fd, 0, os.SEEK_END)
        return b"".join(chunks).decode(errors="replace")

    def __exit__(self, *exc):
        sys.stderr.flush()
        os.dup2(self.saved, 2)
        os.close(self.saved)
        self.text = self._read()
        self.tmp.close()
        # Replay so the runtime's log stays visible when it was wanted.
        if log.isEnabledFor(logging.DEBUG):
            sys.stderr.write(self.text)
        return False

    def reasons(self) -> list:
        seen = []
        for m in _RUNTIME_REASON.finditer(self._read()):
            line = m.group(0).strip()
            if line not in seen:
                seen.append(line)
        return seen


def _compile_gpu(tflite_path: str, enforce_f32: bool = False,
                 allow_cpu_fallback: bool = True):
    """GPU compile: strict GPU first, then GPU|CPU (partial delegation).

    Returns (model, fallback_used, gpu_only_error). The runtime's own
    reason (the op it rejected, the kernel it failed to create) is appended
    to the error message and, on fallback, listed after the exception text.
    """
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator
    first_err = None
    with _CaptureStderr() as cap:
        try:
            model = _compile(tflite_path, HardwareAccelerator.GPU, enforce_f32)
            return model, False, None
        except Exception as e:
            reasons = cap.reasons()
            first_err = RuntimeError(
                f"{e}" + (": " + " | ".join(reasons) if reasons else ""))
    if not allow_cpu_fallback:
        raise first_err
    with _CaptureStderr() as cap:
        try:
            model = _compile(
                tflite_path, HardwareAccelerator.GPU | HardwareAccelerator.CPU,
                enforce_f32)
        except Exception as e:
            reasons = cap.reasons()
            raise RuntimeError(
                f"{e}" + (": " + " | ".join(reasons) if reasons else "")) from None
    return model, True, first_err


FP16_MAX = 65504.0


def _compare(ref: np.ndarray, got: np.ndarray, rtol: float, atol: float,
             fp16_range: bool = False) -> dict:
    """Compare a GPU tensor against its CPU reference.

    Floats: `np.allclose(got, ref, rtol, atol)` with NaN==NaN, so a NaN the
    CPU also produces is not a divergence. Integers/bools: exact equality.

    `nonfinite` counts GPU NaN/Inf where the CPU value is finite. With
    `fp16_range=True` (the GPU runs at fp16), those whose CPU value lies
    outside the fp16 range (|x| > 65504) are expected — an additive mask of
    -1e9 becomes -inf and a reduction past 65504 becomes inf — and only the
    rest, `nonfinite_in_range`, fail the tensor-scale test.
    """
    ref = np.asarray(ref)
    got = np.asarray(got)
    stats = {
        'ok': True, 'scale_ok': True, 'max_abs_diff': 0.0, 'max_rel_diff': 0.0,
        'scale': 0.0, 'mismatch_fraction': 0.0, 'nonfinite': 0,
        'nonfinite_in_range': 0,
    }
    if ref.shape != got.shape:
        stats.update(ok=False, scale_ok=False, error=f"shape {got.shape} vs {ref.shape}")
        return stats
    if np.dtype(ref.dtype).kind == 'f':
        r = ref.astype(np.float64)
        g = got.astype(np.float64)
        ref_finite = np.isfinite(r)
        bad = ~np.isfinite(g) & ref_finite
        stats['nonfinite'] = int(np.sum(bad))
        stats['nonfinite_in_range'] = int(np.sum(bad & (np.abs(r) <= FP16_MAX))) \
            if fp16_range else stats['nonfinite']
        both = ref_finite & np.isfinite(g)
        if np.any(both):
            diff = np.abs(g[both] - r[both])
            stats['max_abs_diff'] = float(diff.max())
            stats['max_rel_diff'] = float(
                (diff / np.maximum(np.abs(r[both]), 1e-12)).max())
            stats['scale'] = float(np.abs(r[both]).max())
        close = np.isclose(g, r, rtol=rtol, atol=atol, equal_nan=True)
        stats['mismatch_fraction'] = float(1.0 - close.mean()) if close.size else 0.0
        stats['ok'] = bool(close.all())
        # Tensor-scale criterion: the worst element against the tensor's
        # own magnitude. fp16 accumulation with cancellation leaves absolute
        # errors of rtol x (partial-sum magnitude) on elements that end near
        # zero, which the elementwise test flags on a correct kernel.
        stats['scale_ok'] = bool(
            stats['nonfinite_in_range'] == 0
            and stats['max_abs_diff'] <= atol + rtol * stats['scale'])
    else:
        eq = (got == ref)
        stats['mismatch_fraction'] = float(1.0 - eq.mean()) if eq.size else 0.0
        stats['ok'] = bool(eq.all())
        stats['scale_ok'] = stats['ok']
    return stats


# --------------------------------------------------------------------------
# Whole-model verification
# --------------------------------------------------------------------------

def check_gpu_compatibility(
    tflite_path: str,
    rtol: float = 1e-2,
    atol: float = 1e-2,
    seed: int = 0,
    inputs: dict = None,
    input_distribution: str = "normal",
    int_high: int = 0,
    enforce_f32: bool = False,
    bisect: bool = False,
    bisect_criterion: str = "auto",
    work_dir: str = None,
) -> dict:
    """Verify a TFLite model on the LiteRT CompiledModel GPU accelerator.

    Compiles the model for GPU, runs every signature on random inputs, and
    compares outputs against a CPU-compiled reference. Note this exercises
    the host GPU — an on-device run can still behave differently, so keep
    comparing device output against CPU before shipping.

    Args:
        tflite_path: Path to the .tflite file.
        rtol/atol: Elementwise tolerances for the GPU-vs-CPU comparison
            (fp16 accumulation on GPU makes bit-exactness unrealistic).
        seed: Seed for the random inputs.
        inputs: Optional `{signature_key: {input_name: array}}` (or a plain
            `{input_name: array}`) to run instead of random inputs.
        input_distribution: "normal" (default) or "uniform" for random floats.
        int_high: When > 0, integer inputs get random ids in [0, int_high)
            instead of zeros (token-input models).
        enforce_f32: Ask the GPU accelerator for fp32 compute. Re-running a
            divergent model with this on separates fp16 precision loss from
            a wrong kernel: a kernel bug reproduces at fp32 too.
        bisect: When a signature's outputs diverge, run
            `bisect_gpu_divergence` on it and attach the result under
            `result['bisect'][signature_key]`.
        bisect_criterion: Passed to `bisect_gpu_divergence` as `criterion`.
        work_dir: Scratch directory for the bisect's prefix models.

    Returns:
        dict with keys:
            - 'compatible': bool — GPU compile + run succeeded on every
              signature and outputs matched CPU within tolerance
            - 'gpu_compile_ok': bool
            - 'gpu_cpu_fallback': bool — GPU-only compile failed but GPU|CPU
              succeeded (some ops fell back to CPU)
            - 'gpu_fully_accelerated': bool | None — the runtime's own
              `is_fully_accelerated()` for the compiled model
            - 'numerics_ok': bool | None — None when the GPU run never happened
            - 'max_abs_diff': float | None — worst output element across signatures
            - 'signatures': dict of {signature_key: {'ran', 'max_abs_diff', 'error'}}
            - 'bisect': dict of {signature_key: bisect result} (only with bisect=True)
            - 'errors': list of str
            - 'warnings': list of str
    """
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator

    result = {
        'compatible': False,
        'gpu_compile_ok': False,
        'gpu_cpu_fallback': False,
        'gpu_fully_accelerated': None,
        'numerics_ok': None,
        'max_abs_diff': None,
        'signatures': {},
        'errors': [],
        'warnings': [],
    }

    # CPU reference.
    try:
        cpu_model = _compile(tflite_path, HardwareAccelerator.CPU)
    except Exception as e:
        result['errors'].append(f"CPU compile failed: {e}")
        log.warning(f"CPU compile failed — cannot verify: {e}")
        return result

    # GPU compile: strict GPU first, then GPU|CPU (partial delegation).
    try:
        gpu_model, fallback, gpu_only_err = _compile_gpu(tflite_path, enforce_f32)
        result['gpu_compile_ok'] = True
        try:
            result['gpu_fully_accelerated'] = bool(gpu_model.is_fully_accelerated())
        except Exception:
            pass
        if fallback:
            result['gpu_cpu_fallback'] = True
            result['warnings'].append(
                f"GPU-only compile failed ({gpu_only_err}); "
                f"compiled with CPU fallback instead"
            )
    except Exception as e:
        result['errors'].append(f"GPU compile failed: {e}")
        log.warning(f"GPU compile failed: {e}")
        return result

    # Run every signature on both accelerators and compare.
    rng = np.random.default_rng(seed)
    max_diff = 0.0
    all_ran = True
    diverged = []
    for signature_key in list(cpu_model.get_signature_list()):
        sig_result = {'ran': False, 'max_abs_diff': None, 'error': None}
        try:
            sig_inputs = _resolve_inputs(
                signature_key, cpu_model.get_input_tensor_details(signature_key),
                inputs, rng, input_distribution, int_high)
            cpu_out = _run_signature(cpu_model, signature_key, sig_inputs)
            gpu_out = _run_signature(gpu_model, signature_key, sig_inputs)
            sig_diff = 0.0
            numerics_ok = True
            nonfinite = 0
            for name, ref in cpu_out.items():
                stats = _compare(ref, gpu_out[name], rtol, atol)
                sig_diff = max(sig_diff, stats['max_abs_diff'])
                nonfinite += stats['nonfinite']
                numerics_ok = numerics_ok and stats['ok']
            sig_result.update(ran=True, max_abs_diff=sig_diff, nonfinite=nonfinite)
            if not numerics_ok:
                sig_result['error'] = (
                    f"outputs diverge from CPU (max abs diff {sig_diff:.3e}"
                    + (f", {nonfinite} non-finite" if nonfinite else "") + ")")
                result['errors'].append(
                    f"Signature '{signature_key}': {sig_result['error']}")
                diverged.append(signature_key)
            max_diff = max(max_diff, sig_diff)
        except Exception as e:
            all_ran = False
            sig_result['error'] = str(e)
            result['errors'].append(
                f"Signature '{signature_key}' failed on GPU: {e}")
        result['signatures'][signature_key] = sig_result

    ran_any = any(s['ran'] for s in result['signatures'].values())
    result['max_abs_diff'] = max_diff if ran_any else None
    result['numerics_ok'] = (
        all(s['ran'] and s['error'] is None
            for s in result['signatures'].values())
        if result['signatures'] else None
    )
    result['compatible'] = bool(
        result['gpu_compile_ok'] and all_ran and result['numerics_ok'])

    if result['compatible']:
        residency = ("with CPU fallback" if result['gpu_cpu_fallback']
                     else "fully on GPU")
        log.info(
            f"GPU verified {residency}: "
            f"{len(result['signatures'])} signature(s), "
            f"max abs diff vs CPU {max_diff:.3e}"
        )
    else:
        log.warning(f"GPU verification FAILED: {result['errors'][:3]}")

    if bisect and diverged:
        # Release the whole-model GPU compile before the prefix compiles.
        del gpu_model, cpu_model
        result['bisect'] = {}
        for signature_key in diverged:
            result['bisect'][signature_key] = bisect_gpu_divergence(
                tflite_path, signature_key=signature_key, rtol=rtol, atol=atol,
                seed=seed, inputs=inputs, input_distribution=input_distribution,
                int_high=int_high, enforce_f32=enforce_f32,
                criterion=bisect_criterion, work_dir=work_dir)

    return result


def print_report(result: dict) -> None:
    """Print a human-readable GPU verification report."""
    print(f"\n{'=' * 60}")
    print("  LiteRT CompiledModel GPU Verification Report")
    print(f"{'=' * 60}")

    if result['compatible']:
        residency = ("via GPU with CPU fallback" if result['gpu_cpu_fallback']
                     else "fully on GPU")
        if result.get('gpu_fully_accelerated') is False and not result['gpu_cpu_fallback']:
            residency = "GPU compile succeeded, runtime reports not fully accelerated"
        print(f"  Status: VERIFIED ({residency})")
        print(f"  Max abs diff vs CPU: {result['max_abs_diff']:.3e}")
    else:
        print("  Status: FAILED")
        for err in result['errors']:
            print(f"    {err}")

    if result['signatures']:
        print("\n  Signatures:")
        for key, sig in result['signatures'].items():
            if sig['ran'] and sig['error'] is None:
                print(f"    {key}: OK (max abs diff {sig['max_abs_diff']:.3e})")
            else:
                print(f"    {key}: {sig['error']}")

    if result['warnings']:
        print("\n  Warnings:")
        for w in result['warnings'][:10]:
            print(f"    {w}")
    print(f"{'=' * 60}\n")

    for key, b in (result.get('bisect') or {}).items():
        print_bisect_report(b)


# --------------------------------------------------------------------------
# Op-level bisect
# --------------------------------------------------------------------------

def _tensor_type_name(model_utils, tensor) -> str:
    try:
        return model_utils.type_to_name(tensor.type).lower()
    except Exception:
        return str(tensor.type)


def _describe_tensor(model, sg, index: int, fu) -> dict:
    t = sg.tensors[index]
    name = t.name.decode(errors="replace") if isinstance(t.name, (bytes, bytearray)) else str(t.name or "")
    buf = model.buffers[t.buffer] if t.buffer < len(model.buffers) else None
    has_data = buf is not None and buf.data is not None and len(buf.data) > 0
    return {
        'index': int(index),
        'name': name,
        'shape': [int(d) for d in (t.shape if t.shape is not None else [])],
        'dtype': _tensor_type_name(fu, t),
        'constant': bool(has_data),
    }


def _describe_op(model, sg, op_index: int, fu) -> dict:
    op = sg.operators[op_index]
    code = model.operatorCodes[op.opcodeIndex]
    name = fu.opcode_to_name(model, op.opcodeIndex)
    return {
        'index': int(op_index),
        'name': name,
        'version': int(code.version),
        'inputs': [_describe_tensor(model, sg, t, fu) for t in op.inputs if t >= 0],
        'outputs': [_describe_tensor(model, sg, t, fu) for t in op.outputs if t >= 0],
    }


def _constant_derived(model, sg) -> set:
    """Tensors that hold weights: constants, or produced only from constants.

    A weight-only quantized graph carries DEQUANTIZE ops whose input is a
    constant; their outputs are weights too, and asking the GPU to emit a
    weight as a graph output fails the run. They are not candidates for a
    numerical fault either, so the bisect neither cuts at such ops nor
    compares such tensors.
    """
    const = set()
    for i, t in enumerate(sg.tensors):
        buf = model.buffers[t.buffer] if t.buffer < len(model.buffers) else None
        if buf is not None and buf.data is not None and len(buf.data) > 0:
            const.add(i)
    for op in sg.operators:
        ins = [t for t in op.inputs if t >= 0]
        if ins and all(t in const for t in ins):
            const.update(t for t in op.outputs if t >= 0)
    return const


def _runtime_ops(sg, const: set) -> list:
    """Indices of ops that consume at least one runtime tensor."""
    return [i for i, op in enumerate(sg.operators)
            if any(t >= 0 and t not in const for t in op.inputs)]


def _frontier_tensors(sg, k: int, const: set = frozenset()) -> list:
    """Tensors produced by ops 0..k that are still needed after the cut.

    That is every tensor produced by an op at or before k which is consumed
    by an op after k or is a graph output — plus op k's own outputs, so the
    op at the cut is always observed. Weight-derived tensors (`const`) are
    left out. Order follows producer order.
    """
    produced_by = {}
    for i, op in enumerate(sg.operators[:k + 1]):
        for t in op.outputs:
            if t >= 0 and t not in const:
                produced_by[t] = i
    consumed_later = set()
    for op in sg.operators[k + 1:]:
        for t in op.inputs:
            if t >= 0:
                consumed_later.add(t)
    graph_outputs = set(int(t) for t in sg.outputs)
    cut_outputs = set(int(t) for t in sg.operators[k].outputs if t >= 0)
    keep = [t for t in produced_by
            if t in consumed_later or t in graph_outputs or t in cut_outputs]
    return keep


def _load_model(tflite_path: str):
    from ai_edge_litert.tools import flatbuffer_utils as fu
    return fu, fu.read_model(tflite_path)


def _signature_subgraph(model, signature_key):
    """Return (signature_def, subgraph_index) for a signature key."""
    sigs = model.signatureDefs or []
    if not sigs:
        raise ValueError("model has no signature defs; bisect needs one")
    if signature_key is None:
        sd = sigs[0]
    else:
        wanted = signature_key.encode() if isinstance(signature_key, str) else signature_key
        matches = [s for s in sigs if s.signatureKey == wanted]
        if not matches:
            raise ValueError(f"signature '{signature_key}' not in model")
        sd = matches[0]
    return sd, int(sd.subgraphIndex)


def _write_prefix(fu, model, sg_index: int, sig_def, k: int, out_path: str,
                  const: set = frozenset()) -> list:
    """Write the model truncated after op k of `sg_index`; return frontier.

    The truncated subgraph keeps ops 0..k, its inputs, and exposes every
    frontier tensor as a graph output named `bisect_t<index>` in the
    signature. Buffers (weights) are shared with the loaded model, not
    copied. Returns [(tensor_index, output_name), ...].

    A frontier tensor that some op inside the prefix consumes is exposed
    through a RESHAPE copy rather than directly. Making such a tensor a
    graph output would create the "output that is also consumed" pattern
    of LiteRT #8599, which the Metal accelerator gets wrong on its own
    (measured on 2.2.0: `ADD` output consumed by `SUM` reads back off by
    3.85 at both precisions; through a same-shape RESHAPE it is exact), and
    the bisect would then report a divergence it caused itself. Tensors
    that were graph outputs in the original model keep their native wiring,
    so a native #8599 case is still observed.
    """
    from ai_edge_litert import schema_py_generated as schema
    sg = model.subgraphs[sg_index]
    frontier = _frontier_tensors(sg, k, const)
    native_outputs = set(int(t) for t in sg.outputs)
    consumed_inside = set()
    for op in sg.operators[:k + 1]:
        consumed_inside.update(int(t) for t in op.inputs if t >= 0)

    new_sg = copy.copy(sg)
    new_sg.tensors = list(sg.tensors)
    new_sg.operators = list(sg.operators[:k + 1])
    new_buffers = list(model.buffers)
    new_opcodes = list(model.operatorCodes)
    reshape_code = None
    for i, code in enumerate(new_opcodes):
        if fu.get_builtin_code_from_operator_code(code) == schema.BuiltinOperator.RESHAPE:
            reshape_code = i
            break
    outputs = []
    named = []
    for t in frontier:
        t = int(t)
        if t in consumed_inside and t not in native_outputs:
            if reshape_code is None:
                code = schema.OperatorCodeT()
                code.builtinCode = schema.BuiltinOperator.RESHAPE
                code.deprecatedBuiltinCode = schema.BuiltinOperator.RESHAPE
                new_opcodes.append(code)
                reshape_code = len(new_opcodes) - 1
            src = sg.tensors[t]
            shape = [int(d) for d in (src.shape if src.shape is not None else [])]
            shape_buf = schema.BufferT()
            shape_buf.data = np.frombuffer(
                np.asarray(shape, np.int32).tobytes(), np.uint8)
            new_buffers.append(shape_buf)
            shape_t = schema.TensorT()
            shape_t.name = f"bisect_shape{t}".encode()
            shape_t.shape = [len(shape)]
            shape_t.type = schema.TensorType.INT32
            shape_t.buffer = len(new_buffers) - 1
            new_sg.tensors.append(shape_t)
            shape_idx = len(new_sg.tensors) - 1
            copy_t = schema.TensorT()
            copy_t.name = f"bisect_copy{t}".encode()
            copy_t.shape = list(shape)
            copy_t.type = src.type
            copy_t.buffer = 0
            copy_t.quantization = src.quantization
            new_sg.tensors.append(copy_t)
            copy_idx = len(new_sg.tensors) - 1
            op = schema.OperatorT()
            op.opcodeIndex = reshape_code
            op.inputs = [t, shape_idx]
            op.outputs = [copy_idx]
            op.builtinOptionsType = schema.BuiltinOptions.ReshapeOptions
            op.builtinOptions = schema.ReshapeOptionsT()
            op.builtinOptions.newShape = list(shape)
            new_sg.operators.append(op)
            outputs.append(copy_idx)
        else:
            outputs.append(t)
        tm = schema.TensorMapT()
        tm.name = f"bisect_t{t}".encode()
        tm.tensorIndex = outputs[-1]
        named.append((t, f"bisect_t{t}", tm))
    new_sg.outputs = outputs

    new_sig = copy.copy(sig_def)
    new_sig.outputs = [tm for _, _, tm in named]

    new_model = copy.copy(model)
    new_model.subgraphs = list(model.subgraphs)
    new_model.subgraphs[sg_index] = new_sg
    new_model.buffers = new_buffers
    new_model.operatorCodes = new_opcodes
    new_model.signatureDefs = [new_sig]
    fu.write_model(new_model, out_path)
    return [(t, name) for t, name, _ in named]


def bisect_gpu_divergence(
    tflite_path: str,
    signature_key: str = None,
    rtol: float = 1e-2,
    atol: float = 1e-2,
    seed: int = 0,
    inputs: dict = None,
    input_distribution: str = "normal",
    int_high: int = 0,
    enforce_f32: bool = False,
    criterion: str = "auto",
    work_dir: str = None,
    max_probes: int = 64,
) -> dict:
    """Find the first op whose GPU result diverges from CPU.

    The graph is cut after op k: the prefix keeps ops 0..k and every tensor
    still live at the cut (consumed later, or a graph output, or produced
    by op k) becomes a graph output. The prefix is compiled on CPU and on
    GPU with the same inputs and the frontier tensors are compared. Binary
    search over k finds the first cut that disagrees; the op at that cut
    is the first op the GPU gets wrong.

    Two things the result says explicitly, because they are what a bisect
    can and cannot claim:

    - The divergence is *first observed* at op k: the prefix ending one op
      earlier matched CPU within tolerance. Later ops are not examined; a
      graph may contain more than one wrong op.
    - If the diverging frontier tensor was produced by an *earlier* op than
      k — it was correct while it was the end of the graph and became wrong
      once op k consumed it — the result marks it `context_dependent`. That
      is the shape of LiteRT #8599 (an ADD that is both output and consumed
      returns an operand), and it means the fault is in how the runtime
      handles the pair, not in op k's arithmetic.

    Args:
        tflite_path: Path to the .tflite file.
        signature_key: Signature to bisect (default: the model's first).
        rtol/atol: Tolerances, same meaning as `check_gpu_compatibility`.
        seed: Seed for random inputs.
        inputs: Optional inputs, same form as `check_gpu_compatibility`.
        input_distribution: "normal" or "uniform" random floats.
        int_high: Random integer inputs in [0, int_high) when > 0 (else zeros).
        enforce_f32: Ask the GPU for fp32 compute (see check_gpu_compatibility).
        criterion: What counts as a diverging frontier tensor.
            - "nonfinite": the GPU returns NaN/Inf where the CPU is finite.
              Overflow bugs (LiteRT #9249) poison every op downstream, so
              this is the symptom to chase when the final output is NaN.
            - "scale": the worst element differs by more than
              `atol + rtol * max|cpu|` of that tensor, or the GPU returns
              NaN/Inf for a CPU value inside the fp16 range. This is the
              default for finite divergences: intermediate tensors on a
              fp16 GPU carry rtol-sized errors relative to the partial sums
              that produced them, and an elementwise test flags those on a
              correct kernel (efficientnet_b0's first CONV_2D fails
              elementwise 1e-2 on Metal fp16 while the SUM eight ops later
              is the real fault). At fp16, a GPU Inf where the CPU value is
              itself beyond 65504 (an additive attention mask, a reduction
              total) is recorded under `nonfinite_sightings` rather than
              treated as the divergence, because the graph's own output was
              finite — with `enforce_f32` every such value counts.
            - "elementwise": `np.allclose(rtol, atol)`, the whole-model
              check's own test. Strictest; use with enforce_f32.
            - "auto": "nonfinite" if the full graph's GPU outputs contain
              NaN/Inf, else "scale".
        work_dir: Where prefix models are written (a temp dir by default,
            removed afterwards).
        max_probes: Safety cap on the number of prefix compiles.

    Returns:
        dict with keys:
            - 'signature': str
            - 'n_ops': int — ops in the signature's subgraph
            - 'diverges': bool — the full graph's outputs differ from CPU
            - 'first_divergent_op': dict | None — {'index', 'name', 'version',
              'inputs', 'outputs'} of the op at the first diverging cut
            - 'clean_prefix_end': int — last op index whose prefix matched
              CPU (-1 if op 0 already diverges)
            - 'diverging_tensors': list of {'tensor', 'name', 'shape', 'dtype',
              'producer_op', 'producer_name', 'context_dependent',
              'max_abs_diff', 'max_rel_diff', 'mismatch_fraction', 'nonfinite'}
              at the first diverging cut
            - 'probes': list of {'cut', 'status', 'max_abs_diff', 'nonfinite',
              'fallback', 'fully_accelerated', 'error'} in the order run
            - 'nonfinite_sightings': list of {'cut', 'tensor', 'producer_op',
              'producer_name', 'nonfinite'} — frontier tensors where the GPU
              returned NaN/Inf for CPU values beyond the fp16 range while the
              graph's output stayed finite (fp16 runs only)
            - 'enforce_f32': bool
            - 'criterion': str — the criterion actually used
            - 'errors': list of str
    """
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator

    if criterion not in ("auto", "nonfinite", "scale", "elementwise"):
        raise ValueError(f"unknown criterion {criterion!r}")
    result = {
        'signature': signature_key,
        'n_ops': None,
        'diverges': None,
        'first_divergent_op': None,
        'clean_prefix_end': None,
        'diverging_tensors': [],
        'probes': [],
        'nonfinite_sightings': [],
        'enforce_f32': bool(enforce_f32),
        'criterion': criterion,
        'errors': [],
    }

    try:
        fu, model = _load_model(tflite_path)
        sig_def, sg_index = _signature_subgraph(model, signature_key)
    except Exception as e:
        result['errors'].append(f"cannot load model for bisect: {e}")
        return result
    signature_key = sig_def.signatureKey.decode()
    result['signature'] = signature_key
    sg = model.subgraphs[sg_index]
    n_ops = len(sg.operators)
    result['n_ops'] = n_ops
    const = _constant_derived(model, sg)
    candidates = _runtime_ops(sg, const)   # cut points, in graph order
    if not candidates:
        result['errors'].append("subgraph has no operators on runtime data")
        return result

    # Inputs come from the original model's signature, once.
    try:
        ref_model = _compile(tflite_path, HardwareAccelerator.CPU)
        rng = np.random.default_rng(seed)
        sig_inputs = _resolve_inputs(
            signature_key, ref_model.get_input_tensor_details(signature_key),
            inputs, rng, input_distribution, int_high)
        del ref_model
    except Exception as e:
        result['errors'].append(f"CPU compile of the model failed: {e}")
        return result

    own_work_dir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="litert_bisect_")
    os.makedirs(work_dir, exist_ok=True)

    def probe(k: int) -> dict:
        """Compile prefix 0..k on CPU and GPU, compare its frontier."""
        rec = {'cut': int(k), 'status': None, 'max_abs_diff': 0.0,
               'nonfinite': 0, 'fallback': False, 'fully_accelerated': None,
               'error': None, 'tensors': []}
        path = os.path.join(work_dir, f"prefix_{k:05d}.tflite")
        try:
            named = _write_prefix(fu, model, sg_index, sig_def, k, path, const)
            cpu = _compile(path, HardwareAccelerator.CPU)
            cpu_out = _run_signature(cpu, signature_key, sig_inputs)
            del cpu
            gpu, fallback, _ = _compile_gpu(path, enforce_f32)
            rec['fallback'] = bool(fallback)
            try:
                rec['fully_accelerated'] = bool(gpu.is_fully_accelerated())
            except Exception:
                pass
            gpu_out = _run_signature(gpu, signature_key, sig_inputs)
            del gpu
            diverged = False
            for t_index, out_name in named:
                stats = _compare(cpu_out[out_name], gpu_out[out_name], rtol, atol,
                                 fp16_range=not enforce_f32)
                rec['max_abs_diff'] = max(rec['max_abs_diff'], stats['max_abs_diff'])
                rec['nonfinite'] += stats['nonfinite']
                if (result['criterion'] == "scale"
                        and stats['nonfinite'] > stats['nonfinite_in_range']):
                    producer = next(
                        (i for i, op in enumerate(sg.operators[:k + 1])
                         if t_index in list(op.outputs)), None)
                    result['nonfinite_sightings'].append({
                        'cut': int(k), 'tensor': int(t_index),
                        'producer_op': producer,
                        'producer_name': (fu.opcode_to_name(
                            model, sg.operators[producer].opcodeIndex)
                            if producer is not None else None),
                        'nonfinite': stats['nonfinite'] - stats['nonfinite_in_range']})
                if result['criterion'] == "nonfinite":
                    bad = stats['nonfinite'] > 0 or 'error' in stats
                elif result['criterion'] == "elementwise":
                    bad = not stats['ok']
                else:
                    bad = not stats['scale_ok']
                if bad:
                    diverged = True
                    desc = _describe_tensor(model, sg, t_index, fu)
                    producer = next(
                        (i for i, op in enumerate(sg.operators[:k + 1])
                         if t_index in list(op.outputs)), None)
                    desc.update(
                        tensor=t_index,
                        producer_op=producer,
                        producer_name=(fu.opcode_to_name(
                            model, sg.operators[producer].opcodeIndex)
                            if producer is not None else None),
                        context_dependent=bool(producer is not None and producer != k),
                        **{key: stats[key] for key in (
                            'max_abs_diff', 'max_rel_diff', 'scale',
                            'mismatch_fraction', 'nonfinite', 'nonfinite_in_range')})
                    desc.pop('index', None)
                    rec['tensors'].append(desc)
            rec['status'] = 'diverged' if diverged else 'clean'
        except Exception as e:
            rec['status'] = 'error'
            rec['error'] = str(e)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        log.info(
            f"bisect cut {k}/{n_ops - 1} ({fu.opcode_to_name(model, sg.operators[k].opcodeIndex)}): "
            f"{rec['status']} max abs diff {rec['max_abs_diff']:.3e}"
            + (f" nonfinite {rec['nonfinite']}" if rec['nonfinite'] else "")
            + (f" [{rec['error']}]" if rec['error'] else ""))
        result['probes'].append({key: rec[key] for key in rec if key != 'tensors'})
        return rec

    def probe_near(pos: int, lo_pos: int, hi_pos: int):
        """Probe candidates[pos]; on a compile/run error walk outward.

        A prefix can fail to compile on the GPU for reasons of its own (the
        Metal backend rejects some cuts with a shader type error), and such
        cuts come in bands, so the walk doubles its step: +-1, +-2, +-4 ...
        """
        rec = probe(candidates[pos])
        step = 1
        while rec['status'] == 'error' and len(result['probes']) < max_probes:
            near = [c for c in (pos + step, pos - step) if lo_pos < c < hi_pos]
            if not near:
                return rec
            for c in near:
                rec = probe(candidates[c])
                if rec['status'] != 'error':
                    break
            step *= 2
        return rec

    try:
        # The full graph must diverge, else there is nothing to bisect.
        if criterion == "auto":
            result['criterion'] = "scale"      # decided after the first probe
        full = probe(candidates[-1])
        if full['status'] == 'error':
            result['errors'].append(f"full-graph probe failed: {full['error']}")
            if os.path.getsize(tflite_path) >= 2 ** 31:
                result['errors'].append(
                    "the model is over 2 GiB: its weights use the buffer-offset "
                    "layout, which the wheel's flatbuffer writer does not emit, so "
                    "prefix models cannot be written")
            return result
        if criterion == "auto" and full['nonfinite'] > 0:
            result['criterion'] = "nonfinite"
            full['status'] = 'diverged'
        result['diverges'] = full['status'] == 'diverged'
        if not result['diverges']:
            result['errors'].append(
                f"full-graph frontier matches CPU under the '{result['criterion']}' "
                f"criterion (max abs diff {full['max_abs_diff']:.3e}); nothing to bisect. "
                "If check_gpu_compatibility reported a divergence, the elementwise "
                "mismatch is within rtol of the tensor's scale — fp16 noise, not a "
                "wrong kernel; confirm with enforce_f32=True or criterion='elementwise'.")
            return result

        # Binary search over positions in `candidates`:
        # lo_pos is clean (or -1 = the empty prefix), hi_pos diverged.
        lo_pos, hi_pos = -1, len(candidates) - 1
        hi_rec = full
        while hi_pos - lo_pos > 1 and len(result['probes']) < max_probes:
            rec = probe_near((lo_pos + hi_pos) // 2, lo_pos, hi_pos)
            if rec['status'] == 'error':
                result['errors'].append(
                    f"could not run any prefix between ops {candidates[lo_pos]} "
                    f"and {candidates[hi_pos]}: {rec['error']}")
                break
            pos = candidates.index(rec['cut'])
            if rec['status'] == 'diverged':
                hi_pos, hi_rec = pos, rec
            else:
                lo_pos = pos

        hi = candidates[hi_pos]
        result['clean_prefix_end'] = candidates[lo_pos] if lo_pos >= 0 else -1
        result['first_divergent_op'] = _describe_op(model, sg, hi, fu)
        result['diverging_tensors'] = hi_rec['tensors']
        failed = [p for p in result['probes'] if p['status'] == 'error']
        if failed:
            result['errors'].append(
                f"{len(failed)} prefix cut(s) could not be compiled or run on the GPU "
                f"(ops {min(p['cut'] for p in failed)}..{max(p['cut'] for p in failed)}; "
                f"first error: {failed[0]['error'][:200]})")
        if hi_pos - lo_pos > 1:
            result['errors'].append(
                f"bisect stopped early: ops {candidates[lo_pos + 1]}..{candidates[hi_pos - 1]} "
                "were not probed")
    finally:
        if own_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
    return result


def print_bisect_report(result: dict) -> None:
    """Print a human-readable bisect report."""
    print(f"\n{'-' * 60}")
    print(f"  GPU divergence bisect — signature '{result['signature']}'"
          f" ({'fp32' if result.get('enforce_f32') else 'fp16'} GPU compute,"
          f" criterion: {result.get('criterion')})")
    print(f"{'-' * 60}")
    op = result.get('first_divergent_op')
    if op is None:
        print("  No op isolated.")
        for err in result['errors']:
            print(f"    {err}")
        print(f"{'-' * 60}\n")
        return

    def fmt(t):
        c = " const" if t.get('constant') else ""
        return f"{t['dtype']}{t['shape']}{c}"

    print(f"  First divergent op: #{op['index']} {op['name']} (v{op['version']})"
          f" of {result['n_ops']} ops")
    print("    inputs : " + ", ".join(fmt(t) for t in op['inputs']))
    print("    outputs: " + ", ".join(fmt(t) for t in op['outputs']))
    print(f"    prefix ending at op #{result['clean_prefix_end']} matched CPU")
    print("  Diverging tensors at that cut:")
    for t in result['diverging_tensors']:
        ctx = (f"  (produced by op #{t['producer_op']} {t['producer_name']};"
               f" was correct before op #{op['index']} was appended)"
               if t.get('context_dependent') else "")
        nf_in = t.get('nonfinite_in_range', t.get('nonfinite', 0))
        nf_out = t.get('nonfinite', 0) - nf_in
        nf = (f", {nf_in} non-finite" if nf_in else "") + (
            f", {nf_out} non-finite where CPU exceeds the fp16 range" if nf_out else "")
        name = (t['name'] or '')[:60]
        print(f"    t{t['tensor']} {name} {fmt(t)}: "
              f"max abs diff {t['max_abs_diff']:.3e} (tensor scale {t.get('scale', 0):.3e}), "
              f"{t['mismatch_fraction'] * 100:.1f}% outside elementwise tolerance{nf}{ctx}")
    if any(p['fallback'] for p in result['probes']):
        print("  Note: some prefixes needed CPU fallback to compile; the op at "
              "those cuts may have run on CPU.")
    sightings = result.get('nonfinite_sightings') or []
    if sightings:
        first = min(sightings, key=lambda x: x['cut'])
        print(f"  Note: GPU NaN/Inf for CPU values beyond the fp16 range, first at "
              f"op #{first['producer_op']} {first['producer_name']} "
              f"({first['nonfinite']} values, cut #{first['cut']}); the graph output "
              f"stayed finite. If the output is wrong from there, that op's fp16 "
              f"accumulation is the cause (LiteRT #9249) — confirm with enforce_f32.")
    print(f"  Probes ({len(result['probes'])}): " + ", ".join(
        f"#{p['cut']}:{p['status'][0]}" for p in result['probes']))
    for err in result['errors']:
        print(f"    {err}")
    print(f"{'-' * 60}\n")


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a .tflite on the LiteRT CompiledModel GPU accelerator "
                    "against CPU, and bisect the first divergent op.")
    parser.add_argument("tflite")
    parser.add_argument("--bisect", action="store_true",
                        help="on divergence, isolate the first op whose GPU result differs")
    parser.add_argument("--signature", default=None,
                        help="bisect only this signature (default: every diverging one)")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--uniform", action="store_true",
                        help="uniform [0,1) random float inputs instead of standard normal")
    parser.add_argument("--int-high", type=int, default=0,
                        help="random integer inputs in [0, N) instead of zeros (token ids)")
    parser.add_argument("--inputs", default=None,
                        help=".npz of input arrays keyed by input name")
    parser.add_argument("--enforce-f32", action="store_true",
                        help="ask the GPU accelerator for fp32 compute")
    parser.add_argument("--criterion", default="auto",
                        choices=["auto", "nonfinite", "scale", "elementwise"],
                        help="what counts as a diverging tensor during the bisect")
    parser.add_argument("--work-dir", default=None,
                        help="keep prefix models here instead of a temp dir")
    parser.add_argument("--json", default=None, help="write the result dict here")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(name)s: %(message)s")
    inputs = None
    if args.inputs:
        with np.load(args.inputs) as npz:
            inputs = {k: npz[k] for k in npz.files}
    distribution = "uniform" if args.uniform else "normal"

    if args.bisect and args.signature:
        result = bisect_gpu_divergence(
            args.tflite, signature_key=args.signature, rtol=args.rtol, atol=args.atol,
            seed=args.seed, inputs=inputs, input_distribution=distribution,
            int_high=args.int_high, enforce_f32=args.enforce_f32,
            criterion=args.criterion, work_dir=args.work_dir)
        print_bisect_report(result)
        ok = result.get('first_divergent_op') is None and result.get('diverges') is False
    else:
        result = check_gpu_compatibility(
            args.tflite, rtol=args.rtol, atol=args.atol, seed=args.seed,
            inputs=inputs, input_distribution=distribution, int_high=args.int_high,
            enforce_f32=args.enforce_f32, bisect=args.bisect,
            bisect_criterion=args.criterion, work_dir=args.work_dir)
        print_report(result)
        ok = result['compatible']

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1, default=str)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main())
