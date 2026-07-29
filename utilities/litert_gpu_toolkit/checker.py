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

The verdict comes from the LiteRT CompiledModel API itself: the model is
compiled for the GPU accelerator, every signature is run on random inputs,
and the outputs are compared against a CPU-compiled reference. The static
op lists below are advisory only — they point at the patch to apply when
GPU compilation fails, but they are not the source of truth and may lag
behind the runtime (e.g. 5D tensor support is still rolling out).
"""

import collections
import logging

import numpy as np

log = logging.getLogger("litert_gpu_toolkit")

# Advisory: ops that have historically failed on CompiledModel GPU (ML Drift).
# Used to suggest patches when GPU compilation fails — not to decide the verdict.
GPU_SUSPECT_OPS = {
    'GATHER_ND', 'GATHER', 'SELECT', 'SELECT_V2',
    'NOT_EQUAL', 'EQUAL', 'GREATER', 'LESS',
    'TOPK_V2', 'CAST', 'PACK', 'SPLIT',
}

# Ops that compile to a GPU delegate fallback rather than failing outright.
GPU_DELEGATED_OPS = {
    'BATCH_MATMUL',  # YOLO26 C2PSA attention; runs via delegate (verified on Mali-G715)
}

# Ops that need specific parameter settings
GPU_CONDITIONAL_OPS = {
    'RESIZE_BILINEAR': 'align_corners must be False',
}

# PyTorch modules known to cause GPU issues (for documentation)
# nn.GroupNorm → ManualGroupNorm (4D reshape approach)
# Conv2d_WS → bake standardized weights into regular Conv2d
# F.normalize → manual sqrt+div (div broadcast issue)
# nn.SiLU/Swish → x * sigmoid(x)
# nn.GELU → x * sigmoid(1.702 * x)


def _static_shape(shape) -> list:
    """Replace dynamic (-1/0) dims with 1 so buffers can be sized."""
    return [int(s) if int(s) > 0 else 1 for s in shape]


def _random_inputs(input_details: dict, rng) -> dict:
    """Build random input arrays keyed by input name.

    Floats get standard-normal noise; integer/bool inputs get zeros (random
    ints could be out of range for index-like inputs).
    """
    inputs = {}
    for name, detail in input_details.items():
        shape = _static_shape(detail['shape'])
        dtype = np.dtype(detail['dtype'])
        if dtype.kind == 'f':
            arr = rng.standard_normal(shape).astype(dtype)
        else:
            arr = np.zeros(shape, dtype=dtype)
        inputs[name] = arr
    return inputs


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


def _scan_ops(tflite_path: str) -> dict:
    """Best-effort static op scan (advisory diagnostics only).

    Uses the LiteRT interpreter when available, falling back to
    tf.lite.Interpreter. Returns an empty dict if neither is installed —
    the CompiledModel verification below does not depend on this.
    """
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        except ImportError:
            return {}

    interp = Interpreter(model_path=tflite_path)
    interp.allocate_tensors()

    details = interp._get_ops_details()
    op_counts = collections.Counter(d.get('op_name', 'UNKNOWN') for d in details)

    warnings = []
    # Rank-5+ tensors: warning only. GPU support for 5D tensors is still
    # rolling out, and some models run despite 5D+ intermediates.
    for detail in interp.get_tensor_details():
        shape = detail.get('shape', [])
        if len(shape) > 4:
            warnings.append(
                f"Tensor '{detail['name']}' has {len(shape)}D shape {list(shape)} — "
                f"rank-5+ GPU support is still rolling out"
            )

    return {
        'total_ops': len(details),
        'op_distribution': dict(op_counts.most_common()),
        'suspect_ops': {k: v for k, v in op_counts.items() if k in GPU_SUSPECT_OPS},
        'delegated_ops': {k: v for k, v in op_counts.items() if k in GPU_DELEGATED_OPS},
        'flex_ops': {k: v for k, v in op_counts.items() if 'Flex' in k},
        'warnings': warnings,
    }


def check_gpu_compatibility(
    tflite_path: str,
    rtol: float = 1e-2,
    atol: float = 1e-2,
    seed: int = 0,
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

    Returns:
        dict with keys:
            - 'compatible': bool — GPU compile + run succeeded on every
              signature and outputs matched CPU within tolerance
            - 'gpu_compile_ok': bool
            - 'gpu_cpu_fallback': bool — GPU-only compile failed but GPU|CPU
              succeeded (some ops fell back to CPU)
            - 'numerics_ok': bool | None — None when the GPU run never happened
            - 'max_abs_diff': float | None — worst output element across signatures
            - 'signatures': dict of {signature_key: {'ran', 'max_abs_diff', 'error'}}
            - 'errors': list of str
            - plus advisory static-scan keys: 'total_ops', 'op_distribution',
              'suspect_ops', 'delegated_ops', 'flex_ops', 'warnings'
    """
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator

    result = {
        'compatible': False,
        'gpu_compile_ok': False,
        'gpu_cpu_fallback': False,
        'numerics_ok': None,
        'max_abs_diff': None,
        'signatures': {},
        'errors': [],
        'total_ops': None,
        'op_distribution': {},
        'suspect_ops': {},
        'delegated_ops': {},
        'flex_ops': {},
        'warnings': [],
    }

    # Advisory static scan first, so diagnostics survive a compile failure.
    try:
        result.update(_scan_ops(tflite_path))
    except Exception as e:  # scan is best-effort by design
        result['warnings'].append(f"Static op scan failed: {e}")

    # CPU reference.
    try:
        cpu_model = CompiledModel.from_file(
            tflite_path, hardware_accel=HardwareAccelerator.CPU)
    except Exception as e:
        result['errors'].append(f"CPU compile failed: {e}")
        log.warning(f"CPU compile failed — cannot verify: {e}")
        return result

    # GPU compile: strict GPU first, then GPU|CPU (partial delegation).
    gpu_model = None
    try:
        gpu_model = CompiledModel.from_file(
            tflite_path, hardware_accel=HardwareAccelerator.GPU)
        result['gpu_compile_ok'] = True
    except Exception as gpu_only_err:
        try:
            gpu_model = CompiledModel.from_file(
                tflite_path,
                hardware_accel=HardwareAccelerator.GPU | HardwareAccelerator.CPU)
            result['gpu_compile_ok'] = True
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
    for signature_key in list(cpu_model.get_signature_list()):
        sig_result = {'ran': False, 'max_abs_diff': None, 'error': None}
        try:
            inputs = _random_inputs(
                cpu_model.get_input_tensor_details(signature_key), rng)
            cpu_out = _run_signature(cpu_model, signature_key, inputs)
            gpu_out = _run_signature(gpu_model, signature_key, inputs)
            sig_diff = 0.0
            numerics_ok = True
            for name, ref in cpu_out.items():
                got = gpu_out[name]
                if np.dtype(ref.dtype).kind == 'f':
                    diff = float(np.max(np.abs(got.astype(np.float64)
                                               - ref.astype(np.float64))))
                    sig_diff = max(sig_diff, diff)
                    if not np.allclose(got, ref, rtol=rtol, atol=atol):
                        numerics_ok = False
                elif not np.array_equal(got, ref):
                    numerics_ok = False
            sig_result.update(ran=True, max_abs_diff=sig_diff)
            if not numerics_ok:
                sig_result['error'] = (
                    f"outputs diverge from CPU (max abs diff {sig_diff:.3e})")
                result['errors'].append(
                    f"Signature '{signature_key}': {sig_result['error']}")
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

    return result


def print_report(result: dict) -> None:
    """Print a human-readable GPU verification report."""
    print(f"\n{'=' * 60}")
    print(f"  LiteRT CompiledModel GPU Verification Report")
    print(f"{'=' * 60}")

    if result['compatible']:
        residency = ("via GPU with CPU fallback" if result['gpu_cpu_fallback']
                     else "fully on GPU")
        print(f"  Status: VERIFIED ({residency})")
        print(f"  Max abs diff vs CPU: {result['max_abs_diff']:.3e}")
    else:
        print("  Status: FAILED")
        for err in result['errors']:
            print(f"    {err}")

    if result['signatures']:
        print(f"\n  Signatures:")
        for key, sig in result['signatures'].items():
            if sig['ran'] and sig['error'] is None:
                print(f"    {key}: OK (max abs diff {sig['max_abs_diff']:.3e})")
            else:
                print(f"    {key}: {sig['error']}")

    if result['suspect_ops']:
        print(f"\n  Suspect ops (advisory — see patches table in README):")
        for op, count in result['suspect_ops'].items():
            print(f"    {op}: {count}")
    if result['flex_ops']:
        print(f"\n  Flex ops (require TF delegate):")
        for op, count in result['flex_ops'].items():
            print(f"    {op}: {count}")
    if result['delegated_ops']:
        print(f"\n  Delegated ops (run via LiteRT GPU delegate):")
        for op, count in result['delegated_ops'].items():
            print(f"    {op}: {count}")

    if result['warnings']:
        print(f"\n  Warnings:")
        for w in result['warnings'][:10]:
            print(f"    {w}")

    if result['op_distribution']:
        print(f"\n  Op distribution (top 10):")
        for op, count in list(result['op_distribution'].items())[:10]:
            print(f"    {op}: {count}")
    print(f"{'=' * 60}\n")
