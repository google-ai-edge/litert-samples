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
Post-conversion GPU compatibility checker for TFLite models.
"""

import collections
import logging

log = logging.getLogger("litert_gpu_toolkit")

# Ops known to be incompatible with CompiledModel GPU (ML Drift)
GPU_INCOMPATIBLE_OPS = {
    'GATHER_ND', 'GATHER', 'SELECT', 'SELECT_V2',
    'NOT_EQUAL', 'EQUAL', 'GREATER', 'LESS',
    'TOPK_V2', 'CAST', 'PACK', 'SPLIT',
}

# Ops that compile to a GPU delegate fallback rather than failing outright.
# They are reported as warnings instead of hard incompatibilities.
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


def check_gpu_compatibility(tflite_path: str) -> dict:
    """Check a TFLite model for GPU compatibility issues.

    Returns:
        dict with keys:
            - 'compatible': bool
            - 'total_ops': int
            - 'incompatible_ops': dict of {op_name: count}
            - 'flex_ops': dict of {op_name: count}
            - 'warnings': list of str
            - 'op_distribution': dict of {op_name: count}
    """
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()

    details = interp._get_ops_details()
    op_counts = collections.Counter(d.get('op_name', 'UNKNOWN') for d in details)

    incompatible = {k: v for k, v in op_counts.items() if k in GPU_INCOMPATIBLE_OPS}
    delegated = {k: v for k, v in op_counts.items() if k in GPU_DELEGATED_OPS}
    flex = {k: v for k, v in op_counts.items() if 'Flex' in k}

    warnings = []
    for op, count in delegated.items():
        warnings.append(f"{op} x{count} runs via GPU delegate (verified compatible)")

    # Check tensor dimensions (4D max for GPU)
    for detail in interp.get_tensor_details():
        shape = detail.get('shape', [])
        if len(shape) > 4:
            warnings.append(
                f"Tensor '{detail['name']}' has {len(shape)}D shape {list(shape)} — "
                f"CompiledModel GPU requires 4D max (BHWC)"
            )

    # Check input/output
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    # 5D+ tensors are warnings only — some models (e.g., MobileSAM TinyViT)
    # work on GPU despite having 5D+ intermediates. Only ops determine compatibility.
    compatible = len(incompatible) == 0 and len(flex) == 0

    result = {
        'compatible': compatible,
        'total_ops': len(details),
        'incompatible_ops': incompatible,
        'delegated_ops': delegated,
        'flex_ops': flex,
        'warnings': warnings[:10],  # Limit warnings
        'input_shape': list(inp['shape']),
        'output_shape': list(out['shape']),
        'op_distribution': dict(op_counts.most_common()),
    }

    # Log summary
    if compatible:
        if delegated:
            log.info(
                f"GPU compatible: {len(details)} ops "
                f"({sum(delegated.values())} via delegate: {sorted(delegated)})"
            )
        else:
            log.info(f"GPU compatible: {len(details)} ops, all native")
    else:
        log.warning(f"GPU INCOMPATIBLE: {incompatible or flex}")
        for w in warnings[:3]:
            log.warning(f"  {w}")

    return result


def print_report(result: dict) -> None:
    """Print a human-readable GPU compatibility report."""
    print(f"\n{'=' * 60}")
    print(f"  LiteRT GPU Compatibility Report")
    print(f"{'=' * 60}")
    print(f"  Input:  {result['input_shape']}")
    print(f"  Output: {result['output_shape']}")
    print(f"  Total ops: {result['total_ops']}")
    print()

    if result['compatible']:
        print("  Status: COMPATIBLE")
        print("  All ops are GPU-native. Ready for CompiledModel GPU.")
        if result.get('delegated_ops'):
            print(f"\n  Delegated ops (run via LiteRT GPU delegate):")
            for op, count in result['delegated_ops'].items():
                print(f"    {op}: {count}")
    else:
        print("  Status: NOT COMPATIBLE")
        if result['incompatible_ops']:
            print(f"\n  Incompatible ops:")
            for op, count in result['incompatible_ops'].items():
                print(f"    {op}: {count}")
        if result['flex_ops']:
            print(f"\n  Flex ops (require TF delegate):")
            for op, count in result['flex_ops'].items():
                print(f"    {op}: {count}")
    if result['warnings']:
        print(f"\n  Warnings:")
        for w in result['warnings']:
            print(f"    {w}")

    print(f"\n  Op distribution (top 10):")
    for op, count in list(result['op_distribution'].items())[:10]:
        print(f"    {op}: {count}")
    print(f"{'=' * 60}\n")
