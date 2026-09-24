# Copyright 2026 The Google AI Edge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Verify GLiNER2.5 graphs against fresh official fp32 extraction results."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from ai_edge_litert import schema_py_generated as schema
from gliner2 import AutoExtractor
from huggingface_hub import snapshot_download
import numpy as np
import torch

import gliner2_5_host as host
from verification_texts import TEXTS

MODEL_ID = 'fastino/gliner2.5-small-v1'
REVISION = 'f1e4d8fdd6fe328f45dee6aca3e6a07c9db4296e'
TOLERANCES = {'fp32': 1e-4, 'wfp16': 5e-3}
# This contract deliberately excludes integer routing and sparse selection.
BANNED_OPS = frozenset({
    'BROADCAST_TO',
    'CAST',
    'CUMSUM',
    'CUSTOM',
    'FILL',
    'GATHER',
    'GATHER_ND',
    'IF',
    'LESS',
    'LESS_EQUAL',
    'LOGICAL_AND',
    'LOGICAL_OR',
    'MAXIMUM',
    'NON_ZERO',
    'ONE_HOT',
    'RANGE',
    'SCATTER_ND',
    'SELECT',
    'SELECT_V2',
    'TILE',
    'TOPK_V2',
    'UNIQUE',
    'WHERE',
    'WHILE',
})


def scan_graph(path, shape):
    """Inspect the flatbuffer for unsupported ops, ranks and the I/O contract.

    Args:
      path: Exported graph path.
      shape: Expected fixed-window host contract.

    Returns:
      Operator counts, failed checks and static contract verdict.
    """
    model = schema.ModelT.InitFromPackedBuf(path.read_bytes(), 0)
    names = {
        value: name
        for name, value in vars(schema.BuiltinOperator).items()
        if isinstance(value, int)
    }
    codes = [
        names.get(max(code.builtinCode, code.deprecatedBuiltinCode), 'UNKNOWN')
        for code in model.operatorCodes
    ]
    histogram = Counter()
    errors = []
    maximum_rank = 0
    matmuls = 0
    for graph in model.subgraphs:
        for tensor in graph.tensors:
            rank = len(tensor.shape) if tensor.shape is not None else 0
            maximum_rank = max(maximum_rank, rank)
        for op in graph.operators:
            name = codes[op.opcodeIndex]
            histogram[name] += 1
            if name in BANNED_OPS or name == 'UNKNOWN':
                errors.append(f'Banned operator: {name}')
            if name == 'BATCH_MATMUL':
                matmuls += 1
                tensors = [graph.tensors[int(i)] for i in op.inputs]
                if any(len(t.shape) != 4 for t in tensors):
                    errors.append('BATCH_MATMUL input is not rank 4')
                data = model.buffers[tensors[0].buffer].data
                if data is not None and len(data):
                    errors.append('BATCH_MATMUL has a constant left input')
        for index in list(graph.inputs) + list(graph.outputs):
            if graph.tensors[index].type != schema.TensorType.FLOAT32:
                errors.append('Graph I/O must be float32')
    if maximum_rank > 4:
        errors.append(f'Tensor rank {maximum_rank} exceeds 4')
    graph = model.subgraphs[0]
    expected_output = shape.contract()['physical_output']['shape']
    if (len(graph.inputs) != 5 or len(graph.outputs) != 1 or
            graph.tensors[graph.outputs[0]].shape.tolist() != expected_output):
        errors.append('Unexpected physical I/O contract')
    return dict(passed=not errors,
                errors=errors,
                operator_histogram=dict(sorted(histogram.items())),
                operator_count=sum(histogram.values()),
                max_tensor_rank=maximum_rank,
                rank4_matmuls=matmuls,
                banned_ops=sorted(BANNED_OPS),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size)


def tensor_metrics(actual, reference):
    """Report finite values, numerical scale and absolute error per field.

    Args:
      actual: Packed graph output.
      reference: Packed native output or published graph output.

    Returns:
      Packed and logical-field error and scale records.
    """

    def metrics(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        # Mask logits are reported separately from the real numerical scale.
        mask = b != -10000.0
        return dict(
            finite=finite,
            elements=int(a.size),
            reference_abs_max=float(np.abs(b).max()) if finite else None,
            actual_abs_max=float(np.abs(a).max()) if finite else None,
            max_abs_diff=float(np.abs(a - b).max()) if finite else None,
            reference_l2_without_mask_logits=float(np.linalg.norm(b[mask])),
            actual_l2_without_mask_logits=float(np.linalg.norm(a[mask])))

    a_parts = host.unpack(torch.from_numpy(actual))
    b_parts = host.unpack(torch.from_numpy(reference))
    return dict(packed=metrics(actual, reference),
                logical={
                    name: metrics(value.numpy(), b_parts[name].numpy())
                    for name, value in a_parts.items()
                })


def write_json(path, value):
    """Write machine-readable evidence, rejecting NaN JSON values.

    Args:
      path: Report path.
      value: JSON-serializable data.
    """
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    """Exit nonzero for any task, finite-value, or operator contract failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path('out'))
    parser.add_argument('--checkpoint', default=MODEL_ID)
    parser.add_argument('--revision')
    parser.add_argument('--windows',
                        type=int,
                        nargs='+',
                        choices=(128, 256, 512),
                        default=[128, 256, 512])
    parser.add_argument('--published-dir', type=Path)
    parser.add_argument('--report-dir', type=Path, default=Path('verification'))
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    revision = args.revision
    if args.checkpoint == MODEL_ID and revision is None:
        revision = REVISION
    source = args.checkpoint
    if not Path(source).is_dir():
        source = snapshot_download(source,
                                   revision=revision,
                                   max_workers=2,
                                   allow_patterns=['*.json', '*.safetensors'])
    official = AutoExtractor.from_pretrained(source,
                                             map_location='cpu').float().eval()
    runtime = host.HostRuntime(args.model_dir / 'host_assets')
    oracle = []
    for text in TEXTS:
        with torch.inference_mode():
            result = official.extract_entities(text,
                                               host.LABELS,
                                               include_spans=True,
                                               include_confidence=True,
                                               threshold=0.5)
        oracle.append(result)
    write_json(
        args.report_dir / 'official_oracle.json',
        dict(texts=TEXTS,
             results=oracle,
             checkpoint=args.checkpoint,
             revision=revision,
             dtype='float32',
             threshold=0.5))
    report = dict(device=platform.platform(),
                  backend='Mac CPU' if platform.system() == 'Darwin' else 'CPU',
                  runtime=importlib.metadata.version('ai-edge-litert'),
                  thresholds=TOLERANCES,
                  graphs=[])
    for seq in args.windows:
        shape = host.Shape(seq)
        prepared = []
        for index, text in enumerate(TEXTS):
            inputs, captured = runtime.prepare(text, seq)
            source_inputs, _ = host.prepare(official, text, seq)
            equality = {
                name: bool(torch.equal(a, b))
                for name, a, b in zip(host.INPUT_NAMES, inputs, source_inputs)
            }
            if not all(equality.values()):
                raise ValueError(f'Host input parity failed: {equality}')
            prepared.append((inputs, captured, equality))
        for variant, tolerance in TOLERANCES.items():
            path = args.model_dir / f'gliner25_small_s{seq}_{variant}.tflite'
            inventory = scan_graph(path, shape)
            row = dict(window=seq,
                       variant=variant,
                       inventory=inventory,
                       cases=[])
            report['graphs'].append(row)
            if not inventory['passed']:
                write_json(args.report_dir / 'verification.json', report)
                raise ValueError(f'Graph inventory failed: {inventory}')
            runner = host.CpuRunner(path, shape)
            published = None
            if variant == 'fp32' and args.published_dir is not None:
                published = host.CpuRunner(args.published_dir / path.name,
                                           shape)
            try:
                for index, (inputs, captured, equality) in enumerate(prepared):
                    actual = runner.run(inputs)
                    finite = bool(np.isfinite(actual).all())
                    decoded = {'entities': {}}
                    if finite:
                        decoded = runtime.decode(captured, actual, inputs)
                    task = host.compare_spans(decoded, oracle[index])
                    case = dict(index=index,
                                text=TEXTS[index],
                                encoded_tokens=int(
                                    captured['batch'].input_ids.shape[1]),
                                text_words=int(inputs[-1].sum()),
                                host_inputs_equal=equality,
                                finite=finite,
                                task=task,
                                passed=finite and task['span_sets_equal'] and
                                task['max_confidence_diff'] <= tolerance)
                    tensors = {'actual': actual}
                    native_path = args.model_dir / (
                        f'rewrite_s{seq}_{index:02d}.npz')
                    if native_path.exists():
                        with np.load(native_path) as native:
                            case['versus_native'] = tensor_metrics(
                                actual, native['source'])
                    if published is not None:
                        reference = published.run(inputs)
                        tensors['published'] = reference
                        case['versus_published'] = tensor_metrics(
                            actual, reference)
                        case['passed'] &= bool(np.isfinite(reference).all())
                    np.savez_compressed(
                        args.report_dir / f's{seq}_{variant}_{index:02d}.npz',
                        **tensors)
                    row['cases'].append(case)
                    write_json(args.report_dir / 'verification.json', report)
            finally:
                runner.close()
                if published is not None:
                    published.close()
            row['passed'] = all(c['passed'] for c in row['cases'])
            row['max_confidence_diff'] = max(
                c['task']['max_confidence_diff'] for c in row['cases'])
            row['exact_span_sets'] = sum(
                c['task']['span_sets_equal'] for c in row['cases'])
            row['all_finite'] = all(c['finite'] for c in row['cases'])
            error = row['max_confidence_diff']
            print(
                f"s{seq} {variant}: {row['exact_span_sets']}/{len(TEXTS)} "
                f"span sets, max confidence diff {error:.9g}",
                flush=True)
    report['passed'] = all(g['passed'] for g in report['graphs'])
    write_json(args.report_dir / 'verification.json', report)
    print('PASS' if report['passed'] else 'FAIL', flush=True)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
