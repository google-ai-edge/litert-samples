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
"""GLiNER2.5 host preprocessing, sparse decoding, and CompiledModel CPU runner.

The sparse proposal, scoring and formatting calls use gliner2 unchanged.
Only the dense producers are replaced by packed graph outputs.
"""
import argparse
from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.compiled_model import HardwareAccelerator
from ai_edge_litert.compiled_model import Options
from ai_edge_litert.compiled_model import CpuOptions
from gliner2.models.base import ExtractorConfig
from gliner2.models.base import load_extractor_tokenizer
from gliner2.models.boundary.encoding import BoundaryEncoding
from gliner2.models.boundary.engine import BoundaryExtractor
from gliner2.models.boundary.heads import BoundaryMarginals
from gliner2.training.trainer import ExtractorCollator
import numpy as np
from safetensors.torch import load_file
import torch
from torch import nn
from torch.nn import functional as F

LABELS = ['person', 'organization', 'location', 'product', 'date']
TEXT = 48
QUERIES = 5
EXAMPLE = (
    'Maya Chen from Orvane Robotics demonstrated the Veltrix 9 in Lisbon '
    'on March 12, 2025.')

INPUT_NAMES = [
    "inputs_embeds", "attention_mask", "text_routing", "query_routing",
    "text_mask"
]
OUTPUT_SHAPES = OrderedDict([
    ("text_states", (1, TEXT, 384)),
    ("query_states", (1, QUERIES, 384)),
    ("boundary_states", (1, TEXT + 1, 128)),
    ("start_logits", (1, QUERIES, TEXT + 1)),
    ("end_logits", (1, QUERIES, TEXT + 1)),
    ("inside_logits", (1, QUERIES, TEXT)),
    ("inside_prefix", (1, QUERIES, TEXT + 1)),
    ("inside_prefix_mean", (1, QUERIES, 1)),
    ("pool_start", (1, TEXT + 1, 128)),
    ("pool_end", (1, TEXT + 1, 128)),
    ("score_start", (1, TEXT + 1, 128)),
    ("score_end", (1, TEXT + 1, 128)),
    ("content_prefix", (1, TEXT + 1, 64)),
    ("score_query", (1, QUERIES, 128)),
    ("film", (1, QUERIES, 256)),
    ("null_logits", (1, QUERIES)),
    ("count_log_rates", (1, QUERIES)),
])
PACKED_SIZE = sum(math.prod(shape) for shape in OUTPUT_SHAPES.values())


def logical_output_shapes(text_capacity=TEXT):
    """Describe the fixed ordering of packed logical fields.
    
    Args:
      text_capacity: Maximum text-word count for this graph.
    
    Returns:
      Ordered mapping from logical field names to tensor shapes."""
    return OrderedDict((name,
                        tuple(text_capacity if d == TEXT else text_capacity +
                              1 if d == TEXT + 1 else d
                              for d in shape))
                       for name, shape in OUTPUT_SHAPES.items())


def unpack(packed):
    """View a packed result as the logical tensors consumed by gliner2.
    
    Args:
      packed: Tensor with exactly one fixed-window packed output.
    
    Returns:
      Ordered tensor views with unchanged values."""
    flat = packed.reshape(-1)
    # Packing has 1108 elements per text slot plus 4574 fixed elements.
    # This is host-only metadata; inference graphs still emit one fixed leaf.
    text_capacity, remainder = divmod(flat.numel() - 4574, 1108)
    assert remainder == 0 and text_capacity > 0
    offset = 0
    out = OrderedDict()
    for name, shape in logical_output_shapes(text_capacity).items():
        size = math.prod(shape)
        out[name] = flat[offset:offset + size].reshape(shape)
        offset += size
    assert offset == flat.numel()
    return out


@dataclass(frozen=True)
class Shape:
    """Encoded-token, word, and packed-output capacities."""

    seq: int

    @property
    def text(self):
        return 48 if self.seq == 128 else self.seq * 3 // 4

    @property
    def outputs(self):
        return logical_output_shapes(self.text)

    @property
    def packed_size(self):
        return sum(math.prod(s) for s in self.outputs.values())

    def contract(self):
        shapes = [(1, self.seq, 384), (1, self.seq), (1, self.text, self.seq),
                  (1, 5, self.seq), (1, self.text)]
        result = dict(seq=self.seq,
                      text_capacity=self.text,
                      inputs=[
                          dict(name=n, shape=list(s), dtype='float32')
                          for n, s in zip(INPUT_NAMES, shapes)
                      ],
                      physical_output=dict(name='packed_output',
                                           shape=[1, 1, 1, self.packed_size],
                                           dtype='float32'),
                      logical_outputs=[])
        offset = 0
        for name, s in self.outputs.items():
            size = math.prod(s)
            result['logical_outputs'].append(
                dict(name=name,
                     shape=list(s),
                     dtype='float32',
                     offset=offset,
                     elements=size))
            offset += size
        return result


def fixed_inputs(model, captured, shape):
    """Create float masks and exact one-hot routing on the host.
    
    Args:
      model: Official or host-only extractor with its embedding table.
      captured: Official batch and decoder metadata.
      shape: Fixed token and text-word capacities.
    
    Returns:
      Five contiguous float32 graph inputs in argument order."""
    batch = captured['batch']
    n = batch.input_ids.shape[1]
    t = batch.text_word_indices.shape[1]
    assert n <= shape.seq and t <= shape.text, (n, t, shape)
    ids = F.pad(batch.input_ids, (0, shape.seq - n),
                value=model.processor.tokenizer.pad_token_id)
    attention = F.pad(batch.attention_mask, (0, shape.seq - n)).float()
    text_mask = F.pad(batch.text_word_mask, (0, shape.text - t)).float()
    text_route = torch.zeros(1, shape.text, shape.seq)
    for i in range(t):
        if batch.text_word_mask[0, i]:
            text_route[0, i, batch.text_word_indices[0, i]] = 1
    assert batch.query_marker_indices.shape == (
        1, 5) and batch.query_marker_mask.all()
    query_route = torch.zeros(1, 5, shape.seq)
    for i in range(5):
        query_route[0, i, batch.query_marker_indices[0, i]] = 1
    with torch.no_grad():
        embeds = model.encoder.embeddings.word_embeddings(ids)
    return tuple(x.detach().contiguous()
                 for x in (embeds, attention, text_route, query_route,
                           text_mask))


@torch.inference_mode()
def decode(model, captured, packed, graph_inputs):
    """Run the unchanged sparse gliner2 pipeline from graph outputs.
    
    Args:
      model: Official or host-only extractor with sparse decoder parameters.
      captured: Official batch, schema metadata and extraction threshold.
      packed: Single graph output containing the dense intermediate values.
      graph_inputs: Five input tensors, including the text mask.
    
    Returns:
      Public extraction result with text, offsets and confidences."""
    outputs = unpack(torch.as_tensor(packed).float())
    text_mask = graph_inputs[-1].bool()
    query_mask = torch.ones(outputs["query_states"].shape[:2], dtype=torch.bool)
    lengths = text_mask.sum(-1).long()
    boundary_mask = torch.arange(text_mask.shape[1] + 1)[None] <= lengths[:,
                                                                          None]
    core = dict(captured["core"])
    core.update(text_states=outputs["text_states"],
                text_mask=text_mask,
                text_lengths=lengths,
                query_states=outputs["query_states"],
                query_mask=query_mask)
    encoding = BoundaryEncoding(outputs["boundary_states"], boundary_mask)
    marginals = BoundaryMarginals(
        **{
            key: outputs[key] for key in (
                "start_logits",
                "end_logits",
                "inside_logits",
                "inside_prefix",
                "inside_prefix_mean",
            )
        })
    h = model.boundary_head
    replacements = [
        (model, "_encode_core", core),
        (h.boundary_encoder, "forward", encoding),
        (h.boundary_query_head, "forward", marginals),
        (h.shared_pool_builder.start_projection, "forward",
         outputs["pool_start"]),
        (h.shared_pool_builder.end_projection, "forward", outputs["pool_end"]),
        (h.shared_pool_scorer.start_projection, "forward",
         outputs["score_start"]),
        (h.shared_pool_scorer.end_projection, "forward", outputs["score_end"]),
        (h.shared_pool_scorer.content_pooler, "build_prefix",
         (outputs["content_prefix"], None)),
        (h.shared_pool_scorer.query_projection, "forward",
         outputs["score_query"]),
        (h.shared_pool_scorer.film, "forward", outputs["film"]),
        (h.null_projection, "forward", outputs["null_logits"].unsqueeze(-1)),
        (h.count_head, "forward", outputs["count_log_rates"].unsqueeze(-1)),
    ]
    with ExitStack() as stack:
        for obj, attribute, value in replacements:
            stack.enter_context(
                patch.object(obj, attribute, lambda *a, _v=value, **kw: _v))
        # Fail loudly if the host accidentally falls back to the dense encoder.
        stack.enter_context(
            patch.object(
                model.encoder,
                "forward",
                side_effect=AssertionError("host attempted encoder inference")))
        results = model._extract_from_batch(
            captured["batch"],
            captured["threshold"],
            captured["metadata_list"],
            True,
            True,
        )
        meta = captured["metadata_list"][0]
        return model.format_results(results[0], True,
                                    meta.get("relation_order", []),
                                    meta.get("classification_tasks", []))


class HostEmbeddingEncoder(nn.Module):
    """Table lookup and placeholder states for routing metadata only."""

    def __init__(self, table):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=384)
        self.embeddings = nn.Module()
        self.embeddings.word_embeddings = nn.Embedding.from_pretrained(
            table, freeze=True)

    def resize_token_embeddings(self, count, *args, **kwargs):
        assert count == self.embeddings.word_embeddings.num_embeddings
        return self.embeddings.word_embeddings

    def forward(self, input_ids, attention_mask=None):
        # Only _encode_core's metadata construction uses this placeholder.
        # decode() patches this method to raise during real decoding.
        return SimpleNamespace(last_hidden_state=torch.zeros(
            *input_ids.shape, 384, dtype=torch.float32))


class HostRuntime:
    """Load only embeddings, tokenizer and the 16 sparse decoder tensors."""

    def __init__(self, assets):
        self.assets = Path(assets)
        info = json.loads((self.assets / 'decoder_parameters.json').read_text())
        embedding = np.memmap(self.assets / 'word_embeddings_fp32.bin',
                              dtype='<f4',
                              mode='c',
                              shape=tuple(
                                  info.get('embedding_shape', [128011, 384])))
        self.embedding = embedding
        # Match the official loader's tokenizer metadata compatibility
        # fallback; the original tokenizer JSON files remain unmodified.
        tokenizer = load_extractor_tokenizer(str(self.assets))
        config = ExtractorConfig.from_json_file(str(self.assets /
                                                    'config.json'))
        encoder = HostEmbeddingEncoder(torch.from_numpy(embedding))
        with patch.object(BoundaryExtractor,
                          '_load_encoder',
                          return_value=encoder):
            model = BoundaryExtractor(config,
                                      tokenizer=tokenizer,
                                      use_flashdeberta=False)
        weights = load_file(str(self.assets /
                                'sparse_decoder_fp32.safetensors'))
        assert set(weights) == set(row['name'] for row in info['tensors'])
        # Missing dense/unused parameters deliberately remain on meta, so an
        # accidental execution outside the documented cut fails instead of using
        # random initialization. The only live encoder weight is the host table.
        for name, parameter in list(model.named_parameters()):
            if name == 'encoder.embeddings.word_embeddings.weight':
                continue
            parent, _, leaf = name.rpartition('.')
            module = model.get_submodule(parent)
            value = weights[name] if name in weights else torch.empty(
                parameter.shape, dtype=parameter.dtype, device='meta')
            setattr(module, leaf, nn.Parameter(value, requires_grad=False))
        for row in info['tensors']:
            if row['kind'] == 'buffer':
                parent, _, leaf = row['name'].rpartition('.')
                setattr(model.get_submodule(parent), leaf, weights[row['name']])
        self.model = model.eval()

    @torch.inference_mode()
    def prepare(self, text, seq=128):
        return prepare(self.model, text, seq)

    def decode(self, captured, packed, inputs):
        return decode(self.model, captured, torch.as_tensor(packed), inputs)


@torch.inference_mode()
def prepare(model, text, seq):
    """Prepare official routing metadata and fixed float graph inputs.

    Args:
      model: Official extractor or host-only extractor.
      text: Input string. Overflowing inputs are rejected.
      seq: Encoded token capacity, including the schema.

    Returns:
      Graph input tuple and metadata for unchanged sparse decoding.
    """
    schema_value = model.create_schema().entities(LABELS)
    dictionaries, metadata = model._build_schema_dicts_and_metadata(
        [schema_value])
    model.processor.change_mode(is_training=False)
    collator = ExtractorCollator(model.processor,
                                 is_training=False,
                                 architecture='boundary')
    batch = collator([(text, dictionaries[0])])
    if (batch.input_ids.shape[1] > seq or
            batch.text_word_indices.shape[1] > Shape(seq).text):
        raise ValueError('Input exceeds fixed capacity; use a larger window.')
    captured = dict(batch=batch,
                    core=model._encode_core(batch),
                    metadata_list=metadata,
                    threshold=0.5)
    return fixed_inputs(model, captured, Shape(seq)), captured


class CpuRunner:
    """Reuse CompiledModel buffers with explicit signature-name ordering."""

    def __init__(self, path, shape):
        model_schema = schema.ModelT.InitFromPackedBuf(
            Path(path).read_bytes(), 0)
        signature = model_schema.signatureDefs[0]
        graph = model_schema.subgraphs[signature.subgraphIndex]
        self.order = []
        for entry in signature.inputs:
            name = entry.name.decode()
            # Named forward parameters are retained by the pinned exporter.
            if name in INPUT_NAMES:
                index = INPUT_NAMES.index(name)
            elif name.startswith('args_'):
                index = int(name.removeprefix('args_'))
            else:
                raise ValueError(f'Unknown graph input name: {name}')
            expected = shape.contract()['inputs'][index]['shape']
            if graph.tensors[entry.tensorIndex].shape.tolist() != expected:
                raise ValueError(f'Wrong input shape for {name}')
            self.order.append(index)
        if sorted(self.order) != list(range(len(INPUT_NAMES))):
            raise ValueError('Graph input signature is not a permutation.')
        self.output_shape = shape.contract()['physical_output']['shape']
        if len(signature.outputs) != 1:
            raise ValueError('Expected exactly one packed output.')
        output_spec = graph.tensors[signature.outputs[0].tensorIndex]
        if output_spec.shape.tolist() != self.output_shape:
            raise ValueError('Packed output shape disagrees with contract.')
        self.model = CompiledModel.from_file(
            str(path),
            options=Options(hardware_accelerators=HardwareAccelerator.CPU,
                            cpu_options=CpuOptions(num_threads=4)))
        self.input_buffers = self.model.create_input_buffers(0)
        self.output_buffers = self.model.create_output_buffers(0)

    def run(self, inputs):
        for buffer, index in zip(self.input_buffers, self.order):
            buffer.write(np.ascontiguousarray(inputs[index], dtype=np.float32))
        self.model.run_by_index(0, self.input_buffers, self.output_buffers)
        return self.output_buffers[0].read(math.prod(
            self.output_shape), np.float32).reshape(self.output_shape).copy()

    def close(self):
        for buffer in self.input_buffers + self.output_buffers:
            buffer.destroy()
        self.model.close()


def span_list(result):
    """Normalize the public extraction result without changing confidences.

    Args:
      result: Official-format entities dictionary.

    Returns:
      Sorted labeled spans with text, offsets and confidence.
    """
    return sorted([
        dict(label=label, **span)
        for label, spans in result['entities'].items()
        for span in spans
    ],
                  key=lambda span: (span['label'], span['start'], span['end']))


def compare_spans(actual, expected):
    """Compare exact labeled text/offset sets and matched confidences.

    Args:
      actual: Decoded extraction result.
      expected: Official fp32 extraction result.

    Returns:
      Exact set equality, differences, maximum confidence error and spans.
    """

    def keyed(result):
        return {
            (s['label'], s['start'], s['end'], s['text']): s
            for s in span_list(result)
        }

    actual_spans, expected_spans = keyed(actual), keyed(expected)
    return dict(span_sets_equal=actual_spans.keys() == expected_spans.keys(),
                missing=sorted(expected_spans.keys() - actual_spans.keys()),
                extra=sorted(actual_spans.keys() - expected_spans.keys()),
                max_confidence_diff=max(
                    (abs(actual_spans[k]['confidence'] -
                         expected_spans[k]['confidence'])
                     for k in actual_spans.keys() & expected_spans.keys()),
                    default=0.0),
                actual_spans=span_list(actual),
                expected_spans=span_list(expected))


def main():
    """Run the example through the graph and sparse host decoder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--window',
                        type=int,
                        choices=(128, 256, 512),
                        default=128)
    parser.add_argument('--variant', choices=('fp32', 'wfp16'), default='wfp16')
    parser.add_argument('--text', default=EXAMPLE)
    args = parser.parse_args()
    torch.set_num_threads(4)
    runtime = HostRuntime(args.model_dir / 'host_assets')
    inputs, captured = runtime.prepare(args.text, args.window)
    path = args.model_dir / (
        f'gliner25_small_s{args.window}_{args.variant}.tflite')
    runner = CpuRunner(path, Shape(args.window))
    try:
        packed = runner.run(inputs)
        if not np.isfinite(packed).all():
            raise ValueError('Graph output contains nonfinite values.')
        result = runtime.decode(captured, packed, inputs)
        print(json.dumps(result, indent=2, allow_nan=False))
    finally:
        runner.close()


if __name__ == '__main__':
    main()
