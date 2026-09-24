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
"""Export GLiNER2.5 Small's exact dense prefix and its host assets.

DeBERTa follows transformers 4.57.6. Boundary math follows gliner2 2.0.0.
The sparse decoder is unchanged. No GELU or attention approximation is used.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import time

# Load converter dialects before transformers model modules.
import litert_torch
from ai_edge_quantizer import algorithm_manager
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import quantizer
from ai_edge_quantizer import recipe_manager
from gliner2 import AutoExtractor
from huggingface_hub import snapshot_download
import numpy as np
from safetensors.torch import save_file
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.deberta_v2 import modeling_deberta_v2

import gliner2_5_host as host

MODEL_ID = 'fastino/gliner2.5-small-v1'
REVISION = 'f1e4d8fdd6fe328f45dee6aca3e6a07c9db4296e'
SPARSE_NAMES = [
    'boundary_head.candidate_encoder.bias',
    'boundary_head.candidate_encoder.weight',
    'boundary_head.shared_pool_scorer.candidate_norm.bias',
    'boundary_head.shared_pool_scorer.candidate_norm.weight',
    'boundary_head.shared_pool_scorer.content_pooler.layer_norm.bias',
    'boundary_head.shared_pool_scorer.content_pooler.layer_norm.weight',
    'boundary_head.shared_pool_scorer.content_projection.bias',
    'boundary_head.shared_pool_scorer.content_projection.weight',
    'boundary_head.shared_pool_scorer.film_output.0.bias',
    'boundary_head.shared_pool_scorer.film_output.0.weight',
    'boundary_head.shared_pool_scorer.film_output.3.bias',
    'boundary_head.shared_pool_scorer.film_output.3.weight',
    'boundary_head.shared_pool_scorer.length_projection.bias',
    'boundary_head.shared_pool_scorer.length_projection.weight',
    'boundary_head.shared_pool_scorer.prior_projection.bias',
    'boundary_head.shared_pool_scorer.prior_projection.weight'
]


@torch.library.custom_op("gliner25::rank4_matmul", mutates_args=())
def rank4_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Multiply matrices while preserving two explicit batch axes.
    
    Args:
      x: Rank-four left tensor with two batch axes.
      y: Rank-four right tensor with the same batch axes.
    
    Returns:
      Batched matrix product without flattening the head axis."""
    assert x.ndim == y.ndim == 4 and x.shape[:2] == y.shape[:2]
    return torch.matmul(x, y)


@rank4_matmul.register_fake
def _rank4_matmul_fake(x, y):
    """Describe the product shape during torch.export tracing.
    
    Args:
      x: Fake left operand.
      y: Fake right operand.
    
    Returns:
      Fake rank-four output with the traced product shape."""
    return x.new_empty((x.shape[0], x.shape[1], x.shape[2], y.shape[3]))


def register_rank4_matmul():
    """Lower the marker to standard StableHLO and TFLite BATCH_MATMUL.
    
    The exported model needs no custom operator or converter changes.
    """
    import jax
    from litert_torch.backend.lowerings._jax_lowerings import lowerings

    @lowerings.lower_by_jax(torch.ops.gliner25.rank4_matmul.default)
    def lower_rank4_matmul(x, y):
        return jax.lax.dot_general(
            x,
            y,
            dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1))),
            precision=jax.lax.Precision.HIGHEST,
        )


class ShapedDebertaLayer(nn.Module):
    """Exact DeBERTa attention with baked relative-position lookups."""

    def __init__(self, layer, rel_embeddings, seq):
        super().__init__()
        self.layer = layer
        self.seq = seq
        a = layer.attention.self
        assert a.share_att_key and set(a.pos_att_type) == {'p2c', 'c2p'}
        self.heads, self.dim = a.num_attention_heads, a.attention_head_size
        self.scale = math.sqrt(self.dim * 3)
        with torch.no_grad():
            offsets = torch.arange(-seq, seq)
            bucket = modeling_deberta_v2.make_log_bucket_position(
                offsets, a.position_buckets, a.max_relative_positions).long()
            ids = (bucket + a.pos_ebd_size).clamp(0, 2 * a.pos_ebd_size - 1)
            q = self.split(a.query_proj(rel_embeddings.unsqueeze(0)))
            k = self.split(a.key_proj(rel_embeddings.unsqueeze(0)))
            # Pre-expand the original lookup constants over signed distances.
            # No lookup or integer operation enters the exported graph.
            self.register_buffer('pos_q', q.index_select(-2, ids).contiguous())
            self.register_buffer('pos_k_reversed',
                                 k.index_select(-2, ids).flip(-2).contiguous())
            dummy = torch.zeros(1, seq, self.heads * self.dim)
            rel = modeling_deberta_v2.build_relative_position(
                dummy, dummy, a.position_buckets, a.max_relative_positions)
            distances = torch.arange(seq)[:, None] - torch.arange(seq)[None, :]
            assert torch.equal(bucket[distances + seq], rel.squeeze(0))

    def split(self, x):
        return x.reshape(1, x.shape[1], self.heads,
                         self.dim).permute(0, 2, 1, 3)

    def relative_shift(self, scores, offset):
        width = scores.shape[-1]
        flat = scores.reshape(1, self.heads, self.seq * width)
        length = self.seq * (width - 1)
        return flat[...,
                    offset:offset + length].reshape(1, self.heads, self.seq,
                                                    width - 1)[..., :self.seq]

    def forward(self, hidden, mask):
        a = self.layer.attention.self
        q, k, v = self.split(a.query_proj(hidden)), self.split(
            a.key_proj(hidden)), self.split(a.value_proj(hidden))
        scores = rank4_matmul(q, k.transpose(-1, -2) / self.scale)
        c2p = self.relative_shift(
            rank4_matmul(q, self.pos_k_reversed.transpose(-1, -2)),
            self.seq - 1) / self.scale
        p2c = self.relative_shift(rank4_matmul(k, self.pos_q.transpose(-1, -2)),
                                  self.seq).transpose(-1, -2) / self.scale
        scores = scores + (c2p + p2c)
        scores = scores * mask + (1.0 - mask) * torch.finfo(torch.float32).min
        context = rank4_matmul(scores.softmax(-1),
                               v).permute(0, 2, 1,
                                          3).reshape(1, self.seq,
                                                     self.heads * self.dim)
        attended = self.layer.attention.output(context, hidden)
        return self.layer.output(self.layer.intermediate(attended), attended)


class ShapedDense(nn.Module):
    """Encoder and boundary projections up to the first sparse selection."""

    def __init__(self, model, shape, native=False):
        super().__init__()
        self.shape = shape
        self.native = native
        self.head = model.boundary_head
        self.embedding_norm = model.encoder.embeddings.LayerNorm
        if native:
            self.native_encoder = model.encoder
        else:
            with torch.no_grad():
                rel = model.encoder.encoder.get_rel_embedding().detach()
            self.layers = nn.ModuleList(
                ShapedDebertaLayer(x, rel, shape.seq)
                for x in model.encoder.encoder.layer)
        t = shape.text
        n = t + 1
        diagonal = torch.eye(n).view(1, 1, n, n)
        self.register_buffer('diagonal', diagonal.contiguous())
        windows = {
            b.window for b in self.head.boundary_encoder.attention_blocks
        }
        assert len(windows) == 1
        window = next(iter(windows))
        positions = torch.arange(n)
        local = ((positions[:, None] - positions[None, :]).abs()
                 <= window).float() if window > 0 else torch.ones(n, n)
        self.register_buffer('local', local.view(1, 1, n, n).contiguous())
        prefix = (torch.arange(t + 1)[:, None]
                  > torch.arange(t)[None, :]).float()
        self.register_buffer('inside_triangle',
                             prefix.T.view(1, 1, t, t + 1).contiguous())
        assert not self.head.shared_pool_scorer.query_layers
        assert not self.head.shared_pool_scorer.content_pooler.use_soft_max_pool

    def boundary(self, text, text_mask):
        m = self.head.boundary_encoder
        n = self.shape.text + 1
        bmask = torch.cat((torch.ones(1, 1), text_mask), 1)
        next_mask = torch.cat((text_mask, torch.zeros(1, 1)), 1)
        eos = (bmask - next_mask).unsqueeze(-1)
        left = torch.cat((m.bos_state.reshape(1, 1, -1), text), 1)
        right = torch.cat((text, m.eos_state.reshape(1, 1, -1)), 1)
        right = right * (1.0 - eos) + m.eos_state.reshape(1, 1, -1) * eos
        states = m.layer_norm(
            m.output_projection(
                torch.cat((m.left_projection(left), m.right_projection(right)),
                          -1)))
        key_mask = bmask.reshape(1, 1, 1, n)
        base = key_mask * self.local
        allowed = base + self.diagonal * (1.0 - base)
        for block in m.attention_blocks:
            qkv = block.qkv_projection(block.norm(states))
            d = states.shape[-1]
            q = qkv[..., :d].reshape(1, n, block.num_heads,
                                     block.head_dim).transpose(1, 2)
            k = qkv[..., d:2 * d].reshape(1, n, block.num_heads,
                                          block.head_dim).transpose(1, 2)
            v = qkv[..., 2 * d:].reshape(1, n, block.num_heads,
                                         block.head_dim).transpose(1, 2)
            scores = rank4_matmul(q * (block.head_dim**-0.5),
                                  k.transpose(-1, -2))
            masked = scores * allowed + (1.0 - allowed) * torch.finfo(
                torch.float32).min
            attended = rank4_matmul(masked.softmax(-1),
                                    v).transpose(1, 2).reshape(1, n, d)
            states = (states +
                      block.output_projection(attended)) * bmask.unsqueeze(-1)
        for block in m.refinement_blocks:
            states = block(states)
        return states * bmask.unsqueeze(-1), bmask

    def marginals(self, boundary, bmask, text, text_mask, queries):
        m = self.head.boundary_query_head

        def score(tokens, query):
            return rank4_matmul(query.unsqueeze(1),
                                tokens.unsqueeze(1).transpose(
                                    -1, -2)).squeeze(1) / math.sqrt(
                                        m.boundary_dim)

        start = score(m.start_boundary_projection(boundary),
                      m.start_query_projection(queries))
        end = score(m.end_boundary_projection(boundary),
                    m.end_query_projection(queries))
        inside = score(m.inside_text_projection(text),
                       m.inside_query_projection(queries))
        bm, tm = bmask.unsqueeze(1), text_mask.unsqueeze(1)
        start = start * bm + (1 - bm) * -10000.0
        end = end * bm + (1 - bm) * -10000.0
        inside = inside * tm + (1 - tm) * -10000.0
        inside_for_prefix = inside * tm
        count = F.relu(tm.sum(-1, keepdim=True) - 1) + 1
        mean = inside_for_prefix.sum(-1, keepdim=True) / count
        centered = (inside_for_prefix - mean) * tm
        prefix = rank4_matmul(centered.unsqueeze(1),
                              self.inside_triangle).squeeze(1)
        return start, end, inside, prefix, mean

    def forward(self, inputs_embeds, attention_mask, text_routing,
                query_routing, text_mask):
        if self.native:
            hidden = self.native_encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask).last_hidden_state
        else:
            hidden = self.embedding_norm(
                inputs_embeds) * attention_mask.unsqueeze(-1)
            mask = attention_mask[:, None, None, :] * attention_mask[:, None, :,
                                                                     None]
            for layer in self.layers:
                hidden = layer(hidden, mask)
        text = rank4_matmul(text_routing.unsqueeze(1),
                            hidden.unsqueeze(1)).squeeze(1)
        query = rank4_matmul(query_routing.unsqueeze(1),
                             hidden.unsqueeze(1)).squeeze(1)
        h = self.head
        if self.native:
            encoding = h.boundary_encoder(text, text_mask.bool())
            boundary = encoding.states
            m = h.boundary_query_head(boundary, encoding.mask, text,
                                      text_mask.bool(), query,
                                      torch.ones(1, 5, dtype=torch.bool))
            marginal = tuple(
                getattr(m, k)
                for k in ('start_logits', 'end_logits', 'inside_logits',
                          'inside_prefix', 'inside_prefix_mean'))
            pooler = h.shared_pool_scorer.content_pooler
            content_prefix, _ = pooler.build_prefix(text, text_mask.bool())
        else:
            boundary, bmask = self.boundary(text, text_mask)
            marginal = self.marginals(boundary, bmask, text, text_mask, query)
            values = h.shared_pool_scorer.content_pooler.value_projection(
                text) * text_mask.unsqueeze(-1)
            content_prefix = rank4_matmul(
                values.unsqueeze(1).transpose(-1, -2),
                self.inside_triangle).transpose(-1, -2).squeeze(1)
        pool, scorer = h.shared_pool_builder, h.shared_pool_scorer
        score_query = scorer.query_projection(query)
        outputs = (text, query, boundary, *marginal,
                   pool.start_projection(boundary),
                   pool.end_projection(boundary),
                   scorer.start_projection(boundary),
                   scorer.end_projection(boundary), content_prefix, score_query,
                   scorer.film(score_query),
                   h.null_projection(query).squeeze(-1),
                   h.count_head(query).squeeze(-1))
        return torch.cat(tuple(x.reshape(-1) for x in outputs)).reshape(
            1, 1, 1, self.shape.packed_size)


def write_json(path, value):
    """Write an indented JSON record.

    Args:
      path: Destination path.
      value: JSON-serializable evidence.
    """
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_checkpoint(checkpoint, revision):
    """Load the official fp32 extractor from a local path or Hub repository.

    Args:
      checkpoint: Compatible GLiNER2.5 Small checkpoint directory or Hub ID.
      revision: Optional Hub revision. The default model is pinned.

    Returns:
      Official extractor and resolved source directory.
    """
    source = Path(checkpoint)
    if not source.is_dir():
        if revision is None and checkpoint == MODEL_ID:
            revision = REVISION
        source = Path(
            snapshot_download(checkpoint,
                              revision=revision,
                              allow_patterns=[
                                  '*.json', '*.safetensors', 'README.md',
                                  '*LICENSE*'
                              ],
                              max_workers=2))
    model = AutoExtractor.from_pretrained(
        str(source), map_location='cpu').float().eval().requires_grad_(False)
    if (model.config.architecture != 'boundary' or
            model.boundary_settings.candidate_pool != 'shared' or
            model.encoder.config.hidden_size != 384):
        raise ValueError('Expected GLiNER2.5 Small with shared boundary pool.')
    embeddings = model.encoder.embeddings
    if (getattr(embeddings, 'position_biased_input', False) or
            getattr(embeddings, 'token_type_embeddings', None) is not None or
            getattr(embeddings, 'embed_proj', None) is not None or
            getattr(model.encoder.encoder, 'conv', None) is not None):
        raise ValueError(
            'Checkpoint changes the supported encoder architecture.')
    head = model.boundary_head
    if (head.boundary_encoder.bos_state.numel() != 384 or
            head.boundary_query_head.boundary_dim != 128 or
            head.shared_pool_scorer.content_pooler.value_projection.out_features
            != 64 or any(b.window != 128
                         for b in head.boundary_encoder.attention_blocks)):
        raise ValueError('Checkpoint changes the boundary architecture.')
    return model, source


def write_host_assets(model, source, destination, checkpoint, revision,
                      windows):
    """Write the embedding, exact sparse tensors, tokenizer and contracts.

    Args:
      model: Loaded official fp32 extractor.
      source: Resolved checkpoint directory.
      destination: Empty host asset directory.
      checkpoint: Original checkpoint identifier.
      revision: Revision used for the Hub source, or None for local input.
      windows: Exported encoded capacities.
    """
    destination.mkdir(parents=True)
    table = model.encoder.embeddings.word_embeddings.weight.detach().numpy()
    table.astype('<f4',
                 copy=False).tofile(destination / 'word_embeddings_fp32.bin')
    for name in ('tokenizer.json', 'tokenizer_config.json', 'config.json'):
        shutil.copyfile(source / name, destination / name)
    shutil.copytree(source / 'encoder_config', destination / 'encoder_config')
    state = model.state_dict()
    tensors = {
        name: state[name].detach().cpu().contiguous() for name in SPARSE_NAMES
    }
    save_file(tensors, str(destination / 'sparse_decoder_fp32.safetensors'))
    rows = [
        dict(name=name,
             kind='parameter',
             shape=list(value.shape),
             dtype='float32',
             bytes=value.numel() * 4) for name, value in tensors.items()
    ]
    write_json(
        destination / 'decoder_parameters.json',
        dict(source=checkpoint,
             source_revision=revision,
             embedding_shape=list(table.shape),
             tensors=rows,
             total_bytes=sum(row['bytes'] for row in rows)))
    for seq in windows:
        write_json(destination / f'graph_contract_s{seq}.json',
                   host.Shape(seq).contract())
    shutil.copyfile(Path(host.__file__), destination / 'gliner2_5_host.py')


def quantize_weights(source, destination):
    """Store supported FC weights in fp16 while retaining float32 activations.

    Args:
      source: Original float32 graph path.
      destination: FLOAT_CASTING output path.
    """
    manager = recipe_manager.RecipeManager()
    manager.add_weight_only_config(
        regex='.*',
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        num_bits=16,
        granularity=qtyping.QuantGranularity.TENSORWISE,
        algorithm_key=algorithm_manager.AlgorithmName.FLOAT_CASTING)
    converter = quantizer.Quantizer(str(source))
    recipe = manager.get_quantization_recipe()
    converter.load_quantization_recipe(recipe)
    converter.quantize().export_model(str(destination))


def main():
    """Build each window after checking source/rewrite task parity."""
    from verification_texts import TEXTS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=MODEL_ID)
    parser.add_argument('--revision')
    parser.add_argument('--output-dir', type=Path, default=Path('out'))
    parser.add_argument('--windows',
                        type=int,
                        nargs='+',
                        choices=(128, 256, 512),
                        default=[128, 256, 512])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output directory must be empty.')
    out.mkdir(parents=True, exist_ok=True)
    revision = args.revision
    if args.checkpoint == MODEL_ID and revision is None:
        revision = REVISION
    model, source = load_checkpoint(args.checkpoint, revision)
    register_rank4_matmul()
    report = dict(checkpoint=args.checkpoint,
                  revision=revision,
                  windows=[],
                  toolchain={
                      name: importlib.metadata.version(name)
                      for name in ('torch', 'gliner2', 'transformers',
                                   'litert-torch', 'litert-converter',
                                   'ai-edge-quantizer', 'ai-edge-litert')
                  })
    for seq in args.windows:
        shape = host.Shape(seq)
        graph = ShapedDense(model, shape).eval().requires_grad_(False)
        native = ShapedDense(model, shape, native=True).eval()
        gate = []
        for index, text in enumerate(TEXTS):
            inputs, captured = host.prepare(model, text, seq)
            with torch.inference_mode():
                expected = native(*inputs)
                actual = graph(*inputs)
                source_result = model.extract_entities(text,
                                                       host.LABELS,
                                                       include_spans=True,
                                                       include_confidence=True,
                                                       threshold=0.5)
                decoded = host.decode(model, captured, actual, inputs)
            task = host.compare_spans(decoded, source_result)
            row = dict(index=index,
                       text=text,
                       finite=bool(
                           torch.isfinite(actual).all() and
                           torch.isfinite(expected).all()),
                       reference_abs_max=float(expected.abs().max()),
                       actual_abs_max=float(actual.abs().max()),
                       max_abs_diff=float((actual - expected).abs().max()),
                       task=task)
            gate.append(row)
            np.savez_compressed(out / f'rewrite_s{seq}_{index:02d}.npz',
                                source=expected.numpy(),
                                rewritten=actual.numpy())
            if not (row['finite'] and task['span_sets_equal'] and
                    task['max_confidence_diff'] <= 1e-4):
                write_json(out / f'rewrite_s{seq}.json', gate)
                raise ValueError(f'Source/rewrite gate failed: {row}')
        write_json(out / f'rewrite_s{seq}.json', gate)
        fp32 = out / f'gliner25_small_s{seq}_fp32.tflite'
        wfp16 = out / f'gliner25_small_s{seq}_wfp16.tflite'
        started = time.perf_counter()
        print(f'Exporting s{seq}', flush=True)
        litert_torch.convert(graph, inputs).export(str(fp32))
        quantize_weights(fp32, wfp16)
        write_json(out / f'graph_contract_s{seq}.json', shape.contract())
        row = dict(window=seq,
                   seconds=time.perf_counter() - started,
                   source_rewrite_max_abs_diff=max(
                       x['max_abs_diff'] for x in gate),
                   files=[
                       dict(name=p.name,
                            bytes=p.stat().st_size,
                            sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                       for p in (fp32, wfp16)
                   ])
        report['windows'].append(row)
        write_json(out / 'build_report.json', report)
        print(json.dumps(row), flush=True)
        del graph, native
        gc.collect()
    write_host_assets(model, source, out / 'host_assets', args.checkpoint,
                      revision, args.windows)
    print('PASS: graphs and host assets written', flush=True)


if __name__ == '__main__':
    main()
