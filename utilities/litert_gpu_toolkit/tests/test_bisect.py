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
"""The bisect must name the op each reference issue describes.

Three reproductions, each a GPU silent miscompute reported against LiteRT:

- google-ai-edge/LiteRT#9272: a rank-3 PAD on a non-innermost axis returns
  row-shifted data (precision is not a factor).
- google-ai-edge/LiteRT#9249: SUM accumulates in fp16, so a reduction whose
  total exceeds 65504 returns NaN — efficientnet_b0's first squeeze-excite
  SUM over [1,112,112,32].
- google-ai-edge/LiteRT#8619: the SAM 2.1 Hiera-Tiny mask decoder exported
  with rank-2/rank-3 tensors miscomputes while the rank-4 export of the
  same weights is correct.

The first two run on synthetic graphs (no download) and, when the network
allows, on the public models. Every GPU test skips where the LiteRT GPU
accelerator is unavailable; the graph-analysis tests run anywhere.

Run:  cd utilities && python -m pytest litert_gpu_toolkit/tests -v
"""

import json
import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("ai_edge_litert")

from litert_gpu_toolkit import checker  # noqa: E402
from litert_gpu_toolkit.tests.graphs import (  # noqa: E402
    GraphBuilder, aliased_add_then_pad_graph, native_aliased_add_graph,
    pad_rank3_graph, sum_overflow_graph)


@pytest.fixture
def tmp_tflite():
    d = tempfile.mkdtemp(prefix="litert_bisect_test_")
    return lambda name: os.path.join(d, name)


# ---------------------------------------------------------------- graph analysis

def test_frontier_and_constant_analysis(tmp_tflite):
    """Cut bookkeeping on the PAD graph, no accelerator involved."""
    fu, model = checker._load_model(pad_rank3_graph(tmp_tflite("pad.tflite")))
    sg = model.subgraphs[0]
    const = checker._constant_derived(model, sg)
    names = lambda ts: [sg.tensors[t].name.decode() for t in ts]
    assert "one" in names(const) and "pad_mid_paddings" in names(const)
    assert checker._runtime_ops(sg, const) == [0, 1, 2, 3, 4]
    # After op 1 only mul1's output is live; after op 2 only the PAD output.
    assert names(checker._frontier_tensors(sg, 1, const)) == ["mul1"]
    assert names(checker._frontier_tensors(sg, 2, const)) == ["pad_mid"]
    # The final cut exposes the graph output.
    assert names(checker._frontier_tensors(sg, 4, const)) == ["mul2"]


def test_weight_only_ops_are_not_cut_points(tmp_tflite):
    """A DEQUANTIZE of a constant is weight handling, not a candidate op."""
    from ai_edge_litert import schema_py_generated as tfl
    g = GraphBuilder()
    x = g.input("x", [1, 4])
    w16 = g.tensor("w16", [4, 4], tfl.TensorType.FLOAT16,
                   np.eye(4, dtype=np.float16))
    w = g.op(tfl.BuiltinOperator.DEQUANTIZE, [w16], "w", [4, 4],
             tfl.BuiltinOptions.DequantizeOptions, tfl.DequantizeOptionsT())
    fc = tfl.FullyConnectedOptionsT()
    y = g.op(tfl.BuiltinOperator.FULLY_CONNECTED, [x, w, -1], "y", [1, 4],
             tfl.BuiltinOptions.FullyConnectedOptions, fc)
    g.outputs = [y]
    fu, model = checker._load_model(g.build(tmp_tflite("wo.tflite")))
    sg = model.subgraphs[0]
    const = checker._constant_derived(model, sg)
    assert w in const                      # dequantized weight is still a weight
    assert checker._runtime_ops(sg, const) == [1]


# ---------------------------------------------------------------- LiteRT #9272

def test_pad_rank3_bisect_names_the_pad(gpu, tmp_tflite):
    path = pad_rank3_graph(tmp_tflite("pad.tflite"))
    for enforce_f32 in (False, True):      # "precision is not a factor"
        r = checker.bisect_gpu_divergence(path, enforce_f32=enforce_f32)
        assert r['diverges'] is True, r['errors']
        op = r['first_divergent_op']
        assert op['name'] == 'PAD' and op['index'] == 2
        assert op['inputs'][0]['shape'] == [16, 32, 1]
        assert op['outputs'][0]['shape'] == [16, 64, 1]
        assert r['clean_prefix_end'] == 1
        assert [t['tensor'] for t in r['diverging_tensors']] == [op['outputs'][0]['index']]
        assert not r['diverging_tensors'][0]['context_dependent']


def test_promoted_tensors_do_not_induce_issue_8599(gpu, tmp_tflite):
    """A cut that exposes a tensor the prefix also consumes goes through a
    RESHAPE copy; the bisect must name the PAD at op 3, not the SUM at op 1."""
    path = aliased_add_then_pad_graph(tmp_tflite("alias_pad.tflite"))
    fu, model = checker._load_model(path)
    # The cut after the SUM exposes t (consumed by SUM and SUB) and s.
    out = tmp_tflite("prefix1.tflite")
    named = checker._write_prefix(fu, model, 0, model.signatureDefs[0], 1, out)
    fu2, prefix = checker._load_model(out)
    psg = prefix.subgraphs[0]
    assert len(psg.operators) == 3                     # ADD, SUM, + one RESHAPE copy
    assert fu2.opcode_to_name(prefix, psg.operators[-1].opcodeIndex) == "RESHAPE"
    assert len(named) == 2                             # t and s
    for enforce_f32 in (False, True):
        r = checker.bisect_gpu_divergence(path, enforce_f32=enforce_f32)
        assert r['diverges'] is True, r['errors']
        assert r['first_divergent_op']['name'] == 'PAD'
        assert r['first_divergent_op']['index'] == 3
        assert r['clean_prefix_end'] == 2


def test_native_output_also_consumed_is_context_dependent(gpu, tmp_tflite):
    """LiteRT #8599 as shipped: the ADD output is a graph output and the SUM
    reads it. On the Metal accelerator the output reads back wrong; the
    bisect attributes it to the SUM being appended, with the ADD's tensor
    marked context_dependent."""
    path = native_aliased_add_graph(tmp_tflite("native_alias.tflite"))
    r = checker.bisect_gpu_divergence(path, enforce_f32=True)
    if not r['diverges']:
        pytest.skip("this accelerator does not reproduce LiteRT #8599")
    assert r['first_divergent_op']['name'] == 'SUM'
    bad = r['diverging_tensors'][0]
    assert bad['producer_name'] == 'ADD' and bad['context_dependent'] is True


# ---------------------------------------------------------------- LiteRT #9249

def test_sum_fp16_overflow_bisect_names_the_sum(gpu, tmp_tflite):
    path = sum_overflow_graph(tmp_tflite("sum.tflite"))
    r = checker.bisect_gpu_divergence(path, input_distribution="uniform")
    assert r['diverges'] is True, r['errors']
    assert r['criterion'] == 'nonfinite'
    op = r['first_divergent_op']
    assert op['name'] == 'SUM' and op['index'] == 1
    assert op['inputs'][0]['shape'] == [1, 112, 112, 32]
    assert op['outputs'][0]['shape'] == [1, 32]
    assert r['diverging_tensors'][0]['nonfinite'] > 0
    # The issue asks for fp32 accumulation; fp32 compute is the control.
    ok = checker.check_gpu_compatibility(path, input_distribution="uniform",
                                         enforce_f32=True)
    assert ok['compatible'], ok['errors']


def test_efficientnet_b0_issue_9249(gpu, public_model):
    """The public model: the first SE SUM over [1,112,112,32] -> [1,32]."""
    path = public_model("efficientnet_b0", "efficientnet_b0.tflite")
    r = checker.check_gpu_compatibility(path, input_distribution="uniform",
                                        bisect=True)
    assert not r['compatible']
    b = r['bisect']['serving_default']
    op = b['first_divergent_op']
    assert op['name'] == 'SUM'
    assert op['inputs'][0]['shape'] == [1, 112, 112, 32]
    assert op['outputs'][0]['shape'] == [1, 32]
    bad = b['diverging_tensors'][0]
    assert bad['nonfinite'] > 0

    # The issue's claim: the NaN channels are exactly those whose true sum
    # exceeds the fp16 maximum. Read that SUM on the CPU interpreter.
    from ai_edge_litert.interpreter import Interpreter
    itp = Interpreter(model_path=path, experimental_preserve_all_tensors=True)
    itp.allocate_tensors()
    inp = itp.get_input_details()[0]
    rng = np.random.default_rng(0)
    x = checker._random_inputs({inp['name']: inp}, rng, "uniform")[inp['name']]
    itp.set_tensor(inp['index'], x)
    itp.invoke()
    cpu_sum = itp.get_tensor(op['outputs'][0]['index']).reshape(-1)
    assert int((cpu_sum > 65504).sum()) == bad['nonfinite']


# ---------------------------------------------------------------- LiteRT #8619

def test_sam2_mask_decoder_issue_8619(gpu, public_model):
    """v1 (rank-2/3 export) diverges, v2 (rank-4 export) does not.

    On the macOS Metal accelerator the first wrong op in v1 is the ADD that
    sums the flattened image embedding with its positional encoding at rank
    2 (`[4096, 256] + const [4096, 256]`); v2 carries the same add at rank 3
    and is correct. Both exports share weights and I/O.
    """
    v1 = public_model("SAM2.1-Hiera-Tiny-Mask-Decoder", "sam2_tiny_mask_decoder_fp16.tflite")
    v2 = public_model("SAM2.1-Hiera-Tiny-Mask-Decoder", "sam2_tiny_mask_decoder_v2_fp16.tflite")
    good = checker.check_gpu_compatibility(v2, enforce_f32=True)
    assert good['compatible'], good['errors']

    r = checker.check_gpu_compatibility(v1, enforce_f32=True, bisect=True)
    assert not r['compatible']
    op = r['bisect']['serving_default']['first_divergent_op']
    assert op['name'] == 'ADD'
    shapes = [tuple(t['shape']) for t in op['inputs']]
    assert shapes == [(4096, 256), (4096, 256)]
    assert [t['constant'] for t in op['inputs']] == [False, True]


# ---------------------------------------------------------------- CLI

def test_cli_json(gpu, tmp_tflite):
    path = pad_rank3_graph(tmp_tflite("pad.tflite"))
    out = tmp_tflite("report.json")
    rc = checker._main([path, "--bisect", "--json", out])
    assert rc == 1
    with open(out) as f:
        report = json.load(f)
    assert report['bisect']['main']['first_divergent_op']['name'] == 'PAD'
