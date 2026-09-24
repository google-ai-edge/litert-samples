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
"""Small .tflite graphs written straight from the schema, no converter.

The bisect tests need graphs whose faulty op is known in advance. Building
them from the flatbuffer schema that ships in the ai-edge-litert wheel keeps
the converter out of the loop, so a result cannot be a converter artifact.
"""

import numpy as np
import flatbuffers
from ai_edge_litert import schema_py_generated as tfl


class GraphBuilder:
    """Append tensors and ops; `build` writes a single-signature model."""

    def __init__(self):
        self.tensors = []
        self.buffers = [tfl.BufferT()]      # buffer 0 is the empty buffer
        self.ops = []
        self.opcodes = []
        self.inputs = []
        self.outputs = []

    def tensor(self, name, shape, ttype=tfl.TensorType.FLOAT32, data=None):
        t = tfl.TensorT()
        t.name = name.encode()
        t.shape = list(shape)
        t.type = ttype
        if data is not None:
            b = tfl.BufferT()
            b.data = np.frombuffer(np.ascontiguousarray(data).tobytes(), np.uint8)
            self.buffers.append(b)
            t.buffer = len(self.buffers) - 1
        else:
            t.buffer = 0
        self.tensors.append(t)
        return len(self.tensors) - 1

    def input(self, name, shape):
        i = self.tensor(name, shape)
        self.inputs.append(i)
        return i

    def const(self, name, array):
        array = np.asarray(array, np.float32)
        return self.tensor(name, array.shape, data=array)

    def _opcode(self, builtin):
        for i, c in enumerate(self.opcodes):
            if c.builtinCode == builtin:
                return i
        c = tfl.OperatorCodeT()
        c.builtinCode = builtin
        c.deprecatedBuiltinCode = min(builtin, 127)
        self.opcodes.append(c)
        return len(self.opcodes) - 1

    def op(self, builtin, inputs, out_name, out_shape, opts_type=None, opts=None):
        o = self.tensor(out_name, out_shape)
        op = tfl.OperatorT()
        op.opcodeIndex = self._opcode(builtin)
        op.inputs = list(inputs)
        op.outputs = [o]
        if opts_type is not None:
            op.builtinOptionsType = opts_type
            op.builtinOptions = opts
        self.ops.append(op)
        return o

    def add(self, a, b, name, shape):
        return self.op(tfl.BuiltinOperator.ADD, [a, b], name, shape,
                       tfl.BuiltinOptions.AddOptions, tfl.AddOptionsT())

    def mul(self, a, b, name, shape):
        return self.op(tfl.BuiltinOperator.MUL, [a, b], name, shape,
                       tfl.BuiltinOptions.MulOptions, tfl.MulOptionsT())

    def pad(self, x, pads, name):
        shape = self.tensors[x].shape
        out = [d + before + after for d, (before, after) in zip(shape, pads)]
        p = self.tensor(name + "_paddings", [len(shape), 2], tfl.TensorType.INT32,
                        np.asarray(pads, np.int32))
        return self.op(tfl.BuiltinOperator.PAD, [x, p], name, out,
                       tfl.BuiltinOptions.PadOptions, tfl.PadOptionsT())

    def sum(self, x, axes, name, keep_dims=False):
        shape = self.tensors[x].shape
        if keep_dims:
            out = [1 if i in axes else d for i, d in enumerate(shape)]
        else:
            out = [d for i, d in enumerate(shape) if i not in axes]
        a = self.tensor(name + "_axes", [len(axes)], tfl.TensorType.INT32,
                        np.asarray(axes, np.int32))
        o = tfl.ReducerOptionsT()
        o.keepDims = keep_dims
        return self.op(tfl.BuiltinOperator.SUM, [x, a], name, out,
                       tfl.BuiltinOptions.ReducerOptions, o)

    def sub(self, a, b, name, shape):
        return self.op(tfl.BuiltinOperator.SUB, [a, b], name, shape,
                       tfl.BuiltinOptions.SubOptions, tfl.SubOptionsT())

    def build(self, path, signature="main"):
        sub = tfl.SubGraphT()
        sub.name = b"main"
        sub.tensors = self.tensors
        sub.inputs = self.inputs
        sub.outputs = self.outputs
        sub.operators = self.ops

        def tensor_map(i):
            m = tfl.TensorMapT()
            m.name = self.tensors[i].name
            m.tensorIndex = i
            return m

        sd = tfl.SignatureDefT()
        sd.signatureKey = signature.encode()
        sd.subgraphIndex = 0
        sd.inputs = [tensor_map(i) for i in self.inputs]
        sd.outputs = [tensor_map(i) for i in self.outputs]

        m = tfl.ModelT()
        m.version = 3
        m.operatorCodes = self.opcodes
        m.subgraphs = [sub]
        m.buffers = self.buffers
        m.signatureDefs = [sd]
        m.description = b"litert_gpu_toolkit test graph"
        b = flatbuffers.Builder(1024)
        b.Finish(m.Pack(b), b"TFL3")
        with open(path, "wb") as f:
            f.write(b.Output())
        return path


def pad_rank3_graph(path):
    """ADD, MUL, PAD(rank 3, middle axis), ADD, MUL — LiteRT #9272 at op 2.

    The PAD pads a non-innermost axis of a rank-3 tensor, the exact shape
    of the issue's self-contained repro (`[16, 32, 1] -> [16, 64, 1]`),
    surrounded by elementwise ops that are correct on their own.
    """
    g = GraphBuilder()
    x = g.input("x", [16, 32, 1])
    one = g.const("one", [1.0])
    two = g.const("two", [2.0])
    a = g.add(x, one, "add1", [16, 32, 1])
    m = g.mul(a, two, "mul1", [16, 32, 1])
    p = g.pad(m, [(0, 0), (0, 32), (0, 0)], "pad_mid")
    a2 = g.add(p, one, "add2", [16, 64, 1])
    g.outputs = [g.mul(a2, two, "mul2", [16, 64, 1])]
    return g.build(path)


def sum_overflow_graph(path):
    """MUL, SUM over HxW of [1,112,112,32], MUL — LiteRT #9249 at op 1.

    With uniform [0, 1) input scaled by 40 the per-channel total is about
    2.5e5, past the fp16 maximum of 65504, so a SUM that accumulates in
    fp16 returns Inf and the graph output is NaN.
    """
    g = GraphBuilder()
    x = g.input("x", [1, 112, 112, 32])
    m = g.mul(x, g.const("gain", [40.0]), "mul1", [1, 112, 112, 32])
    s = g.sum(m, [1, 2], "sum_hw")
    g.outputs = [g.mul(s, g.const("scale", [0.001]), "scale", [1, 32])]
    return g.build(path)


def aliased_add_then_pad_graph(path):
    """ADD, SUM(keep dims), SUB, PAD(rank 3, middle axis) — the PAD is the
    only wrong op, at index 3.

    The ADD output is consumed by the SUM and by the SUB, so the bisect's
    cut after the SUM has to expose it while the SUM also reads it. Exposed
    directly, that is the "output that is also consumed" pattern of LiteRT
    #8599, which the Metal accelerator gets wrong by itself; the bisect
    copies such tensors through a RESHAPE, so it must still name the PAD.
    """
    g = GraphBuilder()
    x = g.input("x", [16, 32, 8])
    y = g.input("y", [16, 32, 8])
    t = g.add(x, y, "t", [16, 32, 8])
    s = g.sum(t, [2], "s", keep_dims=True)
    z = g.sub(t, s, "z", [16, 32, 8])
    g.outputs = [g.pad(z, [(0, 0), (0, 32), (0, 0)], "pad_mid")]
    return g.build(path)


def native_aliased_add_graph(path):
    """ADD whose output is a graph output *and* consumed by a SUM — LiteRT
    #8599 as shipped in a model, at op 0. The bisect must report the ADD's
    output as wrong once the SUM is appended, marked context_dependent.
    """
    g = GraphBuilder()
    x = g.input("x", [1, 8, 256])
    y = g.input("y", [1, 8, 256])
    t = g.add(x, y, "t", [1, 8, 256])
    s = g.sum(t, [2], "s")
    g.outputs = [t, s]
    return g.build(path)
