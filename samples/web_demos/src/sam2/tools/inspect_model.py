#!/usr/bin/env python3
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

"""Inventory of a Tensor-API-authored .tflite: signatures, I/O, op histograms,
and weight sharing across signatures (constant buffers referenced by several
signatures' subgraphs).

  python inspect_model.py model.tflite [--json out.json]
"""
import argparse
import collections
import json
import os

from ai_edge_litert import schema_py_generated as S

OPS = {v: k for k, v in vars(S.BuiltinOperator).items() if not k.startswith('_')}
TYPES = {v: k for k, v in vars(S.TensorType).items() if not k.startswith('_')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    buf = open(a.model, 'rb').read()
    m = S.Model.GetRootAsModel(buf, 0)
    bufsize = {}
    for i in range(m.BuffersLength()):
        b = m.Buffers(i)
        bufsize[i] = b.Size() if b.Size() else b.DataLength()
    users = collections.defaultdict(set)
    sigs = []
    for si in range(m.SignatureDefsLength()):
        sd = m.SignatureDefs(si)
        sg = m.Subgraphs(sd.SubgraphIndex())

        def io(n, get):
            out = []
            for i in range(n):
                tm = get(i)
                t = sg.Tensors(tm.TensorIndex())
                out.append({'name': tm.Name().decode(),
                            'shape': t.ShapeAsNumpy().tolist() if t.ShapeLength() else [],
                            'type': TYPES[t.Type()]})
            return out

        hist = collections.Counter()
        for oi in range(sg.OperatorsLength()):
            code = m.OperatorCodes(sg.Operators(oi).OpcodeIndex())
            b = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
            name = OPS.get(b, str(b))
            if name in ('CUSTOM', 'STABLEHLO_COMPOSITE'):
                name += ':' + (code.CustomCode() or b'').decode()
            hist[name] += 1
        wbytes = 0
        for ti in range(sg.TensorsLength()):
            t = sg.Tensors(ti)
            if bufsize.get(t.Buffer(), 0) > 0:
                users[t.Buffer()].add(sd.SignatureKey().decode())
                wbytes += bufsize[t.Buffer()]
        sigs.append({'key': sd.SignatureKey().decode(), 'subgraph': sd.SubgraphIndex(),
                     'inputs': io(sd.InputsLength(), sd.Inputs), 'outputs': io(sd.OutputsLength(), sd.Outputs),
                     'ops': sg.OperatorsLength(), 'hist': dict(hist.most_common()), 'const_bytes': wbytes})
    total = sum(bufsize[b] for b in users)
    shared = sum(bufsize[b] for b, u in users.items() if len(u) > 1)
    info = {'file': os.path.basename(a.model), 'file_bytes': len(buf), 'signatures': sigs,
            'const_bytes_unique': total, 'const_bytes_shared_by_2plus': shared,
            'const_bytes_if_duplicated': sum(s['const_bytes'] for s in sigs)}
    for s in sigs:
        ins = ', '.join(f"{i['name']}{i['shape']}" for i in s['inputs'][:6]) + (' …' if len(s['inputs']) > 6 else '')
        outs = ', '.join(f"{o['name']}{o['shape']}" for o in s['outputs'])
        top = ', '.join(f'{k}×{v}' for k, v in list(s['hist'].items())[:8])
        print(f"{s['key']:12s} {s['ops']:5d} ops  in({len(s['inputs'])}): {ins}\n{'':18s}out: {outs}\n{'':18s}{top}")
    print(f"file {len(buf) / 1e6:.1f} MB · unique constants {total / 1e6:.1f} MB · shared by 2+ signatures "
          f"{shared / 1e6:.1f} MB · {info['const_bytes_if_duplicated'] / 1e6:.1f} MB if each signature had its own copy")
    if a.json:
        json.dump(info, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
