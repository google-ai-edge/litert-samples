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

"""Copies tensor/backends/tflite/tflite_flatbuffer_conversion.cc for the wasm
build without its in-process TFLite interpreter.

ModelFactory (graph -> .tflite flatbuffer) only needs the flatbuffers schema;
the file's eager `Run(outputs)` helper additionally builds a TFLite
interpreter (+ XNNPACK). The browser runs models through LiteRT.js, so that
helper becomes an Unimplemented stub, and XNN_EXTRA_BYTES (weight tail
padding, 16 outside Hexagon) is defined directly.

  python patch_tflite_conversion.py <in.cc> <out.cc>
"""
import re
import sys

DROP = ('"xnnpack.h"', '"tflite/core/interpreter_builder.h"', '"tflite/core/kernels/register.h"',
        '"tflite/interpreter.h"', '"tflite/model_builder.h"')


def main():
    src, dst = sys.argv[1], sys.argv[2]
    s = open(src).read()
    lines = [ln for ln in s.split('\n') if not (ln.startswith('#include') and any(d in ln for d in DROP))]
    s = '\n'.join(lines)
    s = s.replace('namespace litert::tensor {', '#ifndef XNN_EXTRA_BYTES\n#define XNN_EXTRA_BYTES 16\n#endif\n\n'
                  'namespace litert::tensor {', 1)
    head = 'absl::Status Run(std::vector<TensorHandle> outputs) {'
    i = s.index(head)
    depth, j = 0, i + len(head) - 1
    while True:  # matching brace of the function body
        if s[j] == '{':
            depth += 1
        elif s[j] == '}':
            depth -= 1
            if depth == 0:
                break
        j += 1
    stub = (head + '\n  (void)outputs;\n  return absl::UnimplementedError(\n'
            '      "eager Run needs the TFLite interpreter (not in the wasm build)");\n}')
    s = s[:i] + stub + s[j + 1:]
    assert 'tflite::Interpreter' not in s and not re.search(r'InterpreterBuilder', s)
    open(dst, 'w').write(s)
    print('patched', dst)


if __name__ == '__main__':
    main()
