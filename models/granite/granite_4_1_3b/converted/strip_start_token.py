# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Removes the `start_token` from a .litertlm's LlmMetadata (weights untouched).

Why this exists — measured on granite-4.1-3b:

The bundle builder sets `start_token` from `tokenizer.bos_token` UNCONDITIONALLY
and never consults `add_bos_token`. For a tokenizer that declares
`add_bos_token: False` the runtime then prepends a token the model was never fed
at that position — and when that token is ALSO the EOS (granite:
bos == eos == `<|end_of_text|>`, id 100257), the model reads the prompt as a
finished document and degenerates: it echoes the question back, or emits a run
of backticks, instead of answering.

This is not quantization damage, and the check that proves it costs one minute
(`verify_granite_4_1_3b.py --bos-ab`): feed the SAME rendered prompt to the bf16
PyTorch model with and without the leading BOS. granite answers "There are 7
days in a week." without it and "Answer briefly." with it; a bundle carrying the
start_token reproduces the with-BOS behaviour exactly.

For a fresh export, build_granite_4_1_3b.py already applies the fix. This script
repairs a bundle that is already built (unpack -> delete the `start_token`
block -> pack; ~30 s, weights untouched):

    python strip_start_token.py in.litertlm out.litertlm

Needs the `litert-lm` CLI (pip install litert-lm, >= 0.15 for unpack/pack).
Unpacking writes sections as large as the model — keep --work-dir on a disk
with room.
"""
import argparse
import os
import re
import subprocess


def strip_block(pbtext, field="start_token"):
  """Drops a top-level `field { ... }` block from a text-format proto.

  Brace-counting rather than a regex: the block contains nested braces, and the
  metadata's stop_tokens use the same shape, so a non-greedy regex would eat the
  wrong span.
  """
  m = re.search(rf"^{field}\s*\{{", pbtext, re.M)
  if not m:
    return pbtext, False
  i = pbtext.index("{", m.start())
  depth = 0
  for j in range(i, len(pbtext)):
    if pbtext[j] == "{":
      depth += 1
    elif pbtext[j] == "}":
      depth -= 1
      if depth == 0:
        end = j + 1
        while end < len(pbtext) and pbtext[end] == "\n":
          end += 1
        return pbtext[:m.start()] + pbtext[end:], True
  raise SystemExit(f"unbalanced braces in {field} block")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("src")
  ap.add_argument("dst")
  ap.add_argument("--field", default="start_token")
  ap.add_argument("--litert-lm", default="litert-lm",
                  help="path to the litert-lm CLI — pick the build that matches "
                       "the bundle's builder version")
  ap.add_argument("--work-dir", default=None,
                  help="where to unpack (default: a temp dir next to dst; the "
                       "sections are as big as the model, so keep it on a disk "
                       "with room)")
  args = ap.parse_args()

  work = args.work_dir or os.path.join(os.path.dirname(os.path.abspath(args.dst)),
                                       ".strip_start_token_tmp")
  os.makedirs(work, exist_ok=True)
  unpack = os.path.join(work, "unpack")
  subprocess.run([args.litert_lm, "unpack", args.src, "--output-dir", unpack],
                 check=True)

  pb = os.path.join(unpack, "LlmMetadataProto.pbtext")
  text = open(pb).read()
  new, found = strip_block(text, args.field)
  if not found:
    raise SystemExit(f"no `{args.field}` block in {pb} — nothing to strip")
  open(pb, "w").write(new)

  toml_path = os.path.join(unpack, "model.toml")
  if os.path.exists(args.dst):  # `litert-lm pack` exits 0 without writing otherwise
    os.remove(args.dst)
  subprocess.run([args.litert_lm, "pack", toml_path, "--output", args.dst],
                 check=True)
  print(f"OK: {args.dst} ({args.field} removed)")


if __name__ == "__main__":
  main()
