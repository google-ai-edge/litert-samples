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

"""CPU references for the LFM3-1.7B G6 export, and the check of phone replies.

A bundle is checked by its replies: the phone's reply to each prompt in
prompts/ must be the text the uncompiled export gives on the CPU.

  reference: each prompt through WORK/export on the CPU (XNNPACK through
    ai_edge_litert's interpreter), greedy, at most 256 tokens, fed as
    LiteRT-LM's NPU executor feeds the bundle: the bundle's chat template and
    start token, prefill calls of 128 slots, the last prompt token held back
    for the first decode step, and the auxiliary model's RoPE, masks and cache
    updates. Writes WORK/reference.json.

      python verify_lfm3_1_7b_g6.py reference --work out_g6

  compare: each <prompt>.txt in --replies (what the LiteRT-LM CLI printed for
    that prompt) against WORK/reference.json, with the CLI's [thought] markers
    and the <think> tags taken out. On a difference it prints both texts there
    and how far apart the CPU's two best tokens were at that step: under about
    0.3, 16-bit floats can pick the other token.

      python verify_lfm3_1_7b_g6.py compare --work out_g6 --replies replies

Single turns only: on the NPU, stock LiteRT-LM keeps each turn's thought in the
KV cache, so a later turn sees a context that the chat template does not render.
"""

import argparse
import functools
import json
import sys
from pathlib import Path

from ai_edge_litert import interpreter as interpreter_lib
import jinja2
from litert_lm_builder.runtime.proto import llm_metadata_pb2
import numpy as np
import tokenizers

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"
STEPS = 256  # the phone run's --max_output_tokens


@functools.cache
def interpreter(path, threads):
  # One interpreter per file: XNNPACK packs the weights once per interpreter.
  return interpreter_lib.Interpreter(model_path=str(path), num_threads=threads)


def run(runner, **feeds):
  want = runner.get_input_details()
  return runner(**{k: np.asarray(v, dtype=want[k]["dtype"]).reshape(
      want[k]["shape"]) for k, v in feeds.items() if k in want})


class Export:
  """The uncompiled bundle: export_hf's models, metadata and tokenizer."""

  def __init__(self, export_dir, threads):
    self.dir, self.threads = export_dir, threads
    self.tok = tokenizers.Tokenizer.from_file(str(export_dir / "tokenizer.json"))
    meta = llm_metadata_pb2.LlmMetadata()
    meta.ParseFromString((export_dir / "llm_metadata.pb").read_bytes())
    self.template = meta.jinja_prompt_template
    self.bos = self.tok.token_to_id(meta.start_token.token_str)
    ((self.eos,),) = [s.token_ids.ids for s in meta.stop_tokens]
    sigs = self.interpreter("model_quantized.tflite").get_signature_list()
    self.t = int(next(s for s in sigs if s.startswith("prefill_"))[8:])

  def interpreter(self, name):
    return interpreter(self.dir / name, self.threads)

  def runner(self, name, signature):
    return self.interpreter(name).get_signature_runner(signature)

  def render(self, prompt):
    """The prompt's ids as LiteRT-LM puts them in the cache."""
    env = jinja2.Environment()
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(
        ValueError(m))
    text = env.from_string(self.template).render(
        messages=[{"role": "user", "content": prompt}],
        add_generation_prompt=True, bos_token="")
    return [self.bos] + self.tok.encode(text, add_special_tokens=False).ids

  def generate(self, ids):
    """Greedy tokens after `ids`, and each step's gap between the top two."""
    t = self.t
    prefill = self.runner("model_quantized.tflite", f"prefill_{t}")
    decode = self.runner("model_quantized.tflite", "decode")
    cache = {k: np.zeros(v["shape"], v["dtype"])
             for k, v in prefill.get_input_details().items()
             if k.startswith("kv_cache")}

    def call(main, tokens, positions, valid, time_step):
      n = len(tokens)
      pre, sfx = ("prefill", f"_{n}") if n > 1 else ("decode", "")
      feeds = dict(cache)
      feeds.update(run(self.runner("embedder_direct.tflite",
                                   f"{pre}_embedder{sfx}"),
                       token_ids=tokens[None]))
      feeds.update(run(self.runner("auxiliary.tflite", f"{pre}_rope{sfx}"),
                       input_pos=positions))
      feeds.update(run(self.runner("auxiliary.tflite", f"{pre}_mask{sfx}"),
                       input_tokens=tokens[None], time_step=np.int32(time_step),
                       valid_mask=valid[None]))
      outs = run(main, **feeds)
      slices = {k: v for k, v in outs.items() if k.startswith("kv_slice")}
      cache.update(run(
          self.runner("auxiliary.tflite", f"{pre}_cache_update{sfx}"),
          **cache, **slices, input_pos=positions, valid_mask=valid[None]))
      return outs.get("logits")

    # Each prefill call's last id is held back as the next call's first, so
    # the first call takes t - 1 ids. Padding slots: token 0, position 0.
    body = ids[:-1]
    bounds = sorted({0, len(body)} | set(range(t - 1, len(body), t)))
    for s, e in zip(bounds, bounds[1:]):
      tokens, positions = np.zeros(t, np.int32), np.zeros(t, np.int32)
      tokens[:e - s], positions[:e - s] = body[s:e], np.arange(s, e)
      call(prefill, tokens, positions, np.arange(t) < e - s, s)
    out, gaps, nxt = [], [], ids[-1]
    for pos in range(len(body), len(body) + STEPS):
      logits = call(decode, np.array([nxt], np.int32),
                    np.array([pos], np.int32), np.array([True]), pos)[0, 0]
      first, second = np.argsort(-logits, kind="stable")[:2]
      nxt = int(first)
      out.append(nxt)
      gaps.append(round(float(logits[first] - logits[second]), 3))
      if nxt == self.eos:
        break
    return out, gaps


def bare(text):
  """The reply without the thought markers of the CLI or of the tokenizer."""
  for mark in ("[thought] ", "[thought]", "[/thought]\n", "[/thought]",
               "<think>", "</think>"):
    text = text.replace(mark, "", 1)
  return text.strip()


def reference(args):
  ex = Export(args.work / "export", args.threads)
  refs = {}
  for path in sorted(PROMPTS.glob("*.txt")):
    ids = ex.render(path.read_text())
    out, gaps = ex.generate(ids)
    reply = [i for i in out if i != ex.eos]
    text = ex.tok.decode(reply, skip_special_tokens=False)
    # Where each step's token ends in the bare text, to place a difference.
    ends = [len(bare(ex.tok.decode(reply[:i + 1], skip_special_tokens=False)))
            for i in range(len(reply))]
    refs[path.stem] = {"context_ids": len(ids), "ids": out, "text": text,
                       "gaps": gaps, "ends": ends}
    print(f"{path.stem}: {len(ids)} prompt ids, {len(out)} tokens,"
          f" {'ends with the stop id' if out[-1] == ex.eos else 'no stop id'},"
          f" {sum(g < 0.3 for g in gaps)} steps with the top two under 0.3"
          f" apart (smallest {min(gaps)})", flush=True)
  out_path = args.work / "reference.json"
  out_path.write_text(json.dumps(refs, ensure_ascii=False))
  print(f"wrote {out_path}")


def compare(args):
  refs = json.loads((args.work / "reference.json").read_text())
  equal = 0
  for name, ref in refs.items():
    path = args.replies / f"{name}.txt"
    if not path.exists():
      print(f"{name}: no {path}")
      continue
    want, got = bare(ref["text"]), bare(path.read_text(errors="replace"))
    if want == got:
      equal += 1
      print(f"{name}: EQUAL ({len(want)} characters)")
      continue
    n = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b),
             min(len(want), len(got)))
    step = next((i for i, end in enumerate(ref["ends"]) if end > n),
                len(ref["ends"]) - 1)
    print(f"{name}: DIFFERENT from character {n} (CPU {len(want)}, phone"
          f" {len(got)}), at step {step}, where the CPU's top two tokens are"
          f" {ref['gaps'][step]} apart")
    print(f"  CPU:   {want[max(0, n - 60):n + 60]!r}")
    print(f"  phone: {got[max(0, n - 60):n + 60]!r}")
  print(f"equal {equal} of {len(refs)}")
  return 0 if equal == len(refs) else 1


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("mode", choices=["reference", "compare"])
  ap.add_argument("--work", type=Path, default=Path("out_g6"),
                  help="the build's working directory")
  ap.add_argument("--replies", type=Path, help="compare: folder of <prompt>.txt")
  ap.add_argument("--threads", type=int, default=4,
                  help="reference: CPU threads of the interpreter")
  args = ap.parse_args()
  if args.mode == "compare":
    if not args.replies:
      ap.error("compare needs --replies")
    return compare(args)
  reference(args)
  return 0


if __name__ == "__main__":
  sys.exit(main())
