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

"""Quality gate, metadata and model-specific checks for a converted Nemotron-3-Nano-4B bundle.

Four modes:

  gate (default) — asks 8 fixed, unambiguously checkable questions through the
    `litert-lm` CLI (pip install litert-lm, >= 0.15 for the hybrid state
    binding), greedy, one fresh session per question, and scores correctness
    plus degeneracy. This is a reasoning model: the vendor template opens the
    reply with `<think>`, so the answer is whatever follows the last
    `</think>`; a reply whose thought never closes counts as no answer. Every
    run passes `--cache no` — on this family the GPU path fails outright with
    the compiled-graph cache (see --cache-ab), and on CPU the cache is a
    weight-sized file written next to the model.

      python verify_nemotron_3_nano_4b.py model.litertlm [--backend cpu|gpu]

  --check-metadata — reads the bundle (no unpacking) and asserts the shape
    this recipe produces: NO start_token, <|im_end|> (11) and the exported
    id 2 among the stop ids, the vendor Jinja template embedded byte-equal to
    the repo's chat_template.jinja, an ExecutorMetadata section with the 50
    hybrid state buffers (42 mamba conv+SSM, 8 attention K/V),
    `prefer_activation_type = fp32` declared on the model section, and a
    bytes-per-parameter ratio in the int8 band. Needs litert-lm-builder.

      python verify_nemotron_3_nano_4b.py model.litertlm --check-metadata

  --cache-ab — the one-variable reproducer of the GPU trap: the same
    question on the GPU backend with the runtime's default compiled-graph
    cache and with `--cache no`. On litert-lm 0.16.0 the default arm dies
    (WebGPU "Invalid BindGroup" validation errors) and the `--cache no` arm
    answers. Any file the default arm writes next to the model is removed.

      python verify_nemotron_3_nano_4b.py model.litertlm --cache-ab

  --bos-ab — the BOS finding measured on the unquantized model: HF bf16
    greedy decoding of the SAME rendered prompt with and without a leading
    `<s>`, three probes. The recorded result is byte-identical on 2 of 3 and a
    mid-thought divergence to the same final answer on the third — the
    dropped start_token aligns the stream with training, it does not change
    the verdict at this scale. Needs torch + transformers and the source
    checkpoint (8 GB); `use_cache=False` because transformers' generic cache
    has no `mlp` layer type for this architecture.

      python verify_nemotron_3_nano_4b.py model.litertlm --bos-ab [--device cpu|mps]
"""

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
PARAMS = 3_973_556_832  # every tensor in the checkpoint; embeddings are untied
IM_END_ID, EXPORTED_EOS_ID = 11, 2
# The repo's chat_template.jinja — the one AutoTokenizer resolves. (The same
# repo's tokenizer_config.json carries a different, 10,497-byte copy.)
TEMPLATE_BYTES = 10504
STATE_LINEAR_ATTN, STATE_KV = 42, 8  # 21 mamba layers x2, 4 attention layers x2

SUFFIX = " Answer briefly."
# (label, question, answer-regex, wrong-answer-veto-regex-or-None).
QUESTIONS = [
    ("17+25=42", "What is 17 + 25?", r"\b42\b", None),
    ("capital=Tokyo", "What is the capital of Japan?", r"tokyo", None),
    ("opp(hot)=cold", 'What is the opposite of "hot"?', r"\bcold\b", None),
    ("days/week=7", "How many days are in a week?", r"\bseven\b|\b7\b", None),
    ("thanks(fr)=merci", 'How do you say "thank you" in French?', r"merci", None),
    ("8*7=56", "What is 8 times 7?", r"\b56\b", None),
    ("0.9>0.11", "Which is larger: 0.9 or 0.11?", r"0\.9\b",
     r"0\.11\s*(is|>)\s*(the\s+)?(larg|great|bigg)"),
    ("rhyme=blue", 'Complete the rhyme: "Roses are red, violets are ___"',
     r"\bblue\b", None),
]
BOS_PROBES = ["What is 17 + 25?", "Name the capital of France.",
              "Say 'hello' in Japanese."]


def degenerate(text):
  """True if the output loops or is special-token spam."""
  words = text.split()
  if len(words) >= 10:
    grams = [" ".join(words[i:i + 5]) for i in range(len(words) - 4)]
    top = Counter(grams).most_common(1)[0][1] if grams else 0
    diversity = len(set(words)) / len(words)
    if top >= 3 and (top >= 5 or diversity < 0.50):
      return True
    if diversity < 0.30:
      return True
  if len(text) >= 40 and len(set(text)) < 15:
    return True
  if text.count("<|") >= 5 or text.count("<pad>") >= 5 or text.count("<SPECIAL_") >= 5:
    return True
  return False


def final_answer(text):
  """The part of a reply after its thought. The template already emitted
  `<think>`, so the stream is thought text, `</think>`, then the answer."""
  if "</think>" in text:
    return text.rsplit("</think>", 1)[1].strip()
  if "<think>" in text or text.strip():
    # a thought that never closed — no final answer was produced
    return ""
  return ""


def run_cli(litert_lm, model, prompt, backend, timeout, cache="no"):
  cmd = [litert_lm, "run", str(model), "--backend", backend,
         "--temperature", "0", "--top-k", "1"]
  if cache:
    cmd += ["--cache", cache]
  cmd += ["--prompt", prompt]
  return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def run_one(litert_lm, model, prompt, backend, timeout):
  proc = run_cli(litert_lm, model, prompt, backend, timeout)
  if proc.returncode != 0:
    raise RuntimeError(
        f"litert-lm run failed (exit={proc.returncode}).\n"
        f"--- stderr tail ---\n{proc.stderr[-1500:]}")
  return proc.stdout.strip()


def gate(args):
  results = []
  for label, q, pat, veto in QUESTIONS:
    try:
      raw = run_one(args.litert_lm, args.model, q + SUFFIX, args.backend, args.timeout)
    except Exception as e:  # noqa: BLE001
      print(f"FAIL (harness) on '{label}': {e}", file=sys.stderr)
      return 2
    ans = final_answer(raw)
    low = ans.lower()
    ok = bool(re.search(pat, low)) and not (veto and re.search(veto, low))
    degen = (not ans) or degenerate(raw)
    results.append({"label": label, "question": q, "ok": ok, "degenerate": degen,
                    "answer": ans, "raw": raw})
    shown = " ".join(ans.split())[:70] if ans else "(no answer — thought never closed)"
    print(f"  [{'v' if ok else '.'}]{' DEGEN' if degen else '      '} "
          f"{label:16s} -> {shown!r}  (thought {len(raw.split())} words)")

  score = sum(r["ok"] for r in results)
  any_degen = any(r["degenerate"] for r in results)
  passed = (score >= args.min_correct) and (not any_degen)
  print(f"\n  correct: {score}/8   degenerate: {'YES' if any_degen else 'no'}")
  print(f"  VERDICT: {'PASS' if passed else 'FAIL'} "
        f"(threshold {args.min_correct}/8, non-degenerate)")
  if args.json:
    Path(args.json).write_text(json.dumps({
        "model": str(args.model), "backend": args.backend, "cache": "no",
        "score": score, "of": 8, "degenerate": any_degen, "passed": passed,
        "questions": results}, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {args.json}")
  return 0 if passed else 1


def read_bundle(model, jinja_path=None):
  """(LlmMetadata pbtext, ExecutorMetadata pbtext, [sections]) via peek."""
  from litert_lm_builder import litertlm_peek  # pip install litert-lm-builder

  buf = io.StringIO()
  litertlm_peek.peek_litertlm_file(str(model), None, buf,
                                   jinja_prompt_template_path=jinja_path)
  text = buf.getvalue()
  m = re.search(r"start of LlmMetadata\n(.*?)>{4,} end of LlmMetadata", text, re.S)
  pbtext = m.group(1) if m else ""
  m = re.search(r"start of ExecutorMetadata\n(.*?)>{4,} end of ExecutorMetadata", text, re.S)
  exec_pbtext = m.group(1) if m else ""
  sections = []
  for sm in re.finditer(
      r"Section (\d+):\n(.*?)Begin Offset:\s*(\d+)\n\s*End Offset:\s*(\d+)\n"
      r"\s*Data Type:\s*(\S+)", text, re.S):
    items = sm.group(2)
    act = re.search(r"prefer_activation_type, Value \(String\): (\S+)", items)
    sections.append({"index": int(sm.group(1)), "type": sm.group(5),
                     "activation": act.group(1) if act else None,
                     "bytes": int(sm.group(4)) - int(sm.group(3))})
  return pbtext, exec_pbtext, sections


def check_metadata(args):
  with tempfile.TemporaryDirectory() as td:
    jinja_path = os.path.join(td, "template.jinja")
    pbtext, exec_pbtext, sections = read_bundle(args.model, jinja_path)
    embedded = Path(jinja_path).read_bytes() if os.path.exists(jinja_path) else b""
  if not pbtext:
    print("FAIL: no LlmMetadata section found")
    return 1
  checks = []

  def check(name, ok, detail=""):
    checks.append(ok)
    print(f"  [{'v' if ok else 'X'}] {name}{(': ' + detail) if detail else ''}")

  check("no start_token (add_bos_token False; template opens with <|im_start|>)",
        not re.search(r"^\s*start_token\s*\{", pbtext, re.M))
  stop_ids = {int(x) for x in re.findall(r"ids:\s*(\d+)", pbtext)}
  check(f"<|im_end|> ({IM_END_ID}) and the exported eos ({EXPORTED_EOS_ID}) are stop ids",
        {IM_END_ID, EXPORTED_EOS_ID} <= stop_ids, f"stop ids {sorted(stop_ids)}")

  md5 = hashlib.md5(embedded).hexdigest()
  detail = f"{len(embedded)} B, md5 {md5[:8]}…"
  try:
    from huggingface_hub import hf_hub_download
    repo_tpl = Path(hf_hub_download(MODEL_ID, "chat_template.jinja")).read_bytes()
    check("embedded Jinja is byte-equal to the repo's chat_template.jinja",
          embedded == repo_tpl, detail + f" vs repo {len(repo_tpl)} B")
  except Exception as e:  # noqa: BLE001
    check("embedded Jinja has the repo chat_template.jinja's length",
          len(embedded) == TEMPLATE_BYTES, detail + f" (Hub unreachable: {type(e).__name__})")
  ctx = re.search(r"max_num_tokens:\s*(\d+)", pbtext)
  check("max_num_tokens 4096", bool(ctx) and int(ctx.group(1)) == 4096,
        ctx.group(1) if ctx else "missing")

  types = [s["type"] for s in sections]
  check("ExecutorMetadata section present (litert-lm >= 0.15 binds hybrid state through it)",
        "ExecutorMetadataProto" in types, str(types))
  la = len(re.findall(r"type:\s*TYPE_LINEAR_ATTENTION", exec_pbtext))
  kv = len(re.findall(r"type:\s*TYPE_GLOBAL_(?:KEY|VALUE)_CACHE", exec_pbtext))
  check(f"{STATE_LINEAR_ATTN} mamba conv+SSM buffers + {STATE_KV} attention K/V buffers",
        (la, kv) == (STATE_LINEAR_ATTN, STATE_KV), f"{la} + {kv}")
  seq = {int(x) for x in re.findall(r"maximum_sequence_length:\s*(\d+)", exec_pbtext)}
  check("attention caches sized to the 4096 budget", seq == {4096}, str(sorted(seq)))
  act = [s["activation"] for s in sections if s["type"] == "TFLiteModel"]
  check("prefer_activation_type = fp32 on the model section (GPU keeps the scan in range)",
        act == ["fp32"], str(act))

  total = os.path.getsize(args.model)
  bpp = total / PARAMS
  check("bytes per parameter in the int8 band (0.9–1.35)", 0.9 <= bpp <= 1.35,
        f"{total:,} B / {PARAMS:,} params = {bpp:.2f}")
  print("  (i) sections: " + ", ".join(
      f"{s['type']}:{s['bytes'] / 1024 ** 3:.2f} GiB" for s in sections)
      + " — the weights sit in one section (this recipe does not externalize "
        "the embedder); phone loading is not measured for this bundle")

  ok = all(checks)
  print(f"\n  {'PASS' if ok else 'FAIL'}: {sum(checks)}/{len(checks)} metadata checks")
  return 0 if ok else 1


def cache_ab(args):
  q = QUESTIONS[0][1] + SUFFIX
  model = Path(args.model)
  before = set(os.listdir(model.parent))
  rows = {}
  for arm, cache in (("default cache", None), ("--cache no", "no")):
    try:
      proc = run_cli(args.litert_lm, model, q, "gpu", args.timeout, cache=cache)
      out = proc.stdout.strip()
      err = proc.stderr
      ans = final_answer(out)
      ok = proc.returncode == 0 and bool(re.search(r"\b42\b", ans)) and not degenerate(out)
      rows[arm] = {"exit": proc.returncode, "ok": ok, "answer": ans[:80],
                   "stderr_tail": err[-600:]}
      why = "" if ok else (" — " + (next((l for l in err.splitlines()
                                          if "BindGroup" in l or "rror" in l), "")[:110]))
      print(f"  [{'v' if ok else 'X'}] GPU, {arm:14s} exit={proc.returncode} "
            f"answer={ans[:40]!r}{why}")
    except subprocess.TimeoutExpired:
      rows[arm] = {"exit": None, "ok": False, "answer": "", "stderr_tail": "timeout"}
      print(f"  [X] GPU, {arm:14s} timeout")
  new = sorted(set(os.listdir(model.parent)) - before)
  for f in new:
    p = model.parent / f
    print(f"  removed cache artifact written by the default arm: {f} "
          f"({os.path.getsize(p) / 1e6:.0f} MB)")
    p.unlink() if p.is_file() else None
  a, b = rows["default cache"]["ok"], rows["--cache no"]["ok"]
  verdict = ("trap reproduces: GPU works only with --cache no" if b and not a else
             "both arms answer — the cache trap no longer reproduces on this runtime"
             if a and b else "GPU answers with the default cache but not with --cache no"
             if a and not b else "GPU fails on both arms")
  print(f"\n  VERDICT: {verdict}")
  if args.json:
    Path(args.json).write_text(json.dumps({"model": str(model), "verdict": verdict,
                                           "arms": rows}, indent=2) + "\n")
    print(f"  wrote {args.json}")
  return 0 if b else 1


def bos_ab(args):
  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer

  device = args.device
  if device == "auto":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
  tok = AutoTokenizer.from_pretrained(MODEL_ID)
  model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
  assert tok.bos_token == "<s>" and tok.bos_token_id is not None
  rows, identical = [], 0
  for q in BOS_PROBES:
    text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                   add_generation_prompt=True)
    assert not text.startswith(tok.bos_token), "template now renders a BOS itself"
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
    arms = {"no_bos": ids,
            "with_bos": torch.cat([torch.tensor([[tok.bos_token_id]]), ids], dim=1)}
    row = {"q": q}
    for name, inp in arms.items():
      with torch.no_grad():
        out = model.generate(inp.to(device), max_new_tokens=args.max_new, do_sample=False,
                             use_cache=False, pad_token_id=tok.pad_token_id or 0)
      new = out[0, inp.shape[1]:].tolist()
      row[name] = {"ids": new, "text": tok.decode(new)}
    same = row["no_bos"]["ids"] == row["with_bos"]["ids"]
    row["identical"] = same
    if not same:
      row["diverge_at_token"] = next(i for i, (a, b) in enumerate(
          zip(row["no_bos"]["ids"], row["with_bos"]["ids"])) if a != b)
    fa, fb = final_answer(row["no_bos"]["text"]), final_answer(row["with_bos"]["text"])
    row["same_final_answer"] = fa.split("<|im_end|>")[0].strip() == fb.split("<|im_end|>")[0].strip()
    identical += same
    rows.append(row)
    print(f"  {q:32s} identical={'yes' if same else 'no '}"
          + ("" if same else f" (diverges at token {row['diverge_at_token']}, "
             f"same final answer: {'yes' if row['same_final_answer'] else 'NO'})"))
    print(f"      no BOS  : {row['no_bos']['text'][:150]!r}")
    print(f"      with BOS: {row['with_bos']['text'][:150]!r}")
  robust = all(r["identical"] or r["same_final_answer"] for r in rows)
  print(f"\n  byte-identical: {identical}/{len(rows)}; same final answer on all: "
        f"{'yes' if robust else 'NO'}")
  print("  VERDICT: " + ("robust to the leading <s> at this scale — dropping the "
                         "start_token is a stream-fidelity fix, not a rescue" if robust
                         else "the leading <s> changes an answer — the start_token "
                         "drop is load-bearing here"))
  if args.json:
    Path(args.json).write_text(json.dumps({"device": device, "rows": rows}, indent=1,
                                          ensure_ascii=False) + "\n")
    print(f"  wrote {args.json}")
  return 0


def main():
  ap = argparse.ArgumentParser(
      description="Quality gate / metadata / GPU-cache / BOS checks for Nemotron-3-Nano-4B")
  ap.add_argument("model", help="path to model.litertlm")
  ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
  ap.add_argument("--min-correct", type=int, default=6)
  ap.add_argument("--timeout", type=int, default=900,
                  help="seconds per question (engine init is paid every run; ~25 s "
                       "on CPU for this 4B)")
  ap.add_argument("--litert-lm", default="litert-lm",
                  help="path to the litert-lm CLI (pip install litert-lm)")
  ap.add_argument("--device", default="auto", help="--bos-ab: cpu|mps|auto")
  ap.add_argument("--max-new", type=int, default=96, help="--bos-ab: tokens per arm")
  ap.add_argument("--json", help="write a JSON report here")
  ap.add_argument("--check-metadata", action="store_true")
  ap.add_argument("--cache-ab", action="store_true")
  ap.add_argument("--bos-ab", action="store_true")
  args = ap.parse_args()

  if not args.bos_ab and not Path(args.model).exists():
    print(f"ERROR: model not found: {args.model}", file=sys.stderr)
    return 2
  if args.check_metadata:
    return check_metadata(args)
  if args.cache_ab:
    return cache_ab(args)
  if args.bos_ab:
    return bos_ab(args)
  return gate(args)


if __name__ == "__main__":
  sys.exit(main())
