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

"""Quality gate, metadata and code checks for a converted Qwen2.5-Coder-1.5B bundle.

Three modes:

  gate (default) — asks 8 fixed, unambiguously checkable questions through the
    `litert-lm` CLI (pip install litert-lm), greedy, one fresh session per
    question, and scores correctness plus degeneracy (looping, token spam,
    empty output). This is the publish guardrail: a conversion that collapses
    must not ship. For a code model it certifies little else — see --code-gate.

      python verify_qwen2_5_coder_1_5b.py model.litertlm [--backend cpu|gpu]

  --check-metadata — reads the bundle's LlmMetadata (no unpacking) and asserts
    the shape this recipe produces: the vendor's default system turn folded
    into the user prefix, <|im_end|> (151645) and <|endoftext|> (151643) among
    the stop-token ids, NO start_token, the embedder in its own section, and a
    bytes-per-parameter ratio in the int4 band (< 1.0) — the check that
    catches the 2.53 GB duplicated-embedding export, which is otherwise silent.
    Needs `pip install litert-lm-builder`.

      python verify_qwen2_5_coder_1_5b.py model.litertlm --check-metadata

  --code-gate — the check that matters for this model: six small functions
    (fib, reverse_words, is_prime, largest contiguous sublist sum,
    count_vowels, flatten) where the generated code is EXECUTED against
    assertions rather than read. A task passes only if the code imports and
    every assertion holds — code that looks right and raises is a fail, which
    is the point.

      python verify_qwen2_5_coder_1_5b.py model.litertlm --code-gate [--backend gpu]
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

SUFFIX = " Answer briefly."
# (label, question, answer-regex, wrong-answer-veto-regex-or-None).
# The veto on the 0.9-vs-0.11 question exists because the presence check alone
# is one-sided: "0.11 is larger than 0.9" also contains "0.9" and would score
# as correct. Require the right number AND the absence of the inverted claim.
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

DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
PARAMS = 1_543_714_304  # Qwen2.5-Coder-1.5B-Instruct

# (id, prompt, test source). Kept short: a long prompt measures the harness's
# patience rather than the model.
CODE_TASKS = [
    ("fib",
     "Write a Python function fib(n) that returns the nth Fibonacci number, with fib(0)=0 and fib(1)=1. Code only.",
     "assert fib(0)==0\nassert fib(1)==1\nassert fib(10)==55"),
    ("reverse_words",
     "Write a Python function reverse_words(s) that returns the words of s in reverse order, space separated. Code only.",
     "assert reverse_words('a b c')=='c b a'\nassert reverse_words('hello')=='hello'"),
    ("is_prime",
     "Write a Python function is_prime(n) that returns True if n is a prime number and False otherwise. Code only.",
     "assert is_prime(2)\nassert is_prime(13)\nassert not is_prime(1)\nassert not is_prime(9)"),
    ("max_sublist",
     "Write a Python function max_sum(nums) that returns the largest sum of any contiguous sublist of the list nums. Code only.",
     "assert max_sum([1,-2,3,4])==7\nassert max_sum([-1,-2])==-1"),
    ("count_vowels",
     "Write a Python function count_vowels(s) that returns how many vowels (aeiou, case insensitive) are in s. Code only.",
     "assert count_vowels('Hello')==2\nassert count_vowels('xyz')==0"),
    ("flatten",
     "Write a Python function flatten(lst) that flattens a list of lists into a single list. Code only.",
     "assert flatten([[1,2],[3]])==[1,2,3]\nassert flatten([])==[]"),
]


def degenerate(text):
  """True if the output loops or is special-token spam.

  The 5-gram trigger needs a corroborating diversity collapse: a model may
  legitimately restate a quoted phrase a few times in a clean answer, while
  real loops show heavy repetition AND cratered word diversity."""
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
  if text.count("<|") >= 5 or text.count("<pad>") >= 5:
    return True
  return False


def run_one(litert_lm, model, prompt, backend, timeout):
  """Runs one prompt through `litert-lm run`; the answer is exactly stdout."""
  cmd = [litert_lm, "run", str(model), "--prompt", prompt,
         "--backend", backend, "--temperature", "0", "--top-k", "1",
         "--cache", "no"]
  proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
  if proc.returncode != 0:
    raise RuntimeError(
        f"litert-lm run failed (exit={proc.returncode}).\n"
        f"--- stderr tail ---\n{proc.stderr[-1500:]}")
  return proc.stdout.strip()


def gate(args):
  results = []
  for label, q, pat, veto in QUESTIONS:
    try:
      ans = run_one(args.litert_lm, args.model, q + SUFFIX, args.backend,
                    args.timeout)
    except Exception as e:  # noqa: BLE001
      print(f"FAIL (harness) on '{label}': {e}", file=sys.stderr)
      return 2
    low = ans.lower()
    ok = bool(re.search(pat, low)) and not (veto and re.search(veto, low))
    degen = (not ans.strip()) or degenerate(ans)
    results.append({"label": label, "question": q, "ok": ok,
                    "degenerate": degen, "answer": ans})
    shown = " ".join(ans.split())[:90] if ans.strip() else "(empty)"
    print(f"  [{'v' if ok else '.'}]{' DEGEN' if degen else '      '} "
          f"{label:16s} -> {shown!r}")

  score = sum(r["ok"] for r in results)
  any_degen = any(r["degenerate"] for r in results)
  passed = (score >= args.min_correct) and (not any_degen)
  print(f"\n  correct: {score}/8   degenerate: {'YES' if any_degen else 'no'}")
  print(f"  VERDICT: {'PASS' if passed else 'FAIL'} "
        f"(threshold {args.min_correct}/8, non-degenerate)")

  if args.json:
    Path(args.json).write_text(json.dumps({
        "model": str(args.model), "backend": args.backend,
        "score": score, "of": 8, "degenerate": any_degen, "passed": passed,
        "questions": results,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {args.json}")
  return 0 if passed else 1


def read_metadata(model):
  """Returns (LlmMetadata pbtext, [sections]) without unpacking the bundle."""
  from litert_lm_builder import litertlm_peek  # pip install litert-lm-builder

  buf = io.StringIO()
  litertlm_peek.peek_litertlm_file(str(model), None, buf)
  text = buf.getvalue()
  m = re.search(r"start of LlmMetadata\n(.*?)>{4,} end of LlmMetadata", text, re.S)
  pbtext = m.group(1) if m else ""
  sections = []
  for sm in re.finditer(
      r"Section (\d+):\n(.*?)Begin Offset:\s*(\d+)\n\s*End Offset:\s*(\d+)\n"
      r"\s*Data Type:\s*(\S+)", text, re.S):
    items = sm.group(2)
    mt = re.search(r"model_type, Value \(String\): (\S+)", items)
    sections.append({"index": int(sm.group(1)), "type": sm.group(5),
                     "model_type": mt.group(1) if mt else None,
                     "bytes": int(sm.group(4)) - int(sm.group(3))})
  return pbtext, sections


def check_metadata(args):
  pbtext, sections = read_metadata(args.model)
  if not pbtext:
    print("FAIL: no LlmMetadata section found")
    return 1
  checks = []

  def check(name, ok, detail=""):
    checks.append(ok)
    print(f"  [{'v' if ok else 'X'}] {name}{(': ' + detail) if detail else ''}")

  has_start = bool(re.search(r"^\s*start_token\s*\{", pbtext, re.M))
  check("no start_token (no BOS on this tokenizer)", not has_start)

  stop_ids = {int(x) for x in re.findall(r"ids:\s*(\d+)", pbtext)}
  check("<|im_end|> (151645) and <|endoftext|> (151643) are stop-token ids",
        {151645, 151643} <= stop_ids, f"stop ids {sorted(stop_ids)}")

  def prefix(role):
    m = re.search(role + r"\s*\{\s*prefix:\s*\"((?:[^\"\\]|\\.)*)\"", pbtext)
    return m.group(1) if m else None
  up = prefix("user") or ""
  check("user prefix carries the vendor default system turn",
        DEFAULT_SYSTEM in up and up.endswith(r"<|im_start|>user\n"), repr(up[:60] + "..."))
  check("model prefix", prefix("model") == r"<|im_start|>assistant\n",
        repr(prefix("model")))

  types = [s["model_type"] for s in sections]
  check("embedder in its own section (externalize_embedder)",
        "tf_lite_embedder" in types, str(types))
  total = os.path.getsize(args.model)
  bpp = total / PARAMS
  check("bytes per parameter in the int4 band (< 1.0; duplicated-table build is 1.64)",
        bpp < 1.0, f"{total:,} B / {PARAMS:,} params = {bpp:.2f}")
  big = [s for s in sections if s["bytes"] >= 2 * 1024 ** 3]
  check("every section < 2 GiB (iOS mmap ceiling)", not big,
        ", ".join(f"{s['type']}:{s['bytes'] / 1024 ** 3:.2f} GiB" for s in sections))

  ok = all(checks)
  print(f"\n  {'PASS' if ok else 'FAIL'}: {sum(checks)}/{len(checks)} metadata checks")
  return 0 if ok else 1


def extract_code(text):
  """Prefer a fenced block; fall back to the whole reply."""
  m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
  return m.group(1) if m else text


def run_tests(code, test_src):
  with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
    f.write(code + "\n\n" + test_src + "\nprint('PASS')\n")
    path = f.name
  try:
    r = subprocess.run([sys.executable, path], capture_output=True, text=True,
                       timeout=20)
    if "PASS" in r.stdout:
      return True, ""
    err = r.stderr.strip().split("\n")[-1] if r.stderr.strip() else "no PASS"
    return False, err
  except subprocess.TimeoutExpired:
    return False, "execution timeout (infinite loop?)"
  finally:
    os.unlink(path)


def code_gate(args):
  rows = []
  for tid, prompt, test_src in CODE_TASKS:
    try:
      reply = run_one(args.litert_lm, args.model, prompt, args.backend, args.timeout)
    except Exception as e:  # noqa: BLE001
      print(f"FAIL (harness) on '{tid}': {e}", file=sys.stderr)
      return 2
    ok, err = run_tests(extract_code(reply), test_src)
    rows.append({"id": tid, "ok": ok, "error": err, "reply": reply})
    print(f"  [{'v' if ok else '.'}] {tid:14s} {'' if ok else '— ' + err[:70]}")
  passed = sum(r["ok"] for r in rows)
  print(f"\n  passed: {passed}/{len(CODE_TASKS)}   (threshold {args.min_pass}) — "
        "generated code executed against assertions, not read")
  print(f"  VERDICT: {'PASS' if passed >= args.min_pass else 'FAIL'}")
  if args.json:
    Path(args.json).write_text(json.dumps({
        "model": str(args.model), "backend": args.backend,
        "passed": passed, "total": len(CODE_TASKS), "rows": rows}, indent=2,
        ensure_ascii=False) + "\n")
    print(f"  wrote {args.json}")
  return 0 if passed >= args.min_pass else 1


def main():
  ap = argparse.ArgumentParser(
      description="Quality gate / metadata / code checks for Qwen2.5-Coder-1.5B")
  ap.add_argument("model", help="path to model.litertlm")
  ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
  ap.add_argument("--min-correct", type=int, default=6)
  ap.add_argument("--min-pass", type=int, default=4, help="--code-gate threshold")
  ap.add_argument("--timeout", type=int, default=900,
                  help="seconds per question (engine init is paid every run)")
  ap.add_argument("--litert-lm", default="litert-lm",
                  help="path to the litert-lm CLI (pip install litert-lm)")
  ap.add_argument("--json", help="write a JSON report here (gate / code-gate modes)")
  ap.add_argument("--check-metadata", action="store_true")
  ap.add_argument("--code-gate", action="store_true")
  args = ap.parse_args()

  if not Path(args.model).exists():
    print(f"ERROR: model not found: {args.model}", file=sys.stderr)
    return 2
  if args.check_metadata:
    return check_metadata(args)
  if args.code_gate:
    return code_gate(args)
  return gate(args)


if __name__ == "__main__":
  sys.exit(main())
