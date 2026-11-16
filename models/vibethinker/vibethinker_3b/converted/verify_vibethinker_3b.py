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

"""Quality gate, metadata and math checks for a converted VibeThinker-3B bundle.

Three modes:

  gate (default) — asks 8 fixed, unambiguously checkable questions through the
    `litert-lm` CLI (pip install litert-lm), greedy, one fresh session per
    question, and scores correctness plus degeneracy (looping, token spam,
    empty output). VibeThinker reasons inside <think>...</think> before it
    answers; the gate scores the answer body after the think block. This is
    the publish guardrail: a conversion that collapses must not ship.

      python verify_vibethinker_3b.py model.litertlm [--backend cpu|gpu]

  --check-metadata — reads the bundle's LlmMetadata (no unpacking) and asserts
    the shape this recipe produces: bare ChatML per-role templates, BOTH
    <|im_end|> (151645) and <|endoftext|> (151643) among the stop-token ids
    (the fix in build_vibethinker_3b.py — upstream declares only 151643), NO
    start_token, and the embedder in its own section with every section under
    2 GiB (the iOS single-section mmap ceiling). Needs `pip install
    litert-lm-builder`.

      python verify_vibethinker_3b.py model.litertlm --check-metadata

  --math — the check that matters for this model: six GSM8K-style word
    problems, greedy, scored on the final number after the chain of thought
    (\\boxed{n} / "#### n" / last number). Also reports how many words each
    answer took, which is why the bundle needs its 4096-token context and
    callers need max_tokens >= 2048.

      python verify_vibethinker_3b.py model.litertlm --math [--backend gpu]
"""

import argparse
import io
import json
import re
import subprocess
import sys
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

# (label, problem, answer). Small GSM8K-shaped word problems with one integer answer.
MATH = [
    ("pens", "A shop sells pens at 3 dollars each. Maria buys 4 pens and pays with a "
             "20-dollar bill. How much change does she get?", "8"),
    ("train", "A train travels 60 miles per hour for 2 hours and then 40 miles per hour "
              "for 3 hours. How many miles does it travel in total?", "240"),
    ("apples", "Tom has 3 times as many apples as Jane. Together they have 48 apples. "
               "How many apples does Tom have?", "36"),
    ("tiles", "A rectangular floor is 12 feet by 9 feet. Tiles are 1 square foot each and "
              "come in boxes of 10. How many boxes are needed to cover the floor?", "11"),
    ("savings", "Ana saves 15 dollars a week. After 8 weeks she spends 40 dollars. "
                "How much does she have left?", "80"),
    ("chairs", "A hall has 14 rows of 25 chairs. 60 chairs are removed for a stage. "
               "How many chairs remain?", "290"),
]
COT = ("\n\nSolve this step by step. After your reasoning, write the final answer on "
       "its own line in the exact form:\n#### <number>")


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


def strip_think(text):
  """Drops <think>...</think>; a think block left open counts as no answer."""
  if "<think>" in text and "</think>" not in text:
    return ""
  return re.sub(r"<think>.*?</think>", "", text, flags=re.S)


def gate(args):
  results = []
  for label, q, pat, veto in QUESTIONS:
    try:
      ans = run_one(args.litert_lm, args.model, q + SUFFIX, args.backend,
                    args.timeout)
    except Exception as e:  # noqa: BLE001
      print(f"FAIL (harness) on '{label}': {e}", file=sys.stderr)
      return 2
    body = strip_think(ans)
    low = body.lower()
    ok = bool(re.search(pat, low)) and not (veto and re.search(veto, low))
    degen = (not body.strip()) or degenerate(body)
    results.append({"label": label, "question": q, "ok": ok,
                    "degenerate": degen, "answer": ans})
    shown = " ".join(body.split())[:90] if body.strip() else "(empty)"
    print(f"  [{'v' if ok else '.'}]{' DEGEN' if degen else '      '} "
          f"{label:16s} -> {shown!r}  ({len(ans.split())} words incl. think)")

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
  check("<|im_end|> (151645) is a stop-token id (the eos fix)", 151645 in stop_ids,
        f"stop ids {sorted(stop_ids)}")
  check("<|endoftext|> (151643) is a stop-token id", 151643 in stop_ids)

  def prefix(role):
    m = re.search(role + r"\s*\{\s*prefix:\s*\"((?:[^\"\\]|\\.)*)\"", pbtext)
    return m.group(1) if m else None
  check("user prefix is bare ChatML", prefix("user") == r"<|im_start|>user\n",
        repr(prefix("user")))
  check("model prefix", prefix("model") == r"<|im_start|>assistant\n",
        repr(prefix("model")))

  types = [s["model_type"] for s in sections]
  check("embedder in its own section", "tf_lite_embedder" in types, str(types))
  big = [s for s in sections if s["bytes"] >= 2 * 1024 ** 3]
  check("every section < 2 GiB (iOS mmap ceiling)", not big,
        ", ".join(f"{s['type']}:{s['bytes'] / 1024 ** 3:.2f} GiB" for s in sections))

  ok = all(checks)
  print(f"\n  {'PASS' if ok else 'FAIL'}: {sum(checks)}/{len(checks)} metadata checks")
  return 0 if ok else 1


def extract_number(text):
  """GSM8K-style: prefer '#### n', then \\boxed{n}, then the last number."""
  t = strip_think(text).replace(",", "")
  for pat in (r"####\s*\$?(-?\d+(?:\.\d+)?)", r"\\boxed\{\s*\$?(-?\d+(?:\.\d+)?)",
              r"-?\d+(?:\.\d+)?"):
    m = re.findall(pat, t)
    if m:
      v = m[-1]
      return str(int(float(v))) if float(v) == int(float(v)) else v
  return None


def math_gate(args):
  rows = []
  for label, problem, answer in MATH:
    try:
      out = run_one(args.litert_lm, args.model, problem + COT, args.backend,
                    args.timeout)
    except Exception as e:  # noqa: BLE001
      print(f"FAIL (harness) on '{label}': {e}", file=sys.stderr)
      return 2
    got = extract_number(out)
    ok = got == answer
    words = len(out.split())
    think = re.search(r"<think>(.*?)</think>", out, re.S)
    rows.append({"label": label, "ok": ok, "expected": answer, "got": got,
                 "words": words, "answer": out})
    print(f"  [{'v' if ok else '.'}] {label:9s} expected {answer:>4s} got {str(got):>6s}"
          f"  {words:5d} words"
          f"{'' if think else '  (no closed think block)'}")
  score = sum(r["ok"] for r in rows)
  print(f"\n  correct: {score}/{len(MATH)}   words per answer: "
        f"{min(r['words'] for r in rows)}-{max(r['words'] for r in rows)}")
  print("  This model reasons before it answers — give it max_tokens >= 2048.")
  if args.json:
    Path(args.json).write_text(json.dumps({
        "model": str(args.model), "backend": args.backend,
        "score": score, "of": len(MATH), "rows": rows}, indent=2,
        ensure_ascii=False) + "\n")
    print(f"  wrote {args.json}")
  return 0 if score >= args.min_math else 1


def main():
  ap = argparse.ArgumentParser(
      description="Quality gate / metadata / math checks for VibeThinker-3B")
  ap.add_argument("model", help="path to model.litertlm")
  ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
  ap.add_argument("--min-correct", type=int, default=6)
  ap.add_argument("--min-math", type=int, default=5)
  ap.add_argument("--timeout", type=int, default=1200,
                  help="seconds per question (the model reasons at length "
                       "and engine init is paid every run)")
  ap.add_argument("--litert-lm", default="litert-lm",
                  help="path to the litert-lm CLI (pip install litert-lm)")
  ap.add_argument("--json", help="write a JSON report here (gate / math modes)")
  ap.add_argument("--check-metadata", action="store_true")
  ap.add_argument("--math", action="store_true")
  args = ap.parse_args()

  if not Path(args.model).exists():
    print(f"ERROR: model not found: {args.model}", file=sys.stderr)
    return 2
  if args.check_metadata:
    return check_metadata(args)
  if args.math:
    return math_gate(args)
  return gate(args)


if __name__ == "__main__":
  sys.exit(main())
