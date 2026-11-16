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

"""Quality gate and template checks for a converted SmolLM3-3B bundle.

Three modes:

  gate (default) — asks 8 fixed, unambiguously checkable questions through the
    `litert-lm` CLI (pip install litert-lm), greedy, one fresh session per
    question, and scores correctness plus degeneracy (looping, token spam,
    empty output). This is the publish guardrail: a conversion that collapses
    must not ship.

      python verify_smollm3_3b.py model.litertlm [--backend cpu|gpu]

  --check-metadata — reads the bundle's LlmMetadata (no unpacking) and asserts
    the shape this recipe produces: bare ChatML per-role templates, <|im_end|>
    (128012) among the stop tokens, NO start_token, and the embedder in its own
    section with every section under 2 GiB (the iOS single-section mmap
    ceiling). Needs `pip install litert-lm-builder`.

      python verify_smollm3_3b.py model.litertlm --check-metadata

  --think-ab — shows what the bare-ChatML bundle does with SmolLM3's two
    modes: the same question asked plainly (the bundle's default: the model
    closes <think></think> immediately and answers directly) and with
    SmolLM3's official /think system prompt supplied as a system message
    through the CLI's --preset (the model reasons at length first). Both
    are legitimate; the point is that the bundle carries no default system
    prompt, so the mode is the caller's choice.

      python verify_smollm3_3b.py model.litertlm --think-ab [--backend gpu]
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

# SmolLM3's official reasoning-mode system prompt (the block upstream's chat
# template inserts when no system message is given), minus the date line.
THINK_SYSTEM_PROMPT = (
    "## Metadata\n\nKnowledge Cutoff Date: June 2025\nReasoning Mode: /think\n\n"
    "## Custom Instructions\n\nYou are a helpful AI assistant named SmolLM, "
    "trained by Hugging Face. Your role as an assistant involves thoroughly "
    "exploring questions through a systematic thinking process before providing "
    "the final precise and accurate solutions. This requires engaging in a "
    "comprehensive cycle of analysis, summarizing, exploration, reassessment, "
    "reflection, backtracking, and iteration to develop well-considered thinking "
    "process. Please structure your response into two main sections: Thought and "
    "Solution using the specified format: <think> Thought section </think> "
    "Solution section. In the Thought section, detail your reasoning process in "
    "steps. Each step should include detailed considerations such as analysing "
    "questions, summarizing relevant findings, brainstorming new ideas, verifying "
    "the accuracy of the current steps, refining any errors, and revisiting "
    "previous steps. In the Solution section, based on various attempts, "
    "explorations, and reflections from the Thought section, systematically "
    "present the final solution that you deem correct. The Solution section "
    "should be logical, accurate, and concise and detail necessary steps needed "
    "to reach the conclusion.")


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


def run_one(litert_lm, model, prompt, backend, timeout, preset=None):
  """Runs one prompt through `litert-lm run`; the answer is exactly stdout."""
  cmd = [litert_lm, "run", str(model), "--prompt", prompt,
         "--backend", backend, "--temperature", "0", "--top-k", "1",
         "--cache", "no"]
  if preset:
    cmd += ["--preset", preset]
  proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
  if proc.returncode != 0:
    raise RuntimeError(
        f"litert-lm run failed (exit={proc.returncode}).\n"
        f"--- stderr tail ---\n{proc.stderr[-1500:]}")
  out = proc.stdout
  if preset:
    # --preset echoes a header (the system instruction, then "- Tools:" and one
    # "  - name" line per tool) before the answer; drop it.
    head, sep, tail = out.rpartition("- Tools:\n")
    if sep:
      out = re.sub(r"^  - .*\n", "", tail, flags=re.M)
  return out.strip()


def gate(args):
  results = []
  for label, q, pat, veto in QUESTIONS:
    try:
      ans = run_one(args.litert_lm, args.model, q + SUFFIX, args.backend,
                    args.timeout)
    except Exception as e:  # noqa: BLE001
      print(f"FAIL (harness) on '{label}': {e}", file=sys.stderr)
      return 2
    # SmolLM3 in direct mode opens with an empty think block; score the answer body.
    body = re.sub(r"<think>.*?</think>", "", ans, flags=re.S)
    low = body.lower()
    ok = bool(re.search(pat, low)) and not (veto and re.search(veto, low))
    degen = (not body.strip()) or degenerate(body)
    results.append({"label": label, "question": q, "ok": ok,
                    "degenerate": degen, "answer": ans})
    shown = " ".join(body.split())[:90] if body.strip() else "(empty)"
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
  check("no start_token (SmolLM3 has no BOS)", not has_start)

  stop_ids = {int(x) for x in re.findall(r"ids:\s*(\d+)", pbtext)}
  check("<|im_end|> (128012) is a stop token", 128012 in stop_ids,
        f"stop ids {sorted(stop_ids)}")

  def prefix(role):
    m = re.search(role + r"\s*\{\s*prefix:\s*\"((?:[^\"\\]|\\.)*)\"", pbtext)
    return m.group(1) if m else None
  check("user prefix is bare ChatML", prefix("user") == r"<|im_start|>user\n",
        repr(prefix("user")))
  check("system prefix present (caller can supply /think)",
        prefix("system") == r"<|im_start|>system\n", repr(prefix("system")))
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


def think_ab(args):
  q = "What is 17 + 25?" + SUFFIX
  with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
    f.write("tools = []\nsystem_instruction = " + json.dumps(THINK_SYSTEM_PROMPT) + "\n")
    preset = f.name
  try:
    plain = run_one(args.litert_lm, args.model, q, args.backend, args.timeout)
    think = run_one(args.litert_lm, args.model, q, args.backend, args.timeout,
                    preset=preset)
  finally:
    os.unlink(preset)

  def describe(name, text):
    m = re.search(r"<think>(.*?)</think>", text, re.S)
    thought = m.group(1).strip() if m else None
    body = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    print(f"  {name}:")
    print(f"    think block: {'none' if thought is None else ('empty' if not thought else f'{len(thought.split())} words')}")
    print(f"    answer body: {len(body.split())} words -> {' '.join(body.split())[:120]!r}")
    has42 = re.search(r"\b42\b", body) is not None
    print(f"    contains 42: {'yes' if has42 else 'NO'}")

  print(f"same question, {args.backend}: {q!r}\n")
  describe("plain (bundle default, no system prompt)", plain)
  describe("with SmolLM3's /think system prompt via --preset", think)
  print("\nExpected: the plain run answers directly (empty or no think block); "
        "the /think run reasons first. The bundle carries no default system "
        "prompt, so the mode is the caller's.")
  return 0


def main():
  ap = argparse.ArgumentParser(
      description="Quality gate / metadata / think-mode checks for SmolLM3-3B")
  ap.add_argument("model", help="path to model.litertlm")
  ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
  ap.add_argument("--min-correct", type=int, default=6)
  ap.add_argument("--timeout", type=int, default=900,
                  help="seconds per question (engine init is paid every run)")
  ap.add_argument("--litert-lm", default="litert-lm",
                  help="path to the litert-lm CLI (pip install litert-lm)")
  ap.add_argument("--json", help="write a JSON report here (gate mode)")
  ap.add_argument("--check-metadata", action="store_true")
  ap.add_argument("--think-ab", action="store_true")
  args = ap.parse_args()

  if not Path(args.model).exists():
    print(f"ERROR: model not found: {args.model}", file=sys.stderr)
    return 2
  if args.check_metadata:
    return check_metadata(args)
  if args.think_ab:
    return think_ab(args)
  return gate(args)


if __name__ == "__main__":
  sys.exit(main())
