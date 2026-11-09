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

"""Quality gate and start_token checks for a converted granite-4.1-3b bundle.

Three modes:

  gate (default) — asks 8 fixed, unambiguously checkable questions through the
    `litert-lm` CLI (pip install litert-lm), greedy, one fresh session per
    question, and scores correctness plus degeneracy (looping, token spam,
    empty output). This is the publish guardrail: a conversion that collapses
    must not ship. A bundle with the start_token bug fails it at 5/8 with
    echoed questions; the fixed bundle scores 8/8 on CPU and GPU.

      python verify_granite_4_1_3b.py model.litertlm [--backend cpu|gpu]

  --check-metadata — unpacks the bundle and asserts its LlmMetadata carries NO
    `start_token` block (the fix this recipe exists for). Unpacking writes
    sections as large as the model; point --work-dir at a disk with room.

      python verify_granite_4_1_3b.py model.litertlm --check-metadata

  --bos-ab — the one-minute proof that the start_token trap is a prompt bug and
    not quantization damage: feeds the SAME rendered prompt to the bf16 PyTorch
    model with and without a leading BOS and prints both answers. Needs torch,
    transformers, and the HF checkpoint (no bundle involved).

      python verify_granite_4_1_3b.py --bos-ab [--hf ibm-granite/granite-4.1-3b]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

SUFFIX = " Answer briefly."
# (label, question, answer-regex, wrong-answer-veto-regex-or-None).
# The veto on the 0.9-vs-0.11 question exists because the presence check alone
# is one-sided: "0.11 is larger than 0.9" also contains "0.9" and would score
# as correct — a real false pass observed on another model. Require the right
# number AND the absence of the inverted claim.
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
  proc = subprocess.run(
      [litert_lm, "run", str(model), "--prompt", prompt,
       "--backend", backend, "--temperature", "0", "--top-k", "1",
       "--cache", "no"],
      capture_output=True, text=True, timeout=timeout,
  )
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


def check_metadata(args):
  work = args.work_dir or tempfile.mkdtemp(prefix="g41_unpack_")
  unpack_dir = os.path.join(work, "unpack")
  try:
    subprocess.run(
        [args.litert_lm, "unpack", str(args.model), "--output-dir", unpack_dir],
        check=True)
    pbtext = open(os.path.join(unpack_dir, "LlmMetadataProto.pbtext")).read()
    has_start = bool(re.search(r"^start_token\s*\{", pbtext, re.M))
  finally:
    if not args.work_dir:
      shutil.rmtree(work, ignore_errors=True)
  if has_start:
    print("FAIL: LlmMetadata carries a start_token block — the runtime will "
          "prepend it and this model will echo prompts back. Repair with "
          "strip_start_token.py, or re-export with build_granite_4_1_3b.py.")
    return 1
  print("PASS: no start_token in LlmMetadata.")
  return 0


def bos_ab(args):
  import torch  # deferred: only this mode needs it
  import transformers

  device = args.device
  if device == "auto":
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
  tok = transformers.AutoTokenizer.from_pretrained(args.hf)
  model = transformers.AutoModelForCausalLM.from_pretrained(
      args.hf, dtype=torch.bfloat16).to(device).eval()

  def generate(ids):
    mask = torch.ones_like(ids)
    with torch.no_grad():
      out = model.generate(input_ids=ids, attention_mask=mask,
                           max_new_tokens=args.max_tokens, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

  print(f"bf16 A/B on {args.hf} ({device}): same rendered prompt, with and "
        f"without a leading BOS (<|end_of_text|>, the token the runtime "
        f"prepends when the bundle carries a start_token)\n")
  for q in ["How many days are in a week?", "What is 8 times 7?",
            "What is 17 + 25?"]:
    text = tok.apply_chat_template(
        [{"role": "user", "content": q + SUFFIX}],
        tokenize=False, add_generation_prompt=True)
    # add_special_tokens=False: the rendered template already carries every
    # marker the model expects, and granite sets add_bos_token: False.
    ids = tok(text, add_special_tokens=False,
              return_tensors="pt").input_ids.to(device)
    bos = torch.tensor([[tok.bos_token_id]], device=device)
    no_bos = generate(ids)
    with_bos = generate(torch.cat([bos, ids], dim=1))
    print(f"  Q: {q}")
    print(f"    no BOS  : {' '.join(no_bos.split())[:100]!r}")
    print(f"    with BOS: {' '.join(with_bos.split())[:100]!r}\n")
  print("If the two columns differ like the README table (answers vs echoes), "
        "the trap is live in the prompt path, independent of any quantization.")
  return 0


def main():
  ap = argparse.ArgumentParser(
      description="Quality gate / start_token checks for granite-4.1-3b")
  ap.add_argument("model", nargs="?", help="path to model.litertlm")
  ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
  ap.add_argument("--min-correct", type=int, default=6)
  ap.add_argument("--timeout", type=int, default=900,
                  help="seconds per question (engine init is paid every run)")
  ap.add_argument("--litert-lm", default="litert-lm",
                  help="path to the litert-lm CLI (pip install litert-lm)")
  ap.add_argument("--json", help="write a JSON report here (gate mode)")
  ap.add_argument("--check-metadata", action="store_true")
  ap.add_argument("--work-dir", help="unpack scratch dir for --check-metadata")
  ap.add_argument("--bos-ab", action="store_true")
  ap.add_argument("--hf", default="ibm-granite/granite-4.1-3b",
                  help="HF id or local checkout (--bos-ab mode)")
  ap.add_argument("--device", default="auto", help="--bos-ab device")
  ap.add_argument("--max-tokens", type=int, default=64,
                  help="--bos-ab generation budget")
  args = ap.parse_args()

  if args.bos_ab:
    return bos_ab(args)
  if not args.model:
    ap.error("model.litertlm is required unless --bos-ab")
  if not Path(args.model).exists():
    print(f"ERROR: model not found: {args.model}", file=sys.stderr)
    return 2
  if args.check_metadata:
    return check_metadata(args)
  return gate(args)


if __name__ == "__main__":
  sys.exit(main())
