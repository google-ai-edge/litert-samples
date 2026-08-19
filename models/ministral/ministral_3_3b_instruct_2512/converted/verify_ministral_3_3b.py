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

"""Quality gate, metadata and start_token checks for a converted Ministral-3-3B bundle.

Three modes:

  gate (default) — asks 8 fixed, unambiguously checkable questions through the
    `litert-lm` CLI (pip install litert-lm), greedy, one fresh session per
    question, and scores correctness plus degeneracy (looping, token spam,
    empty output). This is the publish guardrail: a conversion that collapses
    must not ship — and for this model, the check that catches a ChatML
    export (which runs away with <|im_start|> spam after the right answer).

      python verify_ministral_3_3b.py model.litertlm [--backend cpu|gpu]

  --check-metadata — reads the bundle's LlmMetadata (no unpacking) and asserts
    the shape this recipe produces: Mistral [INST] / [/INST] per-role
    templates, </s> (id 2) among the stop tokens, a start_token of "<s>"
    (present on purpose — see --bos-ab), and the embedder in its own section
    with every section under 2 GiB (the iOS single-section mmap ceiling that
    the un-externalized 2.55 GiB build hit). Needs `pip install
    litert-lm-builder`.

      python verify_ministral_3_3b.py model.litertlm --check-metadata

  --bos-ab — the one-minute check that the start_token is legitimate here:
    feeds the SAME rendered [INST] prompt to the bf16 PyTorch text decoder
    with and without a leading <s> and prints both answers. For granite-4.1-3b
    (bos == eos) the with-BOS column breaks; for Ministral (bos <s> != eos
    </s>) both columns answer, and <s>-first is what the model was trained on.
    Needs torch, transformers and the extracted text decoder (no bundle
    involved).

      python verify_ministral_3_3b.py --bos-ab [--hf ministral3_text]
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

  m = re.search(r"start_token\s*\{\s*token_str:\s*\"(.*?)\"", pbtext)
  check('start_token is "<s>" (Mistral convention, bos != eos — kept on purpose)',
        bool(m) and m.group(1) == "<s>", repr(m.group(1)) if m else "absent")

  stop_ids = {int(x) for x in re.findall(r"ids:\s*(\d+)", pbtext)}
  stop_strs = set(re.findall(r"token_str:\s*\"((?:[^\"\\]|\\.)*)\"", pbtext))
  check("</s> is a stop token", 2 in stop_ids or "</s>" in stop_strs,
        f"ids {sorted(stop_ids)}")
  check("no ChatML markers among the stops (tekken has no <|im_end|>)",
        not any("<|im_end|>" in s for s in stop_strs))

  def part(role, which):
    m = re.search(role + r"\s*\{(.*?)\}", pbtext, re.S)
    if not m:
      return None
    p = re.search(which + r":\s*\"((?:[^\"\\]|\\.)*)\"", m.group(1))
    return p.group(1) if p else ""
  check("user turn is [INST] ... [/INST]",
        part("user", "prefix") == "[INST]" and part("user", "suffix") == "[/INST]",
        f"{part('user', 'prefix')!r} ... {part('user', 'suffix')!r}")
  check("model turn ends with </s>", part("model", "suffix") == "</s>",
        repr(part("model", "suffix")))
  check("system turn is [SYSTEM_PROMPT] ... [/SYSTEM_PROMPT]",
        part("system", "prefix") == "[SYSTEM_PROMPT]",
        repr(part("system", "prefix")))

  types = [s["model_type"] for s in sections]
  check("embedder in its own section", "tf_lite_embedder" in types, str(types))
  big = [s for s in sections if s["bytes"] >= 2 * 1024 ** 3]
  check("every section < 2 GiB (iOS mmap ceiling)", not big,
        ", ".join(f"{s['type']}:{s['bytes'] / 1024 ** 3:.2f} GiB" for s in sections))

  ok = all(checks)
  print(f"\n  {'PASS' if ok else 'FAIL'}: {sum(checks)}/{len(checks)} metadata checks")
  return 0 if ok else 1


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
  bos_id = tok.bos_token_id
  eos_id = tok.eos_token_id

  def generate(ids):
    mask = torch.ones_like(ids)
    with torch.no_grad():
      out = model.generate(input_ids=ids, attention_mask=mask,
                           max_new_tokens=args.max_tokens, do_sample=False,
                           eos_token_id=eos_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

  print(f"bf16 A/B on {args.hf} ({device}): the same rendered [INST] prompt, "
        f"with and without a leading {tok.bos_token!r} (id {bos_id}); "
        f"eos {tok.eos_token!r} (id {eos_id})\n")
  for q in ["How many days are in a week?", "What is 8 times 7?",
            "What is 17 + 25?"]:
    text = "[INST]" + q + SUFFIX + "[/INST]"   # what the bundle's template renders
    ids = tok(text, add_special_tokens=False,
              return_tensors="pt").input_ids.to(device)
    bos = torch.tensor([[bos_id]], device=device)
    no_bos = generate(ids)
    with_bos = generate(torch.cat([bos, ids], dim=1))
    print(f"  Q: {q}")
    print(f"    no BOS  : {' '.join(no_bos.split())[:100]!r}")
    print(f"    with BOS: {' '.join(with_bos.split())[:100]!r}\n")
  print("Expected here: both columns answer (with-BOS is the trained "
        "convention), which is why this recipe keeps the start_token. If the "
        "with-BOS column echoed the question, the field would have to go — "
        "that is the granite-4.1-3b case (bos == eos), not this one.")
  return 0


def main():
  ap = argparse.ArgumentParser(
      description="Quality gate / metadata / start_token checks for Ministral-3-3B")
  ap.add_argument("model", nargs="?", help="path to model.litertlm")
  ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
  ap.add_argument("--min-correct", type=int, default=6)
  ap.add_argument("--timeout", type=int, default=900,
                  help="seconds per question (engine init is paid every run)")
  ap.add_argument("--litert-lm", default="litert-lm",
                  help="path to the litert-lm CLI (pip install litert-lm)")
  ap.add_argument("--json", help="write a JSON report here (gate mode)")
  ap.add_argument("--check-metadata", action="store_true")
  ap.add_argument("--bos-ab", action="store_true")
  ap.add_argument("--hf", default="ministral3_text",
                  help="the extracted text decoder dir (--bos-ab mode)")
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
