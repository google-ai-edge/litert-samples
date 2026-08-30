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

"""Quality gate, metadata and model-specific checks for a converted Hy-MT2-1.8B bundle.

Five modes:

  gate (default) — asks 8 fixed, unambiguously checkable questions through the
    `litert-lm` CLI (pip install litert-lm), greedy, one fresh session per
    question, and scores correctness plus degeneracy (looping, token spam,
    empty output). This is the publish guardrail: a conversion that collapses
    must not ship. Hy-MT2 is a translation model, so this certifies little
    about its real job — the published bundle scores 6/8 on both backends with
    the same two misses ("Cool", "pink"); --translate is the check that
    matters.

      python verify_hy_mt2_1_8b.py model.litertlm [--backend cpu|gpu]

  --check-metadata — reads the bundle's LlmMetadata (no unpacking) and asserts
    the shape this recipe produces: NO start_token (the template renders the
    BOS literal itself and the engine prepends the metadata start_token
    unconditionally), <｜hy_place▁holder▁no▁2｜> (120020) among the stop ids,
    the vendor Jinja template embedded byte-equal to the repo's
    chat_template.jinja, a bytes-per-parameter ratio in the int8 band, and
    every section under the iOS ~2 GiB single-section mmap ceiling.
    Needs `pip install litert-lm-builder`.

      python verify_hy_mt2_1_8b.py model.litertlm --check-metadata

  --translate — the model's actual task: three translation probes with the
    source card's default prompt, greedy, HF bf16 reference vs the bundle
    through the runtime. Reports byte-identity per probe (the published
    bundle matches on 1 of 3; the other two are fluent int8-class alternates)
    and passes when every runtime output is a non-degenerate translation.
    Needs torch + transformers and the source checkpoint (3.6 GB).

      python verify_hy_mt2_1_8b.py model.litertlm --translate [--device cpu|mps]

  --rope-ab — the one-variable proof behind the rope bake: loads the model
    twice on CPU, once from the vendor config (rope_scaling dynamic/alpha)
    and once from the baked config (rope_theta 11,158,839.925…, no
    rope_scaling), and asserts inv_freq and teacher-forced logits are
    bitwise-equal. About a minute; the bundle is not needed.

      python verify_hy_mt2_1_8b.py model.litertlm --rope-ab

  --bos-discriminator WITH_START_TOKEN.litertlm — decides, inside the runtime
    and with no HF/int8 confound, whether the engine prepends the metadata
    start_token. Three streams over the same weights (CPU, greedy):
      X  : the default bundle, --no-template, prompt = rendered text minus
           its leading BOS string            -> [start_token?] + rest
      Y' : this bundle, normal templating    -> [template BOS] + rest
      X' : this bundle, --no-template, same BOS-stripped text -> rest only
    If the engine prepends: X == Y' on every probe and X != X' somewhere.
    Build the default bundle with `build_hy_mt2_1_8b.py --keep-start-token`.

      python verify_hy_mt2_1_8b.py model.litertlm --bos-discriminator out_int8_bos/Hy-MT2-1.8B_int8.litertlm
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

MODEL_ID = "tencent/Hy-MT2-1.8B"
EOS_ID = 120020
# 2,038,515,712 tensors in the checkpoint minus the tied lm_head copy
# (120818 x 2048 = 247,435,264) that the bundle stores once.
UNIQUE_PARAMS = 2_038_515_712 - 120_818 * 2_048
# The repo's chat_template.jinja (the only template the repo ships).
TEMPLATE_BYTES = 654
TEMPLATE_MD5 = "6d883d86c23cbdc6879c208513770884"

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

# The source model card's default translation prompt, verbatim.
TRANSLATE_PROMPT = ("Translate the following text into {lang}. Note that you "
                    "should **only output the translated result without any "
                    "additional explanation**:\n\n{text}")
# (target language, source text, regex the translation must match)
PROBES = [
    ("Japanese", "The weather is nice today, so let's go for a walk in the park.",
     r"[぀-ヿ一-鿿]"),
    ("French", "Machine translation has improved dramatically in recent years.",
     r"traduction"),
    ("English", "今日は天気がいいので、公園を散歩しましょう。", r"\bpark\b"),
]


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
  if text.count("<|") >= 5 or text.count("<pad>") >= 5:
    return True
  return False


def run_one(litert_lm, model, prompt, backend, timeout, no_template=False):
  """Runs one prompt through `litert-lm run`, greedy; the answer is stdout."""
  cmd = [litert_lm, "run", str(model), "--backend", backend,
         "--temperature", "0", "--top-k", "1", "--cache", "no"]
  if no_template:
    cmd.append("--no-template")
  cmd += ["--prompt", prompt]
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


def read_metadata(model, jinja_path=None):
  """Returns (LlmMetadata pbtext, [sections]) without unpacking the bundle.

  With jinja_path, the embedded Jinja template is written there (the peek
  API writes only that file when no dump dir is given)."""
  from litert_lm_builder import litertlm_peek  # pip install litert-lm-builder

  buf = io.StringIO()
  litertlm_peek.peek_litertlm_file(str(model), None, buf,
                                   jinja_prompt_template_path=jinja_path)
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
  with tempfile.TemporaryDirectory() as td:
    jinja_path = os.path.join(td, "template.jinja")
    pbtext, sections = read_metadata(args.model, jinja_path)
    embedded = Path(jinja_path).read_bytes() if os.path.exists(jinja_path) else b""
  if not pbtext:
    print("FAIL: no LlmMetadata section found")
    return 1
  checks = []

  def check(name, ok, detail=""):
    checks.append(ok)
    print(f"  [{'v' if ok else 'X'}] {name}{(': ' + detail) if detail else ''}")

  has_start = bool(re.search(r"^\s*start_token\s*\{", pbtext, re.M))
  check("no start_token (template renders the BOS literal; engine prepends)",
        not has_start)

  stop_ids = {int(x) for x in re.findall(r"ids:\s*(\d+)", pbtext)}
  check(f"<｜hy_place▁holder▁no▁2｜> ({EOS_ID}) is a stop-token id",
        EOS_ID in stop_ids, f"stop ids {sorted(stop_ids)}")

  md5 = hashlib.md5(embedded).hexdigest()
  detail = f"{len(embedded)} B, md5 {md5[:8]}…"
  try:  # live byte comparison when the Hub is reachable; else the recorded hash
    from huggingface_hub import hf_hub_download
    repo_tpl = Path(hf_hub_download(MODEL_ID, "chat_template.jinja")).read_bytes()
    check("embedded Jinja template is byte-equal to the repo's chat_template.jinja",
          embedded == repo_tpl, detail + f" vs repo {len(repo_tpl)} B")
  except Exception as e:  # noqa: BLE001
    check("embedded Jinja template matches the recorded repo template hash",
          len(embedded) == TEMPLATE_BYTES and md5 == TEMPLATE_MD5,
          detail + f" (Hub unreachable: {type(e).__name__})")

  ctx = re.search(r"max_num_tokens:\s*(\d+)", pbtext)
  check("max_num_tokens 4096", bool(ctx) and int(ctx.group(1)) == 4096,
        ctx.group(1) if ctx else "missing")

  total = os.path.getsize(args.model)
  bpp = total / UNIQUE_PARAMS
  check("bytes per unique parameter in the int8 band (0.9–1.35)",
        0.9 <= bpp <= 1.35, f"{total:,} B / {UNIQUE_PARAMS:,} params = {bpp:.2f}")
  big = [s for s in sections if s["bytes"] >= 2 * 1024 ** 3]
  check("every section < 2 GiB (iOS mmap ceiling)", not big,
        ", ".join(f"{s['type']}:{s['bytes'] / 1024 ** 3:.2f} GiB" for s in sections))
  types = [s["type"] for s in sections]
  check("dense bundle: LlmMetadata + tokenizer + one prefill/decode model, "
        "no ExecutorMetadata needed", "ExecutorMetadataProto" not in types
        and "TFLiteModel" in types, str(types))

  sp = re.search(r"sampler_params \{.*?\n\}", pbtext, re.S)
  if sp:
    print("  (i) sampler_params from the vendor generation_config: "
          + " ".join(sp.group(0).split()) + " — the gate overrides with "
          "--temperature 0 --top-k 1")

  ok = all(checks)
  print(f"\n  {'PASS' if ok else 'FAIL'}: {sum(checks)}/{len(checks)} metadata checks")
  return 0 if ok else 1


def load_reference(device):
  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer

  tok = AutoTokenizer.from_pretrained(MODEL_ID)
  model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
  return tok, model.to(device).eval()


def translate(args):
  import torch

  device = args.device
  if device == "auto":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
  tok, model = load_reference(device)
  rows, matches, passed = [], 0, 0
  for lang, text, must in PROBES:
    q = TRANSLATE_PROMPT.format(lang=lang, text=text)
    rendered = tok.apply_chat_template([{"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True)
    ids = tok(rendered, add_special_tokens=False, return_tensors="pt").input_ids
    with torch.no_grad():
      out = model.generate(ids.to(device), max_new_tokens=128, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    ref = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    try:
      rt = run_one(args.litert_lm, args.model, q, args.backend, args.timeout)
    except Exception as e:  # noqa: BLE001
      print(f"FAIL (harness) on {lang}: {e}", file=sys.stderr)
      return 2
    same = rt == ref
    ok = bool(rt) and not degenerate(rt) and bool(re.search(must, rt, re.I))
    matches += same
    passed += ok
    rows.append({"lang": lang, "source": text, "hf_bf16": ref, "runtime": rt,
                 "byte_identical": same, "ok": ok})
    print(f"  [{'v' if ok else '.'}] {lang:9s} byte-identical={'yes' if same else 'no '}")
    print(f"        HF bf16 : {ref!r}")
    print(f"        runtime : {rt!r}")
  print(f"\n  byte-identical to HF bf16 greedy: {matches}/{len(PROBES)}; "
        f"valid non-degenerate translations: {passed}/{len(PROBES)}")
  print(f"  VERDICT: {'PASS' if passed == len(PROBES) else 'FAIL'} "
        f"(identity is reported, not required — int8 greedy may take a fluent "
        "alternate)")
  if args.json:
    Path(args.json).write_text(json.dumps({
        "model": str(args.model), "backend": args.backend, "device": device,
        "byte_identical": matches, "passed": passed, "rows": rows},
        indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {args.json}")
  return 0 if passed == len(PROBES) else 1


def rope_ab(args):
  import torch
  from huggingface_hub import snapshot_download
  from transformers import AutoModelForCausalLM, AutoTokenizer

  sys.path.insert(0, str(Path(__file__).resolve().parent))
  from build_hy_mt2_1_8b import normalize_rope  # noqa: E402

  snap = Path(snapshot_download(MODEL_ID, ignore_patterns=["train/*", "imgs/*", "*.png"]))
  with tempfile.TemporaryDirectory() as td:
    norm = normalize_rope(snap, Path(td))
    a = AutoModelForCausalLM.from_pretrained(str(snap), dtype=torch.bfloat16).eval()
    b = AutoModelForCausalLM.from_pretrained(str(norm), dtype=torch.bfloat16).eval()
  ra, rb = a.model.rotary_emb, b.model.rotary_emb
  inv_equal = torch.equal(ra.inv_freq.view(torch.int32), rb.inv_freq.view(torch.int32))
  tok = AutoTokenizer.from_pretrained(str(snap))
  msgs = [{"role": "user", "content":
           "Translate to French: The quick brown fox jumps over the lazy dog."}]
  ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                return_dict=True, return_tensors="pt")["input_ids"]
  with torch.no_grad():
    la = a(ids, use_cache=False).logits
    lb = b(ids, use_cache=False).logits
  logits_equal = torch.equal(la, lb)
  print(f"  rope_type: {ra.rope_type} -> {rb.rope_type}; "
        f"attention_scaling {float(ra.attention_scaling)} / {float(rb.attention_scaling)}")
  print(f"  [{'v' if inv_equal else 'X'}] inv_freq bitwise-equal "
        f"(max |diff| {float((ra.inv_freq - rb.inv_freq).abs().max()):g})")
  print(f"  [{'v' if logits_equal else 'X'}] teacher-forced logits bitwise-equal "
        f"over {int(ids.shape[1])} tokens (max |diff| {float((la - lb).abs().max()):g})")
  ok = inv_equal and logits_equal
  print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'} — the bake is "
        f"{'exact' if ok else 'NOT exact; do not ship'}")
  return 0 if ok else 1


def bos_discriminator(args):
  from transformers import AutoTokenizer

  tok = AutoTokenizer.from_pretrained(MODEL_ID)
  with_bos, nobos = Path(args.bos_discriminator), args.model
  if not with_bos.exists():
    print(f"ERROR: {with_bos} not found (build it with --keep-start-token)",
          file=sys.stderr)
    return 2
  rows = []
  for lang, text, _ in PROBES:
    q = TRANSLATE_PROMPT.format(lang=lang, text=text)
    rendered = tok.apply_chat_template([{"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True)
    assert rendered.startswith(tok.bos_token), "template no longer renders a BOS literal"
    stripped = rendered[len(tok.bos_token):]
    try:
      x = run_one(args.litert_lm, with_bos, stripped, "cpu", args.timeout, no_template=True)
      yp = run_one(args.litert_lm, nobos, q, "cpu", args.timeout)
      xp = run_one(args.litert_lm, nobos, stripped, "cpu", args.timeout, no_template=True)
    except Exception as e:  # noqa: BLE001
      print(f"FAIL (harness) on {lang}: {e}", file=sys.stderr)
      return 2
    rows.append({"lang": lang, "X": x, "Yp": yp, "Xp": xp,
                 "X_eq_Yp": x == yp, "X_eq_Xp": x == xp})
    print(f"  {lang:9s} X==Y' {'yes' if x == yp else 'NO '}   X==X' {'yes' if x == xp else 'no '}")
    print(f"        X  [start_token]+rest : {x!r}")
    print(f"        Y' [template BOS]+rest: {yp!r}")
    print(f"        X' rest only          : {xp!r}")
  all_xy = all(r["X_eq_Yp"] for r in rows)
  all_xx = all(r["X_eq_Xp"] for r in rows)
  verdict = ("engine prepends start_token (X == Y' everywhere, X != X' somewhere) "
             "-> the default export fed BOS twice; dropping it is correct"
             if all_xy and not all_xx else
             "engine does NOT prepend (X == X' everywhere, X != Y' somewhere)"
             if all_xx and not all_xy else "inconclusive on these probes")
  print(f"\n  VERDICT: {verdict}")
  if args.json:
    Path(args.json).write_text(json.dumps({"with_start_token": str(with_bos),
                                           "model": str(nobos), "verdict": verdict,
                                           "rows": rows}, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {args.json}")
  return 0 if (all_xy and not all_xx) else 1


def main():
  ap = argparse.ArgumentParser(
      description="Quality gate / metadata / task checks for Hy-MT2-1.8B")
  ap.add_argument("model", help="path to model.litertlm")
  ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
  ap.add_argument("--min-correct", type=int, default=6)
  ap.add_argument("--timeout", type=int, default=900,
                  help="seconds per prompt (engine init is paid every run)")
  ap.add_argument("--litert-lm", default="litert-lm",
                  help="path to the litert-lm CLI (pip install litert-lm)")
  ap.add_argument("--device", default="auto",
                  help="--translate: HF reference device (cpu|mps|auto; the "
                       "recorded numbers are CPU)")
  ap.add_argument("--json", help="write a JSON report here")
  ap.add_argument("--check-metadata", action="store_true")
  ap.add_argument("--translate", action="store_true")
  ap.add_argument("--rope-ab", action="store_true")
  ap.add_argument("--bos-discriminator", metavar="WITH_START_TOKEN_BUNDLE")
  args = ap.parse_args()

  if not args.rope_ab and not Path(args.model).exists():
    print(f"ERROR: model not found: {args.model}", file=sys.stderr)
    return 2
  if args.check_metadata:
    return check_metadata(args)
  if args.translate:
    return translate(args)
  if args.rope_ab:
    return rope_ab(args)
  if args.bos_discriminator:
    return bos_discriminator(args)
  return gate(args)


if __name__ == "__main__":
  sys.exit(main())
