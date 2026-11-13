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

  --sections OTHER.litertlm — compares two files section by section: bytes
    and sha256 per section, every LlmMetadata field (the stops as a set and in
    order), the tokenizer, the ExecutorMetadata, and each TFLite graph
    (signatures, operators per kind, quantized tensors, the bytes of every
    weight buffer). Needs `pip install litert-lm-builder ai-edge-litert`.

      python verify_vibethinker_3b.py model.litertlm \\
          --sections published.litertlm
"""

import argparse
import hashlib
import io
import json
import mmap
import os
import re
import subprocess
import sys
import zlib
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
    ("thanks(fr)=merci", 'How do you say "thank you" in French?', r"merci",
     None),
    ("8*7=56", "What is 8 times 7?", r"\b56\b", None),
    ("0.9>0.11", "Which is larger: 0.9 or 0.11?", r"0\.9\b",
     r"0\.11\s*(is|>)\s*(the\s+)?(larg|great|bigg)"),
    ("rhyme=blue", 'Complete the rhyme: "Roses are red, violets are ___"',
     r"\bblue\b", None),
]

# (label, problem, answer). Small GSM8K-shaped word problems with one integer
# answer.
MATH = [
    ("pens",
     "A shop sells pens at 3 dollars each. Maria buys 4 pens and pays with a "
     "20-dollar bill. How much change does she get?", "8"),
    ("train",
     "A train travels 60 miles per hour for 2 hours and then 40 miles per hour "
     "for 3 hours. How many miles does it travel in total?", "240"),
    ("apples",
     "Tom has 3 times as many apples as Jane. Together they have 48 apples. "
     "How many apples does Tom have?", "36"),
    ("tiles",
     "A rectangular floor is 12 feet by 9 feet. Tiles are 1 square foot each"
     " and come in boxes of 10. How many boxes are needed to cover the floor?",
     "11"),
    ("savings",
     "Ana saves 15 dollars a week. After 8 weeks she spends 40 dollars. "
     "How much does she have left?", "80"),
    ("chairs",
     "A hall has 14 rows of 25 chairs. 60 chairs are removed for a stage. "
     "How many chairs remain?", "290"),
]
COT = ("\n\nSolve this step by step. After your reasoning, write the final"
       " answer on its own line in the exact form:\n#### <number>")
CPU_THREADS = 4  # litert-lm --cpu-thread-count for every CPU run
# sha256 of the Jinja template the recipe writes (vibethinker_extension.py
# BUNDLE_TEMPLATE, the published file's at revision 087512d2).
BUNDLE_TEMPLATE_SHA256 = (
    "8a9b6a72a2f27bf693877b0d3ab81cf44d9a31e1719a81471ee3819788a5adb7")


def degenerate(text):
  """True if the output loops or is special-token spam.

  The 5-gram trigger needs a corroborating diversity collapse: a model may
  legitimately restate a quoted phrase a few times in a clean answer, while
  real loops show heavy repetition AND cratered word diversity.

  Args:
    text: The reply to score.
  """
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
  """Runs one prompt through `litert-lm run`; the answer is exactly stdout.

  Args:
    litert_lm: The `litert-lm` command.
    model: The `.litertlm` file.
    prompt: The prompt text.
    backend: "cpu" or "gpu".
    timeout: Seconds before the run is stopped.
  """
  cmd = [litert_lm, "run", str(model), "--prompt", prompt,
         "--backend", backend, "--temperature", "0", "--top-k", "1",
         "--cache", "no"]
  if backend == "cpu":
    # A near-tie in greedy decoding can flip with the CPU thread count, so
    # every CPU run uses the same count.
    cmd += ["--cpu-thread-count", str(CPU_THREADS)]
  # `litert-lm run` appends a non-TTY stdin to --prompt and waits for its EOF;
  # an inherited pipe would hang every run, so give it none.
  proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                        stdin=subprocess.DEVNULL)
  if proc.returncode != 0:
    raise RuntimeError(
        f"litert-lm run failed (exit={proc.returncode}).\n"
        f"--- stderr tail ---\n{proc.stderr[-1500:]}")
  return proc.stdout.strip()


def strip_think(text):
  """Drops the thought: the CLI prints the bundle's thought channel as
  `[thought] ... [/thought]` (LiteRT-LM 0.19), a file without the channel
  leaves `<think>...</think>` in the text; a thought left open counts as no
  answer.

  Args:
    text: The CLI's output.
  """
  text = re.sub(r"\[thought\].*?\[/thought\]", "", text, flags=re.S)
  if "[thought]" in text:
    return ""
  if "<think>" in text and "</think>" not in text:
    return ""
  return re.sub(r"<think>.*?</think>", "", text, flags=re.S)


def gate(args):
  """Asks the 8 questions, one process each, and scores the replies.

  Args:
    args: The parsed command line.
  """
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
  """Returns (LlmMetadata pbtext, [sections]) without unpacking the bundle.

  Args:
    model: The `.litertlm` file.
  """
  from litert_lm_builder import litertlm_peek  # pip install litert-lm-builder

  buf = io.StringIO()
  litertlm_peek.peek_litertlm_file(str(model), None, buf)
  text = buf.getvalue()
  m = re.search(
      r"start of LlmMetadata\n(.*?)>{4,} end of LlmMetadata", text, re.S)
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
  """Checks the LlmMetadata and the sections against this recipe's form.

  Args:
    args: The parsed command line.
  """
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
  check("<|im_end|> (151645) is a stop-token id (the eos fix)",
        151645 in stop_ids, f"stop ids {sorted(stop_ids)}")
  check("<|endoftext|> (151643) is a stop-token id", 151643 in stop_ids)

  def prefix(role):
    m = re.search(role + r"\s*\{\s*prefix:\s*\"((?:[^\"\\]|\\.)*)\"", pbtext)
    return m.group(1) if m else None
  check("user prefix is bare ChatML", prefix("user") == r"<|im_start|>user\n",
        repr(prefix("user")))
  check("model prefix", prefix("model") == r"<|im_start|>assistant\n",
        repr(prefix("model")))

  meta = read_llm_metadata(args.model)
  template = meta.jinja_prompt_template
  check("Jinja template = the recipe's (default system turn, content parts)",
        hashlib.sha256(template.encode()).hexdigest() == BUNDLE_TEMPLATE_SHA256,
        f"{len(template)} chars")
  check("thought channel <think> ... </think>",
        [(c.channel_name, c.start, c.end) for c in meta.channels]
        == [("thought", "<think>", "</think>")])

  types = [s["model_type"] for s in sections]
  check("embedder in its own section", "tf_lite_embedder" in types, str(types))
  big = [s for s in sections if s["bytes"] >= 2 * 1024 ** 3]
  check("every section < 2 GiB (iOS mmap ceiling)", not big,
        ", ".join(f"{s['type']}:{s['bytes'] / 1024 ** 3:.2f} GiB"
                  for s in sections))

  ok = all(checks)
  print(f"\n  {'PASS' if ok else 'FAIL'}: {sum(checks)}/{len(checks)}"
        " metadata checks")
  return 0 if ok else 1


def extract_number(text):
  """GSM8K-style: prefer '#### n', then \\boxed{n}, then the last number.

  Args:
    text: The model's reply, thought included.
  """
  t = strip_think(text).replace(",", "")
  for pat in (r"####\s*\$?(-?\d+(?:\.\d+)?)",
              r"\\boxed\{\s*\$?(-?\d+(?:\.\d+)?)",
              r"-?\d+(?:\.\d+)?"):
    m = re.findall(pat, t)
    if m:
      v = m[-1]
      return str(int(float(v))) if float(v) == int(float(v)) else v
  return None


def math_gate(args):
  """Asks the six word problems and scores the final numbers.

  Args:
    args: The parsed command line.
  """
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
    think = re.search(r"<think>(.*?)</think>|\[thought\](.*?)\[/thought\]", out,
                      re.S)
    rows.append({"label": label, "ok": ok, "expected": answer, "got": got,
                 "words": words, "answer": out})
    print(f"  [{'v' if ok else '.'}] {label:9s} expected {answer:>4s}"
          f" got {str(got):>6s}"
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


def bundle_sections(model):
  """[{type, model_type, begin, end}] from the file header.

  Args:
    model: The `.litertlm` file.
  """
  # pylint: disable-next=g-import-not-at-top
  from litert_lm_builder import litertlm_core, litertlm_peek

  header = litertlm_peek.read_litertlm_header(str(model), io.StringIO())
  table = header.SectionMetadata()
  rows = []
  for i in range(table.ObjectsLength()):
    s = table.Objects(i)
    rows.append({
        "type": litertlm_core.any_section_data_type_to_string(s.DataType()),
        "model_type": litertlm_peek.get_model_type(s),
        "begin": s.BeginOffset(), "end": s.EndOffset()})
  return rows


def section_bytes(model, section):
  """The bytes of one section.

  Args:
    model: The `.litertlm` file.
    section: A row of `bundle_sections`.
  """
  with open(model, "rb") as f:
    f.seek(section["begin"])
    return f.read(section["end"] - section["begin"])


def read_llm_metadata(model):
  """The LlmMetadata proto of a bundle.

  Args:
    model: The `.litertlm` file.
  """
  # pylint: disable-next=g-import-not-at-top
  from litert_lm_builder.runtime.proto import llm_metadata_pb2

  section = next(s for s in bundle_sections(model)
                 if s["type"] == "LlmMetadataProto")
  meta = llm_metadata_pb2.LlmMetadata()
  meta.ParseFromString(section_bytes(model, section))
  return meta


def tflite_summary(model, section, weights=True):
  """Signatures, operators and composites of a TFLite section (lazy read);
  with `weights` (the weights inside the section), also its quantized tensors
  and the hash of every weight buffer.

  Args:
    model: The `.litertlm` file.
    section: A row of `bundle_sections` with a TFLite model.
    weights: Whether the weights are inside the section.
  """
  # pylint: disable-next=g-import-not-at-top
  from ai_edge_litert import schema_py_generated as fb

  with open(model, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
  buf = memoryview(mm)[section["begin"]:section["end"]]
  m = fb.Model.GetRootAs(buf, 0)
  names = {getattr(fb.BuiltinOperator, k): k
           for k in dir(fb.BuiltinOperator) if not k.startswith("_")}
  types = {getattr(fb.TensorType, k): k
           for k in dir(fb.TensorType) if not k.startswith("_")}
  codes = []
  for i in range(m.OperatorCodesLength()):
    oc = m.OperatorCodes(i)
    codes.append(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()))
  ops, composites, quantized = Counter(), Counter(), Counter()
  for s in range(m.SubgraphsLength()):
    sg = m.Subgraphs(s)
    for o in range(sg.OperatorsLength()):
      op = sg.Operators(o)
      code = codes[op.OpcodeIndex()]
      ops[names.get(code, str(code))] += 1
      if (code == fb.BuiltinOperator.STABLEHLO_COMPOSITE
          and op.BuiltinOptions2Type()
          == fb.BuiltinOptions2.StableHLOCompositeOptions):
        opt = fb.StableHLOCompositeOptions()
        t = op.BuiltinOptions2()
        opt.Init(t.Bytes, t.Pos)
        composites[opt.Name().decode()] += 1
    seen = set()
    for t in range(sg.TensorsLength()):
      tensor = sg.Tensors(t)
      if (weights
          and tensor.Type() in (fb.TensorType.INT4, fb.TensorType.INT8)
          and tensor.Buffer() not in seen
          and m.Buffers(tensor.Buffer()).Size()):
        seen.add(tensor.Buffer())
        quantized[types[tensor.Type()]] += 1
  hashes = Counter()
  for i in range(m.BuffersLength() if weights else 0):
    b = m.Buffers(i)
    if b.Offset() > 1:
      data = buf[b.Offset():b.Offset() + b.Size()]
    elif b.DataLength():
      data = b.DataAsNumpy()
    else:
      continue
    hashes[(len(data), hashlib.sha256(data).hexdigest())] += 1
  signatures = sorted(m.SignatureDefs(i).SignatureKey().decode()
                      for i in range(m.SignatureDefsLength()))
  return {"signatures": signatures, "subgraphs": m.SubgraphsLength(),
          "operators": dict(ops), "composites": dict(composites),
          "quantized_weight_tensors": dict(quantized), "buffer_hashes": hashes}


def sections_compare(args):
  """This file against another, section by section.

  Args:
    args: The parsed command line.
  """
  from google.protobuf import text_format  # pylint: disable=g-import-not-at-top

  files = {"this": Path(args.model), "other": Path(args.sections)}
  secs = {k: bundle_sections(p) for k, p in files.items()}
  report = {k: str(p) for k, p in files.items()}
  for k, p in files.items():
    print(f"  {k:5s}: {p} ({os.path.getsize(p):,} B)")
  print("  sections (type, model_type, bytes, sha256), this | other:")
  rows = []
  for i in range(max(len(secs["this"]), len(secs["other"]))):
    row = {}
    for k in files:
      if i < len(secs[k]):
        s = secs[k][i]
        row[k] = (s["type"], s["model_type"], s["end"] - s["begin"],
                  hashlib.sha256(section_bytes(files[k], s)).hexdigest())
    rows.append(row)
    print("    " + " | ".join(
        f"{r[0]} {r[1] or ''} {r[2]:,} {r[3][:12]}" if r else "-"
        for r in (row.get("this"), row.get("other"))))
  report["sections"] = rows

  def first(k, kind):
    return next((s for s in secs[k] if s["type"] == kind), None)

  meta = {k: read_llm_metadata(p) for k, p in files.items()}
  fields = sorted({f.name for k in files for f, _ in meta[k].ListFields()})
  diffs = []
  for name in fields:
    a, b = (getattr(meta[k], name) for k in files)
    if name == "stop_tokens":
      sa = sorted(text_format.MessageToString(t, as_one_line=True) for t in a)
      sb = sorted(text_format.MessageToString(t, as_one_line=True) for t in b)
      same_order = list(a) == list(b)
      if sa != sb:
        diffs.append(name)
      print(f"  LlmMetadata.stop_tokens: {len(a)} vs {len(b)}, same set: "
            f"{sa == sb}, same order: {same_order}")
      for t in sorted(set(sa) ^ set(sb)):
        print(f"      only {'this' if t in sa else 'other'}: {t}")
      continue
    has = [any(f.name == name for f, _ in meta[k].ListFields()) for k in files]
    same = has[0] == has[1] and a == b
    if not same:
      diffs.append(name)
      show = lambda v: (
          repr(v)[:200] if not hasattr(v, "ListFields")
          else text_format.MessageToString(v, as_one_line=True)[:200])
      print(f"  LlmMetadata.{name}: differs; this: "
            f"{show(a) if has[0] else '(unset)'}; other: "
            f"{show(b) if has[1] else '(unset)'}")
  print(f"  LlmMetadata: {len(fields) - len(diffs)} of {len(fields)}"
        " fields equal"
        f"{'' if not diffs else '; differ: ' + ', '.join(diffs)}")
  report["llm_metadata_differs"] = diffs

  toks = {}
  for k in files:
    data = section_bytes(files[k], first(k, "HF_Tokenizer_Zlib"))
    toks[k] = zlib.decompress(data[8:])
  same_tok = toks["this"] == toks["other"]
  print(f"  tokenizer.json: {len(toks['this']):,} vs {len(toks['other']):,} B, "
        f"{'byte-equal' if same_tok else 'differs'}")
  if not same_tok:
    ja, jb = (json.loads(toks[k]) for k in files)
    keys = sorted(k for k in set(ja) | set(jb) if ja.get(k) != jb.get(k))
    print(f"      top-level keys that differ after parsing: {keys or 'none'}")
    report["tokenizer_keys_differ"] = keys
  report["tokenizer_equal"] = same_tok

  ex = {k: first(k, "ExecutorMetadataProto") for k in files}
  ex_state = ("absent in both" if not any(ex.values()) else
              "only in this" if not ex["other"] else
              "only in other" if not ex["this"] else
              "byte-equal" if section_bytes(files["this"], ex["this"])
              == section_bytes(files["other"], ex["other"]) else "differs")
  print(f"  ExecutorMetadata: {ex_state}")
  report["executor_metadata"] = ex_state

  report["graphs"] = {}
  for mt in ("tf_lite_prefill_decode", "tf_lite_embedder"):
    g, inline = {}, {}
    for k in files:
      model_sec = next((s for s in secs[k] if s["type"] == "TFLiteModel"
                        and s["model_type"] == mt), None)
      inline[k] = not any(s["type"] == "TFLiteWeights" and s["model_type"] == mt
                          for s in secs[k])
      g[k] = None if model_sec is None else tflite_summary(
          files[k], model_sec, weights=inline[k])
    if not all(g.values()):
      print(f"  {mt}: not compared (no section)")
      continue
    ha, hb = (g[k].pop("buffer_hashes") for k in files)
    keys = ["signatures", "subgraphs", "composites"]
    if all(inline.values()):
      common = sum((ha & hb).values())
      print(f"  {mt}: weight buffers {sum(ha.values())} vs {sum(hb.values())}, "
            f"{common} byte-equal; the rest hold "
            f"{sum(n * c for (n, _), c in (ha - hb).items()):,} B vs "
            f"{sum(n * c for (n, _), c in (hb - ha).items()):,} B")
      keys.append("quantized_weight_tensors")
    else:
      common = None
      print(f"  {mt}: graph only (weights outside the graph in "
            f"{' and '.join(k for k in files if not inline[k])})")
    for key in keys:
      a, b = g["this"][key], g["other"][key]
      print(f"      {key}: {'equal' if a == b else 'differs'}: {a}"
            f"{'' if a == b else ' vs ' + str(b)}")
    oa, ob = g["this"]["operators"], g["other"]["operators"]
    odiff = {o: (oa.get(o, 0), ob.get(o, 0)) for o in sorted(set(oa) | set(ob))
             if oa.get(o, 0) != ob.get(o, 0)}
    print(f"      operators: {sum(oa.values())} vs {sum(ob.values())}; "
          f"differing counts (this, other): {odiff or 'none'}")
    report["graphs"][mt] = {
        "this": g["this"], "other": g["other"],
        "buffers": (None if common is None else
                    [sum(ha.values()), sum(hb.values()), common]),
        "operators_differ": odiff}
  if args.json:
    Path(args.json).write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"  wrote {args.json}")
  return 0


def main():
  """Parses the arguments and runs the chosen mode."""
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
  ap.add_argument("--json", help="write a JSON report here "
                  "(gate / math / sections modes)")
  ap.add_argument("--check-metadata", action="store_true")
  ap.add_argument("--math", action="store_true")
  ap.add_argument("--sections", metavar="OTHER_BUNDLE")
  args = ap.parse_args()

  if not Path(args.model).exists():
    print(f"ERROR: model not found: {args.model}", file=sys.stderr)
    return 2
  if args.check_metadata:
    return check_metadata(args)
  if args.math:
    return math_gate(args)
  if args.sections:
    return sections_compare(args)
  return gate(args)


if __name__ == "__main__":
  sys.exit(main())
