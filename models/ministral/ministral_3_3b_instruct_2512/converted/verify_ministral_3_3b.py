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

"""Quality gate, metadata and start_token checks for a Ministral-3-3B bundle.

Four modes:

  gate (default) — asks 8 fixed, unambiguously checkable questions through the
    `litert-lm` CLI (pip install litert-lm), greedy, one fresh session per
    question, and scores correctness plus degeneracy (looping, token spam,
    empty output). This is the publish guardrail: a conversion that collapses
    must not ship — and for this model, the check that catches a ChatML
    export (which runs away with <|im_start|> spam after the right answer).

      python verify_ministral_3_3b.py model.litertlm [--backend cpu|gpu]

  --check-metadata — reads the bundle's LlmMetadata (no unpacking) and asserts
    the shape this recipe produces: the published chat template (sha256 from
    ministral3_extension.py), Mistral [INST] / [/INST] per-role templates,
    </s> (id 2) as the only stop token, a start_token of "<s>" (present on
    purpose — see --bos-ab), no pad token and no channel, a 4,096-token limit,
    and the embedder in its own section with every section under 2 GiB (the
    iOS single-section mmap ceiling that the un-externalized 2.55 GiB build
    hit). Needs `pip install litert-lm-builder`.

      python verify_ministral_3_3b.py model.litertlm --check-metadata

  --bos-ab — the check that the start_token is legitimate here: feeds the
    SAME rendered [INST] prompt to the bf16 PyTorch model with and without a
    leading <s> and prints both answers. For granite-4.1-3b (bos == eos) the
    with-BOS column breaks; for Ministral (bos <s> != eos </s>) both columns
    answer, and <s>-first is what the model was trained on. Needs torch,
    transformers and the BF16 checkpoint (no bundle involved); the prompts
    are text only, so the checkpoint's own class runs its text decoder.

      python verify_ministral_3_3b.py --bos-ab [--hf <BF16 checkpoint dir>]

  --sections OTHER.litertlm — compares two files section by section: bytes
    and sha256 per section, every LlmMetadata field (the stops as a set and in
    order), the tokenizer, the ExecutorMetadata, and each TFLite graph
    (signatures, operators per kind, quantized tensors, the bytes of every
    weight buffer). Needs `pip install litert-lm-builder ai-edge-litert`.

      python verify_ministral_3_3b.py model.litertlm \\
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
CPU_THREADS = 4  # litert-lm --cpu-thread-count for every CPU run
BF16_REPO = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
BF16_REVISION = "b6d637bef2393152b3da2b2fde72eecdee30557e"
# (label, question, answer-regex, wrong-answer-veto-regex-or-None).
# The veto on the 0.9-vs-0.11 question exists because the presence check alone
# is one-sided: "0.11 is larger than 0.9" also contains "0.9" and would score
# as correct. Require the right number AND the absence of the inverted claim.
QUESTIONS = [
    ("17+25=42", "What is 17 + 25?", r"\b42\b", None),
    ("capital=Tokyo", "What is the capital of Japan?", r"tokyo", None),
    ("opp(hot)=cold", 'What is the opposite of "hot"?', r"\bcold\b", None),
    ("days/week=7", "How many days are in a week?", r"\bseven\b|\b7\b", None),
    ("thanks(fr)=merci", 'How do you say "thank you" in French?',
     r"merci", None),
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
  real loops show heavy repetition AND cratered word diversity.

  Args:
    text: the reply.

  Returns:
    True for a degenerate text.
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
    litert_lm: the litert-lm command.
    model: the .litertlm file.
    prompt: the user message.
    backend: "cpu" or "gpu".
    timeout: seconds before the process is stopped.

  Returns:
    The reply: stdout, stripped.
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


def gate(args):
  """The 8 questions, one process each.

  Args:
    args: the parsed command line (model, backend, timeout, min_correct, json).

  Returns:
    0 when the file passes, 1 when it does not, 2 on a harness failure.
  """
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


# The same value as ministral3_extension.PUBLISHED_TEMPLATE_SHA256: the chat
# template of litert-community/Ministral-3-3B-Instruct-2512 at revision
# 613e437f.
PUBLISHED_TEMPLATE_SHA256 = (
    "78bf773541dfe55d0bdd234eaeaf9f178760f1a29fcc86631a78bc73bb7f5e9e")


def read_llm_metadata(model):
  """(LlmMetadata or None, [sections]) from the file, without unpacking it.

  Args:
    model: the .litertlm file.

  Returns:
    (the LlmMetadata message or None, the sections of bundle_sections).
  """
  # pylint: disable-next=g-import-not-at-top
  from litert_lm_builder.runtime.proto import llm_metadata_pb2

  sections = bundle_sections(model)
  section = next((s for s in sections if s["type"] == "LlmMetadataProto"),
                 None)
  if section is None:
    return None, sections
  meta = llm_metadata_pb2.LlmMetadata()
  meta.ParseFromString(section_bytes(model, section))
  return meta, sections


def check_metadata(args):
  """The file's metadata against the published file's form, check by check.

  Args:
    args: the parsed command line (model).

  Returns:
    0 when every check passes, else 1.
  """
  meta, sections = read_llm_metadata(args.model)
  if meta is None:
    print("FAIL: no LlmMetadata section found")
    return 1
  checks = []

  def check(name, ok, detail=""):
    checks.append(ok)
    print(f"  [{'v' if ok else 'X'}] {name}{(': ' + detail) if detail else ''}")

  start = meta.start_token.token_str if meta.HasField("start_token") else None
  check('start_token is "<s>" (Mistral convention, bos != eos — kept on '
        "purpose)", start == "<s>", repr(start))
  stops = [list(t.token_ids.ids) or t.token_str for t in meta.stop_tokens]
  check("</s> (id 2) is the only stop token", stops == [[2]], repr(stops))
  check("no pad token", not meta.HasField("pad_token"))
  for role, want in (("user", ("[INST]", "[/INST]")),
                     ("model", ("", "</s>")),
                     ("system", ("[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]"))):
    t = getattr(meta.prompt_templates, role)
    check(f"{role} turn is {want[0]!r} ... {want[1]!r}",
          (t.prefix, t.suffix) == want, repr((t.prefix, t.suffix)))
  digest = hashlib.sha256(
      meta.jinja_prompt_template.encode("utf-8")).hexdigest()
  check("chat template is the published one (default system turn, content "
        "parts)", digest == PUBLISHED_TEMPLATE_SHA256, digest[:12])
  check("no channel", len(meta.channels) == 0,
        repr([c.channel_name for c in meta.channels]))
  check("max_num_tokens 4096", meta.max_num_tokens == 4096,
        str(meta.max_num_tokens))
  types = [s["model_type"] for s in sections]
  check("embedder in its own section", "tf_lite_embedder" in types, str(types))
  size = lambda s: s["end"] - s["begin"]
  big = [s for s in sections if size(s) >= 2 * 1024 ** 3]
  check("every section < 2 GiB (iOS mmap ceiling)", not big,
        ", ".join(f"{s['type']}:{size(s) / 1024 ** 3:.2f} GiB"
                  for s in sections))

  ok = all(checks)
  print(f"\n  {'PASS' if ok else 'FAIL'}: {sum(checks)}/{len(checks)} "
        "metadata checks")
  return 0 if ok else 1


def bos_ab(args):
  """The BF16 model on the same rendered prompt with and without a leading BOS.

  Args:
    args: the parsed command line (hf, device).

  Returns:
    0. The two columns are printed, not judged.
  """
  import torch  # deferred: only this mode needs it
  import transformers

  device = args.device
  if device == "auto":
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
  src = args.hf
  if not Path(src).is_dir():
    # pylint: disable-next=g-import-not-at-top
    from huggingface_hub import snapshot_download

    src = snapshot_download(
        src, revision=BF16_REVISION,
        allow_patterns=["*.json", "model-*.safetensors", "chat_template.jinja"])
  tok = transformers.AutoTokenizer.from_pretrained(src)
  # The checkpoint's own class; with a text-only prompt it runs its text
  # decoder alone.
  model = transformers.AutoModelForImageTextToText.from_pretrained(
      src, dtype=torch.bfloat16).to(device).eval()
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
    # What the bundle's template renders.
    text = "[INST]" + q + SUFFIX + "[/INST]"
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


def bundle_sections(model):
  """[{type, model_type, begin, end}] from the file header.

  Args:
    model: the .litertlm file.

  Returns:
    One dict per section, in the header's order.
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
    model: the .litertlm file.
    section: a dict of bundle_sections.

  Returns:
    The section's bytes.
  """
  with open(model, "rb") as f:
    f.seek(section["begin"])
    return f.read(section["end"] - section["begin"])


def tflite_summary(model, section, weights=True):
  """Signatures, operators and composites of a TFLite section (lazy read);
  with `weights` (the weights inside the section), also its quantized tensors
  and the hash of every weight buffer.

  Args:
    model: the .litertlm file.
    section: a dict of bundle_sections, a TFLite section.
    weights: also count the quantized tensors and hash the weight buffers.

  Returns:
    A dict: signatures, subgraphs, operators, composites,
    quantized_weight_tensors and buffer_hashes.
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
    args: the parsed command line (model, sections = the other file, json).

  Returns:
    0. The comparison is printed (and written to --json).
  """
  from google.protobuf import text_format  # pylint: disable=g-import-not-at-top
  # pylint: disable-next=g-import-not-at-top
  from litert_lm_builder.runtime.proto import llm_metadata_pb2

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

  meta = {}
  for k in files:
    meta[k] = llm_metadata_pb2.LlmMetadata()
    meta[k].ParseFromString(
        section_bytes(files[k], first(k, "LlmMetadataProto")))
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
            f"{show(a) if has[0] else '(unset)'}; "
            f"other: {show(b) if has[1] else '(unset)'}")
  print(f"  LlmMetadata: {len(fields) - len(diffs)} of {len(fields)} "
        f"fields equal"
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
  """Runs the check the command line names; the exit code is its result."""
  ap = argparse.ArgumentParser(
      description="Quality gate / metadata / start_token checks for "
                  "Ministral-3-3B")
  ap.add_argument("model", nargs="?", help="path to model.litertlm")
  ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
  ap.add_argument("--min-correct", type=int, default=6)
  ap.add_argument("--timeout", type=int, default=900,
                  help="seconds per question (engine init is paid every run)")
  ap.add_argument("--litert-lm", default="litert-lm",
                  help="path to the litert-lm CLI (pip install litert-lm)")
  ap.add_argument("--json",
                  help="write a JSON report here (gate / sections modes)")
  ap.add_argument("--check-metadata", action="store_true")
  ap.add_argument("--sections", metavar="OTHER_BUNDLE")
  ap.add_argument("--bos-ab", action="store_true")
  ap.add_argument("--hf", default=BF16_REPO,
                  help="the BF16 checkpoint: a local dir, or the repo id "
                       "(fetched at the pinned revision) (--bos-ab mode)")
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
  if args.sections:
    return sections_compare(args)
  return gate(args)


if __name__ == "__main__":
  sys.exit(main())
