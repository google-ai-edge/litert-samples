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

"""Quality gate, metadata, template and section checks for an LFM3-1.7B bundle.

Four modes:

  gate (default) — asks 8 fixed, unambiguously checkable questions through the
    `litert-lm` CLI (pip install litert-lm), greedy, thinking off, one fresh
    process per question, and scores correctness plus degeneracy (looping,
    token spam, empty output). This is the publish guardrail: a conversion
    that collapses must not ship.

      python verify_lfm3_1_7b.py model.litertlm [--backend cpu|gpu]

  --check-metadata — reads the bundle (no unpacking) and asserts the shape
    this recipe produces: <|startoftext|> as start_token (the template renders
    `bos_token`, which the runtime leaves empty), <|im_end|> (124900) as the
    only stop, TOP_P with k 50 / p 1 / temperature 0.1, the context length of
    the file name, the generic model type and a thought channel, the embedded
    Jinja equal to the checkpoint's chat_template.jinja after the exporter's
    rewrites and lfm3_extension.history_stable_template, the tokenizer section
    equal to the checkpoint's tokenizer.json, an ExecutorMetadata section with
    a convolution state for each of the 13 conv layers and global K / V caches
    for the 11 attention layers, the seven signatures (prefill 1024 / 256 / 64
    / 16 / 4 / 1 and decode), the int4 tensors of the mixed int4 recipe (48:
    the MLP gate and up projections of the 24 layers; none in an int8 file)
    with every block scale at or above 2^-14 (XNNPACK refuses a zero or subnormal fp16 scale),
    and every section under the iOS ~2 GiB single-section mmap ceiling. Needs
    `pip install litert-lm-builder`.

      python verify_lfm3_1_7b.py model.litertlm --check-metadata

  --render — the runtime's own rendering of the bundle's template
    (`Conversation.render_message_to_string`, litert-lm Python package)
    against transformers' `apply_chat_template` on the checkpoint's template:
    one turn, three turns and a tool round trip, thinking off and on. Two
    differences are expected and accounted for: the start token (the runtime
    renders `bos_token` empty and adds the metadata start_token itself) and,
    in the tool round trip, the thought of the assistant turn after the last
    user message (lfm3_extension.history_stable_template drops it). Needs
    litert-lm and transformers in one environment.

      python verify_lfm3_1_7b.py model.litertlm --render

  --sections OTHER.litertlm — compares the two files section by section:
    bytes and sha256 per section, the LlmMetadata fields, the template, and a
    summary of each TFLite graph (signatures, operators, int4 tensors).

      python verify_lfm3_1_7b.py model.litertlm --sections published/model.litertlm
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
import tempfile
import zlib
from collections import Counter
from pathlib import Path

MODEL_ID = "LiquidAI/LFM3-1.7B"
REVISION = "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"
CPU_THREADS = 4  # litert-lm --cpu-thread-count for every CPU run
START_TOKEN = "<|startoftext|>"
EOS_ID = 124900  # <|im_end|>
PAD_TOKEN = "<|pad|>"
CONV_LAYERS, ATTENTION_LAYERS = 13, 11
INT4_TENSORS = 48  # w1 and w3 of the 24 MLPs
INT4_SCALE_FLOOR = 2.0 ** -14
SIGNATURES = {"prefill_1024", "prefill_256", "prefill_64", "prefill_16",
              "prefill_4", "prefill_1", "decode"}
GIB = 1024 ** 3

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


def source_file(args, name):
  """A file of the checkpoint: from --source, else the Hub at REVISION."""
  if args.source:
    return Path(args.source) / name
  from huggingface_hub import hf_hub_download

  return Path(hf_hub_download(MODEL_ID, name, revision=REVISION))


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


def answer_part(text):
  """The reply without a thought block; '' when a thought never closed."""
  text = re.sub(r"\[thought\].*?\[/thought\]", "", text, flags=re.S)
  if "<think>" in text and "</think>" not in text:
    return ""
  return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def run_one(litert_lm, model, prompt, backend, timeout):
  """Runs one prompt through `litert-lm run`, greedy, thinking off."""
  cmd = [litert_lm, "run", str(model), "--backend", backend,
         "--temperature", "0", "--top-k", "1", "--cache", "no",
         "--thinking", "false"]
  if backend == "cpu":
    # A near-tie in greedy decoding can flip with the CPU thread count, so
    # every CPU run uses the same count.
    cmd += ["--cpu-thread-count", str(CPU_THREADS)]
  cmd += ["--prompt", prompt]
  # `litert-lm run` appends a non-TTY stdin to --prompt and waits for its EOF;
  # an inherited pipe or socket would hang every run, so give it none.
  proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                        stdin=subprocess.DEVNULL)
  if proc.returncode != 0:
    raise RuntimeError(
        f"litert-lm run failed (exit={proc.returncode}).\n"
        f"--- stderr tail ---\n{proc.stderr[-1500:]}")
  return proc.stdout.strip()


def gate(args):
  results = []
  for label, q, pat, veto in QUESTIONS:
    try:
      raw = run_one(args.litert_lm, args.model, q + SUFFIX, args.backend,
                    args.timeout)
    except Exception as e:  # noqa: BLE001
      print(f"FAIL (harness) on '{label}': {e}", file=sys.stderr)
      return 2
    ans = answer_part(raw)
    low = ans.lower()
    ok = bool(re.search(pat, low)) and not (veto and re.search(veto, low))
    degen = (not ans.strip()) or degenerate(ans)
    results.append({"label": label, "question": q, "ok": ok,
                    "degenerate": degen, "answer": raw,
                    "thought_in_reply": ans != raw.strip()})
    shown = " ".join(raw.split())[:90] if raw.strip() else "(empty)"
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


def peek(model, jinja_path=None):
  """Returns (LlmMetadata pbtext, ExecutorMetadata pbtext, [sections])."""
  from litert_lm_builder import litertlm_peek  # pip install litert-lm-builder

  buf = io.StringIO()
  litertlm_peek.peek_litertlm_file(str(model), None, buf,
                                   jinja_prompt_template_path=jinja_path)
  text = buf.getvalue()
  m = re.search(r"start of LlmMetadata\n(.*?)>{4,} end of LlmMetadata", text, re.S)
  llm = m.group(1) if m else ""
  m = re.search(r"start of ExecutorMetadata\n(.*?)>{4,} end of ExecutorMetadata",
                text, re.S)
  executor = m.group(1) if m else ""
  sections = []
  for sm in re.finditer(
      r"Section (\d+):\n(.*?)Begin Offset:\s*(\d+)\n\s*End Offset:\s*(\d+)\n"
      r"\s*Data Type:\s*(\S+)", text, re.S):
    mt = re.search(r"model_type, Value \(String\): (\S+)", sm.group(2))
    sections.append({"index": int(sm.group(1)), "type": sm.group(5),
                     "model_type": mt.group(1) if mt else None,
                     "begin": int(sm.group(3)), "end": int(sm.group(4))})
  return llm, executor, sections


def read_template(model):
  with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "template.jinja")
    llm, executor, sections = peek(model, path)
    template = Path(path).read_text() if os.path.exists(path) else ""
  return llm, executor, sections, template


def exporter_rewrite(template):
  """The two rewrites litert-torch's packager applies to an embedded template
  (core/litert_lm_builder.py, build_llm_metadata)."""
  template = re.sub(r"\{%-?\s*(?:end)?generation\s*-?%\}", "", template)
  return re.sub(r"\.get\(\s*(['\"][^'\"]+['\"])\s*\)", r"[\1]", template)


def expected_template(args):
  sys.path.insert(0, str(Path(__file__).resolve().parent))
  import lfm3_extension  # pylint: disable=g-import-not-at-top

  vendor = source_file(args, "chat_template.jinja").read_text()
  return lfm3_extension.history_stable_template(exporter_rewrite(vendor))


def section_bytes(model, section):
  with open(model, "rb") as f:
    f.seek(section["begin"])
    return f.read(section["end"] - section["begin"])


def tflite_summary(model, section, contents=False):
  """Signatures, operators and int4 tensors of a TFLite section (lazy read).

  With `contents`, also the sha256 of every buffer and the tensor names.
  """
  from ai_edge_litert import schema_py_generated as fb  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top

  # The flatbuffer objects keep views into the mapping, so it stays open until
  # they are collected.
  with open(model, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
  buf = memoryview(mm)[section["begin"]:section["end"]]
  m = fb.Model.GetRootAs(buf, 0)
  names = {getattr(fb.BuiltinOperator, k): k
           for k in dir(fb.BuiltinOperator) if not k.startswith("_")}
  codes = []
  for i in range(m.OperatorCodesLength()):
    oc = m.OperatorCodes(i)
    codes.append(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()))
  ops, composites = Counter(), Counter()
  int4_buffers, scale_min = set(), None
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
    for t in range(sg.TensorsLength()):
      tensor = sg.Tensors(t)
      if tensor.Type() != fb.TensorType.INT4:
        continue
      int4_buffers.add(tensor.Buffer())
      q = tensor.Quantization()
      scales = None
      if q is not None and q.DetailsType() == fb.QuantizationDetails.BlockwiseQuantization:
        bq = fb.BlockwiseQuantization()
        d = q.Details()
        bq.Init(d.Bytes, d.Pos)
        st = sg.Tensors(bq.Scales())
        dtype = {fb.TensorType.FLOAT16: np.float16,
                 fb.TensorType.FLOAT32: np.float32}[st.Type()]
        b = m.Buffers(st.Buffer())
        if b.Offset() > 1:
          raw = buf[b.Offset():b.Offset() + b.Size()]
        else:
          raw = b.DataAsNumpy()
        scales = np.frombuffer(raw, dtype=dtype)
      elif q is not None and q.ScaleLength():
        scales = q.ScaleAsNumpy()
      if scales is not None and scales.size:
        low = float(np.min(scales.astype(np.float32)))
        scale_min = low if scale_min is None else min(scale_min, low)
  signatures = sorted(m.SignatureDefs(i).SignatureKey().decode()
                      for i in range(m.SignatureDefsLength()))
  summary = {"signatures": signatures, "subgraphs": m.SubgraphsLength(),
             "operators": dict(ops), "composites": dict(composites),
             "int4_tensors": len(int4_buffers), "int4_scale_min": scale_min}
  if contents:
    hashes = Counter()
    for i in range(m.BuffersLength()):
      b = m.Buffers(i)
      if b.Offset() > 1:
        data = buf[b.Offset():b.Offset() + b.Size()]
      elif b.DataLength():
        data = b.DataAsNumpy()
      else:
        continue
      hashes[(len(data), hashlib.sha256(data).hexdigest())] += 1
    tensor_names = set()
    for s in range(m.SubgraphsLength()):
      sg = m.Subgraphs(s)
      for t in range(sg.TensorsLength()):
        tensor_names.add(sg.Tensors(t).Name().decode(errors="replace"))
    summary["buffer_hashes"] = hashes
    summary["tensor_names"] = tensor_names
  return summary


def check_metadata(args):
  llm, executor, sections, template = read_template(args.model)
  if not llm:
    print("FAIL: no LlmMetadata section found")
    return 1
  checks = []

  def check(name, ok, detail=""):
    checks.append(ok)
    print(f"  [{'v' if ok else 'X'}] {name}{(': ' + detail) if detail else ''}")

  start = re.search(r'start_token \{\s*token_str: "([^"]*)"', llm)
  check(f"start_token {START_TOKEN} (the runtime renders bos_token empty)",
        bool(start) and start.group(1) == START_TOKEN,
        start.group(1) if start else "missing")
  stops = re.findall(r"stop_tokens \{(.*?)\n    \}", llm, re.S)
  stop_ids = [int(x) for s in stops for x in re.findall(r"ids:\s*(\d+)", s)]
  str_stops = [s for s in stops if "token_str" in s]
  check(f"the only stop is <|im_end|> ({EOS_ID}) as an id",
        stop_ids == [EOS_ID] and not str_stops,
        f"ids {stop_ids}, {len(str_stops)} string stops")
  pad = re.search(r'pad_token \{\s*token_str: "([^"]*)"', llm)
  check(f"pad_token {PAD_TOKEN}", bool(pad) and pad.group(1) == PAD_TOKEN)
  sp = re.search(r"sampler_params \{(.*?)\n    \}", llm, re.S)
  sp_text = " ".join(sp.group(1).split()) if sp else ""
  check("sampler TOP_P, k 50, p 1, temperature 0.1",
        sp_text == "type: TOP_P k: 50 p: 1.0 temperature: 0.1", sp_text or "missing")
  ctx = re.search(r"max_num_tokens:\s*(\d+)", llm)
  want_ctx = re.search(r"ekv(\d+)", Path(args.model).name)
  want_ctx = int(want_ctx.group(1)) if want_ctx else args.ctx
  check(f"max_num_tokens {want_ctx}", bool(ctx) and int(ctx.group(1)) == want_ctx,
        ctx.group(1) if ctx else "missing")
  check("generic model type", "generic_model" in llm)
  check("thought channel <think> ... </think>",
        bool(re.search(r'channels \{\s*channel_name: "thought"\s*start: "<think>"'
                       r'\s*end: "</think>"', llm)))
  try:
    want = expected_template(args)
    check("embedded Jinja = the checkpoint's template after the exporter's "
          "rewrites and history_stable_template",
          template == want, f"{len(template.encode())} B, sha256 "
          f"{hashlib.sha256(template.encode()).hexdigest()[:12]}… vs "
          f"{len(want.encode())} B")
  except Exception as e:  # noqa: BLE001
    check("embedded Jinja (checkpoint template unreachable)", False,
          f"{type(e).__name__}: {e}")

  by_type = Counter(s["type"] for s in sections)
  check("sections: LlmMetadata, ExecutorMetadata, tokenizer, one TFLite model",
        by_type == Counter({"LlmMetadataProto": 1, "ExecutorMetadataProto": 1,
                            "HF_Tokenizer_Zlib": 1, "TFLiteModel": 1}),
        str(dict(by_type)))
  big = [s for s in sections if s["end"] - s["begin"] >= 2 * GIB]
  check("every section < 2 GiB (iOS mmap ceiling)", not big,
        ", ".join(f"{s['type']}:{(s['end'] - s['begin']) / GIB:.2f} GiB"
                  for s in sections))
  tok = next((s for s in sections if s["type"] == "HF_Tokenizer_Zlib"), None)
  if tok:
    # The packer writes the uncompressed size (8 bytes, little endian), then
    # the zlib stream of the tokenizer.json it was given.
    blob = section_bytes(args.model, tok)
    data = zlib.decompress(blob[8:])
    size_ok = int.from_bytes(blob[:8], "little") == len(data)
    try:
      repo = source_file(args, "tokenizer.json").read_bytes()
      check("tokenizer section = the checkpoint's tokenizer.json",
            size_ok and data == repo, f"{len(data):,} B vs {len(repo):,} B")
    except Exception as e:  # noqa: BLE001
      check("tokenizer section (checkpoint tokenizer unreachable)", False,
            f"{type(e).__name__}: {e}")

  kinds = Counter(re.findall(r"type:\s*(TYPE_\w+)", executor))
  lengths = sorted({int(x) for x in re.findall(r"maximum_sequence_length:\s*(\d+)", executor)})
  check(f"ExecutorMetadata: {CONV_LAYERS} conv states, {ATTENTION_LAYERS} + "
        f"{ATTENTION_LAYERS} global K/V caches at {want_ctx}",
        kinds == Counter({"TYPE_LINEAR_ATTENTION": CONV_LAYERS,
                          "TYPE_GLOBAL_KEY_CACHE": ATTENTION_LAYERS,
                          "TYPE_GLOBAL_VALUE_CACHE": ATTENTION_LAYERS})
        and lengths == [want_ctx], f"{dict(kinds)}, lengths {lengths}")

  model_sec = next(s for s in sections if s["type"] == "TFLiteModel")
  g = tflite_summary(args.model, model_sec)
  check("signatures prefill 1024 / 256 / 64 / 16 / 4 / 1 and decode",
        set(g["signatures"]) == SIGNATURES, ", ".join(g["signatures"]))
  mixed = "int4" in Path(args.model).name or args.int4
  want_int4 = INT4_TENSORS if mixed else 0
  check(f"{want_int4} int4 tensors ({'MLP gate and up projections' if mixed else 'int8 file'})",
        g["int4_tensors"] == want_int4, str(g["int4_tensors"]))
  if mixed:
    low = g["int4_scale_min"]
    check("every int4 block scale >= 2^-14 (no zero or subnormal fp16 scale for XNNPACK)",
          low is not None and low >= INT4_SCALE_FLOOR, f"min {low!r}")
  gpu = [n for n in g["composites"] if re.search(r"sdpa|attention|cache|short_conv|rope", n)]
  check("no attention, cache, short-conv or rope composite (plain operators)",
        not gpu, f"composites {g['composites'] or 'none'}")
  ok = all(checks)
  print(f"\n  {'PASS' if ok else 'FAIL'}: {sum(checks)}/{len(checks)} metadata checks")
  if args.json:
    Path(args.json).write_text(json.dumps({
        "model": str(args.model), "passed": ok, "checks": len(checks),
        "graph": g}, indent=2) + "\n")
  return 0 if ok else 1


CODE_WORD = "TANGERINE-7319"
TURNS = [
    f"Remember this code word for later: {CODE_WORD}. Reply with one short sentence.",
    "What is 12 + 30? Answer briefly.",
    "What was the code word I gave you at the start? Answer with the code word only.",
]
TOOL = {"type": "function", "function": {
    "name": "lookup_secret",
    "description": "Returns the secret value stored under a key in the user's vault.",
    "parameters": {"type": "object",
                   "properties": {"key": {"type": "string", "description": "Vault key"}},
                   "required": ["key"]}}}
TOOL_QUESTION = ('Use the lookup_secret tool to read the value stored under the '
                 'key "alpha", then tell me that value.')
TOOL_CALL = ("<tool_call>\n<function=lookup_secret>\n<parameter=key>\nalpha\n"
             "</parameter>\n</function>\n</tool_call>")
TOOL_THOUGHT = "I should call the tool with key alpha."


def scenarios():
  """(name, runtime messages, transformers messages, tools, thinking)."""
  ru = lambda t: {"role": "user", "content": [{"type": "text", "text": t}]}
  hu = lambda t: {"role": "user", "content": t}
  ra = lambda c, r: {"role": "assistant", "content": [{"type": "text", "text": c}],
                     "channels": {"thought": r}, "reasoning_content": r}
  ha = lambda c, r: {"role": "assistant", "content": c, "reasoning_content": r}
  a1, r1 = "Nice to meet you. I will remember it.", "The user gave a code word to remember."
  a2, r2 = "12 + 30 = 42.", "Simple addition: 12 plus 30 is 42."
  out = []
  for think in (False, True):
    tag = "on" if think else "off"
    out.append((f"1turn_{tag}", [ru(TURNS[0])], [hu(TURNS[0])], None, think))
    out.append((f"3turn_{tag}",
                [ru(TURNS[0]), ra(a1, r1), ru(TURNS[1]), ra(a2, r2), ru(TURNS[2])],
                [hu(TURNS[0]), ha(a1, r1), hu(TURNS[1]), ha(a2, r2), hu(TURNS[2])],
                None, think))
    out.append((f"tool_{tag}",
                [ru(TOOL_QUESTION), ra(TOOL_CALL, TOOL_THOUGHT),
                 {"role": "tool", "content": [{"type": "text", "text": "48213"}]}],
                [hu(TOOL_QUESTION), ha(TOOL_CALL, TOOL_THOUGHT),
                 {"role": "tool", "content": "48213"}],
                [TOOL], think))
  return out


def render(args):
  import litert_lm  # pylint: disable=g-import-not-at-top
  import transformers  # pylint: disable=g-import-not-at-top

  tok_dir = source_file(args, "tokenizer.json").parent
  for name in ("tokenizer_config.json", "special_tokens_map.json",
               "chat_template.jinja"):
    source_file(args, name)
  tok = transformers.AutoTokenizer.from_pretrained(str(tok_dir),
                                              trust_remote_code=True)
  rows, ok = [], True
  with litert_lm.Engine(str(args.model), backend=litert_lm.Backend.CPU(),
                        cache_dir=":nocache") as engine:
    for name, rt_msgs, hf_msgs, tools, think in scenarios():
      conv = engine.create_conversation(
          messages=rt_msgs[:-1] or None,
          extra_context={"tools": tools} if tools else None,
          thinking_config=litert_lm.ThinkingConfig(enable_thinking=think))
      runtime = conv.render_message_to_string(rt_msgs[-1])
      conv.close()
      want = tok.apply_chat_template(hf_msgs, tools=tools, tokenize=False,
                                     add_generation_prompt=True,
                                     enable_thinking=think)
      accounted = want
      notes = []
      if accounted.startswith(START_TOKEN) and not runtime.startswith(START_TOKEN):
        accounted = accounted[len(START_TOKEN):]
        notes.append("start token from the metadata")
      if name.startswith("tool") and TOOL_THOUGHT in accounted:
        accounted = accounted.replace(f"<think>{TOOL_THOUGHT}</think>", "<think></think>")
        notes.append("thought after the last user message dropped")
      same = runtime == accounted
      ok &= same
      rows.append({"scenario": name, "equal_after_accounted": same,
                   "accounted": notes, "runtime": runtime, "transformers": want})
      print(f"  [{'v' if same else 'X'}] {name:9s} runtime == transformers"
            f"{' except: ' + '; '.join(notes) if notes else ''}")
      if not same:
        import difflib  # pylint: disable=g-import-not-at-top
        for line in difflib.unified_diff(accounted.splitlines(), runtime.splitlines(),
                                         "transformers", "runtime", lineterm="", n=1):
          print("      " + line)
  print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'} ({sum(r['equal_after_accounted'] for r in rows)}"
        f"/{len(rows)} scenarios)")
  if args.json:
    Path(args.json).write_text(json.dumps({"model": str(args.model), "passed": ok,
                                           "rows": rows}, indent=2, ensure_ascii=False) + "\n")
  return 0 if ok else 1


def sections_compare(args):
  other = Path(args.sections)
  a_llm, a_exec, a_secs, a_tpl = read_template(args.model)
  b_llm, b_exec, b_secs, b_tpl = read_template(other)
  print(f"  this : {args.model} ({os.path.getsize(args.model):,} B)")
  print(f"  other: {other} ({os.path.getsize(other):,} B)")
  rows = []
  for sa, sb in zip(a_secs, b_secs):
    da, db = section_bytes(args.model, sa), section_bytes(other, sb)
    ha, hb = hashlib.sha256(da).hexdigest(), hashlib.sha256(db).hexdigest()
    rows.append({"type": sa["type"], "other_type": sb["type"], "bytes": len(da),
                 "other_bytes": len(db), "sha256": ha, "other_sha256": hb,
                 "equal": ha == hb})
    print(f"  {sa['type']:22s} {len(da):>14,} B  {ha[:12]}…  "
          f"{'==' if ha == hb else '!='}  {hb[:12]}…  {len(db):>14,} B")
  strip = lambda t: re.sub(r"\n\s*jinja_prompt_template:.*?(?=\n\s*\w+[ :{])", "", t, flags=re.S)
  la = [l.strip() for l in strip(a_llm).splitlines() if l.strip()]
  lb = [l.strip() for l in strip(b_llm).splitlines() if l.strip()]
  if la == lb:
    print("  LlmMetadata fields other than the template: equal")
  else:
    import difflib  # pylint: disable=g-import-not-at-top
    print("  LlmMetadata fields other than the template differ:")
    for line in difflib.unified_diff(lb, la, "other", "this", lineterm="", n=0):
      print("      " + line)
  if a_tpl == b_tpl:
    print("  template: byte-equal")
  else:
    import difflib  # pylint: disable=g-import-not-at-top
    print(f"  template: {len(a_tpl.encode())} B vs {len(b_tpl.encode())} B; the lines that differ:")
    for line in difflib.unified_diff(b_tpl.splitlines(), a_tpl.splitlines(), "other",
                                     "this", lineterm="", n=0):
      print("      " + repr(line))
  print(f"  ExecutorMetadata text: {'equal' if a_exec == b_exec else 'differs'}")
  ga = tflite_summary(args.model, next(s for s in a_secs if s["type"] == "TFLiteModel"), True)
  gb = tflite_summary(other, next(s for s in b_secs if s["type"] == "TFLiteModel"), True)
  ha, hb = ga.pop("buffer_hashes"), gb.pop("buffer_hashes")
  common = sum((ha & hb).values())
  only_a = sum(n * c for (n, _), c in (ha - hb).items())
  only_b = sum(n * c for (n, _), c in (hb - ha).items())
  print(f"  graph buffers with data: {sum(ha.values())} vs {sum(hb.values())}; "
        f"{common} have the same bytes in both; the rest hold {only_a:,} B "
        f"vs {only_b:,} B")
  na, nb = ga.pop("tensor_names"), gb.pop("tensor_names")
  print(f"  graph tensor names: {len(na)} vs {len(nb)}; {len(na & nb)} in both")
  for label, names in (("only here", sorted(na - nb)[:3]), ("only there", sorted(nb - na)[:3])):
    for n in names:
      print(f"      {label}: {n[:160]}")
  for key in ("signatures", "subgraphs", "int4_tensors", "int4_scale_min", "composites"):
    same = ga[key] == gb[key]
    print(f"  graph {key}: {'equal' if same else 'differs'}: {ga[key]}"
          f"{'' if same else ' vs ' + str(gb[key])}")
  ops = sorted(set(ga["operators"]) | set(gb["operators"]))
  diff = {o: (ga["operators"].get(o, 0), gb["operators"].get(o, 0)) for o in ops
          if ga["operators"].get(o, 0) != gb["operators"].get(o, 0)}
  print(f"  graph operators: {sum(ga['operators'].values())} vs "
        f"{sum(gb['operators'].values())}; differing counts (this, other): {diff or 'none'}")
  if args.json:
    Path(args.json).write_text(json.dumps({
        "model": str(args.model), "other": str(other), "sections": rows,
        "llm_metadata_equal_but_template": la == lb, "template_equal": a_tpl == b_tpl,
        "executor_metadata_equal": a_exec == b_exec, "graph": ga,
        "other_graph": gb}, indent=2) + "\n")
  return 0


def main():
  ap = argparse.ArgumentParser(
      description="Quality gate / metadata / template / section checks for LFM3-1.7B")
  ap.add_argument("model", help="path to the .litertlm")
  ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
  ap.add_argument("--min-correct", type=int, default=7)
  ap.add_argument("--timeout", type=int, default=900,
                  help="seconds per prompt (engine init is paid every run)")
  ap.add_argument("--litert-lm", default="litert-lm",
                  help="path to the litert-lm CLI (pip install litert-lm)")
  ap.add_argument("--source", help="local copy of the checkpoint (default: the "
                                   "Hub at the pinned revision)")
  ap.add_argument("--ctx", type=int, default=2048,
                  help="--check-metadata: the context length when the file "
                       "name does not say it")
  ap.add_argument("--int4", action="store_true",
                  help="--check-metadata: expect the mixed int4 recipe when "
                       "the file name does not say it")
  ap.add_argument("--json", help="write a JSON report here")
  ap.add_argument("--check-metadata", action="store_true")
  ap.add_argument("--render", action="store_true")
  ap.add_argument("--sections", metavar="OTHER_BUNDLE")
  args = ap.parse_args()

  if not Path(args.model).exists():
    print(f"ERROR: model not found: {args.model}", file=sys.stderr)
    return 2
  if args.check_metadata:
    return check_metadata(args)
  if args.render:
    return render(args)
  if args.sections:
    return sections_compare(args)
  return gate(args)


if __name__ == "__main__":
  sys.exit(main())
