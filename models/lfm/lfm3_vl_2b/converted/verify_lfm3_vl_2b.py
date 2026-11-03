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

"""Quality gate, metadata, template and section checks for an LFM3-VL-2B bundle.

Four modes:

  gate (default) — 8 questions about 4 pictures and 8 text questions through
    the `litert-lm` CLI (pip install litert-lm), greedy, thinking off, one
    fresh process per question, scored for correctness and degeneracy
    (looping, token spam, empty output). The pictures are the 512 x 512
    fixtures of litert-community/LFM3-VL-2B-LiteRT (two NASA photographs in
    the public domain, a drawn chart and a drawn sign); `--photos-json` takes
    other rows.

      python verify_lfm3_vl_2b.py model.litertlm [--backend cpu|gpu]

  --check-metadata — reads the bundle (no unpacking) and asserts the shape
    this recipe produces: <|startoftext|> as start_token, <|im_end|> (124900)
    as the only stop, no sampler, a 2,048-token context, model type fast_vlm
    with a 512 x 512 image, a thought channel, the embedded Jinja equal to the
    checkpoint's chat_template.jinja after the exporter's rewrites and the two
    extensions' edits, the tokenizer section equal to the checkpoint's
    tokenizer.json, an ExecutorMetadata section with a convolution state for
    each of the 13 conv layers and global K / V caches for the 11 attention
    layers, the four graphs (token embedder with an int8 table; text decoder
    with prefill 1024 / 256 / 64 / 16 / 4 / 1 and decode, int4 weights with
    every block scale at or above 2^-14 and an int8 output projection; vision
    encoder [1, 512, 512, 3] -> [1, 1024, 1024] with float32 weights;
    projector [1, 1024, 1024] -> [1, 256, 2048] with float16 weights), and
    every section under 2 GiB. Needs `pip install litert-lm-builder`.

      python verify_lfm3_vl_2b.py model.litertlm --check-metadata

  --render — the runtime's own rendering of the bundle's template
    (`Conversation.render_message_to_string`, litert-lm Python package)
    against transformers' `apply_chat_template` on the checkpoint's template:
    one turn with a picture and three turns with two pictures, thinking off
    and on. Two differences are expected and accounted for: the start token
    (the runtime renders `bos_token` empty and adds the metadata start_token
    itself) and `<|image|>`, which the bundle renders as `<image_soft_token>`
    (the runtime splices the projector's 256 embeddings there; the checkpoint's
    processor expands `<|image|>` to the same count).

      python verify_lfm3_vl_2b.py model.litertlm --render

  --sections OTHER.litertlm — compares the two files section by section:
    bytes and sha256 per section, the LlmMetadata fields, the template, and a
    summary of each TFLite graph (signatures, operators, weight types).

      python verify_lfm3_vl_2b.py model.litertlm --sections published/model.litertlm
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

MODEL_ID = "LiquidAI/LFM3-VL-2B"
REVISION = "f83d2b0a2296aaa277f394990a9b8b949ddf9691"
FIXTURE_REPO = "litert-community/LFM3-VL-2B-LiteRT"
FIXTURE_REVISION = "580859278a59710b3c93b298982f3f9e884d80e8"
HERE = Path(__file__).resolve().parent
DECODER_EXTENSION_DIR = HERE.parent.parent / "lfm3_1_7b" / "converted"
CPU_THREADS = 4  # litert-lm --cpu-thread-count for every CPU run
START_TOKEN = "<|startoftext|>"
BOS_ID = 124894  # <|startoftext|>
EOS_ID = 124900  # <|im_end|>
PAD_TOKEN = "<|pad|>"
IMAGE_SIZE = 512
CONTEXT = 2048
CONV_LAYERS, ATTENTION_LAYERS = 13, 11
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
# (id, picture, question, expected answer) on the published fixtures.
PHOTOS = [
    ("jungle_count", "nasa_jungle.png",
     "How many people are in this image? Answer with a number.", "3"),
    ("jungle_color", "nasa_jungle.png",
     "What color are the people's clothes? Answer with one word.", "orange"),
    ("truck_sign", "nasa_truck.png",
     "What words are written on the yellow sign? Answer with the words only.",
     "oversize load"),
    ("truck_color", "nasa_truck.png",
     "What color is the truck? Answer with one word.", "red"),
    ("bar_max", "syn_bar.png",
     "Which day has the tallest bar in this chart? Answer with the day label "
     "only.", "wed"),
    ("bar_count", "syn_bar.png",
     "How many bars are in this chart? Answer with a number.", "4"),
    ("exit_text", "syn_exit.png",
     "What text is written on the sign? Answer with the text only.", "exit 12"),
    ("exit_color", "syn_exit.png",
     "What color is the sign? Answer with one word.", "green"),
]
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                "nineteen", "twenty"]


def source_file(args, name):
  """A file of the checkpoint: from --source, else the Hub at REVISION."""
  if args.source:
    return Path(args.source) / name
  from huggingface_hub import hf_hub_download

  return Path(hf_hub_download(MODEL_ID, name, revision=REVISION))


def fixture(args, name):
  """A picture: from --photos-dir, else the published fixtures."""
  if args.photos_dir:
    return Path(args.photos_dir) / name
  from huggingface_hub import hf_hub_download

  return Path(hf_hub_download(FIXTURE_REPO, f"fixtures/{name}",
                              revision=FIXTURE_REVISION))


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


def photo_correct(answer, expect):
  """The expected answer as a whole word, number words read as digits."""
  low = answer.lower()
  for i, word in enumerate(NUMBER_WORDS):
    low = re.sub(rf"\b{word}\b", str(i), low)
  return bool(re.search(rf"\b{re.escape(expect.lower())}\b", low))


def run_one(args, prompt, picture=None):
  """Runs one prompt through `litert-lm run`, greedy, thinking off."""
  cmd = [args.litert_lm, "run", str(args.model), "--backend", args.backend,
         "--vision-backend", args.backend, "--temperature", "0",
         "--top-k", "1", "--cache", "no", "--thinking", "false"]
  if args.backend == "cpu":
    # A near-tie in greedy decoding can flip with the CPU thread count, so
    # every CPU run uses the same count.
    cmd += ["--cpu-thread-count", str(CPU_THREADS)]
  if picture is not None:
    cmd += ["--attachment", str(picture)]
  cmd += ["--prompt", prompt]
  # `litert-lm run` appends a non-TTY stdin to --prompt and waits for its EOF;
  # an inherited pipe or socket would hang every run, so give it none.
  proc = subprocess.run(cmd, capture_output=True, text=True,
                        timeout=args.timeout, stdin=subprocess.DEVNULL)
  if proc.returncode != 0:
    raise RuntimeError(
        f"litert-lm run failed (exit={proc.returncode}).\n"
        f"--- stderr tail ---\n{proc.stderr[-1500:]}")
  return proc.stdout.strip()


def photo_rows(args):
  if not args.photos_json:
    return [{"id": i, "image": p, "question": q, "expect": e}
            for i, p, q, e in PHOTOS]
  rows = json.loads(Path(args.photos_json).read_text())
  if isinstance(rows, dict):
    rows = rows["rows"]
  # Rows without an expected answer (boxes, descriptions) are not scored.
  return [{**r, "image": r["image"] if Path(r["image"]).suffix
           else r["image"] + ".png"} for r in rows if r.get("expect")]


def gate(args):
  results = {"text": [], "photos": []}
  try:
    for label, q, pat, veto in QUESTIONS:
      raw = run_one(args, q + SUFFIX)
      ans = answer_part(raw)
      low = ans.lower()
      ok = bool(re.search(pat, low)) and not (veto and re.search(veto, low))
      results["text"].append({"label": label, "question": q, "ok": ok,
                              "degenerate": (not ans.strip()) or degenerate(ans),
                              "answer": raw})
    for row in photo_rows(args):
      raw = run_one(args, row["question"], fixture(args, row["image"]))
      ans = answer_part(raw)
      results["photos"].append({**row, "ok": photo_correct(ans, row["expect"]),
                                "degenerate": (not ans.strip()) or degenerate(ans),
                                "answer": raw})
  except Exception as e:  # noqa: BLE001
    print(f"FAIL (harness): {e}", file=sys.stderr)
    return 2
  for part in ("photos", "text"):
    for r in results[part]:
      shown = " ".join(r["answer"].split())[:80] if r["answer"].strip() else "(empty)"
      name = r.get("id") or r.get("label")
      print(f"  [{'v' if r['ok'] else '.'}]{' DEGEN' if r['degenerate'] else '      '} "
            f"{part:6s} {name:16s} -> {shown!r}")
  photos = sum(r["ok"] for r in results["photos"])
  text = sum(r["ok"] for r in results["text"])
  any_degen = any(r["degenerate"] for p in results.values() for r in p)
  n_photos = len(results["photos"])
  passed = (photos >= min(args.min_photos, n_photos)
            and text >= args.min_correct and not any_degen)
  print(f"\n  photos: {photos}/{n_photos}   text: {text}/8   degenerate: "
        f"{'YES' if any_degen else 'no'}")
  print(f"  VERDICT: {'PASS' if passed else 'FAIL'} (thresholds photos "
        f"{args.min_photos}, text {args.min_correct}/8, non-degenerate)")
  if args.json:
    Path(args.json).write_text(json.dumps({
        "model": str(args.model), "backend": args.backend, "photos": photos,
        "photos_of": n_photos, "text": text, "text_of": 8,
        "degenerate": any_degen, "passed": passed, "results": results,
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
    act = re.search(r"prefer_activation_type, Value \(String\): (\S+)", sm.group(2))
    sections.append({"index": int(sm.group(1)), "type": sm.group(5),
                     # peek prints the enum name: tf_lite_prefill_decode.
                     "model_type": (mt.group(1).removeprefix("tf_lite_")
                                    if mt else None),
                     "activation": act.group(1) if act else None,
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


def extensions():
  for path in (HERE, DECODER_EXTENSION_DIR):
    if str(path) not in sys.path:
      sys.path.insert(0, str(path))
  import lfm3_extension  # pylint: disable=g-import-not-at-top
  import lfm3_vl_extension  # pylint: disable=g-import-not-at-top

  return lfm3_extension, lfm3_vl_extension


def expected_template(args):
  decoder, vision = extensions()
  vendor = source_file(args, "chat_template.jinja").read_text()
  return decoder.history_stable_template(
      vision.image_soft_token_template(exporter_rewrite(vendor)))


def section_bytes(model, section):
  with open(model, "rb") as f:
    f.seek(section["begin"])
    return f.read(section["end"] - section["begin"])


def tflite_summary(model, section, contents=False):
  """Signatures, operators and weight types of a TFLite section (lazy read).

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
  types = {getattr(fb.TensorType, k): k
           for k in dir(fb.TensorType) if not k.startswith("_")}
  codes = []
  for i in range(m.OperatorCodesLength()):
    oc = m.OperatorCodes(i)
    codes.append(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()))
  ops, composites = Counter(), Counter()
  int4_buffers, scale_min = set(), None
  # Stored types of the weights (input 1) of FULLY_CONNECTED, CONV_2D,
  # DEPTHWISE_CONV_2D and EMBEDDING_LOOKUP, each stored tensor counted once.
  weight_types = Counter()
  fc_weights = []  # (output tensor name, stored weight type) of every FC.
  seen_weights = set()
  for s in range(m.SubgraphsLength()):
    sg = m.Subgraphs(s)
    producer = {}
    for o in range(sg.OperatorsLength()):
      op = sg.Operators(o)
      for j in range(op.OutputsLength()):
        producer[op.Outputs(j)] = op
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
      if code in (fb.BuiltinOperator.FULLY_CONNECTED, fb.BuiltinOperator.CONV_2D,
                  fb.BuiltinOperator.DEPTHWISE_CONV_2D,
                  fb.BuiltinOperator.EMBEDDING_LOOKUP) and op.InputsLength() > 1:
        idx = op.Inputs(1)
        # A float16 weight reaches the op through a DEQUANTIZE (float16 ->
        # float32): count the stored tensor, the DEQUANTIZE's input.
        src = producer.get(idx)
        if (src is not None and codes[src.OpcodeIndex()]
            == fb.BuiltinOperator.DEQUANTIZE):
          idx = src.Inputs(0)
        w = sg.Tensors(idx)
        key = (s, idx)
        if key not in seen_weights:
          seen_weights.add(key)
          weight_types[types.get(w.Type(), str(w.Type()))] += 1
        if code == fb.BuiltinOperator.FULLY_CONNECTED:
          out = sg.Tensors(op.Outputs(0)).Name().decode(errors="replace")
          fc_weights.append((out, types.get(w.Type(), str(w.Type()))))
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
  signatures, io_shapes = [], {}
  for i in range(m.SignatureDefsLength()):
    sd = m.SignatureDefs(i)
    key = sd.SignatureKey().decode()
    signatures.append(key)
    sg = m.Subgraphs(sd.SubgraphIndex())
    shape = lambda tm: [int(x) for x in sg.Tensors(tm.TensorIndex()).ShapeAsNumpy()]
    io_shapes[key] = {
        "inputs": {sd.Inputs(j).Name().decode(): shape(sd.Inputs(j))
                   for j in range(sd.InputsLength())},
        "outputs": {sd.Outputs(j).Name().decode(): shape(sd.Outputs(j))
                    for j in range(sd.OutputsLength())}}
  summary = {"signatures": sorted(signatures), "io": io_shapes,
             "subgraphs": m.SubgraphsLength(), "operators": dict(ops),
             "composites": dict(composites), "weight_types": dict(weight_types),
             "int4_tensors": len(int4_buffers), "int4_scale_min": scale_min,
             "fc_weights": fc_weights}
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


def graphs(model, sections, contents=False):
  return {s["model_type"]: tflite_summary(model, s, contents)
          for s in sections if s["type"] == "TFLiteModel"}


def check_metadata(args):
  llm, executor, sections, template = read_template(args.model)
  if not llm:
    print("FAIL: no LlmMetadata section found")
    return 1
  checks = []

  def check(name, ok, detail=""):
    checks.append(ok)
    print(f"  [{'v' if ok else 'X'}] {name}{(': ' + detail) if detail else ''}")

  # The stock packager writes the token as a string; an id is the same token.
  start = re.search(r'start_token \{\s*(?:token_str: "([^"]*)"|'
                    r'token_ids \{\s*ids: (\d+)\s*\})', llm)
  start_ok = bool(start) and (start.group(1) == START_TOKEN
                              or start.group(2) == str(BOS_ID))
  check(f"start_token {START_TOKEN} (the runtime renders bos_token empty)",
        start_ok, (f"token_str {start.group(1)}" if start and start.group(1)
                   else f"id {start.group(2)}" if start else "missing"))
  stops = re.findall(r"stop_tokens \{(.*?)\n    \}", llm, re.S)
  stop_ids = [int(x) for s in stops for x in re.findall(r"ids:\s*(\d+)", s)]
  str_stops = [s for s in stops if "token_str" in s]
  check(f"the only stop is <|im_end|> ({EOS_ID}) as an id",
        stop_ids == [EOS_ID] and not str_stops,
        f"ids {stop_ids}, {len(str_stops)} string stops")
  pad = re.search(r'pad_token \{\s*token_str: "([^"]*)"', llm)
  check(f"pad_token {PAD_TOKEN}", bool(pad) and pad.group(1) == PAD_TOKEN)
  check("no sampler (the runtime's default sampling)", "sampler_params" not in llm)
  ctx = re.search(r"max_num_tokens:\s*(\d+)", llm)
  check(f"max_num_tokens {CONTEXT}", bool(ctx) and int(ctx.group(1)) == CONTEXT,
        ctx.group(1) if ctx else "missing")
  vlm = re.search(r"fast_vlm \{\s*image_tensor_height:\s*(\d+)\s*"
                  r"image_tensor_width:\s*(\d+)", llm)
  check(f"model type fast_vlm {IMAGE_SIZE} x {IMAGE_SIZE}",
        bool(vlm) and vlm.groups() == (str(IMAGE_SIZE), str(IMAGE_SIZE)),
        "x".join(vlm.groups()) if vlm else "missing")
  check("thought channel <think> ... </think>",
        bool(re.search(r'channels \{\s*channel_name: "thought"\s*start: "<think>"'
                       r'\s*end: "</think>"', llm)))
  try:
    want = expected_template(args)
    check("embedded Jinja = the checkpoint's template after the exporter's "
          "rewrites, image_soft_token_template and history_stable_template",
          template == want, f"{len(template.encode())} B, sha256 "
          f"{hashlib.sha256(template.encode()).hexdigest()[:12]}… vs "
          f"{len(want.encode())} B")
  except Exception as e:  # noqa: BLE001
    check("embedded Jinja (checkpoint template unreachable)", False,
          f"{type(e).__name__}: {e}")

  by_type = Counter(s["type"] for s in sections)
  models = sorted(s["model_type"] for s in sections if s["type"] == "TFLiteModel")
  check("sections: LlmMetadata, ExecutorMetadata, tokenizer, four TFLite models "
        "(embedder, prefill_decode, vision_adapter, vision_encoder)",
        by_type == Counter({"LlmMetadataProto": 1, "ExecutorMetadataProto": 1,
                            "HF_Tokenizer_Zlib": 1, "TFLiteModel": 4})
        and models == ["embedder", "prefill_decode", "vision_adapter",
                       "vision_encoder"], f"{dict(by_type)}, {models}")
  big = [s for s in sections if s["end"] - s["begin"] >= 2 * GIB]
  check("every section < 2 GiB", not big,
        ", ".join(f"{s['model_type'] or s['type']}:"
                  f"{(s['end'] - s['begin']) / GIB:.2f} GiB" for s in sections))
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
        f"{ATTENTION_LAYERS} global K/V caches at {CONTEXT}",
        kinds == Counter({"TYPE_LINEAR_ATTENTION": CONV_LAYERS,
                          "TYPE_GLOBAL_KEY_CACHE": ATTENTION_LAYERS,
                          "TYPE_GLOBAL_VALUE_CACHE": ATTENTION_LAYERS})
        and lengths == [CONTEXT], f"{dict(kinds)}, lengths {lengths}")

  g = graphs(args.model, sections)
  dec = g.get("prefill_decode", {})
  check("decoder signatures prefill 1024 / 256 / 64 / 16 / 4 / 1 and decode",
        set(dec.get("signatures", [])) == SIGNATURES,
        ", ".join(dec.get("signatures", [])))
  fc = dec.get("fc_weights", [])
  int4_fc = sum(t == "INT4" for _, t in fc)
  head = [t for name, t in fc if "logits" in name]
  check("decoder: every fully connected weight int4 but the output "
        "projection, which is int8",
        bool(fc) and int4_fc == len(fc) - len(head) and head
        and set(head) == {"INT8"},
        f"{int4_fc} int4 of {len(fc)} FC ops, output projection {head}")
  low = dec.get("int4_scale_min")
  check("every int4 block scale >= 2^-14 (XNNPACK takes only normal fp16 scales)",
        low is not None and low >= INT4_SCALE_FLOOR, f"min {low!r}")
  gpu = [n for n in dec.get("composites", {})
         if re.search(r"sdpa|attention|cache|short_conv|rope", n)]
  check("decoder: no attention, cache, short-conv or rope composite",
        not gpu, f"composites {dec.get('composites') or 'none'}")
  emb = g.get("embedder", {})
  emb_io = next(iter(emb.get("io", {}).values()), {})
  check("embedder: token_ids [1, 1] -> embeddings [1, 1, 2048], int8 table",
        emb_io == {"inputs": {"token_ids": [1, 1]},
                   "outputs": {"embeddings": [1, 1, 2048]}}
        and emb.get("weight_types") == {"INT8": 1},
        f"{emb_io}, {emb.get('weight_types')}")
  enc = g.get("vision_encoder", {})
  enc_io = list(enc.get("io", {}).values())
  check(f"vision encoder: one signature, [1, {IMAGE_SIZE}, {IMAGE_SIZE}, 3] -> "
        "[1, 1024, 1024], float32 weights",
        len(enc_io) == 1
        and list(enc_io[0]["inputs"].values()) == [[1, IMAGE_SIZE, IMAGE_SIZE, 3]]
        and list(enc_io[0]["outputs"].values()) == [[1, 1024, 1024]]
        and set(enc.get("weight_types", {})) == {"FLOAT32"},
        f"{enc_io}, {enc.get('weight_types')}")
  adp = g.get("vision_adapter", {})
  adp_io = list(adp.get("io", {}).values())
  check("projector: one signature, [1, 1024, 1024] -> [1, 256, 2048], float16 "
        "fully connected weights",
        len(adp_io) == 1
        and list(adp_io[0]["inputs"].values()) == [[1, 1024, 1024]]
        and list(adp_io[0]["outputs"].values()) == [[1, 256, 2048]]
        and adp.get("weight_types") == {"FLOAT16": 2},
        f"{adp_io}, {adp.get('weight_types')}")
  ok = all(checks)
  print(f"\n  {'PASS' if ok else 'FAIL'}: {sum(checks)}/{len(checks)} metadata checks")
  if args.json:
    Path(args.json).write_text(json.dumps({
        "model": str(args.model), "passed": ok, "checks": len(checks),
        "sections": sections,
        "graphs": {k: {kk: vv for kk, vv in v.items() if kk != "fc_weights"}
                   for k, v in g.items()}}, indent=2) + "\n")
  return 0 if ok else 1


def scenarios(picture):
  """(name, runtime messages, transformers messages, thinking)."""
  rimg = {"type": "image", "path": str(picture)}
  himg = {"type": "image"}
  rt = lambda t: {"type": "text", "text": t}
  ru = lambda *c: {"role": "user", "content": list(c)}
  ra = lambda c, r: {"role": "assistant", "content": [{"type": "text", "text": c}],
                     "channels": {"thought": r}, "reasoning_content": r}
  ha = lambda c, r: {"role": "assistant", "content": c, "reasoning_content": r}
  q1, q2, q3 = ("What is written on the sign?", "And its color?",
                "Does this picture show the same sign?")
  a1, r1 = "EXIT 12.", "The sign shows text."
  a2, r2 = "Green.", "The sign is green."
  out = []
  for think in (False, True):
    tag = "on" if think else "off"
    out.append((f"1turn_{tag}", [ru(rimg, rt(q1))], [ru(himg, rt(q1))], think))
    out.append((f"3turn_{tag}",
                [ru(rimg, rt(q1)), ra(a1, r1), ru(rt(q2)), ra(a2, r2),
                 ru(rimg, rt(q3))],
                [ru(himg, rt(q1)), ha(a1, r1), ru(rt(q2)), ha(a2, r2),
                 ru(himg, rt(q3))], think))
  return out


def render(args):
  import litert_lm  # pylint: disable=g-import-not-at-top
  import transformers  # pylint: disable=g-import-not-at-top

  tok_dir = source_file(args, "tokenizer.json").parent
  for name in ("tokenizer_config.json", "chat_template.jinja"):
    source_file(args, name)
  tok = transformers.AutoTokenizer.from_pretrained(str(tok_dir),
                                                   trust_remote_code=True)
  picture = fixture(args, PHOTOS[-1][1])
  rows, ok = [], True
  with litert_lm.Engine(str(args.model), backend=litert_lm.Backend.CPU(),
                        vision_backend=litert_lm.Backend.CPU(),
                        cache_dir=":nocache") as engine:
    for name, rt_msgs, hf_msgs, think in scenarios(picture):
      conv = engine.create_conversation(
          messages=rt_msgs[:-1] or None,
          thinking_config=litert_lm.ThinkingConfig(enable_thinking=think))
      runtime = conv.render_message_to_string(rt_msgs[-1])
      conv.close()
      want = tok.apply_chat_template(hf_msgs, tokenize=False,
                                     add_generation_prompt=True,
                                     enable_thinking=think)
      accounted = want
      notes = []
      if accounted.startswith(START_TOKEN) and not runtime.startswith(START_TOKEN):
        accounted = accounted[len(START_TOKEN):]
        notes.append("start token from the metadata")
      if "<|image|>" in accounted:
        accounted = accounted.replace("<|image|>", "<image_soft_token>")
        notes.append("<|image|> as <image_soft_token>")
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
  key = lambda s: (s["type"], s["model_type"] or "")
  b_by = {key(s): s for s in b_secs}
  rows = []
  for sa in a_secs:
    sb = b_by.get(key(sa))
    da = section_bytes(args.model, sa)
    db = section_bytes(other, sb) if sb else b""
    ha, hb = hashlib.sha256(da).hexdigest(), hashlib.sha256(db).hexdigest()
    rows.append({"type": sa["type"], "model_type": sa["model_type"],
                 "activation": sa["activation"],
                 "other_activation": sb["activation"] if sb else None,
                 "bytes": len(da), "other_bytes": len(db), "sha256": ha,
                 "other_sha256": hb, "equal": ha == hb})
    print(f"  {(sa['model_type'] or sa['type']):22s} {len(da):>14,} B  {ha[:12]}…  "
          f"{'==' if ha == hb else '!='}  {hb[:12]}…  {len(db):>14,} B  "
          f"activation {sa['activation']} / {sb['activation'] if sb else '-'}")
  missing = sorted(set(b_by) - {key(s) for s in a_secs})
  if missing:
    print(f"  only in the other file: {missing}")
  strip = lambda t: re.sub(r"\n\s*jinja_prompt_template:.*?(?=\n\s*\w+[ :{])", "", t, flags=re.S)
  la = [l.strip() for l in strip(a_llm).splitlines() if l.strip()]
  lb = [l.strip() for l in strip(b_llm).splitlines() if l.strip()]
  import difflib  # pylint: disable=g-import-not-at-top
  if la == lb:
    print("  LlmMetadata fields other than the template: equal")
  else:
    print("  LlmMetadata fields other than the template differ:")
    for line in difflib.unified_diff(lb, la, "other", "this", lineterm="", n=0):
      print("      " + line)
  if a_tpl == b_tpl:
    print("  template: byte-equal")
  else:
    print(f"  template: {len(a_tpl.encode())} B vs {len(b_tpl.encode())} B; the lines that differ:")
    for line in difflib.unified_diff(b_tpl.splitlines(), a_tpl.splitlines(), "other",
                                     "this", lineterm="", n=0):
      print("      " + repr(line))
  print(f"  ExecutorMetadata text: {'equal' if a_exec == b_exec else 'differs'}")
  ga, gb = graphs(args.model, a_secs, True), graphs(other, b_secs, True)
  report = {}
  for kind in sorted(set(ga) | set(gb)):
    x, y = ga.get(kind), gb.get(kind)
    if not x or not y:
      print(f"  graph {kind}: only in {'this' if x else 'the other'} file")
      continue
    hx, hy = x.pop("buffer_hashes"), y.pop("buffer_hashes")
    nx, ny = x.pop("tensor_names"), y.pop("tensor_names")
    x.pop("fc_weights"), y.pop("fc_weights")
    common = sum((hx & hy).values())
    only_x = sum(n * c for (n, _), c in (hx - hy).items())
    only_y = sum(n * c for (n, _), c in (hy - hx).items())
    ops = sorted(set(x["operators"]) | set(y["operators"]))
    diff = {o: (x["operators"].get(o, 0), y["operators"].get(o, 0)) for o in ops
            if x["operators"].get(o, 0) != y["operators"].get(o, 0)}
    print(f"  graph {kind}: buffers with data {sum(hx.values())} vs {sum(hy.values())}, "
          f"{common} with the same bytes, the rest {only_x:,} B vs {only_y:,} B; "
          f"tensor names {len(nx)} vs {len(ny)}, {len(nx & ny)} in both; "
          f"operators {sum(x['operators'].values())} vs {sum(y['operators'].values())}, "
          f"differing counts {diff or 'none'}")
    for field in ("signatures", "io", "subgraphs", "weight_types", "int4_tensors",
                  "int4_scale_min", "composites"):
      if x[field] != y[field]:
        print(f"      {field}: {x[field]} vs {y[field]}")
    report[kind] = {"this": x, "other": y, "buffers_same": common,
                    "buffers_this": sum(hx.values()), "buffers_other": sum(hy.values()),
                    "bytes_only_this": only_x, "bytes_only_other": only_y,
                    "names_both": len(nx & ny), "names_this": len(nx),
                    "names_other": len(ny), "operator_count_diff": diff}
  if args.json:
    Path(args.json).write_text(json.dumps({
        "model": str(args.model), "other": str(other), "sections": rows,
        "llm_metadata_equal_but_template": la == lb,
        "llm_metadata_diff": list(difflib.unified_diff(lb, la, n=0)),
        "template_equal": a_tpl == b_tpl,
        "template_diff": list(difflib.unified_diff(b_tpl.splitlines(), a_tpl.splitlines(), n=0)),
        "executor_metadata_equal": a_exec == b_exec, "graphs": report},
        indent=2, default=str) + "\n")
  return 0


def main():
  ap = argparse.ArgumentParser(
      description="Quality gate / metadata / template / section checks for LFM3-VL-2B")
  ap.add_argument("model", help="path to the .litertlm")
  ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu",
                  help="the text decoder's and the vision encoder's backend")
  ap.add_argument("--min-correct", type=int, default=7)
  ap.add_argument("--min-photos", type=int, default=7)
  ap.add_argument("--photos-json", help="other picture rows: a JSON list of "
                  '{"id", "image", "question", "expect"} (or {"rows": [...]}), '
                  "image = a file name in --photos-dir (.png when it has no "
                  "suffix); rows with no expect are skipped")
  ap.add_argument("--photos-dir", help="folder of the pictures (default: the "
                  "published fixtures)")
  ap.add_argument("--timeout", type=int, default=900,
                  help="seconds per prompt (engine init is paid every run)")
  ap.add_argument("--litert-lm", default="litert-lm",
                  help="path to the litert-lm CLI (pip install litert-lm)")
  ap.add_argument("--source", help="local copy of the checkpoint (default: the "
                                   "Hub at the pinned revision)")
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
