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

"""Quality gate, metadata, template and section checks for an LFM3-Audio-2B
bundle (the text path: speech or text in, text out).

Four modes:

  gate (default) — through the LiteRT-LM Python API (pip install litert-lm),
    greedy, one engine per question, scored for correctness and degeneracy
    (looping, token spam, empty output): 8 text questions always; with
    `--asr-json`, transcription of English clips (system prompt `Perform ASR
    in English.`, word error rate against the reference with the FLEURS
    rule); with `--voicemail-json`, recorded messages with the instruction
    `Summarize this voicemail in one or two short sentences: who called, what
    they want, and the number to call back.`, right when the answer names the
    caller's first and last name and has the callback number's digits in a
    row.

      python verify_lfm3_audio_2b.py model.litertlm [--backend cpu|gpu] \
        [--asr-json clips.json] [--voicemail-json voicemails.json]

  --check-metadata — reads the bundle (no unpacking) and asserts the shape
    this recipe produces: <|startoftext|> as start_token, <|im_end|> (124900)
    and <|text_end|> (125014) as the only stops, no sampler, a 2,048-token
    context, model type generic_model with the audio fields (<|audio|>
    delimiter, raw 16 kHz mono PCM in 160-sample frames), a thought channel,
    the embedded Jinja equal to the checkpoint's chat_template.jinja after
    the exporter's rewrites and the decoder extension's edit, the tokenizer
    section equal to the checkpoint's tokenizer.json, an ExecutorMetadata
    section with a convolution state for each of the 13 conv layers and K / V
    caches for the 11 attention layers, the three graphs (token embedder with
    an int8 table; text decoder with prefill 1024 / 256 / 64 / 16 / 4 / 1 and
    decode, a `hidden` output beside `logits`, int4 weights with every block
    scale at or above 2^-14 and an int8 output projection; audio section
    `audio [1, 3000, 160]` -> `features [1, 375, 2048]` + `mask [1, 375]`
    with float16 fully connected weights), and every section under 2 GiB.
    Needs `pip install litert-lm-builder`.

      python verify_lfm3_audio_2b.py model.litertlm --check-metadata

  --render — the runtime's own rendering of the bundle's template
    (`Conversation.render_message_to_string`) against transformers'
    `apply_chat_template` on the checkpoint's template: a transcription
    request, a question about a clip, and three text turns with a thought.
    One difference is expected and accounted for: the start token (the
    runtime renders `bos_token` empty and adds the metadata start_token).

      python verify_lfm3_audio_2b.py model.litertlm --render --wav some.wav

  --sections OTHER.litertlm — compares the two files section by section:
    bytes and sha256 per section, the LlmMetadata fields, the template, and a
    summary of each TFLite graph (signatures, operators, weight types).

      python verify_lfm3_audio_2b.py model.litertlm --sections published/model.litertlm
"""

import argparse
import hashlib
import io
import json
import mmap
import os
import re
import sys
import tempfile
import time
import zlib
from collections import Counter
from pathlib import Path

MODEL_ID = "LiquidAI/LFM3-Audio-2B"
REVISION = "9d49701c372d6011d900f6abb40c6b296fb706dd"
HERE = Path(__file__).resolve().parent
DECODER_EXTENSION_DIR = HERE.parent.parent / "lfm3_1_7b" / "converted"
CPU_THREADS = 4  # Every CPU run uses the same thread count.
START_TOKEN = "<|startoftext|>"
BOS_ID = 124894  # <|startoftext|>
STOP_IDS = [124900, 125014]  # <|im_end|>, <|text_end|>
PAD_TOKEN = "<|pad|>"
CONTEXT = 2048
CONV_LAYERS, ATTENTION_LAYERS = 13, 11
INT4_SCALE_FLOOR = 2.0 ** -14
SIGNATURES = {"prefill_1024", "prefill_256", "prefill_64", "prefill_16",
              "prefill_4", "prefill_1", "decode"}
GIB = 1024 ** 3
ASR_SYSTEM = "Perform ASR in English."
AUDIO_FIELDS = {"audio_enabled": "true", "delimiter_regex": r'"(<\\|audio\\|>)"',
                "audio_token_regex": r'"<\\|audio\\|>"',
                "skip_mel_spectrogram_extraction": "true",
                "audio_sample_rate_hz": "16000", "audio_num_channels": "1",
                "audio_frame_length": "160", "audio_hop_length": "160",
                "audio_input_scale": "1.0", "add_audio_end": "false"}

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
# The instruction of the LFM3-Audio voicemail sample's English answers.
VOICEMAIL_INSTRUCTION = ("Summarize this voicemail in one or two short "
                         "sentences: who called, what they want, and the "
                         "number to call back.")
DIGIT_WORDS = {"zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2",
               "three": "3", "four": "4", "five": "5", "six": "6",
               "seven": "7", "eight": "8", "nine": "9"}


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
  import lfm3_audio_extension  # pylint: disable=g-import-not-at-top
  import lfm3_extension  # pylint: disable=g-import-not-at-top

  return lfm3_extension, lfm3_audio_extension


def expected_template(args):
  decoder, audio = extensions()
  vendor = source_file(args, "chat_template.jinja").read_text()
  return audio.check_audio_template(
      decoder.history_stable_template(exporter_rewrite(vendor)))


def norm_words(text):
  """The FLEURS scoring rule: upper case, '-' as a space, anything but
  A-Z, 0-9 and the apostrophe as a space, split on spaces."""
  text = text.upper().replace("-", " ")
  return re.sub(r"[^A-Z0-9' ]+", " ", text).split()


def word_errors(ref, hyp):
  """Levenshtein distance over words."""
  prev = list(range(len(hyp) + 1))
  for i, r in enumerate(ref, 1):
    cur = [i] + [0] * len(hyp)
    for j, h in enumerate(hyp, 1):
      cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
    prev = cur
  return prev[-1]


def words(text):
  return re.findall(r"[a-z0-9]+", text.lower())


def digit_stream(text):
  """The digits of `text`, digit words read as digits ("five five five",
  "double two")."""
  out, w, i = [], words(text), 0
  while i < len(w):
    t = w[i]
    if t.isdigit():
      out.append(t)
    elif t in DIGIT_WORDS:
      out.append(DIGIT_WORDS[t])
    elif t in ("double", "triple") and i + 1 < len(w) and w[i + 1] in DIGIT_WORDS:
      out.append(DIGIT_WORDS[w[i + 1]] * (2 if t == "double" else 3))
      i += 1
    i += 1
  return "".join(out)


def voicemail_correct(answer, caller, callback):
  """Both names of the caller as words, and the number's digits in a row."""
  have = set(words(answer))
  number = re.sub(r"\D", "", callback)
  return (all(n in have for n in words(caller))
          and number in digit_stream(answer))


def ask(args, litert_lm, contents, system=None, max_tokens=512):
  """One question on a fresh engine (a second conversation on one engine
  can start from the first one's state)."""
  backend = (litert_lm.Backend.CPU(thread_count=CPU_THREADS)
             if args.backend == "cpu" else litert_lm.Backend.GPU())
  audio_backend = (litert_lm.Backend.CPU(thread_count=CPU_THREADS)
                   if args.audio_backend == "cpu" else litert_lm.Backend.GPU())
  t0 = time.time()
  with litert_lm.Engine(str(args.model), backend=backend,
                        audio_backend=audio_backend,
                        cache_dir=":nocache") as engine:
    conv = engine.create_conversation(
        messages=[{"role": "system", "content": system}] if system else None,
        sampler_config=litert_lm.SamplerConfig(top_k=1, top_p=1.0,
                                               temperature=0.0),
        max_output_tokens=max_tokens)
    reply = conv.send_message({"role": "user", "content": contents})
    conv.close()
  # The thought channel, if the model opened one, stays out of the answer.
  return str(reply.contents), round(time.time() - t0, 2)


def asr_rows(args):
  doc = json.loads(Path(args.asr_json).read_text())
  clips = doc["clips"] if isinstance(doc, dict) else doc
  if args.asr_line_set and any("parts_line" in c for c in clips):
    clips = [c for c in clips if c.get("parts_line")]
  base = Path(args.asr_dir) if args.asr_dir else Path(args.asr_json).parent
  rows = []
  for c in clips:
    ref = c.get("reference") or c["reference_text"]
    if isinstance(ref, dict):  # FLEURS rows: the raw transcription
      ref = ref.get("raw_transcription") or ref.get("reference")
    rows.append({"id": c["name"], "wav": str(base / c["path"]), "reference": ref})
  return rows


def voicemail_rows(args):
  doc = json.loads(Path(args.voicemail_json).read_text())
  items = doc["items"] if isinstance(doc, dict) else doc
  base = Path(args.voicemail_dir) if args.voicemail_dir else Path(args.voicemail_json).parent / "wav"
  return [{"id": it["id"], "wav": str(base / f"{it['id']}.wav"),
           "caller": it["caller"], "callback": it["callback"]} for it in items]


def gate(args):
  import litert_lm  # pylint: disable=g-import-not-at-top

  results = {"text": [], "asr": [], "voicemail": []}
  try:
    for label, q, pat, veto in QUESTIONS:
      raw, secs = ask(args, litert_lm, [{"type": "text", "text": q + SUFFIX}])
      ans = answer_part(raw)
      low = ans.lower()
      ok = bool(re.search(pat, low)) and not (veto and re.search(veto, low))
      results["text"].append({"label": label, "question": q, "ok": ok,
                              "degenerate": (not ans.strip()) or degenerate(ans),
                              "answer": raw, "seconds": secs})
      print(f"  text {label:16s} ok={ok} {' '.join(raw.split())[:70]!r}", flush=True)
    if args.asr_json:
      for row in asr_rows(args):
        raw, secs = ask(args, litert_lm, [{"type": "audio", "path": row["wav"]}],
                        system=ASR_SYSTEM, max_tokens=1024)
        ans = answer_part(raw)
        ref, hyp = norm_words(row["reference"]), norm_words(ans)
        errors = word_errors(ref, hyp)
        results["asr"].append({**row, "answer": raw, "words": len(ref),
                               "errors": errors, "seconds": secs,
                               "degenerate": (not ans.strip()) or degenerate(ans)})
        print(f"  asr  {row['id']:24s} {errors}/{len(ref)} {' '.join(raw.split())[:60]!r}", flush=True)
    if args.voicemail_json:
      for row in voicemail_rows(args):
        raw, secs = ask(args, litert_lm, [{"type": "audio", "path": row["wav"]},
                                          {"type": "text", "text": VOICEMAIL_INSTRUCTION}])
        ans = answer_part(raw)
        ok = voicemail_correct(ans, row["caller"], row["callback"])
        results["voicemail"].append({**row, "answer": raw, "ok": ok, "seconds": secs,
                                     "degenerate": (not ans.strip()) or degenerate(ans)})
        print(f"  vm   {row['id']:6s} ok={ok} {' '.join(raw.split())[:70]!r}", flush=True)
  except Exception as e:  # noqa: BLE001
    print(f"FAIL (harness): {type(e).__name__}: {e}", file=sys.stderr)
    return 2
  text = sum(r["ok"] for r in results["text"])
  words = sum(r["words"] for r in results["asr"])
  errors = sum(r["errors"] for r in results["asr"])
  wer = errors / words if words else None
  vm = sum(r["ok"] for r in results["voicemail"])
  any_degen = any(r["degenerate"] for p in results.values() for r in p)
  passed = (text >= args.min_correct and not any_degen
            and (wer is None or wer <= args.max_wer)
            and vm >= min(args.min_voicemail, len(results["voicemail"])))
  print(f"\n  text: {text}/8   asr: {errors}/{words} word errors"
        f"{f' (WER {wer:.4f})' if wer is not None else ''}   voicemail:"
        f" {vm}/{len(results['voicemail'])}   degenerate: {'YES' if any_degen else 'no'}")
  print(f"  VERDICT: {'PASS' if passed else 'FAIL'} (thresholds text"
        f" {args.min_correct}/8, WER <= {args.max_wer}, voicemail"
        f" {args.min_voicemail}, non-degenerate)")
  if args.json:
    Path(args.json).write_text(json.dumps({
        "model": str(args.model), "backend": args.backend,
        "audio_backend": args.audio_backend, "text": text, "text_of": 8,
        "asr_errors": errors, "asr_words": words, "wer": wer,
        "voicemail": vm, "voicemail_of": len(results["voicemail"]),
        "degenerate": any_degen, "passed": passed, "results": results,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {args.json}")
  return 0 if passed else 1


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
  check(f"start_token {START_TOKEN} (the runtime renders bos_token empty)",
        bool(start) and (start.group(1) == START_TOKEN
                         or start.group(2) == str(BOS_ID)),
        (f"token_str {start.group(1)}" if start and start.group(1)
         else f"id {start.group(2)}" if start else "missing"))
  stops = re.findall(r"stop_tokens \{(.*?)\n    \}", llm, re.S)
  stop_ids = [int(x) for s in stops for x in re.findall(r"ids:\s*(\d+)", s)]
  str_stops = [s for s in stops if "token_str" in s]
  check(f"the stops are <|im_end|> and <|text_end|> ({STOP_IDS}) as ids",
        sorted(stop_ids) == STOP_IDS and not str_stops,
        f"ids {stop_ids}, {len(str_stops)} string stops")
  pad = re.search(r'pad_token \{\s*token_str: "([^"]*)"', llm)
  check(f"pad_token {PAD_TOKEN} or none", "pad_token" not in llm
        or (bool(pad) and pad.group(1) == PAD_TOKEN),
        pad.group(1) if pad else "none")
  check("no sampler (the runtime's default sampling)", "sampler_params" not in llm)
  ctx = re.search(r"max_num_tokens:\s*(\d+)", llm)
  check(f"max_num_tokens {CONTEXT}", bool(ctx) and int(ctx.group(1)) == CONTEXT,
        ctx.group(1) if ctx else "missing")
  gm = re.search(r"generic_model \{(.*?)\n\s*\}", llm, re.S)
  fields = dict(re.findall(r"(\w+):\s*(\S+)", gm.group(1))) if gm else {}
  check("model type generic_model with the audio fields (<|audio|>, raw 16 kHz"
        " mono PCM, 160-sample frames, no runtime mel)",
        fields == AUDIO_FIELDS, json.dumps(fields))
  check("thought channel <think> ... </think>",
        bool(re.search(r'channels \{\s*channel_name: "thought"\s*start: "<think>"'
                       r'\s*end: "</think>"', llm)))
  try:
    want = expected_template(args)
    check("embedded Jinja = the checkpoint's template after the exporter's "
          "rewrites and history_stable_template, audio as <|audio|>",
          template == want, f"{len(template.encode())} B, sha256 "
          f"{hashlib.sha256(template.encode()).hexdigest()[:12]}… vs "
          f"{len(want.encode())} B")
  except Exception as e:  # noqa: BLE001
    check("embedded Jinja (checkpoint template unreachable)", False,
          f"{type(e).__name__}: {e}")

  by_type = Counter(s["type"] for s in sections)
  models = sorted(s["model_type"] for s in sections if s["type"] == "TFLiteModel")
  check("sections: LlmMetadata, ExecutorMetadata, tokenizer, three TFLite "
        "models (audio_encoder_hw, embedder, prefill_decode)",
        by_type == Counter({"LlmMetadataProto": 1, "ExecutorMetadataProto": 1,
                            "HF_Tokenizer_Zlib": 1, "TFLiteModel": 3})
        and models == ["audio_encoder_hw", "embedder", "prefill_decode"],
        f"{dict(by_type)}, {models}")
  big = [s for s in sections if s["end"] - s["begin"] >= 2 * GIB]
  check("every section < 2 GiB", not big,
        ", ".join(f"{s['model_type'] or s['type']}:"
                  f"{(s['end'] - s['begin']) / GIB:.2f} GiB" for s in sections))
  tok = next((s for s in sections if s["type"] == "HF_Tokenizer_Zlib"), None)
  if tok:
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
        f"{ATTENTION_LAYERS} K/V caches at {CONTEXT}",
        kinds == Counter({"TYPE_LINEAR_ATTENTION": CONV_LAYERS,
                          "TYPE_GLOBAL_KEY_CACHE": ATTENTION_LAYERS,
                          "TYPE_GLOBAL_VALUE_CACHE": ATTENTION_LAYERS})
        and lengths == [CONTEXT], f"{dict(kinds)}, lengths {lengths}")

  g = graphs(args.model, sections)
  dec = g.get("prefill_decode", {})
  check("decoder signatures prefill 1024 / 256 / 64 / 16 / 4 / 1 and decode",
        set(dec.get("signatures", [])) == SIGNATURES,
        ", ".join(dec.get("signatures", [])))
  io_ok, io_bad = True, []
  for sig, io_ in dec.get("io", {}).items():
    outs = {k: v for k, v in io_["outputs"].items() if not k.startswith("kv_cache")}
    ins = {k: v for k, v in io_["inputs"].items() if not k.startswith("kv_cache")}
    t = 1 if sig == "decode" else int(sig.split("_")[1])
    want_out = {"hidden": [1, t, 2048]}
    if sig == "decode":
      want_out["logits"] = [1, 1, 128000]
    want_in = {"embeddings": [1, t, 2048], "input_pos": [t],
               "mask": [1, 1, t, CONTEXT]}
    if outs != want_out or ins != want_in:
      io_ok = False
      io_bad.append(f"{sig}: {ins} -> {outs}")
  check("decoder: embeddings in; `hidden` out of every signature, `logits` "
        "out of decode", io_ok and bool(dec.get("io")), "; ".join(io_bad))
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
  aud = g.get("audio_encoder_hw", {})
  aud_io = aud.get("io", {})
  fc_types = Counter(t for _, t in aud.get("fc_weights", []))
  check("audio section: one signature `encode`, audio [1, 3000, 160] -> "
        "features [1, 375, 2048] + mask [1, 375], float16 fully connected "
        "weights",
        aud_io == {"encode": {"inputs": {"audio": [1, 3000, 160]},
                              "outputs": {"features": [1, 375, 2048],
                                          "mask": [1, 375]}}}
        and set(fc_types) == {"FLOAT16"},
        f"{aud_io}, FC {dict(fc_types)}, weights {aud.get('weight_types')}")
  ok = all(checks)
  print(f"\n  {'PASS' if ok else 'FAIL'}: {sum(checks)}/{len(checks)} metadata checks")
  if args.json:
    Path(args.json).write_text(json.dumps({
        "model": str(args.model), "passed": ok, "checks": len(checks),
        "sections": sections,
        "graphs": {k: {kk: vv for kk, vv in v.items() if kk != "fc_weights"}
                   for k, v in g.items()}}, indent=2) + "\n")
  return 0 if ok else 1


def scenarios(wav):
  """(name, runtime messages, transformers messages)."""
  raud = {"type": "audio", "path": str(wav)}
  haud = {"type": "audio"}
  rt = lambda t: {"type": "text", "text": t}
  ru = lambda *c: {"role": "user", "content": list(c)}
  sys_ = {"role": "system", "content": ASR_SYSTEM}
  ra = lambda c, r: {"role": "assistant", "content": [{"type": "text", "text": c}],
                     "channels": {"thought": r}, "thinking": r}
  ha = lambda c, r: {"role": "assistant", "content": c, "thinking": r}
  q = "What is the speaker asking for?"
  return [
      ("asr", [sys_, ru(raud)], [sys_, ru(haud)]),
      ("audio_question", [ru(raud, rt(q))], [ru(haud, rt(q))]),
      ("text_3turn",
       [ru(rt("What is 17 + 25?")), ra("42.", "Add them."), ru(rt("And 8 times 7?"))],
       [ru(rt("What is 17 + 25?")), ha("42.", "Add them."), ru(rt("And 8 times 7?"))]),
  ]


def render(args):
  import litert_lm  # pylint: disable=g-import-not-at-top
  import transformers  # pylint: disable=g-import-not-at-top

  tok_dir = source_file(args, "tokenizer.json").parent
  for name in ("tokenizer_config.json", "chat_template.jinja"):
    source_file(args, name)
  tok = transformers.AutoTokenizer.from_pretrained(str(tok_dir),
                                                   trust_remote_code=True)
  rows, ok = [], True
  with litert_lm.Engine(str(args.model), backend=litert_lm.Backend.CPU(),
                        audio_backend=litert_lm.Backend.CPU(),
                        cache_dir=":nocache") as engine:
    for name, rt_msgs, hf_msgs in scenarios(args.wav):
      conv = engine.create_conversation(messages=rt_msgs[:-1] or None)
      runtime = conv.render_message_to_string(rt_msgs[-1])
      conv.close()
      want = tok.apply_chat_template(hf_msgs, tokenize=False,
                                     add_generation_prompt=True)
      accounted = want
      notes = []
      if accounted.startswith(START_TOKEN) and not runtime.startswith(START_TOKEN):
        accounted = accounted[len(START_TOKEN):]
        notes.append("start token from the metadata")
      same = runtime == accounted
      ok &= same
      rows.append({"scenario": name, "equal_after_accounted": same,
                   "accounted": notes, "runtime": runtime, "transformers": want})
      print(f"  [{'v' if same else 'X'}] {name:15s} runtime == transformers"
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
      description="Quality gate / metadata / template / section checks for LFM3-Audio-2B")
  ap.add_argument("model", help="path to the .litertlm")
  ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu",
                  help="the text decoder's backend")
  ap.add_argument("--audio-backend", choices=["cpu", "gpu"],
                  help="the audio section's backend (default: --backend)")
  ap.add_argument("--min-correct", type=int, default=7)
  ap.add_argument("--asr-json", help="clips to transcribe: a JSON list (or "
                  '{"clips": [...]}) of {"name", "path", "reference"}, path '
                  "relative to --asr-dir (default: the JSON's folder)")
  ap.add_argument("--asr-dir")
  ap.add_argument("--asr-line-set", action="store_true",
                  help='only the clips marked "parts_line": true')
  ap.add_argument("--max-wer", type=float, default=0.10)
  ap.add_argument("--voicemail-json", help='recorded messages: {"items": [{"id",'
                  ' "caller", "callback"}]}, audio <id>.wav in --voicemail-dir '
                  "(default: wav/ next to the JSON)")
  ap.add_argument("--voicemail-dir")
  ap.add_argument("--min-voicemail", type=int, default=5)
  ap.add_argument("--wav", help="a 16 kHz mono WAV for --render")
  ap.add_argument("--source", help="local copy of the checkpoint (default: the "
                                   "Hub at the pinned revision)")
  ap.add_argument("--json", help="write a JSON report here")
  ap.add_argument("--check-metadata", action="store_true")
  ap.add_argument("--render", action="store_true")
  ap.add_argument("--sections", metavar="OTHER_BUNDLE")
  args = ap.parse_args()
  args.audio_backend = args.audio_backend or args.backend

  if not Path(args.model).exists():
    print(f"ERROR: model not found: {args.model}", file=sys.stderr)
    return 2
  if args.check_metadata:
    return check_metadata(args)
  if args.render:
    if not args.wav:
      print("ERROR: --render needs --wav", file=sys.stderr)
      return 2
    return render(args)
  if args.sections:
    return sections_compare(args)
  return gate(args)


if __name__ == "__main__":
  sys.exit(main())
