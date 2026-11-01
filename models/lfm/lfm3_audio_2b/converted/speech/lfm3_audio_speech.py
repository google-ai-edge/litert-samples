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

"""LFM3-Audio-2B speech output on LiteRT: the host loop around the `.litertlm` decoder and the speech graphs.

LiteRT-LM feeds the decoder token ids. In speech mode the decoder's next input is an embedding, the sum of the 12
codebook embeddings of the frame just spoken, so this loop drives the graphs itself through the LiteRT
CompiledModel API (the generation loop of LiquidAI/LFM3-Audio-2B@9d49701c, `Lfm3AudioForConditionalGeneration._generate`,
ported to numpy for one sequence):

  decoder      the PREFILL_DECODE and EMBEDDER sections of the `.litertlm` file, read by offset from its header:
               embeddings in, logits and the last hidden state (`hidden`, after the final norm) out
  depth_proj   hidden -> depth_in, once per 80 ms speech frame
  depth_pass   depth_in + the codes known so far + mask -> 12 x 2017 logits; three passes reveal 4 + 4 + 4
               codebooks in a random (MaskGIT) order
  frame_embed  the frame's 12 codes -> the decoder's next input
  codec        12 frames of codes + streaming state -> windowed time frames; the host overlap-adds them at hop 320
               and divides by the window's overlap-added square (24 kHz audio)

Modes, as the source model: TEXT; AUDIO after <|audio_start|> until a frame holds the end-of-audio code (2016);
INTERLEAVED after <|mixed_start|>: blocks of 6 text tokens and 12 speech frames until <|text_end|>, then speech until
the end-of-audio frame. <|im_end|> in a text slot ends the turn.
Sampling, in the source model's order: repetition penalty over the last `window` frames of that codebook ->
temperature -> top-k -> softmax -> one draw. The defaults are the source card's (audio temperature 1.0, top-k 128,
repetition penalty 1.1 over 4,096 frames; text greedy); --greedy takes the argmax after the penalty.

  python lfm3_audio_speech.py --litertlm lfm3-audio-2b_int4_b128_c2048_h30.litertlm --graphs GRAPHS --hf SRC \\
      --text "The quick brown fox jumps over the lazy dog." --out fox.wav

GRAPHS holds the speech graphs and codec_window_f32.npy (export_speech_graphs.py writes both; the window can also be
read from SRC/audio_tokenizer/model.safetensors). SRC is a local copy of LiquidAI/LFM3-Audio-2B: the prompt is
rendered with its chat_template.jinja and tokenizer.json.
"""
import argparse
import collections
import json
import pathlib
import time

import numpy as np

TOKENS = {"im_end": "<|im_end|>", "audio_start": "<|audio_start|>", "text_end": "<|text_end|>",
          "mixed_start": "<|mixed_start|>", "audio": "<|audio|>", "audio_out": "<|audio_out|>"}
NUM_CB, CB_SIZE, AUDIO_EOS = 12, 2017, 2016
MASKGIT_SCHEDULE = (4, 4, 4)
INTERLEAVED_TEXT, INTERLEAVED_AUDIO = 6, 12
CODEC = {"sample_rate": 24000, "frame_stack": 6, "n_fft": 1280, "hop": 320,
         "state": {"k0": (1, 2, 29, 64), "v0": (1, 2, 29, 64), "k1": (1, 2, 29, 64), "v1": (1, 2, 29, 64),
                   "k2": (1, 2, 29, 64), "v2": (1, 2, 29, 64), "c0": (1, 6, 512), "c1": (1, 6, 512),
                   "c2": (1, 6, 512), "c3": (1, 6, 512), "c4": (1, 6, 512)}}
# LiquidAI/LFM3-Audio-2B generation_config.json (= its card's "Audio sampling parameters"); text: greedy, the
# source model's default text repetition window 32 at penalty 1.0 (no effect).
SAMPLING = {"audio_temperature": 1.0, "audio_top_k": 128, "audio_repetition_penalty": 1.1,
            "audio_repetition_window": 4096, "text_temperature": None, "text_top_k": None,
            "text_repetition_penalty": 1.0, "text_repetition_window": 32}
GRAPH_FILES = {"depth_proj": "lfm3_audio_depth_proj_r4b.tflite",
               "depth_pass": {"int8": "lfm3_audio_depth_pass_r4b_int8.tflite",
                              "f32": "lfm3_audio_depth_pass_r4b.tflite"},
               "frame_embed": "lfm3_audio_frame_embed_f32.tflite",
               "codec": "lfm3_audio_codec_decoder_f12_r4b2.tflite",
               "window": "codec_window_f32.npy"}
MASK_FILL = np.float32(-0.7 * np.finfo(np.float32).max)


# ---------------------------------------------------------------- LiteRT graphs
def compile_model(path_or_bytes, accel="cpu", threads=4):
  from ai_edge_litert.compiled_model import CompiledModel  # pylint: disable=g-import-not-at-top
  from ai_edge_litert.hardware_accelerator import HardwareAccelerator  # pylint: disable=g-import-not-at-top
  from ai_edge_litert.options import CpuOptions, Options  # pylint: disable=g-import-not-at-top
  hw = {"cpu": HardwareAccelerator.CPU, "gpu": HardwareAccelerator.GPU | HardwareAccelerator.CPU}[accel]
  opts = Options(hardware_accelerators=hw, cpu_options=CpuOptions(num_threads=threads))
  if isinstance(path_or_bytes, (bytes, bytearray)):
    return CompiledModel.from_buffer(bytes(path_or_bytes), options=opts)
  return CompiledModel.from_file(str(path_or_bytes), options=opts)


class Sig:
  """One signature of a compiled model; buffers by tensor name (state buffers can be left to the caller)."""

  def __init__(self, model, name, with_state=True):
    self.model, self.name = model, name
    self.ind = model.get_input_tensor_details(name)
    self.outd = model.get_output_tensor_details(name)
    keep = (lambda n: True) if with_state else (lambda n: not n.startswith("kv_cache_"))
    self.inb = {n: model.create_input_buffer_by_name(name, n) for n in self.ind if keep(n)}
    self.outb = {n: model.create_output_buffer_by_name(name, n) for n in self.outd if keep(n)}

  @staticmethod
  def shape(det):
    return tuple(int(s) for s in det["shape"])

  def read(self, name):
    det = self.outd[name]
    dtype = det["dtype"] if isinstance(det["dtype"], np.dtype) else np.dtype(det["dtype"])
    return self.outb[name].read(int(np.prod(self.shape(det))), dtype).reshape(self.shape(det))


class Graph:
  """A single-signature .tflite: call with inputs by name, get the outputs by name."""

  def __init__(self, path, accel="cpu", threads=4):
    self.model = compile_model(path, accel, threads)
    sigs = list(self.model.get_signature_list())
    assert len(sigs) == 1, sigs
    self.sig = Sig(self.model, sigs[0])

  def __call__(self, **inputs):
    for name, value in inputs.items():
      self.sig.inb[name].write(np.ascontiguousarray(value))
    self.model.run_by_name(self.sig.name, self.sig.inb, self.sig.outb)
    return {n: self.sig.read(n) for n in self.sig.outd}


def litertlm_sections(path):
  """The sections of a .litertlm file: type, model_type (TFLite sections), begin and end offsets."""
  from litert_lm_builder import litertlm_core, litertlm_peek  # pylint: disable=g-import-not-at-top
  from litert_lm_builder import litertlm_header_schema_py_generated as hschema  # pylint: disable=g-import-not-at-top
  at = litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET
  with open(path, "rb") as f:
    prefix = f.read(32)
    if not prefix.startswith(b"LITERTLM"):
      raise ValueError(f"{path} is not a .litertlm file")
    end = int.from_bytes(prefix[at:at + 8], "little")
    f.seek(0)
    head = f.read(end)
  md = hschema.LiteRTLMMetaData.GetRootAs(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:], 0)
  out = []
  for i in range(md.SectionMetadata().ObjectsLength()):
    s = md.SectionMetadata().Objects(i)
    items = {}
    for j in range(s.ItemsLength()):
      d = litertlm_peek.kvp_to_dict(s.Items(j))
      items[d["key"]] = d["value"]
    out.append({"type": litertlm_core.any_section_data_type_to_string(s.DataType()),
                "model_type": items.get("model_type"), "begin": s.BeginOffset(), "end": s.EndOffset()})
  return out


def read_section(path, model_type):
  rows = [s for s in litertlm_sections(path) if s["model_type"] == model_type]
  if len(rows) != 1:
    raise ValueError(f"{path}: {len(rows)} sections of type {model_type}")
  with open(path, "rb") as f:
    f.seek(rows[0]["begin"])
    return f.read(rows[0]["end"] - rows[0]["begin"])


class DecoderLM:
  """The decoder of the .litertlm file driven step by step: prefill in the largest prefill signatures that fit, the
  prompt's last row and every later step through `decode` (as LiteRT-LM does), the cache state passed from one call
  to the next. Positions are absolute; the mask lets row i see the cache up to its own position."""

  def __init__(self, litertlm, accel="cpu", threads=4):
    types = {s["model_type"] for s in litertlm_sections(litertlm)}
    missing = [t for t in ("tf_lite_prefill_decode", "tf_lite_embedder") if t not in types]
    if missing:
      raise ValueError(f"{litertlm} has no {' or '.join(missing)} section: speech output needs the LFM3-Audio file, "
                       "whose decoder takes embeddings and returns `hidden`, with the token embedder in its own section")
    self._dec_bytes = read_section(litertlm, "tf_lite_prefill_decode")  # kept alive with the model
    self._emb_bytes = read_section(litertlm, "tf_lite_embedder")
    self.model = compile_model(self._dec_bytes, accel, threads)
    sigs = self.model.get_signature_list()
    self.prefill_lengths = sorted((int(s.split("_")[1]) for s in sigs if s.startswith("prefill_")), reverse=True)
    self.sigs = {s: Sig(self.model, s, with_state=False) for s in sigs}
    dec = self.sigs["decode"]
    if "hidden" not in dec.outd:
      raise ValueError("the decoder has no `hidden` output: speech output needs the build that returns it")
    self.state_names = [n for n in dec.ind if n.startswith("kv_cache_")]
    self.cache_length = Sig.shape(dec.ind["mask"])[-1]
    self.hidden_size = Sig.shape(dec.ind["embeddings"])[-1]
    # two buffer sets swapped per call: a call reads one set and writes the other
    self.sets = [{n: self.model.create_input_buffer_by_name("decode", n) for n in self.state_names},
                 {n: self.model.create_output_buffer_by_name("decode", n) for n in self.state_names}]
    self.embedder = Graph(self._emb_bytes, "cpu", threads)
    self.last_hidden = self.last_logits = None
    self.reset()

  def reset(self):
    self.pos, self.cur = 0, 0
    for n in self.state_names:
      z = np.zeros(Sig.shape(self.sigs["decode"].ind[n]), np.float32)
      self.sets[0][n].write(z)
      self.sets[1][n].write(z)

  def _run(self, sig, embeddings, input_pos, mask):
    s = self.sigs[sig]
    s.inb["embeddings"].write(np.ascontiguousarray(embeddings, np.float32))
    s.inb["input_pos"].write(np.ascontiguousarray(input_pos, np.int32))
    s.inb["mask"].write(np.ascontiguousarray(mask, np.float32))
    src, dst = self.sets[self.cur], self.sets[1 - self.cur]
    self.model.run_by_name(sig, {**s.inb, **src}, {**s.outb, **dst})
    self.cur = 1 - self.cur
    return s

  def _mask(self, start, rows):
    m = np.full((1, 1, rows, self.cache_length), MASK_FILL, np.float32)
    for i in range(rows):
      m[0, 0, i, :start + i + 1] = 0.0
    return m

  def _decode(self, vec):
    if self.pos >= self.cache_length:
      raise RuntimeError(f"context full ({self.cache_length} tokens)")
    s = self._run("decode", np.asarray(vec, np.float32).reshape(1, 1, -1), np.array([self.pos], np.int32),
                  self._mask(self.pos, 1))
    self.pos += 1
    self.last_hidden = s.read("hidden").reshape(-1).copy()
    self.last_logits = s.read("logits").reshape(-1).copy()
    return self.last_hidden

  def embed_token(self, token_id):
    return self.embedder(token_ids=np.array([[int(token_id)]], np.int32))["embeddings"].reshape(-1).copy()

  def prefill(self, ids):
    """Prefills the prompt's token ids; returns the hidden state of its last position."""
    rows = np.stack([self.embed_token(t) for t in ids]).astype(np.float32)
    self.reset()
    i, n = 0, rows.shape[0] - 1
    while i < n:
      size = next(b for b in self.prefill_lengths if b <= n - i)
      self._run(f"prefill_{size}", rows[i:i + size][None], np.arange(self.pos, self.pos + size, dtype=np.int32),
                self._mask(self.pos, size))
      self.pos += size
      i += size
    return self._decode(rows[-1])

  def step(self, vec):
    return self._decode(vec)

  def text_logits(self, hidden):
    if hidden is not self.last_hidden:
      raise ValueError("text_logits reads the logits of the call that returned this hidden state")
    return self.last_logits


# ---------------------------------------------------------------- sampling and the depth decoder
def sample_logits(logits, temperature, top_k, penalty=None, history=None, suppress=(), rng=None):
  """One row, the source model's order: repetition penalty -> suppress -> (argmax if temperature 0 / no warper) ->
  temperature -> top-k -> softmax -> one draw from `rng`."""
  scores = logits.astype(np.float32).copy()
  if history is not None and len(history) and penalty not in (None, 1.0):
    ids = np.unique(np.asarray(history, np.int64))
    s = scores[ids]
    scores[ids] = np.where(s < 0, s * penalty, s / penalty)
  if suppress:
    scores[list(suppress)] = -np.inf
  if temperature == 0 or (temperature is None and not top_k):
    return int(np.argmax(scores))
  if temperature:
    scores = scores / np.float32(temperature)
  if top_k:
    k = min(int(top_k), scores.shape[-1])
    kth = np.partition(scores, -k)[-k]
    scores = np.where(scores < kth, -np.inf, scores)
  p = np.exp(scores - scores.max())
  p = p / p.sum()
  return int(rng.choice(p.shape[0], p=p))


class DepthSampler:
  """One speech frame: depth_proj once, then the MaskGIT passes of depth_pass, each revealing its codebooks."""

  def __init__(self, proj, dpass):
    self.proj, self.dpass = proj, dpass

  def sample_frame(self, hidden, s, history, rng):
    depth_in = self.proj(hidden=hidden.reshape(1, -1).astype(np.float32))["depth_in"]
    codes = np.zeros((1, NUM_CB), np.int32)
    revealed = np.zeros((1, NUM_CB), bool)
    order = rng.permutation(NUM_CB)
    start = 0
    for count in MASKGIT_SCHEDULE:
      pick = order[start:start + count]
      start += count
      logits = self.dpass(depth_in=depth_in, codes=codes, masked=(~revealed).astype(np.float32))["logits"][0]
      for k in pick:
        hist = None if history is None else history[:, k]
        codes[0, k] = sample_logits(logits[k], s["audio_temperature"], s["audio_top_k"],
                                    s["audio_repetition_penalty"], hist, rng=rng)
      revealed[0, pick] = True
    return codes[0]


# ---------------------------------------------------------------- codec
class CodecDecoder:
  """Streaming codec decode, 12 frames per call (the last call zero-padded), and the host overlap-add: hop 320,
  divided by the overlap-added window square, the first 480 samples dropped, the last 960 held until the next call
  (the last 480 dropped at the end), as LiquidCodecModel.decode streams."""

  def __init__(self, graph, window, frames_per_call=12):
    self.g, self.f = graph, frames_per_call
    self.state = {n: np.zeros(shape, np.float32) for n, shape in CODEC["state"].items()}
    self.pending = np.zeros((0, NUM_CB), np.int32)
    self.done = 0
    self.n_fft, self.hop = CODEC["n_fft"], CODEC["hop"]
    self.trim = (self.n_fft - self.hop) // 2
    self.w2 = np.asarray(window, np.float32) ** 2
    self.tail = self.tail_env = None
    self.started = False

  def _ola(self, frames, finalize):
    t = frames.shape[0]
    size = (t - 1) * self.hop + self.n_fft
    audio, env = np.zeros(size, np.float32), np.zeros(size, np.float32)
    for i in range(t):
      audio[i * self.hop:i * self.hop + self.n_fft] += frames[i]
      env[i * self.hop:i * self.hop + self.n_fft] += self.w2
    if self.tail is not None:
      audio[:2 * self.trim] += self.tail
      env[:2 * self.trim] += self.tail_env
    if not self.started:
      audio, env = audio[self.trim:], env[self.trim:]
      self.started = True
    if finalize:
      audio, env = audio[:-self.trim], env[:-self.trim]
      self.tail = self.tail_env = None
    else:
      self.tail, self.tail_env = audio[-2 * self.trim:].copy(), env[-2 * self.trim:].copy()
      audio, env = audio[:-2 * self.trim], env[:-2 * self.trim]
    return audio / env

  def _call(self, chunk, finalize):
    valid = chunk.shape[0]
    if valid < self.f:
      chunk = np.concatenate([chunk, np.zeros((self.f - valid, NUM_CB), np.int32)])
    out = self.g(codes=chunk.T[None].astype(np.int32), start=np.array([self.done * CODEC["frame_stack"]], np.int32),
                 **self.state)
    self.state = {n: out[n + "_out"] for n in self.state}
    self.done += valid
    return self._ola(out["frames"][0][:valid * CODEC["frame_stack"]], finalize)

  def push(self, frames):
    self.pending = np.concatenate([self.pending, np.asarray(frames, np.int32).reshape(-1, NUM_CB)])
    outs = []
    while self.pending.shape[0] >= self.f:
      outs.append(self._call(self.pending[:self.f], finalize=False))
      self.pending = self.pending[self.f:]
    return np.concatenate(outs) if outs else np.zeros(0, np.float32)

  def finish(self):
    if self.pending.shape[0]:
      out = self._call(self.pending, finalize=True)
      self.pending = self.pending[:0]
      return out
    if self.tail is None:
      return np.zeros(0, np.float32)
    audio, env = self.tail[:-self.trim], self.tail_env[:-self.trim]
    self.tail = self.tail_env = None
    return audio / env


# ---------------------------------------------------------------- prompt
class PromptBuilder:
  """The source repository's chat_template.jinja and tokenizer.json, rendered as transformers'
  apply_chat_template(..., continue_final_message=True) renders them."""

  def __init__(self, hf_dir):
    import jinja2  # pylint: disable=g-import-not-at-top
    from jinja2 import nodes  # pylint: disable=g-import-not-at-top
    from jinja2.ext import Extension  # pylint: disable=g-import-not-at-top
    from jinja2.sandbox import ImmutableSandboxedEnvironment  # pylint: disable=g-import-not-at-top
    from tokenizers import Tokenizer  # pylint: disable=g-import-not-at-top

    class GenerationTag(Extension):
      # transformers' {% generation %}...{% endgeneration %}: the body renders unchanged
      tags = {"generation"}

      def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.CallBlock(self.call_method("_render", []), [], [], body).set_lineno(lineno)

      def _render(self, caller):
        return caller()

    def raise_exception(msg):
      raise ValueError(msg)

    hf_dir = pathlib.Path(hf_dir)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[GenerationTag, jinja2.ext.loopcontrols])
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False, **kw)
    self.template = env.from_string((hf_dir / "chat_template.jinja").read_text())
    self.tok = Tokenizer.from_file(str(hf_dir / "tokenizer.json"))
    self.bos = json.loads((hf_dir / "tokenizer_config.json").read_text())["bos_token"]

  def token_id(self, text):
    i = self.tok.token_to_id(text)
    if i is None:
      raise ValueError(f"{text} is not a token of this tokenizer")
    return i

  def ids(self, messages):
    text = self.template.render(messages=messages, bos_token=self.bos, add_generation_prompt=False)
    final = messages[-1]["content"].strip()
    text = text[:text.rindex(final) + len(final)]  # continue_final_message
    return self.tok.encode(text, add_special_tokens=False).ids


# ---------------------------------------------------------------- the generation loop
class SpeechGenerator:
  TEXT, AUDIO, INTERLEAVED = "text", "audio", "interleaved"

  def __init__(self, lm, depth, frame_embed, tokens):
    self.lm, self.depth, self.frame_embed, self.t = lm, depth, frame_embed, tokens

  def generate(self, prompt_ids, max_new_tokens=512, settings=None, rng=None, on_frame=None, on_token=None):
    t, s = self.t, dict(SAMPLING)
    s.update(settings or {})
    rng = rng if rng is not None else np.random.default_rng(0)
    suppress = [t["audio"], t["audio_out"]]  # placeholder ids are never sampled in a text slot
    eos = (t["im_end"], t["text_end"])
    last = prompt_ids[-1]
    mode = self.AUDIO if last in (t["audio_start"], t["audio_out"]) else \
        self.INTERLEAVED if last == t["mixed_start"] else self.TEXT
    hidden = self.lm.prefill(prompt_ids)
    tokens, frames = [], []
    text_hist = collections.deque(maxlen=s["text_repetition_window"])
    audio_hist = collections.deque(maxlen=s["audio_repetition_window"])
    slots_left, speaking, text_done, finished = INTERLEAVED_TEXT, False, False, False
    for step in range(max_new_tokens):
      interleaving = mode == self.INTERLEAVED
      if mode == self.AUDIO or (interleaving and speaking):
        codes = self.depth.sample_frame(hidden, s, np.stack(audio_hist) if audio_hist else None, rng)
        if (codes == AUDIO_EOS).any():
          codes = np.full_like(codes, AUDIO_EOS)
          mode = self.TEXT
        else:
          audio_hist.append(codes.copy())
          frames.append(codes.copy())
          tokens.append(t["audio_out"])
          if on_frame is not None:
            on_frame(codes)
        next_embeds = self.frame_embed(codes=codes.reshape(1, -1).astype(np.int32))["embeds"].reshape(-1)
      else:
        token = sample_logits(self.lm.text_logits(hidden), s["text_temperature"], s["text_top_k"],
                              s["text_repetition_penalty"], np.asarray(text_hist) if text_hist else None, suppress,
                              rng=rng)
        text_hist.append(token)
        tokens.append(token)
        if on_token is not None:
          on_token(token)
        if token == t["text_end"] and interleaving:
          text_done, speaking = True, True
        elif token in eos:
          finished = True
        elif not interleaving and token == t["audio_start"]:
          mode = self.AUDIO
        elif not interleaving and token == t["mixed_start"]:
          mode, slots_left, speaking, text_done = self.INTERLEAVED, INTERLEAVED_TEXT, False, False
        next_embeds = self.lm.embed_token(token)
      if interleaving and not text_done:
        slots_left -= 1
        if slots_left == 0:
          speaking = not speaking
          slots_left = INTERLEAVED_AUDIO if speaking else INTERLEAVED_TEXT
      if finished or step == max_new_tokens - 1:
        break
      hidden = self.lm.step(next_embeds)
    return {"tokens": tokens, "frames": np.asarray(frames, np.int32).reshape(-1, NUM_CB), "finished": finished}


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--litertlm", required=True, help="lfm3-audio-2b_int4_b128_c2048_h30.litertlm or a rebuild")
  ap.add_argument("--graphs", required=True, help="folder of the speech graphs and codec_window_f32.npy")
  ap.add_argument("--hf", required=True, help="local copy of LiquidAI/LFM3-Audio-2B (template and tokenizer)")
  ap.add_argument("--text", required=True)
  ap.add_argument("--system", default="Perform TTS in American English using a female voice.",
                  help="the source card's TTS prompt; 'Respond with interleaved text and audio.' with --start mixed")
  ap.add_argument("--start", default="audio", choices=["audio", "mixed"],
                  help="how the assistant turn starts: <|audio_start|> (speech) or <|mixed_start|> (text + speech)")
  ap.add_argument("--depth-pass", default="int8", choices=["int8", "f32"])
  ap.add_argument("--codec-accel", default="cpu", choices=["cpu", "gpu"])
  ap.add_argument("--threads", type=int, default=4)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--greedy", action="store_true", help="audio codes by argmax after the repetition penalty")
  ap.add_argument("--max-new-tokens", type=int, default=512)
  ap.add_argument("--out", default="speech.wav")
  ap.add_argument("--codes-out", help="write the speech codes [frames, 12] to this .npy")
  a = ap.parse_args()
  graphs = pathlib.Path(a.graphs)
  t0 = time.time()
  lm = DecoderLM(a.litertlm, "cpu", a.threads)
  depth = DepthSampler(Graph(graphs / GRAPH_FILES["depth_proj"], "cpu", a.threads),
                       Graph(graphs / GRAPH_FILES["depth_pass"][a.depth_pass], "cpu", a.threads))
  frame_embed = Graph(graphs / GRAPH_FILES["frame_embed"], "cpu", a.threads)
  codec = CodecDecoder(Graph(graphs / GRAPH_FILES["codec"], a.codec_accel, a.threads),
                       np.load(graphs / GRAPH_FILES["window"]))
  pb = PromptBuilder(a.hf)
  tokens = {k: pb.token_id(v) for k, v in TOKENS.items()}
  start = "<|audio_start|>" if a.start == "audio" else "<|mixed_start|>"
  ids = pb.ids([{"role": "system", "content": a.system}, {"role": "user", "content": a.text},
                {"role": "assistant", "content": start}])
  load_s = time.time() - t0
  pcm, first_audio = [], []
  t1 = time.time()

  def on_frame(codes):
    chunk = codec.push(codes[None])
    if chunk.size and not first_audio:
      first_audio.append(time.time() - t1)
    pcm.append(chunk)

  gen = SpeechGenerator(lm, depth, frame_embed, tokens)
  out = gen.generate(ids, a.max_new_tokens, {"audio_temperature": 0.0} if a.greedy else None,
                     np.random.default_rng(a.seed), on_frame=on_frame)
  pcm.append(codec.finish())
  wall = time.time() - t1
  wav = np.concatenate(pcm)
  import soundfile as sf  # pylint: disable=g-import-not-at-top
  sf.write(a.out, np.clip(wav, -1, 1), CODEC["sample_rate"], subtype="PCM_16")
  if a.codes_out:
    np.save(a.codes_out, out["frames"])
  text = pb.tok.decode([x for x in out["tokens"] if x != tokens["audio_out"]], skip_special_tokens=True)
  audio_s = len(wav) / CODEC["sample_rate"]
  print(json.dumps({"frames": int(out["frames"].shape[0]), "audio_s": round(audio_s, 2), "text": text,
                    "finished": out["finished"], "prompt_tokens": len(ids), "load_s": round(load_s, 1),
                    "wall_s": round(wall, 1), "first_audio_s": round(first_audio[0], 2) if first_audio else None,
                    "s_per_audio_s": round(wall / audio_s, 3) if audio_s else None, "out": a.out}))


if __name__ == "__main__":
  main()
