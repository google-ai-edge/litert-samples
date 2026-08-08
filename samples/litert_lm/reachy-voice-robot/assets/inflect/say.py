"""Inflect-Nano-v2 on LiteRT — drop-in text-to-speech module.

    from say import InflectTTS
    tts = InflectTTS(models_dir=".", frontend_dir="frontend")
    pcm = tts.say("Hello! How can I help you today?")        # float32 @ 24 kHz
    for sentence, pcm in tts.stream(long_text):              # sentence-streaming
        play(pcm)

CLI:
    python say.py "Hello there." -o hello.wav [--speed 1.0] [--variation 0.667]
                  [--seed 0] [--precision fp16] [--threads 4]

Runtime deps: numpy + ai-edge-litert + the upstream text frontend (Apache-2.0,
files from huggingface.co/owensong/Inflect-Nano-v2: inflect_vits_frontend.py,
inflect_nano_v2_frontend.py, runtime/text/) which needs
phonemizer + num2words + Unidecode (espeak-ng loaded in-process; GPL-3.0).

Mirrors the upstream inference.py behavior: sentence splitting, per-sentence
seeded variation noise, punctuation-dependent pauses, 5 ms edge fades.
"""
import argparse
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter

SR = 24000
NOISE_SCALE_W_UNUSED = None  # use_sdp=false checkpoint: duration is deterministic
DEFAULT_FRONTEND = Path(__file__).resolve().parent.parent / "checkpoint"


def load_frontend(frontend_dir):
    fd = Path(frontend_dir)
    for p in (fd, fd / "runtime"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from inflect_vits_frontend import run_vits_frontend  # noqa: E402
    from text import cleaned_text_to_sequence  # noqa: E402

    def to_tokens(sentence):
        seq = cleaned_text_to_sequence(run_vits_frontend(sentence).phoneme_text)
        interspersed = [0] * (2 * len(seq) + 1)
        interspersed[1::2] = seq
        return np.array([interspersed], dtype=np.int32)

    return to_tokens


def split_text(text, limit=280):
    normalized = " ".join(text.split())
    sentences = [p.strip() for p in re.split(r"(?<=[.!?;:])\s+", normalized) if p.strip()]
    chunks = []
    for sentence in sentences or [normalized]:
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit + 1)
            cut = cut if cut >= limit // 2 else limit
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            chunks.append(sentence)
    return chunks


def pause_seconds(chunk):
    ending = chunk.rstrip()[-1:]
    return {"?": 0.28, "!": 0.24, ".": 0.22, ";": 0.16, ":": 0.13, ",": 0.09}.get(ending, 0.08)


def edge_fade(wav, ms=5.0):
    n = min(round(SR * ms / 1000.0), wav.size // 2)
    if n <= 0:
        return wav
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    wav = wav.copy()
    wav[:n] *= ramp
    wav[-n:] *= ramp[::-1]
    return wav


class _Graph:
    def __init__(self, path, threads):
        self.it = Interpreter(model_path=str(path), num_threads=threads)
        self.inp = self.it.get_input_details()[0]
        self.outs = self.it.get_output_details()

    def __call__(self, x):
        self.it.resize_tensor_input(self.inp["index"], list(x.shape))
        self.it.allocate_tensors()
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        return [self.it.get_tensor(o["index"]) for o in self.outs]


class InflectTTS:
    def __init__(self, models_dir=".", frontend_dir=DEFAULT_FRONTEND,
                 precision="fp32", threads=4):
        models_dir = Path(models_dir)
        suffix = "" if precision == "fp32" else "_fp16"
        self.encoder = _Graph(models_dir / f"inflect_text_encoder{suffix}.tflite", threads)
        self.decoder = _Graph(models_dir / f"inflect_decoder{suffix}.tflite", threads)
        self.to_tokens = load_frontend(frontend_dir)

    def _synthesize(self, sentence, speed, variation, seed):
        e_out = self.encoder(self.to_tokens(sentence))
        m_p = [v for v in e_out if v.shape[-1] == 128][0]
        logs_p = [v for v in e_out if v.shape[-1] == 128][1]
        logw = [v for v in e_out if v.shape[-1] == 1][0]
        durations = np.ceil(np.exp(logw[0, :, 0]) / speed).astype(np.int64)
        durations = np.maximum(durations, 0)
        m_p_exp = np.repeat(m_p[0], durations, axis=0)[None]
        logs_p_exp = np.repeat(logs_p[0], durations, axis=0)[None]
        noise = np.random.RandomState(seed).randn(*m_p_exp.shape).astype(np.float32)
        z_p = (m_p_exp + noise * np.exp(logs_p_exp) * variation).astype(np.float32)
        return edge_fade(self.decoder(z_p)[0][0])

    def stream(self, text, speed=1.0, variation=0.667, seed=0):
        """Yield (sentence, float32 PCM @ 24 kHz) as each sentence is ready."""
        for i, sentence in enumerate(split_text(text)):
            yield sentence, self._synthesize(sentence, speed, variation, seed + i)

    def say(self, text, speed=1.0, variation=0.667, seed=0):
        pieces = []
        for i, (sentence, pcm) in enumerate(self.stream(text, speed, variation, seed)):
            if i:
                pieces.append(np.zeros(round(SR * pause_seconds(prev)), dtype=np.float32))
            pieces.append(pcm)
            prev = sentence
        out = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        return np.clip(out, -1.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description="Inflect-Nano-v2 on LiteRT")
    ap.add_argument("text")
    ap.add_argument("-o", "--output", default="say.wav")
    ap.add_argument("--models-dir", default=str(Path(__file__).resolve().parent.parent / "out"))
    ap.add_argument("--frontend-dir", default=str(DEFAULT_FRONTEND))
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--variation", type=float, default=0.667)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    tts = InflectTTS(args.models_dir, args.frontend_dir, args.precision, args.threads)
    t0 = time.perf_counter()
    pieces, prev = [], None
    for sentence, pcm in tts.stream(args.text, args.speed, args.variation, args.seed):
        print(f"  [{time.perf_counter()-t0:6.2f}s] {len(pcm)/SR:5.2f}s  {sentence[:60]}")
        if prev is not None:
            pieces.append(np.zeros(round(SR * pause_seconds(prev)), dtype=np.float32))
        pieces.append(pcm)
        prev = sentence
    wav = np.clip(np.concatenate(pieces), -1.0, 1.0)
    dt = time.perf_counter() - t0
    with wave.open(args.output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((wav * 32767).astype(np.int16).tobytes())
    print(f"wrote {args.output}: {len(wav)/SR:.2f}s audio in {dt:.2f}s")


if __name__ == "__main__":
    main()
