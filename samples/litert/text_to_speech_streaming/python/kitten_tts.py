# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""KittenTTS nano 0.8 on LiteRT — streaming text-to-speech engine (Python).

Python twin of the Android sample in ../kotlin_cpu: the same five model files,
the same espeak-free G2P (dictionary + neural fallback), and the same
sentence-level streaming granularity. Runs anywhere the ai-edge-litert pip
package does, including a Raspberry Pi 5 (64-bit OS).

LiteRT API split, mirroring the app:

  - G2P and vocoder run on the CompiledModel API. The vocoder has a dynamic
    sequence length and is resized per sentence with `resize_input_tensor`.
  - Predictor and prosody run on the classic Interpreter API. Their fused
    dynamic-length LSTM kernels keep hidden state in TFLite variable tensors,
    which the CompiledModel model loader does not accept yet (b/365299994).
    The Interpreter path must call `reset_all_variables()` before every
    invoke, or state left over from the previous sentence corrupts the
    predicted durations.

Runtime dependencies: numpy + ai-edge-litert only.
"""

import gzip
import json
import re
import time
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator
from ai_edge_litert.interpreter import Interpreter
from ai_edge_litert.tensor_buffer import TensorBuffer

SAMPLE_RATE = 24000

# The 8 bundled voices, in voices.bin order (see conversion/export_voices.py).
VOICES = (
    "expr-voice-2-m",
    "expr-voice-2-f",
    "expr-voice-3-m",
    "expr-voice-3-f",
    "expr-voice-4-m",
    "expr-voice-4-f",
    "expr-voice-5-m",
    "expr-voice-5-f",
)

_PREDICTOR_MODEL = "kitten_predictor_fp16.tflite"
_PROSODY_MODEL = "kitten_prosody_fp16.tflite"
_VOCODER_MODEL = "kitten_vocoder_fp16.tflite"
_G2P_MODEL = "dp_g2p_matcha_fp16.tflite"
_VOICES_FILE = "voices.bin"

_STYLE_ROWS = 400
_STYLE_DIM = 256
_D_DIM = 256
_ASR_DIM = 128
_SAMPLES_PER_FRAME = 600

# The upstream pip package trims this many samples from the end of every chunk.
_TAIL_TRIM = 5000
_MIN_SAMPLES = 1200

# Sentence chunking (a faithful port of the pip package's chunk_text; also the
# streaming granularity — see ../kotlin_cpu SentenceChunker.kt).
_SENTENCE_END = re.compile(r"[.!?]+")
_MAX_CHUNK_CHARS = 400
_CHUNK_PUNCTUATION = ".!?,;:"

# Case-aware tokens: ACRONYM (2+ caps) | NUMBER | word | punctuation.
_TOKEN = re.compile(r"[A-Z]{2,}|\d[\d,]*(?:\.\d+)?|[A-Za-z']+|[.,!?;:—…\"]")
_ACRONYM = re.compile(r"[A-Z]{2,}\Z")
_WORD = re.compile(r"[A-Za-z']+\Z")

# espeak letter-name IPA — acronyms are spelled out ("GPU" -> dʒˈiːpˈiːjˈuː).
_LETTER_IPA = {
    "a": "ˈeɪ", "b": "bˈiː", "c": "sˈiː", "d": "dˈiː",
    "e": "ˈiː", "f": "ˈɛf", "g": "dʒˈiː", "h": "ˈeɪtʃ",
    "i": "ˈaɪ", "j": "dʒˈeɪ", "k": "kˈeɪ", "l": "ˈɛl",
    "m": "ˈɛm", "n": "ˈɛn", "o": "ˈoʊ", "p": "pˈiː",
    "q": "kjˈuː", "r": "ˈɑːɹ", "s": "ˈɛs", "t": "tˈiː",
    "u": "jˈuː", "v": "vˈiː", "w": "dˈʌbəljˌuː", "x": "ˈɛks",
    "y": "wˈaɪ", "z": "zˈiː",
}
_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety")
_SCALES = ("", "thousand", "million", "billion", "trillion")


def chunk_text(text: str) -> list[str]:
    """Splits text into per-sentence synthesis chunks.

    Upstream quirks kept on purpose (the model's prosody was tuned against
    this frontend): sentence-final `.!?` are consumed by the split, and every
    chunk is then terminated with a comma, so the model always sees `,` as
    the chunk-final token.

    Args:
        text: Input text of any length.

    Returns:
        Sentence-sized chunks, each ending in punctuation.
    """
    chunks = []
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= _MAX_CHUNK_CHARS:
            chunks.append(_ensure_punctuation(sentence))
            continue
        # Overlong sentence: split on word boundaries.
        piece = ""
        for word in sentence.split():
            if piece and len(piece) + len(word) + 1 > _MAX_CHUNK_CHARS:
                chunks.append(_ensure_punctuation(piece))
                piece = ""
            piece = f"{piece} {word}" if piece else word
        if piece:
            chunks.append(_ensure_punctuation(piece))
    return chunks


def _ensure_punctuation(sentence: str) -> str:
    return sentence if sentence[-1] in _CHUNK_PUNCTUATION else sentence + ","


def _number_to_words(raw: str) -> list[str]:
    """'1,234.5' -> ['one','thousand','two','hundred','thirty','four',...]."""
    token = raw.replace(",", "")
    if "." in token:
        integer_part, _, fraction = token.partition(".")
        words = (_integer_to_words(int(integer_part)) if integer_part
                 else ["zero"])
        words.append("point")
        words.extend(_ONES[int(digit)] for digit in fraction if digit.isdigit())
        return words
    try:
        return _integer_to_words(int(token))
    except ValueError:
        return []


def _words_under_thousand(value: int) -> list[str]:
    words = []
    if value >= 100:
        words += [_ONES[value // 100], "hundred"]
        value %= 100
    if value >= 20:
        words.append(_TENS[value // 10])
        value %= 10
    if value > 0:
        words.append(_ONES[value])
    return words


def _integer_to_words(value: int) -> list[str]:
    if value == 0:
        return ["zero"]
    if value < 0:
        return ["minus"] + _integer_to_words(-value)
    groups = []
    while value > 0:
        groups.append(value % 1000)
        value //= 1000
    if len(groups) > len(_SCALES):
        # Too big to name; read digit by digit.
        digits = "".join(str(g).zfill(3) for g in reversed(groups)).lstrip("0")
        return [_ONES[int(d)] for d in digits]
    words = []
    for index in reversed(range(len(groups))):
        if groups[index] == 0:
            continue
        words.extend(_words_under_thousand(groups[index]))
        if _SCALES[index]:
            words.append(_SCALES[index])
    return words


class KittenG2P:
    """Espeak-free English G2P, shared with the Android sample.

    1. A 275k-entry espeak-IPA dictionary (primary, covers ~all common words
       correctly), and
    2. a DeepPhonemizer ForwardTransformer (OpenPhonemizer's espeak-IPA
       checkpoint, Clear BSD / MIT) for out-of-dictionary words, converted to
       a fixed-shape LiteRT graph run on the CompiledModel CPU accelerator.

    KittenTTS was trained on espeak en-us IPA over the same 178-symbol
    StyleTTS2 table this dictionary targets, so the IPA maps 1:1 onto the
    model's symbol inventory. Punctuation is emitted as its own
    space-separated token, matching the upstream pip package's
    `basic_english_tokenize`, which the duration predictor was tuned against.
    """

    def __init__(self, model_path: Path, assets_dir: Path):
        """Loads the neural G2P graph and the host tables.

        Args:
            model_path: Path to dp_g2p_matcha_fp16.tflite.
            assets_dir: Directory with g2p_dict.txt.gz, g2p_meta.json and
                config.json (the Android app's assets directory).
        """
        self._model = CompiledModel.from_file(
            str(model_path), HardwareAccelerator.CPU)
        self._inputs = self._model.create_input_buffers(0)
        self._outputs = self._model.create_output_buffers(0)

        meta = json.loads((assets_dir / "g2p_meta.json").read_text())
        self._char_to_index = {
            char: index for char, index in meta["char2idx"].items()
            if len(char) == 1
        }
        self._index_to_phoneme = {
            int(index): phoneme for index, phoneme in meta["idx2ph"].items()
        }
        self._char_repeats: int = meta["char_repeats"]
        self._start_id: int = meta["start"]
        self._end_id: int = meta["end"]
        self._max_tokens: int = meta["MAXT"]
        self._num_phonemes: int = meta["n_phonemes"]
        self._special_tokens = set(meta["special"])

        # StyleTTS2 symbol -> id (178-symbol list from config.json).
        config = json.loads((assets_dir / "config.json").read_text())
        self._symbol_to_id = {
            symbol: index for index, symbol in enumerate(config["symbols"])
            if len(symbol) == 1
        }

        # espeak-IPA dictionary (primary G2P): word<TAB>ipa per line.
        self._dictionary = {}
        with gzip.open(assets_dir / "g2p_dict.txt.gz", "rt",
                       encoding="utf-8") as dictionary_file:
            for line in dictionary_file:
                word, tab, ipa = line.rstrip("\n").partition("\t")
                if tab:
                    self._dictionary[word] = ipa

    def phonemize(self, text: str) -> list[int]:
        """Converts one sentence to KittenTTS symbol ids (no BOS/EOS).

        Host-side text normalization: ALL-CAPS acronyms are spelled
        letter-by-letter (GPU -> "gee pee you") and numbers are read as words
        (4090 -> "four thousand ninety").

        Args:
            text: One sentence.

        Returns:
            Symbol ids into the 178-symbol StyleTTS2 table.
        """
        pieces = []
        for match in _TOKEN.finditer(text):
            token = match.group()
            if _ACRONYM.match(token):
                pieces.append("".join(
                    _LETTER_IPA[c] for c in token.lower() if c in _LETTER_IPA))
            elif token[0].isdigit():
                for word in _number_to_words(token):
                    pieces.append(
                        self._dictionary.get(word) or self.phonemize_word(word))
            elif _WORD.match(token):
                # Dictionary first; the neural G2P handles the rest.
                lower = token.lower()
                pieces.append(
                    self._dictionary.get(lower) or self.phonemize_word(lower))
            else:
                # Punctuation: its own space-separated token.
                pieces.append(token)
        ipa = " ".join(piece for piece in pieces if piece)
        return [self._symbol_to_id[c] for c in ipa if c in self._symbol_to_id]

    def phonemize_word(self, word: str) -> str:
        """Converts one word to espeak-style IPA with the neural G2P."""
        ids = [self._start_id]
        for char in word:
            index = self._char_to_index.get(char)
            if index is not None:
                ids.extend([index] * self._char_repeats)
        ids.append(self._end_id)
        length = min(len(ids), self._max_tokens)
        model_input = np.zeros(self._max_tokens, dtype=np.float32)
        model_input[:length] = ids[:length]

        self._inputs[0].write(model_input)
        self._model.run_by_index(0, self._inputs, self._outputs)
        logits = np.asarray(
            self._outputs[0].read(self._max_tokens * self._num_phonemes,
                                  np.float32)
        ).reshape(self._max_tokens, self._num_phonemes)

        result = []
        previous = -1
        for best in np.argmax(logits[:length], axis=1):
            if best == previous:
                continue
            previous = best
            phoneme = self._index_to_phoneme.get(int(best))
            if not phoneme or phoneme in self._special_tokens or best == 0:
                continue
            result.append(phoneme.replace("-", ""))  # Strip acronym hyphens.
        return "".join(result)

    def close(self) -> None:
        for buffer in self._inputs + self._outputs:
            buffer.destroy()
        self._model.close()


class _LstmGraph:
    """A dynamic-length graph on the Interpreter API (predictor / prosody).

    These graphs contain fused dynamic-length LSTM kernels whose hidden state
    lives in TFLite variable tensors. The CompiledModel loader rejects
    variable tensors (b/365299994), so the Interpreter drives them: resize to
    the sentence's shapes, re-allocate, reset the variable tensors, invoke.
    The reset is required — without it a same-length invoke starts from the
    previous sentence's final LSTM state and corrupts the output.
    """

    def __init__(self, path: Path, threads: int):
        self._interpreter = Interpreter(
            model_path=str(path), num_threads=threads)
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    @staticmethod
    def _canonical(name: str) -> str:
        name = name.split(":")[0]
        prefix = "serving_default_"
        return name[len(prefix):] if name.startswith(prefix) else name

    def __call__(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Runs the graph; returns outputs keyed by tensor-name suffix."""
        for detail in self._input_details:
            feed = feeds[self._canonical(detail["name"])]
            self._interpreter.resize_tensor_input(
                detail["index"], list(feed.shape))
        self._interpreter.allocate_tensors()
        self._interpreter.reset_all_variables()
        for detail in self._input_details:
            self._interpreter.set_tensor(
                detail["index"], feeds[self._canonical(detail["name"])])
        self._interpreter.invoke()
        return {
            detail["name"]: self._interpreter.get_tensor(detail["index"])
            for detail in self._output_details
        }


class _DynamicVocoder:
    """The conv-only vocoder on the CompiledModel API with per-call resize.

    The working dynamic-shape sequence on the CompiledModel API is:

      1. `resize_input_tensor` for every input, with the sentence's shapes.
      2. Create the input buffers *after* the resize (they are sized from the
         resized tensors).
      3. Provide an output buffer of the exact final size, wrapped from host
         memory — output shapes propagate only when the model allocates
         during run, after the default output buffers would have been sized.
    """

    def __init__(self, path: Path):
        self._model = CompiledModel.from_file(
            str(path), HardwareAccelerator.CPU)
        signatures = self._model.get_signature_list()
        self._input_names = next(iter(signatures.values()))["inputs"]

    def __call__(self, feeds: dict[str, np.ndarray]) -> np.ndarray:
        """Runs the vocoder; returns the waveform as [samples] float32."""
        for index, name in enumerate(self._input_names):
            self._model.resize_input_tensor(index, list(feeds[name].shape))
        inputs = self._model.create_input_buffers(0)
        for index, name in enumerate(self._input_names):
            inputs[index].write(np.ascontiguousarray(feeds[name].ravel()))
        frames = feeds["asr"].shape[1]
        waveform = np.zeros(_SAMPLES_PER_FRAME * frames, dtype=np.float32)
        outputs = [TensorBuffer.create_from_host_memory(waveform)]
        self._model.run_by_index(0, inputs, outputs)
        for buffer in inputs + outputs:
            buffer.destroy()
        return waveform


class SentenceResult(NamedTuple):
    """One synthesized sentence, plus per-stage metrics for the demo."""

    sentence: str
    audio: np.ndarray  # float32 PCM @ 24 kHz
    tokens: int
    frames: int
    synthesis_seconds: float


class KittenTTS:
    """KittenTTS nano 0.8 (15M params, 24 kHz) on LiteRT, CPU.

    ```
    token ids [1,N] + style [1,256] + speed [1]
      -> predictor -> d [1,N,256], t_en [1,N,128], durations [N]
      -> (host: repeat rows by durations) -> en [1,T,256], asr [1,T,128]
      -> prosody  -> f0 [1,2T], noise [1,2T], harmonics [1,120T+1,22]
      -> vocoder  -> waveform [1,600T] @ 24 kHz
    ```

    All graphs have a dynamic sequence length — any sentence runs on the same
    graphs with no padding buckets, so compute scales with the text.
    """

    def __init__(self, models_dir: str | Path, assets_dir: str | Path,
                 voice: str = "expr-voice-5-m", threads: int = 4):
        """Loads the five model files and the host tables.

        Args:
            models_dir: Directory with the three synthesis graphs, the G2P
                graph and voices.bin (see the sample README).
            assets_dir: The Android app's assets directory (host tables).
            voice: One of VOICES.
            threads: Interpreter thread count for the LSTM graphs.

        Raises:
            FileNotFoundError: If a model file is missing.
            ValueError: If the voice name is unknown.
        """
        models_dir = Path(models_dir)
        assets_dir = Path(assets_dir)
        for name in (_PREDICTOR_MODEL, _PROSODY_MODEL, _VOCODER_MODEL,
                     _G2P_MODEL, _VOICES_FILE):
            if not (models_dir / name).exists():
                raise FileNotFoundError(
                    f"{models_dir / name} not found — build the models with "
                    "../conversion/ first (see README.md)")
        if voice not in VOICES:
            raise ValueError(f"Unknown voice {voice!r}; expected one of "
                             f"{', '.join(VOICES)}")

        self.g2p = KittenG2P(models_dir / _G2P_MODEL, assets_dir)
        self._predictor = _LstmGraph(models_dir / _PREDICTOR_MODEL, threads)
        self._prosody = _LstmGraph(models_dir / _PROSODY_MODEL, threads)
        self._vocoder = _DynamicVocoder(models_dir / _VOCODER_MODEL)
        self._voice = VOICES.index(voice)

        voices = np.fromfile(models_dir / _VOICES_FILE, dtype="<f4")
        expected = len(VOICES) * _STYLE_ROWS * _STYLE_DIM
        if voices.size != expected:
            raise ValueError(f"Unexpected {_VOICES_FILE} size {voices.size}")
        self._voice_table = voices.reshape(len(VOICES), _STYLE_ROWS,
                                           _STYLE_DIM)

    def warm_up(self) -> None:
        """Runs one short synthesis so first-sentence timing is honest."""
        self.synthesize("warm up,")

    def synthesize(self, sentence: str, speed: float = 1.0) -> SentenceResult:
        """Synthesizes one sentence.

        Args:
            sentence: One sentence (one chunk from chunk_text).
            speed: Speech speed; 1.0 = normal.

        Returns:
            The synthesized audio plus metrics.
        """
        start = time.perf_counter()
        # KittenTTS ships one style row per *text length*: the lookup uses the
        # sentence's character count, same as the upstream pip package.
        row = min(len(sentence), _STYLE_ROWS - 1)
        style = np.ascontiguousarray(
            self._voice_table[self._voice, row:row + 1])

        symbol_ids = self.g2p.phonemize(sentence)
        ids = np.array([[0] + symbol_ids + [0]], dtype=np.int32)

        predicted = self._predictor({
            "input_ids": ids,
            "style": style,
            "speed": np.array([speed], dtype=np.float32),
        })
        t_en = self._output(predicted, ":0")  # [1,N,128]
        durations = self._output(predicted, ":1")  # [N]
        d = self._output(predicted, ":2")  # [1,N,256]

        # Length-regulate: repeat each token's feature row by its duration.
        frames = int(durations.sum())
        en = np.repeat(d[0], durations, axis=0)[None]
        asr = np.repeat(t_en[0], durations, axis=0)[None]

        prosody = self._prosody({"en": en, "style": style})
        f0 = self._output(prosody, ":0")  # [1,2T]
        noise = self._output(prosody, ":1")  # [1,2T]
        harmonics = self._output(prosody, ":2")  # [1,120T+1,22]

        waveform = self._vocoder({
            "asr": asr,
            "f0": f0,
            "n": noise,
            "har": harmonics,
            "style": style,
        })

        # The upstream pip package trims the synthesis tail of every chunk.
        samples = max(waveform.size - _TAIL_TRIM,
                      min(_MIN_SAMPLES, waveform.size))
        audio = np.clip(waveform[:samples], -1.0, 1.0)
        return SentenceResult(
            sentence=sentence,
            audio=audio,
            tokens=ids.shape[1],
            frames=frames,
            synthesis_seconds=time.perf_counter() - start,
        )

    def stream(self, text: str,
               speed: float = 1.0) -> Iterator[SentenceResult]:
        """Yields one SentenceResult per chunk, as soon as each is ready."""
        for sentence in chunk_text(text):
            yield self.synthesize(sentence, speed)

    @staticmethod
    def _output(outputs: dict[str, np.ndarray], suffix: str) -> np.ndarray:
        for name, value in outputs.items():
            if name.endswith(suffix):
                return value
        raise KeyError(f"Output {suffix} not found in {list(outputs)}")

    def close(self) -> None:
        self.g2p.close()
