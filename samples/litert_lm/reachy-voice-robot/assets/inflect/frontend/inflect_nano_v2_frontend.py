from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from num2words import num2words


MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

WORD_OVERRIDES = {
    "Qwen3": "Qwen three",
    "Qwen": "Qwen",
    "PyTorch": "pie torch",
    "SQLite": "ess cue lite",
    "USB-C": "you ess bee see",
    "RTX 3060": "ar tee ex thirty sixty",
    "RTX 3090": "ar tee ex thirty ninety",
    "RTX 4090": "ar tee ex forty ninety",
    "RTX 5080": "ar tee ex fifty eighty",
    "RTX 5090": "ar tee ex fifty ninety",
}

LETTER_NAMES = {
    "A": "ay",
    "B": "bee",
    "C": "see",
    "D": "dee",
    "E": "ee",
    "F": "eff",
    "G": "gee",
    "H": "aitch",
    "I": "eye",
    "J": "jay",
    "K": "kay",
    "L": "ell",
    "M": "em",
    "N": "en",
    "O": "oh",
    "P": "pee",
    "Q": "cue",
    "R": "ar",
    "S": "ess",
    "T": "tee",
    "U": "you",
    "V": "vee",
    "W": "double you",
    "X": "ex",
    "Y": "why",
    "Z": "zee",
}

ABBREVIATIONS = {
    "Dr.": "doctor",
    "Mr.": "mister",
    "Mrs.": "missus",
    "Ms.": "miss",
    "Prof.": "professor",
    "St.": "saint",
    "vs.": "versus",
    "etc.": "et cetera",
    "e.g.": "for example",
    "i.e.": "that is",
}

PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": ", ",
        "\u2026": "...",
        "(": ", ",
        ")": ", ",
        "[": ", ",
        "]": ", ",
        "{": ", ",
        "}": ", ",
    }
)

_ESPEAK_CONFIGURED = False
_ESPEAK_BACKEND = None


@dataclass
class FrontendOutput:
    raw_text: str
    normalized_text: str
    phoneme_text: str
    tokens: list[str]
    token_count: int


def _words(value: int | float, *, ordinal: bool = False) -> str:
    if ordinal:
        text = num2words(value, to="ordinal")
    else:
        text = num2words(value)
    return text.replace("-", " ").replace(",", "")


def _digit_words(text: str) -> str:
    return " ".join(_words(int(ch)) for ch in text if ch.isdigit())


def _identifier_digits(text: str) -> str:
    words = []
    for index, character in enumerate(text):
        if not character.isdigit():
            continue
        words.append("oh" if character == "0" and index > 0 else _words(int(character)))
    return " ".join(words)


def _expand_identifier_token(token: str) -> str:
    match = re.fullmatch(r"([A-Za-z]?)(\d+)([A-Za-z]?)", token)
    if match is None:
        return token
    prefix, digits, suffix = match.groups()
    pieces = []
    if prefix:
        pieces.append(LETTER_NAMES[prefix.upper()])
    if len(digits) == 3 or digits.startswith("0"):
        pieces.append(_identifier_digits(digits))
    else:
        pieces.append(_words(int(digits)))
    if suffix:
        pieces.append(LETTER_NAMES[suffix.upper()])
    return " ".join(pieces)


def _expand_labeled_identifier(match: re.Match[str]) -> str:
    return f"{match.group(1)} {_expand_identifier_token(match.group(2))}"


def _expand_street_number(match: re.Match[str]) -> str:
    return _identifier_digits(match.group(1))


def _expand_money(match: re.Match[str]) -> str:
    raw = match.group(1).replace(",", "")
    dollars, _, cents = raw.partition(".")
    dollar_count = int(dollars)
    parts = [_words(dollar_count), "dollar" if dollar_count == 1 else "dollars"]
    if cents:
        cents = cents[:2].ljust(2, "0")
        cent_count = int(cents)
        if cent_count:
            parts.extend(["and", _words(cent_count), "cent" if cent_count == 1 else "cents"])
    return " ".join(parts)


def _expand_date_slash(match: re.Match[str]) -> str:
    month = int(match.group(1))
    day = int(match.group(2))
    year = int(match.group(3))
    try:
        date(year, month, day)
    except ValueError:
        return match.group(0)
    return f"{MONTHS[month - 1]} {_words(day, ordinal=True)} {_words(year)}"


def _expand_time(match: re.Match[str]) -> str:
    hour = int(match.group(1))
    minute = int(match.group(2))
    suffix = match.group(3) or ""
    pieces = [_words(hour)]
    if minute == 0:
        pieces.append("o clock")
    elif minute < 10:
        pieces.extend(["oh", _words(minute)])
    else:
        pieces.append(_words(minute))
    if suffix:
        suffix = suffix.lower().replace(".", "")
        pieces.extend(list(suffix))
    return " ".join(pieces)


def _expand_bare_hour_time(match: re.Match[str]) -> str:
    hour = int(match.group(1))
    suffix = re.sub(r"[^A-Za-z]", "", match.group(2)).lower()
    return f"{_words(hour)} {' '.join(suffix)}"


def _expand_version(match: re.Match[str]) -> str:
    return " point ".join(_words(int(part)) for part in match.group(0).split("."))


def _expand_decimal(match: re.Match[str]) -> str:
    whole, frac = match.group(1), match.group(2)
    return f"{_words(int(whole))} point {_digit_words(frac)}"


def _expand_ordinal(match: re.Match[str]) -> str:
    return _words(int(match.group(1)), ordinal=True)


def _expand_number(match: re.Match[str]) -> str:
    value = match.group(0).replace(",", "")
    if len(value) >= 5 and not value.startswith("20"):
        return _digit_words(value)
    return _words(int(value))


def _expand_phone(match: re.Match[str]) -> str:
    left, right = match.group(1), match.group(2)
    return f"{_digit_words(left)}, {_digit_words(right)}"


def _expand_acronym(match: re.Match[str]) -> str:
    acronym = match.group(0)
    if len(acronym) <= 1:
        return acronym
    return " ".join(LETTER_NAMES.get(ch, ch) for ch in acronym)


def normalize_text(text: str) -> str:
    text = text.translate(PUNCT_TRANSLATION)
    text = re.sub(r"\s+", " ", text).strip()

    for src, dst in WORD_OVERRIDES.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text)
    for src, dst in ABBREVIATIONS.items():
        text = re.sub(rf"\b{re.escape(src)}", dst, text, flags=re.IGNORECASE)

    text = re.sub(r"\b([A-Z])(?:\.([A-Z]))+\.", lambda m: " ".join(re.findall(r"[A-Z]", m.group(0))), text)
    text = re.sub(
        r"\b(apartment|apt\.?|suite|unit|room|flight|extension|order|invoice|locker|aisle|gate)\s+([A-Za-z]?\d{1,4}[A-Za-z]?)\b",
        _expand_labeled_identifier,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(\d{3})(?=\s+(?:North|South|East|West)\b)",
        _expand_street_number,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\$(\d[\d,]*(?:\.\d{1,2})?)", _expand_money, text)
    text = re.sub(r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(20\d{2}|19\d{2})\b", _expand_date_slash, text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp]\.?\s*[Mm]\.?)?\b", _expand_time, text)
    text = re.sub(r"\b(\d{1,2})\s*([AaPp]\.?\s*[Mm]\.?)\b", _expand_bare_hour_time, text)
    text = re.sub(r"\b(\d{3})-(\d{4})\b", _expand_phone, text)
    text = re.sub(r"\b\d+(?:\.\d+){2,}\b", _expand_version, text)
    text = re.sub(r"\b(\d+)\.(\d+)\b", _expand_decimal, text)
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", _expand_ordinal, text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d[\d,]*\b", _expand_number, text)
    text = re.sub(r"\b[A-Z]{2,}\b", _expand_acronym, text)
    text = re.sub(r",(?:\s*,)+", ",", text)
    text = re.sub(r",\s*([.!?])", r"\1", text)
    text = re.sub(r"\s+([,;:.!?])", r"\1", text)
    text = re.sub(r"([,;:.!?])(?=\S)", r"\1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def _configure_espeak() -> None:
    global _ESPEAK_CONFIGURED
    if _ESPEAK_CONFIGURED:
        return

    # Prefer the persistent distro library for long Linux preprocessing jobs.
    # espeakng-loader extracts a temporary shared object, which can exhaust mmap
    # resources when phonemizer repeatedly creates backends over a large corpus.
    system_libraries = (
        Path("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1"),
        Path("/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1"),
        Path("/usr/lib64/libespeak-ng.so.1"),
    )
    system_library = next((path for path in system_libraries if path.is_file()), None)
    if system_library is not None:
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", str(system_library))
    else:
        import espeakng_loader

        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", espeakng_loader.get_library_path())
        os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
        espeakng_loader.make_library_available()
        espeakng_loader.load_library()
    _ESPEAK_CONFIGURED = True


def phonemize_normalized_text(normalized_text: str) -> str:
    global _ESPEAK_BACKEND
    _configure_espeak()
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator

    if _ESPEAK_BACKEND is None:
        _ESPEAK_BACKEND = EspeakBackend(
            language="en-us",
            preserve_punctuation=True,
            with_stress=True,
            language_switch="remove-flags",
        )
    return _ESPEAK_BACKEND.phonemize(
        [normalized_text],
        separator=Separator(phone=" ", word=" | ", syllable=""),
        strip=True,
        njobs=1,
    )[0]


def tokenize_phoneme_text(phoneme_text: str) -> list[str]:
    text = phoneme_text.replace("|", " <word> ")
    text = re.sub(r"([,;:.!?])", r" \1 ", text)
    tokens = [tok for tok in re.split(r"\s+", text.strip()) if tok]
    return tokens


def run_frontend(text: str) -> FrontendOutput:
    normalized = normalize_text(text)
    phoneme_text = phonemize_normalized_text(normalized)
    tokens = tokenize_phoneme_text(phoneme_text)
    return FrontendOutput(
        raw_text=text,
        normalized_text=normalized,
        phoneme_text=phoneme_text,
        tokens=tokens,
        token_count=len(tokens),
    )


def _iter_input_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                row = json.loads(line)
                text = row.get("target_text") or row.get("text") or row.get("source_text")
                if not text:
                    raise ValueError(f"No text field found at {path}:{line_number}")
                yield row, str(text)
            else:
                yield {"line_number": line_number}, line


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Inflect-Nano-v2 English text frontend.")
    ap.add_argument("--text", help="Single text string to normalize and phonemize.")
    ap.add_argument("--input", type=Path, help="Text file or JSONL to process.")
    ap.add_argument("--out", type=Path, help="Output JSONL path for --input.")
    args = ap.parse_args()

    if bool(args.text) == bool(args.input):
        raise SystemExit("Provide exactly one of --text or --input.")

    if args.text:
        print(json.dumps(asdict(run_frontend(args.text)), ensure_ascii=False, indent=2))
        return

    if not args.out:
        raise SystemExit("--out is required with --input.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for metadata, text in _iter_input_rows(args.input):
            result = asdict(run_frontend(text))
            result["metadata"] = metadata
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
