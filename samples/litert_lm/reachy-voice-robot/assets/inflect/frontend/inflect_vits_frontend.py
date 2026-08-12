from __future__ import annotations

import re
from dataclasses import dataclass

from inflect_nano_v2_frontend import _configure_espeak, normalize_text


# eSpeak is the general fallback. This table contains verified exceptions only;
# every entry is covered by a regression test and listening audit.
PHONEME_OVERRIDES = {
    "sˈæskɐtʃˌuːən": "sɐskˈætʃəwən",
    "flʊɹɹˈɛsənt": "flʊˈɹɛsənt",
}


@dataclass(frozen=True)
class VitsFrontendOutput:
    raw_text: str
    normalized_text: str
    phoneme_text: str


def phonemize_normalized(normalized_text: str) -> str:
    return phonemize_normalized_batch([normalized_text], jobs=1)[0]


def _apply_phoneme_overrides(phoneme_text: str) -> str:
    for source, replacement in PHONEME_OVERRIDES.items():
        phoneme_text = phoneme_text.replace(source, replacement)
    return re.sub(r"\s+", " ", phoneme_text).strip()


def phonemize_normalized_batch(normalized_texts: list[str], *, jobs: int = 1) -> list[str]:
    if not normalized_texts:
        return []
    _configure_espeak()
    from phonemizer import phonemize

    phoneme_texts = phonemize(
        normalized_texts,
        language="en-us",
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
        njobs=jobs,
    )
    return [_apply_phoneme_overrides(text) for text in phoneme_texts]


def run_vits_frontend_batch(texts: list[str], *, jobs: int = 1) -> list[VitsFrontendOutput]:
    normalized = [normalize_text(text) for text in texts]
    phonemes = phonemize_normalized_batch(normalized, jobs=jobs)
    return [
        VitsFrontendOutput(raw_text=raw, normalized_text=norm, phoneme_text=phones)
        for raw, norm, phones in zip(texts, normalized, phonemes, strict=True)
    ]


def run_vits_frontend(text: str) -> VitsFrontendOutput:
    normalized = normalize_text(text)
    return VitsFrontendOutput(
        raw_text=text,
        normalized_text=normalized,
        phoneme_text=phonemize_normalized(normalized),
    )
