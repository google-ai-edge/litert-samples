# Inflect-Nano-v2 — LiteRT on Raspberry Pi 5

## Drop-in synthesis

    pip install numpy ai-edge-litert phonemizer espeakng-loader num2words Unidecode
    python say.py "Hello! How can I help you today?" --models-dir . --frontend-dir frontend -o hello.wav

    from say import InflectTTS
    tts = InflectTTS(models_dir=".", frontend_dir="frontend")
    for sentence, pcm in tts.stream(text):   # float32 @ 24 kHz per sentence
        play(pcm)

frontend/ contains the upstream Apache-2.0 text frontend
(huggingface.co/owensong/Inflect-Nano-v2); it phonemizes with espeak-ng
(GPL-3.0, in-process).
