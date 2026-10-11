# LiteRT-LM Samples

Usage examples and inference workflows for LiteRT-LM.

## Samples

| Sample | Task | API | Platform | Model |
|---|---|---|---|---|
| [`reachy-voice-robot/`](reachy-voice-robot/) | A voice robot on a Raspberry Pi: it sees, hears, reasons and speaks | LiteRT-LM and LiteRT | Raspberry Pi (Python) | [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) (imported with `litert-lm import`); [litert-community/moonshine-tiny](https://huggingface.co/litert-community/moonshine-tiny) and [litert-community/Inflect-Nano-v2](https://huggingface.co/litert-community/Inflect-Nano-v2), downloaded on first use; a YOLO26 detector exported into `assets/yolo/` |
| [`voice_assistant/`](voice_assistant/) | A phone that hears a request, acts on it with its own tools (alarm, timer, calendar) and answers aloud, offline | LiteRT-LM and LiteRT | Android (Kotlin) | [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm), [litert-community/Zipformer-medium-CR-CTC-LiteRT](https://huggingface.co/litert-community/Zipformer-medium-CR-CTC-LiteRT) and [litert-community/kitten-tts-nano-0.8](https://huggingface.co/litert-community/kitten-tts-nano-0.8) with the G2P files of [litert-community/Matcha-TTS](https://huggingface.co/litert-community/Matcha-TTS), downloaded on first use |
