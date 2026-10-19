# Models built into the app

## YOLO26n detector — `yolo26n_fp16_rawhead.tflite`

- 10 361 332 bytes, SHA-256 `5ddd5eebad18587d56500a30b0995c07c9e1a241640750f568e4a974f66ed80a`.
- A derivative of **YOLO26n** by Ultralytics (Ultralytics YOLO26), converted to LiteRT by Arm
  ([Arm/yolo26n-fp16-litert](https://huggingface.co/Arm/yolo26n-fp16-litert), file
  `yolo26n_conv2d_f16_weights.tflite`).
- Derived with [`tool/prune_yolo26n_head.py`](../../tool/prune_yolo26n_head.py). The script cuts the head at
  `[1, 8400, 84]` (ADD → v1) so the whole graph runs on the GPU. It does not change any weights.
- **Licence: GNU Affero General Public License v3.0** (AGPL-3.0), as for the originals
  ([full text](https://www.gnu.org/licenses/agpl-3.0.html)). An app that distributes this file must make its
  corresponding source available under AGPL-compatible terms. That source includes the derivation script.
- Shown in the app under Home › More › Licences.

## EmbeddingGemma 2 740m (knowledge base) — `embeddinggemma-2-740m.litertlm`

- `embeddinggemma-2-740m.litertlm`: 484 622 336 bytes, SHA-256
  `e7a8a2204b91e0f96e92960e84a09a89212e1633dcb7575a9bf3378b4df77f4c`. One self-contained bundle: the model and its
  tokenizer.
- From [litert-community/embeddinggemma-2-740m-litert-lm](https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm/tree/24d962e906c7d332c6428e71c9676855024569e2)
  (revision 24d962e9), the LiteRT-LM packaging of
  [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2).
- **Licence: Apache License 2.0** ([full text](https://www.apache.org/licenses/LICENSE-2.0)). Redistributed
  unmodified. The app uses its text, audio and image embeddings.
- Shown in the app under Home › More › Licences.

## Whisper base (speech recognition, Demo 1) — `whisper_base_30s_i8.tflite`, `whisper_base_tokenizer.json`

- `whisper_base_30s_i8.tflite`: 77 012 960 bytes, SHA-256
  `f6943d9d293138850b729e074057956c664891c57837692b1bac4608c4506cd1`, from
  [litert-community/whisper-base](https://huggingface.co/litert-community/whisper-base) (revision ba2c6136).
- `whisper_base_tokenizer.json`: 2 480 466 bytes, SHA-256
  `27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566`, the `tokenizer.json` of
  [openai/whisper-base](https://huggingface.co/openai/whisper-base) (revision e37978b9).
- **Licence: Apache License 2.0** ([full text](https://www.apache.org/licenses/LICENSE-2.0)). Whisper by OpenAI.
  Redistributed unmodified.

## moonshine-tiny (speech recognition, Demo 3) — `moonshine_tiny_5s_f32.tflite`, `moonshine_tiny_tokenizer.json`

- `moonshine_tiny_5s_f32.tflite`: 109 373 140 bytes, SHA-256
  `16f281f1d3d23124e6adbdead8730d46f97cd105c299a9b94608d033c1151b12`, from
  [litert-community/moonshine-tiny](https://huggingface.co/litert-community/moonshine-tiny) (revision 4d0fd016).
- `moonshine_tiny_tokenizer.json`: 1 985 534 bytes, SHA-256
  `ed2324b3f699d8ba18a4030f33ba205d50b30ff46ff73f2b7d22661cc850efb4`, the `ctranslate2/tiny/tokenizer.json` of
  [moonshine-ai/moonshine](https://huggingface.co/moonshine-ai/moonshine) (revision 48b4e427).
- **Licence: MIT** (Copyright (c) Useful Sensors / Moonshine AI). Redistributed unmodified.

## Inflect-nano-v2 (speech synthesis) — `inflect/`

- `inflect_text_encoder_fp16.tflite` (1 782 268 bytes, SHA-256
  `c778f516b8557c69734f4d4308084c411b8d3a4575566385d163209bf504f083`) and `inflect_decoder_fp16.tflite`
  (6 375 948 bytes, SHA-256 `155d5072f104248983015e82d1b90968ab2e87bd7f96c4016928fc83231efbd0`), from
  [litert-community/Inflect-Nano-v2](https://huggingface.co/litert-community/Inflect-Nano-v2) (revision
  5c184d02). **Licence: Apache License 2.0.**
- The four G2P files it reuses, from [litert-community/Matcha-TTS](https://huggingface.co/litert-community/Matcha-TTS)
  (revision ee321481): `config.json` (2 071 bytes, SHA-256
  `7363e9e4dda1613aebff7005f2e8c0c76d9b0a1cac31de0f3483bef3089c6906`), `g2p_dict.txt.gz` (1 762 038 bytes, SHA-256
  `5b3493a8cd4d20b72c7b91415afaf3f32335ebd81f349698e1cedc898c59f979`), `dp_g2p_matcha_fp16.tflite` (25 785 872 bytes,
  SHA-256 `6e4b481f6874dfabc32ce73bf6f0ea1ba6ab5986ee6f76a27779364be8a53c73`), `g2p_meta.json` (1 904 bytes, SHA-256
  `7b87bfeaaa072be236e8491d771b0cb97cc92c3e5d83e3558fff8849868810f5`).
- **Licences of the G2P files**, as the Matcha-TTS card at that revision gives them ("Model: MIT (Matcha-TTS /
  HiFi-GAN). G2P dict: Clear BSD (OpenPhonemizer) + MIT (DeepPhonemizer)"):
  - `g2p_dict.txt.gz` is the espeak-IPA dictionary from [OpenPhonemizer](https://github.com/NeuralVox/OpenPhonemizer):
    **Clear BSD License (BSD-3-Clause-Clear)**, Copyright (c) 2024 mrfakename, NeuralVox, OpenPhonemizer
    Contributors. Its licence text, with the conditions and the disclaimer it requires to be reproduced:
    [`licenses/OpenPhonemizer-BSD-3-Clause-Clear.txt`](../../licenses/OpenPhonemizer-BSD-3-Clause-Clear.txt)
    (OpenPhonemizer's `LICENSE` at commit 15ffa723).
  - `config.json`, `dp_g2p_matcha_fp16.tflite` and `g2p_meta.json`: **MIT**.
- Redistributed unmodified. Shown in the app under Home › More › Licences.
