---
title: On-device speech models for LiteRT
source: https://huggingface.co/openai/whisper-tiny/blob/169d4a4341b33bc18d8881c4b69c2e104e1cc0af/README.md ; https://huggingface.co/openai/whisper-base/blob/e37978b90ca9030d5170a5c07aadb050351a65bb/README.md ; https://huggingface.co/litert-community/whisper-base/blob/ba2c613646edc2b72f5a51fa0b5b0b322e436101/README.md ; https://huggingface.co/UsefulSensors/moonshine-tiny/blob/390624ed33d594443aa4aa221f5b9f283b545b5a/README.md ; https://huggingface.co/litert-community/moonshine-tiny/blob/4d0fd016df005c889aaa738b4765ede92736424f/README.md ; https://huggingface.co/litert-community/Inflect-Nano-v2/blob/5c184d02d4d1a7aacfa1e54f791f1e495cf52124/README.md ; https://huggingface.co/litert-community/Matcha-TTS/blob/8d650e794583c0b0869c87027c2f3a7c293902cb/README.md ; https://github.com/google-ai-edge/litert-samples/blob/367ddbdee22a8d73fba6e519c4a3f18d19008b05/samples/litert/speech_recognition/README.md
license: Apache-2.0 (Whisper cards, litert-community/whisper-base, litert-community/Inflect-Nano-v2, litert-samples) ; MIT (Moonshine cards, litert-community/Matcha-TTS)
---

# On-device speech models for LiteRT

This document covers speech models that run on device with LiteRT: the Whisper and Moonshine speech recognition (ASR) models and their LiteRT exports, the LiteRT speech recognition sample app, and the Inflect-Nano-v2 and Matcha-TTS text-to-speech (TTS) models.

## Whisper speech recognition model

Whisper is a pre-trained model for automatic speech recognition (ASR) and speech translation. Trained on 680k hours of labelled data, Whisper models demonstrate a strong ability to generalise to many datasets and domains without the need for fine-tuning. Whisper was proposed in the paper "Robust Speech Recognition via Large-Scale Weak Supervision" by Alec Radford et al. from OpenAI.

Whisper is a Transformer-based encoder-decoder model, also referred to as a sequence-to-sequence model. The models were trained on either English-only data or multilingual data. The English-only models were trained on speech recognition. The multilingual models were trained on both speech recognition and speech translation: for speech recognition the model predicts transcriptions in the same language as the audio, and for speech translation it predicts transcriptions in a different language to the audio. The Whisper models on Hugging Face are released under the Apache-2.0 license.

## Whisper model sizes

Whisper checkpoints come in five configurations of varying model sizes. The smallest four are trained on either English-only or multilingual data; the largest checkpoints are multilingual only.

| Size | Parameters | English-only | Multilingual |
| --- | --- | --- | --- |
| tiny | 39 M | yes | yes |
| base | 74 M | yes | yes |
| small | 244 M | yes | yes |
| medium | 769 M | yes | yes |
| large | 1550 M | no | yes |
| large-v2 | 1550 M | no | yes |

On LibriSpeech test-clean, Whisper tiny has a word error rate (WER) of 7.54 and Whisper base 5.01; on LibriSpeech test-other, tiny scores 17.15 and base 12.85. On Common Voice 11.0 Hindi, the reported WER is 141 for tiny and 131 for base.

## How Whisper chooses the language and task

The model is used with a `WhisperProcessor`, which pre-processes the audio inputs by converting them to log-Mel spectrograms and post-processes the model outputs by converting tokens to text. The model is told which task to perform through "context tokens" given to the decoder at the start of decoding, in this order:

1. The transcription always starts with the `<|startoftranscript|>` token.
2. The second token is the language token, for example `<|en|>` for English.
3. The third token is the task token: `<|transcribe|>` for speech recognition or `<|translate|>` for speech translation.
4. A `<|notimestamps|>` token is added if the model should not predict timestamps.

These tokens can be forced or un-forced. Forcing them controls the output language and task; if they are un-forced, Whisper automatically predicts the output language and task itself.

## The Whisper 30-second window and long-form audio

The Whisper model is intrinsically designed to work on audio samples of up to 30 s in duration. By using a chunking algorithm, it can transcribe audio samples of arbitrary length: in the Transformers pipeline, chunking is enabled by setting `chunk_length_s=30`, which also allows batched inference and sequence-level timestamps with `return_timestamps=True`. The model card notes that Whisper models cannot be used for real-time transcription out of the box, although their speed and size suggest that others may be able to build near-real-time speech recognition and translation on top of them.

The pre-trained model generalises well, but its predictive capabilities can be improved further for certain languages and tasks through fine-tuning, with as little as 5 hours of labelled data.

## Whisper training data and language coverage

The models are trained on 680,000 hours of audio and transcripts collected from the internet. 65% of this data (438,000 hours) is English-language audio with English transcripts, roughly 18% (126,000 hours) is non-English audio with English transcripts, and the final 17% (117,000 hours) is non-English audio with the corresponding transcript. The non-English data represents 98 different languages. Performance on transcription in a given language is directly correlated with the amount of training data in that language. The models show strong ASR results in about 10 languages, and are potentially quite useful as an ASR solution for developers, especially for English speech recognition.

## Whisper limitations and risks

The models show improved robustness to accents, background noise and technical language, and near state-of-the-art accuracy on speech recognition and translation. However, because they are trained in a weakly supervised manner on large-scale noisy data, predictions may include text that is not actually spoken in the audio (hallucination). The models perform unevenly across languages, with lower accuracy on low-resource languages, and show disparate performance on different accents and dialects. The sequence-to-sequence architecture makes them prone to generating repetitive text, which beam search and temperature scheduling mitigate only partly. The model card cautions against transcribing recordings of individuals taken without their consent and against use in high-risk decision-making contexts.

## Whisper base as a LiteRT model

The Hugging Face repo `litert-community/whisper-base` contains a LiteRT `.tflite` export of `openai/whisper-base`, licensed Apache-2.0. Its file `whisper_base_30s_f32.tflite` is an FP32 LiteRT model with two signatures:

- `encode`: takes a `float32[1,80,3000]` log-Mel input and returns encoder states `float32[1,1500,512]`.
- `decode`: takes the encoder states `float32[1,1500,512]`, token ids `int32[1,128]` and a mask `float32[1,1,128,128]`, and returns logits `float32[1,128,51865]`.

## Moonshine speech recognition model

Moonshine models are automatic speech recognition models trained and released by Useful Sensors (now Moonshine AI), described in the paper "Moonshine: Speech Recognition for Live Transcription and Voice Commands" (arXiv 2410.15608) and released in October 2024 under the MIT license. They transcribe English speech audio into English text. Useful Sensors developed them to support real-time speech transcription products based on low-cost hardware. There are two English-only sizes: tiny with 27 M parameters and base with 61 M parameters.

The primary intended users are AI developers who want to deploy English speech recognition on platforms that are severely constrained in memory capacity and computational resources. The models are trained on 200,000 hours of audio and transcripts collected from the internet, plus openly available datasets on Hugging Face. According to the Moonshine paper, Moonshine Tiny matches Whisper tiny.en word error rates across standard evaluation datasets at about 5x less compute. Like Whisper, Moonshine can hallucinate and repeat text, and this may be worse for short audio segments or segments where parts of words are cut off at the beginning or end. To avoid hallucination loops, the reference code limits generated length to about 6.5 tokens per second of audio.

## Moonshine Tiny as a LiteRT model

The Hugging Face repo `litert-community/moonshine-tiny` packages Moonshine Tiny for LiteRT under the MIT license: a float32 model (`moonshine_tiny_5s_f32.tflite`, 109 MB), an int8 model (`moonshine_tiny_5s_i8.tflite`, 52 MB, a float32 encoder with a dynamic-range int8 decoder), and ahead-of-time compiled variants for a range of MediaTek and Qualcomm SoCs so the model can run on the device NPU. Each file has two signatures:

- `encode`: raw audio `[1, 80000]` float32 (5 s at 16 kHz, zero-padded) to encoder states `[1, 207, 288]`.
- `decode`: states, tokens `[1, 64]` int32 and an additive causal mask `[1, 1, 64, 64]` to logits `[1, 64, 32768]`.

The audio frontend is inside the graph, so the model takes a raw 16 kHz waveform in `[-1, 1]` and needs no mel-spectrogram extraction. The window is fixed at 5 seconds: longer audio is transcribed in consecutive 5 s windows, and shorter audio is zero-padded. Decoding is greedy, with start token 1, EOS token 2 and at most 64 tokens per window. The decoder re-scores the full token buffer each step (no KV cache), so decode time grows with the number of emitted tokens. The tokenizer (`tokenizer.json`) is loaded from the base model repository.

## Moonshine Tiny performance on CPU, GPU and NPU

Measured on one 5 s window of continuous speech (11 output tokens), CPU inference, median of 10 runs:

| Device | Variant | Window total | Real-time factor |
| --- | --- | --- | --- |
| iPhone 17 Pro | f32 | 81.2 ms | 0.016 |
| iPhone 17 Pro | i8 | 80.0 ms | 0.016 |
| Apple M4 Max (macOS) | f32 | 87.5 ms | 0.017 |
| Raspberry Pi 5 | f32 | 495.3 ms | 0.099 |
| Raspberry Pi 5 | i8 | 317.7 ms | 0.064 |

The real-time factor is processing time divided by audio duration; below 1.0 is faster than real time. Decode dominates and scales with the number of emitted tokens. The i8 model is about 1.6x faster than f32 on the Pi 5, while on Apple silicon the two are equally fast. On a Samsung Galaxy S26 (Snapdragon 8 Elite Gen 5) with LiteRT `CompiledModel` 2.2.0, the GPU running the f32 file encodes the 5 s window in 5.4 ms, the only fast accelerated path on that device; the Hexagon NPU runs the i8 encoder in about 1.26 s. The encoder is kept in float32 deliberately, because the convolutional audio frontend on the raw waveform does not survive dynamic-range quantization.

## The LiteRT speech recognition sample app

The litert-samples repository has an Android sample that runs open-weight ASR models with LiteRT on CPU, GPU, Google Tensor TPU and Qualcomm or MediaTek NPUs. Supported models are Parakeet TDT, Parakeet CTC, Parakeet TDT-CTC for Japanese, Moonshine, Whisper and Qwen3-ASR. Moonshine and Whisper run on CPU and GPU; Parakeet TDT also runs on the Pixel 10 TPU and Parakeet CTC on the Galaxy S23 and S24 NPU. A `convert` folder holds Python tools to convert PyTorch models to `.tflite` (with dynamic range quantization), compile them for NPUs, and verify them against the reference models. All models are pre-converted and AOT-compiled for NPUs and uploaded to litert-community; for a production app, Play AI packs are suggested for distributing models.

### How the sample processes audio

None of the supported models are streaming models; they get audio in overlapping windows. From a file, each chunk is 5 seconds with 2 seconds of overlap; from the microphone, each chunk is 5 seconds with 4 seconds of overlap, and the first chunks are pre-padded with silence to show text as early as possible. Except Moonshine, which takes raw audio, all models take log-Mel spectrogram input. Encoder outputs are passed to the decoder with zero copy by reusing the output `TensorBuffer`s. Parakeet TDT decodes statefully with cached LSTM states, while Moonshine, Whisper and Qwen3-ASR decode statelessly, feeding all tokens decoded so far; for Qwen3-ASR the sample notes that LiteRT-LM, with its KV cache management, would be the better API. Because chunks overlap, duplicated tokens are aligned with timestamps or a Levenshtein edit-distance heuristic and then merged.

## Inflect-Nano-v2 text-to-speech for LiteRT

The Hugging Face repo `litert-community/Inflect-Nano-v2` is a LiteRT conversion of `owensong/Inflect-Nano-v2`, a small English text-to-speech model with 4.0 million parameters, licensed Apache-2.0. It is a VITS-family end-to-end model with one fixed male voice that outputs 24 kHz audio. The conversion re-authors the VITS inference graph in TensorFlow, loads the upstream PyTorch weights, and converts it with the official `TFLiteConverter`, keeping both sequence axes dynamic so one graph handles any sentence length. It targets small CPUs through XNNPACK.

| File | Stage | Precision | Size |
| --- | --- | --- | --- |
| `inflect_text_encoder.tflite` | phoneme tokens to the acoustic distribution and log-durations | fp32 | 3.5 MB |
| `inflect_decoder.tflite` | latent frames to the 24 kHz waveform | fp32 | 12.6 MB |
| `inflect_text_encoder_fp16.tflite` | the same encoder with fp16 weights | fp16 | 1.8 MB |
| `inflect_decoder_fp16.tflite` | the same decoder with fp16 weights | fp16 | 6.4 MB |

Between the two graphs the host does the glue: durations come from the encoder's log-durations, the means and log-variances are repeated per frame, and the latent is sampled with noise generated on the host for reproducibility. The decoder is fully convolutional with no normalisation layers, so decoding in overlapping chunks and discarding the overlap reproduces the full decode exactly, which gives true streaming within a sentence. The repo also ships `inflect_decoder_static228.tflite`, a decoder for fixed 228-frame chunks, for accelerators that need static shapes.

### Inflect-Nano-v2 speed and caveats

On a Raspberry Pi 5 with four CPU threads the fp32 graphs reach a real-time factor of about 0.111, with about 220 ms to the first audio — the speed class of Piper. The output matches the PyTorch reference (waveform correlation 1.000000), and streamed output matches the full decode. The fp16 variants run at the same speed but showed a real quality break on one test sentence, because the flow layers are sensitive to fp16; the card recommends deploying fp32. On a Snapdragon 8 Elite Gen 5 phone the Hexagon NPU runs the published decoder in about 121 ms and the encoder in about 25 ms, while the phone GPU compiles only the static 228-frame decoder, which it runs in about 8.7 ms.

The English text frontend is not part of the graphs. The upstream frontend phonemizes with espeak-ng, which is GPL-3.0; the card suggests running espeak as a separate process or using a neural grapheme-to-phoneme model for a GPL-free stack.

## Matcha-TTS text-to-speech for LiteRT

The Hugging Face repo `litert-community/Matcha-TTS` (MIT) provides on-device English text-to-speech for Android via the LiteRT `CompiledModel` API. Matcha-TTS pairs a conditional flow-matching acoustic model with a HiFi-GAN time-domain vocoder, so there is no FFT or iSTFT anywhere in the synthesis path. It outputs 22.05 kHz audio with the LJSpeech voice and uses fp16 weights, converted with litert-torch and re-authored to be clean for the ML Drift GPU delegate. It has four graphs: a text encoder (15 MB), a flow-matching decoder (23 MB), a vocoder (29 MB) and a neural grapheme-to-phoneme model (26 MB). Shapes are fixed at 256 phonemes and 512 mel frames, about 5.9 s of audio, with a runtime mask so one compiled graph handles any length. Grapheme-to-phoneme conversion avoids GPL espeak by using a 275k-entry espeak-IPA dictionary from OpenPhonemizer plus DeepPhonemizer on the CPU for out-of-dictionary words.

### Matcha-TTS performance and backend split

On an Apple M4 Max (CPU, XNNPACK, 8 threads), one 512-frame chunk takes 903 ms of graph time for 5.94 s of audio, a real-time factor of 0.15; the vocoder is 77% of the total. On a Pixel 8a the text encoder and vocoder run on the GPU and the decoder runs on the CPU, because the Mali ML Drift GPU delegate mis-fuses the decoder's transformer blocks; the pipeline stays real-time at a real-time factor of about 0.8. On a Samsung Galaxy S26, the Hexagon NPU ran the text encoder 1.49x and the decoder 1.46x faster than the GPU, and loaded them 5.89x and 10.04x faster, while the vocoder did not compile for the NPU.
