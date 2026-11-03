# LFM3-Audio-2B speech output: graphs and a host loop

The `.litertlm` file of this recipe answers in text through LiteRT-LM. Speech output runs outside LiteRT-LM: LiteRT
graphs and a loop on the host, laid out like the [Qwen3-TTS recipe](../../../../qwen/qwen3_tts/converted) (export
scripts next to a host runner).

## Why speech output is not in the `.litertlm` path

- LiteRT-LM's decoder loop works in token ids: `Decode` returns token ids, and the executor's inputs are token ids
  plus embedding rows for image and audio placeholders in the prompt (LiteRT-LM v0.19.0,
  `runtime/executor/llm_executor_base.h` lines 63–80; `runtime/executor/llm_executor_io_types.h`: `ExecutorTextData`
  line 166, `ExecutorVisionData` line 215, `ExecutorAudioData` line 276).
  In speech mode LFM3-Audio's next decoder input is not a token: it is the sum of the 12 codebook embeddings of the
  frame just spoken.
- Each 80 ms speech frame needs a depth decoder that reads the decoder's last hidden state and reveals the frame's 12
  codes in three sampled passes, and a codec decoder that turns codes into 24 kHz audio. In v0.19.0, speech output
  runs only through model-specific sessions (`omni/text2speech/kokoro.cc`, `omni/text2speech/qwen3_tts.cc`, and
  `omni/any2any/any2any_session.h`, whose `CreateWithInputSource` each model implements; the any2any model types list
  MiniCPM-o only). None of them is LFM3-Audio's loop.
- A litert-torch `ModelExtension` decides what is exported and packed. It does not add a decoding loop to the
  runtime, so the speech graphs are exported by `export_speech_graphs.py` and driven by `lfm3_audio_speech.py`.

## Files

| File | Signature: inputs → outputs | When it runs |
| --- | --- | --- |
| `lfm3_audio_depth_proj_r4b.tflite` | `depth_proj`: hidden [1,2048] → depth_in [1,12,1024] | once per speech frame |
| `lfm3_audio_depth_pass_r4b.tflite` | `depth_pass`: depth_in, codes [1,12] int32, masked [1,12] → logits [1,12,2017] | three passes per frame (4 + 4 + 4 codebooks) |
| `lfm3_audio_depth_pass_r4b_int8.tflite` | the same pass with int8 weights and float32 activations | in place of the float32 pass on a CPU |
| `lfm3_audio_frame_embed_f32.tflite` | `frame_embed`: codes [1,12] int32 → embeds [1,1,2048] | once per speech frame |
| `lfm3_audio_codec_decoder_f12_r4b2.tflite` | `decode`: codes [1,12,12], start [1], 11 state tensors → frames [1,72,1280] + state | every 12 frames (0.96 s) |
| `codec_window_f32.npy` | the codec's synthesis window (`decoder.istft.window`) | the host overlap-add |

The decoder itself is the `.litertlm` file's `PREFILL_DECODE` section: it returns the last hidden state as a second
output (`hidden`), which the depth decoder reads. The runner loads that section and the token embedder straight from
the `.litertlm` file, so text and speech share one copy of the decoder weights.

The file, module and class names (`r1b_graphs`, `r4b_parts`, `…R4b`) are the ones the published files were traced
with. litert-torch writes them into the tensor names, and keeping them keeps a rebuild byte-identical.

## Rebuild the graphs

```sh
python3.12 -m venv .venv-speech
.venv-speech/bin/pip install litert-torch==0.10.0 litert-converter==0.5.0 ai-edge-litert==2.3.0 \
    ai-edge-quantizer==0.10.0 torch==2.14.0 transformers==5.20.2 numpy==2.5.4 safetensors==0.8.0 \
    litert-lm-builder==0.18.1 tokenizers jinja2 soundfile
.venv-speech/bin/python export_speech_graphs.py --hf SRC --out-dir graphs
```

`SRC` is a local copy of [LiquidAI/LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) at
`9d49701c372d6011d900f6abb40c6b296fb706dd`. Each part runs in its own process and takes under a minute on a Mac
(peak 3.1 GB). `--ref-dir` compares the files with the published ones by sha256.

With these versions the five `.tflite` files come out byte for byte as published in
[litert-community/LFM3-Audio-2B-LiteRT](https://huggingface.co/litert-community/LFM3-Audio-2B-LiteRT). With the
toolchain of the `.litertlm` path (litert-torch built from main, litert-converter 0.6.0.dev20261027) the four float
graphs come out 160 to 1,204 bytes larger: the same buffers, tensors, operators, signatures and metadata, laid out
differently in the flatbuffer. The int8 pass comes out byte for byte as published with both toolchains. The package
line above, installed in a fresh venv, rebuilt all five files byte for byte.

The published NPU file (`lfm3_audio_depth_pass_r5b_aot.tflite`, the depth pass compiled ahead of time for the
Snapdragon SM8850) is not rebuilt here: an ahead-of-time compile writes different bytes every time, so the published
file is the one that was measured on the phone.

## Run speech output

```sh
.venv-speech/bin/python lfm3_audio_speech.py \
    --litertlm lfm3-audio-2b_int4_b128_c2048_h30.litertlm --graphs graphs --hf SRC \
    --text "The quick brown fox jumps over the lazy dog." --out fox.wav
```

The loop follows the source model's `generate`: `<|audio_start|>` speaks until a frame holds the end-of-audio code;
`--start mixed` with the system prompt `Respond with interleaved text and audio.` alternates 6 text tokens and 12
speech frames. The sampling defaults are the source card's (audio temperature 1.0, top-k 128, repetition penalty 1.1
over 4,096 frames); `--greedy` takes the argmax. `--depth-pass f32` uses the float32 pass, `--codec-accel gpu` the
GPU for the codec.

## What was measured

On a Mac (M4 Max, CPU, 2026-11-02), with the published `.litertlm` file and one English sentence of 21 words, seed 0
and the source card's sampling:

- The published graphs, the rebuilt graphs and the rebuilt float32 pass each spoke 9.1 s of audio (114 frames of 12
  codes), and Whisper base.en read all three back with no word errors.
- With one CPU thread, the published and the rebuilt graphs gave the same 122 frames of codes. With four threads the
  codes changed from run to run with the same files, so compare codes at one thread.
- With the `.litertlm` file this recipe builds, the same run at one thread gave the same 122 frames of codes as with
  the published file, and Whisper read the speech back with no word errors.
- Four threads took 0.36 to 0.43 s per second of speech, the first audio after about 0.6 s (single runs, not a
  benchmark).
- `--codec-accel gpu` (Metal) spoke the sentence with no word errors. `--start mixed` with a question wrote
  "The tallest mountain in the world is Mount Everest." and spoke the same words.

Phone numbers for these graphs are on the model card.
