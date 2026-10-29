# Voicemail summary: LFM3-Audio-2B on LiteRT (CompiledModel API)

This sample listens to an English voicemail on an Android phone and tells you, in your language, as text and speech, who called, what they want and the number to call back. It works with no network connection: the recording never leaves the phone. The model is [LiquidAI/LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) (LFM Open License v1.0), one speech-in, speech-out language model for the whole answer (no separate speech recognizer, translator or voice), run with [LiteRT](https://github.com/google-ai-edge/litert)'s Compiled Model API.

Most people let unknown numbers go to voicemail (67% in a 2020 [Pew Research Center](https://www.pewresearch.org/short-reads/2020/12/14/most-americans-dont-answer-cellphone-calls-from-unknown-numbers/) survey), and for the 67.8 million people in the U.S. who spoke a language other than English at home in 2019 ([U.S. Census Bureau](https://www.census.gov/library/stories/2022/12/languages-we-speak-in-united-states.html)), that English voicemail from a school or a clinic may not be in the language they read best. You choose the answer's language; English and Spanish are the ones measured here (Performance).

## What you can give it

1. **An example voicemail**: three English voicemails of 21 to 23 s are in the app (Voicemail 1 to 3).
2. **A recording**: Record takes up to 30 s from the microphone (your own voicemail, read aloud or played from another phone).
3. **A file**: Open file picks an audio file (WAV, FLAC or M4A; the phone's decoders read other formats too), and the app also appears under "Open with" and "Share" for audio files from other apps. A voicemail longer than 30 s is read up to its first 30 s, and the screen says so. The screen also says when the phone decodes less of a file than its stated length: on a Galaxy S26, the phone's FLAC reader returned 18.0 s of a 19.1 s FLAC test file (its last 1.05 s missing), while WAV and M4A files came through whole.
4. **Answer in**: the language of the answer, from the model card's speech languages (English by default). Choosing one puts that language's default instruction in the instruction field.
5. **The instruction**: the default asks for the caller, the reason for the call and the number to call back, in the chosen language. Edit it to ask something else about the voicemail.

Play plays the voicemail. Summarize answers: the text appears as the model writes it (the app renders the model's bold and list marks), the screen follows the end of the answer until you touch it, and the speech plays as it is generated. When the answer has a phone number written in digits, a "Call back" line shows it with a Call button, which opens the phone's dialer. The app asks for one permission, the microphone (for Record); it has no internet permission, so the recording stays on the phone.

## How it runs

The prompts are the model card's. An answer is made in one of two ways (`Answer.kt` sets which one a language uses): English answers are interleaved, the other languages use two stages, because in this sample's development runs on a Galaxy S26, Whisper heard English in the speech of all 6 interleaved answers to a Spanish request, including the 3 whose text was Spanish.

- **Interleaved**, the card's voice chat: the system prompt `Respond with interleaved text and audio.`, the user turn = the recording, then the instruction (the card's audio-understanding turn), the answer opened with `<|mixed_start|>`. The model answers with six text tokens, then twelve speech frames (80 ms of audio each), and so on until the text ends and the speech finishes.
- **Two stages of the same model**: the card's audio understanding (`Answer questions about the provided audio clip. Base every answer only on what can be heard.`, the same user turn, text only) writes the answer, then the card's text to speech (`Perform TTS in <locale> using a female voice.`) speaks it. For Spanish, the text that goes to the speech stage loses its markdown and has the phone number written in words the way it is said aloud, in groups ("555-0161" becomes "quinientos cincuenta y cinco, cero uno, sesenta y uno"), because in the development runs the model's text to speech garbled digit strings and looped on repeated digit words; the screen shows the answer as the model wrote it. The other languages go to the speech stage as written.

The speech path is a host loop around LiteRT graphs:

1. **Audio encoder**: 16 kHz audio, up to 30 s, becomes the language model's input rows at the `<|audio|>` positions (the bundle's audio section, on the GPU).
2. **Language model**: the decoder and its token embedder, read from the sections of the `.litertlm` bundle (the file LiteRT-LM runs for text answers). The app copies the three sections it needs into files of their own once, because LiteRT's Kotlin API opens a model by path. Each step returns the last hidden state and, before a text slot, the text logits; a speech step runs the bundle's `prefill_1` signature, which is `decode` without its tied text head. On the CPU the KV and conv state live in one set of buffers that each call updates in place.
3. **Depth decoder**: the hidden state of a speech step goes through `depth_proj` once and `depth_pass` three times; each pass reveals four of the twelve codebooks (MaskGIT order, schedule 4 + 4 + 4). With the NPU build on an SM8850 phone (Galaxy S26) the pass runs on the Hexagon NPU from an ahead-of-time compiled graph; otherwise on the CPU.
4. **Frame embedding**: the twelve codes of a frame become the next input of the language model (the sum of their codebook embeddings).
5. **Codec decoder**: every twelve frames (0.96 s) become 24 kHz audio on a thread of its own while the model writes the next frames; the app overlap-adds the decoder's windowed frames and plays them as they come.

Everything between the graphs is Kotlin: the byte-level BPE tokenizer (inflated from the bundle), the chat prompt, the sampling of the model card (audio temperature 1.0, top-k 128, repetition penalty 1.1 over 4,096 frames; text greedy), the text / speech turns of the reference implementation, the overlap-add, and the audio input (WAV in the app, other formats through MediaCodec, resampled to 16 kHz mono).

| File | Role |
|---|---|
| `MainActivity.kt` | the screen: the inputs, Play, Summarize, the answer, the Call back line |
| `Summarizer.kt` | one answer: the prompts of the two ways, the generation, the speech decoded as it comes, the timings |
| `Answer.kt` | the answer languages, their default instructions and which way each one is made |
| `SpeechGenerator.kt` | the text / speech state machine, sampling, MaskGIT order |
| `LmRunner.kt` | prefill (exact cover by the prefill signatures) and decode, the state in place (CPU) |
| `Parts.kt` | depth decoder, frame embedding, codec decoder + overlap-add, audio encoder |
| `Lfm3Tokenizer.kt` | `tokenizer.json` (byte-level BPE, the file's split regex, `ignore_merges`) and the chat prompt |
| `Lfm3AudioEngine.kt` | the files, the bundle's sections, the graphs, their accelerators and the memory check before each stage |
| `AudioIO.kt` | reading, checking (length, silence, the 30 s cut), recording and playing audio |
| `Callback.kt` | the phone number in an answer, and the measuring tool's checks (English and Spanish number words) |
| `Voicemails.kt` | the example voicemails in the assets |
| `AutoRun.kt` | the measuring tool (below) |

## Files

| File | Bytes | SHA-256 |
|---|---|---|
| `lfm3-audio-2b_int4_b128_c2048_h30.litertlm` | 1,531,232,624 | `5168b79a795d1737e789f22a84f5830e808f45a3ba2d3a02099b4d291bda4f02` |
| `lfm3_audio_depth_proj_r4b.tflite` | 100,713,664 | `66789aa754112d69fcc7bb857d317466af52af8b1686c1b327fa9fa11217476d` |
| `lfm3_audio_depth_pass_r4b.tflite` | 334,104,044 | `5f684777daf8c73dc4a95b314fe559cb329f8a7976791ed7ecfdfa78ae93dfe7` |
| `lfm3_audio_frame_embed_f32.tflite` | 198,281,752 | `d042be62b439630e55feed86bc9ae630db70674ba924d4752c4d5dfc3a44472d` |
| `lfm3_audio_codec_decoder_f12_r4b2.tflite` | 152,498,860 | `9dbbbcf870c1be4e166b77c1948ca4e34e8e069780236c15066f1d3c8ed52cbd` |
| `lfm3_audio_depth_pass_r5b_aot.tflite` (optional, Galaxy S26 NPU build) | 169,584,336 | `e3bdb0ed185190ba7b0881af250e585385e04360c6a614e712801287a8f1bfd4` |

The bundle holds the int4 (blockwise 128) language model with a 2,048-token context, its int8 embedder, the audio encoder with fp16 linear weights and the repo's `tokenizer.json`. The graphs are fp32; the optional sixth file is the depth pass compiled ahead of time for the SM8850's Hexagon NPU. The suffixes in the file names are build labels; the app reads these names as they are.

The files are on Hugging Face at [litert-community/LFM3-Audio-2B-LiteRT](https://huggingface.co/litert-community/LFM3-Audio-2B-LiteRT) (2.5 GB):

```bash
pip install -U huggingface_hub
hf download litert-community/LFM3-Audio-2B-LiteRT --include "*.litertlm" --include "*.tflite" --local-dir lfm3_audio
```

(One `--include` per pattern: the form of `hf` 1.x and later.)

## Build and install

Android Studio, or JDK 17 on the command line; the Gradle wrapper fetches Gradle 8.11.1. minSdk 26, arm64-v8a only. Download the model files first (Files).

```bash
cd kotlin_cpu_gpu/android
./gradlew :app:installDebug
./install_to_device.sh <path to lfm3_audio>
```

`install_to_device.sh` pushes the files into the app's external files dir (`/sdcard/Android/data/com.google.ai.edge.examples.voicemail_summary/files`) with `adb push`. Open the app: the first start copies three of the bundle's sections into files of their own (1.0 s on a Galaxy S26; the whole start took 5.5 s), then says Ready. Without the files, the app names the missing ones and Summarize stays off.

Accelerators: the language model, the depth projection and the frame embedding run on the CPU (4 threads), the codec decoder and the audio encoder on the GPU in fp32 (each falls back to the CPU if its GPU path cannot be created). Before each loading stage the app checks the phone's free memory and stops with a message instead of starting a stage that does not fit.

### NPU (Galaxy S26)

The depth decoder's pass runs on the Hexagon NPU of an SM8850 phone from the ahead-of-time compiled graph. The QNN runtime libraries are Qualcomm's and are not in this repository: build with LiteRT 2.3.0's Qualcomm dispatch library (Maven `litert-npu-runtime-qualcomm`) and a local copy of the QNN libraries from the Qualcomm AI Runtime (QAIRT) 2.50 SDK:

```bash
./gradlew :app:installDebug -Pnpu.libs=<dir with libQnnHtp.so, libQnnSystem.so, libQnnHtpV81Stub.so, libQnnHtpV81Skel.so, ...>
```

The NPU build uses the NPU when the phone is an SM8850 and the compiled file is there, and falls back to the CPU pass otherwise.

## Performance

Galaxy S26 (SM8850, Android 16), airplane mode, the NPU build, Answer in = Spanish, the three example voicemails, one answer each after a warm-up answer, seed 0:

| | Voicemail 1 | Voicemail 2 | Voicemail 3 |
|---|---|---|---|
| Tap to the first sound of the answer | 5.28 s | 4.68 s | 5.37 s |
| Length of the spoken answer | 22.9 s | 17.4 s | 20.2 s |
| Time of the whole answer per second of speech | 0.59 s | 0.61 s | 0.62 s |
| Caller's name and the number in the text | yes | yes | yes |
| The number heard in the speech (Whisper base, Spanish) | yes | yes | yes |

The first sound comes about 5 s after the tap (5.1 s on average): a Spanish answer is written as text first (3.3 to 3.9 s on the CPU), then spoken (about 1.4 s more to the first audio). An English answer is interleaved and starts sooner: 2.1 to 2.5 s in three development runs on the same phone. Loading takes 3.6 s (5.5 s on the first start, which copies the bundle's sections out once), and the app then holds 3.1 to 3.2 GB of memory. The default build (no NPU libraries) runs the depth decoder's pass on the CPU; on the same phone, one Spanish answer to a development voicemail started speaking 4.4 s after the tap and took 0.75 s per second of speech.

## The measuring tool

The same activity started with a plan runs the screen without a finger, through the functions the buttons call, and writes a report:

```bash
adb shell am start -n com.google.ai.edge.examples.voicemail_summary/.MainActivity \
  --es plan plans/example.json --es report example.json
```

The plan (in the app's files dir) lists items `{"id", "voicemail": "<example id>" | "file": "<path in the app's files dirs>", "answer_in", "instruction", "path", "press", "seed", "action": "record", "rest_ms", "dump_input"}`; an item without `press` only sets the screen, and `dump_input` writes the decoded input (16 kHz mono float32) next to the report. The report `<tag>.json` has every item's screen lines (the input, the notice, the answer), each answer's record (text, the number shown, the time from the tap to the first sound out of the speaker, the length of the speech, the real-time factor, the timings of each graph, AudioTrack's underruns, and for the examples a check that the caller's name and the callback number are in the text), and `out_<tag>.f32` the speech of every answer (24 kHz float32). Every answer, tapped or planned, also leaves its record in `files/runs/`.

## The example voicemails

The three voicemails were written for this sample; the names, the companies and the numbers are fictional (555-0100 to 555-0199 are reserved for fiction). They were spoken by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0, revision `f3ff3571`) and stored as 16 kHz mono WAV files.

| Example | Voice | Transcript |
|---|---|---|
| Voicemail 1 | `af_heart` | Hi, this is Emily Carter calling from Lakeside Dental. I'm calling to confirm your cleaning appointment this Thursday at two thirty in the afternoon. If that time no longer works for you, please call us back at five five five, oh one four seven, so we can find a new time. Again, that's five five five, oh one four seven. Thank you, and have a great day. |
| Voicemail 2 | `am_michael` | Good morning, this is Daniel Foster from Hillcrest Plumbing. We have you down for a visit tomorrow to look at the leak under your kitchen sink. Our plumber can be there between nine and eleven in the morning. Could you please call me back at five five five, oh one two six, to confirm that someone will be home? Many thanks, bye. |
| Voicemail 3 | `af_bella` | Hi, this is Jennifer Brooks from Oakwood Elementary School, calling about your son, Jacob. He's not feeling well and is resting in the nurse's office right now. It's nothing serious, but we think he should go home early today. Please call the school office back at five five five, oh one eight three, as soon as you can. Thank you. |

## License

The sample code is Apache-2.0. LFM3-Audio-2B's weights are under the LFM Open License v1.0 (LiquidAI); the converted files keep that license. The example voicemails were synthesized with Kokoro-82M (Apache-2.0) from scripts written for this sample.
