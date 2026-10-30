# Photo voice chat: LFM3-Omni-2B on LiteRT (CompiledModel API)

This sample lets you show an Android phone a photo, ask about it out loud, and hear the answer, then ask again about the same photo, all with no network connection: the photo and your voice never leave the phone. The model is [LiquidAI/LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B) (LFM Open License v1.0), one model that sees the photo, hears the question and speaks the answer (no separate speech recognizer, vision model or voice), run with [LiteRT](https://github.com/google-ai-edge/litert)'s Compiled Model API.

People who cannot see a label well already photograph things to ask about them. [Be My Eyes](https://www.bemyeyes.com/about/) has over a million blind and low-vision users, and Google says the average TalkBack user comes across 90 unlabeled images a day ([Android Developers Blog, 2024](https://android-developers.googleblog.com/2024/09/talkback-uses-gemini-nano-to-increase-low-vision-accessibility.html)). Sending the photo to a person has a cost: in a survey of 155 visually impaired people, the authors note that people "can (inadvertently) reveal sensitive information" to the volunteers who answer ([Akter et al., USENIX Security 2020](https://www.usenix.org/conference/usenixsecurity20/presentation/akter)). Here the photo, the question and the answer stay on the phone.

## What you can give it

1. **A photo**: Camera takes one with the phone's camera app, Choose photo picks one from the phone, and the app also appears under "Open with" and "Share" for pictures from other apps. Three example photos are in the app (Fruit bowl, Coffee, Umbrella). The app looks at the center square of the photo (the square on the screen is what the model sees).
2. **A question, spoken**: hold Hold to ask while you speak and let go (or tap once to start and tap again to finish), up to 30 s. Each example photo has three example questions; tapping one plays its recording as if you asked it.
3. **A question, typed**: type it and tap Ask.
4. **New conversation**: starts over about the same photo. A new photo also starts a new conversation.

The answer's text appears as the model writes it and its speech plays as it is generated. The next question goes on from the answers before it ("What color is it?" after "What is this thing?"). A conversation is limited by the model's 4,096-token context; each three-question example conversation used 731–892 tokens of it. The answers can be wrong in the details: asked "What fruit is in this bowl?" about the example Fruit bowl, the model answered on a Galaxy S26 that there were five apples, where the photo shows seven. The app asks for one permission, the microphone (for Hold to ask); it has no internet permission and does not ask for the camera (the phone's camera app takes the photo).

## How it runs

The prompt is the model card's for a spoken question about an image: the system prompt `Answer questions about the provided image. Base every answer only on what can be seen. Respond with interleaved text and audio.`, the user turn = the photo, then the question (as in the card's code example: nothing else is written), and the answer opened with `<think></think><|mixed_start|>` (the template's empty thinking, then text and speech). The model then writes six text tokens, then twelve speech frames (80 ms each), and so on until the text ends and the speech finishes. A conversation goes on in the language model's state, as the reference implementation's `generate` continues from its cache: the next question is added after the last answer, whose text and speech frames are already in the state, so a follow-up question costs only its own tokens.

The speech path is a host loop around LiteRT graphs:

1. **Vision encoder and adapter**: the photo at 512 x 512 becomes 256 rows at the `<|image|>` positions (the bundle's two vision sections). This runs when you choose the photo, together with the system prompt, so a question only waits for itself.
2. **Audio encoder**: the spoken question (16 kHz, up to 30 s) becomes the rows at the `<|audio|>` positions (the bundle's audio section).
3. **Language model**: the decoder and its token embedder, read from the sections of the `.litertlm` bundle (the file LiteRT-LM runs for text answers). The app copies the five sections it needs into files of their own once, because LiteRT's Kotlin API opens a model by path. Each step returns the last hidden state and, before a text slot, the text logits; a speech step uses the bundle's `prefill_1` signature, which returns the hidden state without the text logits. The KV and conv state live in one set of buffers that each call updates in place.
4. **Depth decoder**: the hidden state of a speech step goes through `depth_step` three times; each pass reveals four of the twelve codebooks (MaskGIT order, schedule 4 + 4 + 4).
5. **Frame embedding**: the twelve codes of a frame become the next input of the language model (the sum of their codebook embeddings).
6. **Codec decoder**: every twelve frames (0.96 s) become 24 kHz audio on a thread of its own while the model writes the next frames; the app overlap-adds the decoder's windowed frames and plays them as they come.

Everything between the graphs is Kotlin: the byte-level BPE tokenizer (inflated from the bundle), the chat prompt, the sampling of the model card (audio temperature 0.4, repetition penalty 1.2 over the last 64 frames; text greedy), the text / speech turns of the reference implementation and the overlap-add.

| File | Role |
|---|---|
| `MainActivity.kt` | the screen: the photo, the questions, the answers |
| `Conversation.kt` | a conversation about one photo: the photo prefilled, each question asked after the answers before it, the timings |
| `SpeechGenerator.kt` | the text / speech state machine, sampling, MaskGIT order |
| `LmRunner.kt` | prefill in chunks that the prefill signatures cover exactly, from the current position; decode; the state in place |
| `Parts.kt` | depth decoder, frame embedding, codec decoder + overlap-add, audio encoder, vision encoder |
| `Lfm3Tokenizer.kt` | `tokenizer.json` (byte-level BPE, the file's split regex, `ignore_merges`) and the chat prompt |
| `Lfm3OmniEngine.kt` | the files, the bundle's sections, the graphs and the memory check before each stage |
| `Photo.kt` | a photo from a file, the camera or the assets: upright, the center square, 512 x 512 |
| `AudioIO.kt` | recording, checking (length, silence) and playing audio |
| `Examples.kt` | the example photos and questions in the assets |
| `AutoRun.kt` | the measuring tool (below) |

## Files

The model files are on Hugging Face at [litert-community/LFM3-Omni-2B-LiteRT](https://huggingface.co/litert-community/LFM3-Omni-2B-LiteRT) (the four files below, 4.0 GB):

| File | Bytes | SHA-256 |
|---|---|---|
| `LFM3-Omni-2B_int8.litertlm` | 3,226,750,192 | `959617973a7ddcb3d887fa0d9891d79556a266490d5c432b02a18ffcac3124ed` |
| `speech/LFM3-Omni-2B_depth_step_f32.tflite` | 434,852,308 | `9e61c47319d8907a60aebeac9605b9b4df3ea3aadb232b708897263d4ef07a21` |
| `speech/LFM3-Omni-2B_frame_embed_f32.tflite` | 198,281,972 | `71b68529124ef770a495554ea6928f42a66c8c0f2756df2452b15b22bdbc707d` |
| `speech/LFM3-Omni-2B_codec_decoder_f12_f32.tflite` | 152,370,912 | `7134523fb54c2fd9b31697373b37e1a5bd1669c3a432ab7896a8a7d070f38ed3` |

```bash
pip install -U huggingface_hub
hf download litert-community/LFM3-Omni-2B-LiteRT --include "*.litertlm" --include "speech/*depth_step*" \
  --include "speech/*frame_embed*" --include "speech/*f12*" --local-dir lfm3_omni
```

(`hf` 1.x and later take one `--include` per pattern.) The bundle holds the language model (its weights rescaled per channel and quantized to int8) with a 4,096-token context, its embedder, the audio encoder, the vision encoder and adapter, and the repo's `tokenizer.json`. The three graphs are float32.

## Build and install

Android Studio, or JDK 17 and the Android SDK (platform 35) on the command line; the Gradle wrapper fetches Gradle 8.11.1. minSdk 26, arm64-v8a only. Download the model files first (Files).

```bash
cd kotlin_cpu_gpu/android
./gradlew :app:installDebug
./install_to_device.sh <path to lfm3_omni>
```

With more than one phone on adb, set `ANDROID_SERIAL` to the phone's serial first: `./gradlew` and `adb` then use that phone only. `install_to_device.sh` pushes the four files into the app's external files dir (`/sdcard/Android/data/com.google.ai.edge.examples.photo_voice_chat/files`) with `adb push`. Open the app: the first start copies five of the bundle's sections into files of their own, then says Ready (40 s on a Pixel 8a). Without the files, the app names the missing ones; push them, come back to the app, and it loads them.

Every part runs on the CPU (4 threads), as the model card of the LiteRT files does (its speech loop runs on the CPU, and it keeps the codec decoder there). The weights are packed into XNNPACK's cache files on the first start and mapped from them, so the app takes little of the phone's free memory (its resident size was 1.1 GB on a Galaxy S26 and on a Pixel 8a). Before each loading stage the app checks the free memory and stops with a message instead of starting a stage that does not fit.

## Performance

Galaxy S26 in airplane mode, every part on the CPU with 4 threads, the three example photos with their three example questions each (one conversation per photo, after one warm-up answer, October 2026):

| | Galaxy S26 |
|---|---|
| First sound after the end of the question | 3.1 s on average (2.5–3.7 s) |
| Time to make one second of speech | 1.15 s on average (0.88–1.30 s) |
| Length of a spoken answer | 2.6–14.1 s |
| Answers whose speech paused | 4 of 9 |

On the S26's CPU the speech is made a little slower than it plays (1.15 s of work per second of speech), so the speech pauses often once the phone is warm (all four answers with a pause came late in the run, in the second and third conversations). On a cool phone it kept up: in a later run the first sound came 3.1, 2.8 and 3.0 s after the question in the three Umbrella answers, and the speech did not pause. More threads did not help: one answer took 1.25 s per second of speech with 6 threads and 1.21 s with 8, against 1.02–1.09 s with 4, and 2.0 s with the language model on the GPU; so the app keeps 4 threads on the CPU. On a Pixel 8a (Android 16, airplane mode, the app built from this directory) the app runs, but slowly: in 11 answers the model took 2.2–3.5 s to make each second of speech (2.8 s on average), the first sound came 5.1–10.3 s after the question, and the speech of 8 of the 11 answers paused.

## What was run on a phone

On the Pixel 8a, with the app built from this directory, once each:

- the steps above as written: `hf download`, `./gradlew :app:installDebug`, `./install_to_device.sh` before the first start, the app opened from the launcher
- an example question; a typed question; Stop; New conversation
- Hold to ask, and tap then tap again, with a question played into the microphone from a computer's speaker (both answers fit the question, also when transcribed)
- a question too short to use (0.45 s, through the measuring tool's `record` item)
- Camera (a photo taken with the phone's camera app); Choose photo (the system photo picker); Open with and Share (a JPEG from the phone's photos)
- a model file pushed after the app had named it missing
- the measuring tool's example below

On the Galaxy S26: the nine example questions of the table above and the three Umbrella answers of the later run.

## The measuring tool

The same activity started with a plan runs the screen without a finger, through the functions the buttons call, and writes a report. The plan is a JSON file in the app's internal files dir (`/data/data/com.google.ai.edge.examples.photo_voice_chat/files/plans/`), copied there with `adb shell run-as` (the debug build allows it), for example `example.json` = `{"items": [{"id": "photo", "photo": "umbrella"}, {"id": "q1", "ask": "umbrella_q1"}, {"id": "q2", "ask_text": "What color is it?"}]}`:

```bash
adb push example.json /data/local/tmp/example.json
adb shell run-as com.google.ai.edge.examples.photo_voice_chat mkdir -p files/plans
adb shell run-as com.google.ai.edge.examples.photo_voice_chat cp /data/local/tmp/example.json files/plans/example.json
adb shell am start -n com.google.ai.edge.examples.photo_voice_chat/.MainActivity \
  --es plan plans/example.json --es report example.json
```

The report is written when the plan has run (each answer takes as long as on the screen); then copy it out:

```bash
adb exec-out run-as com.google.ai.edge.examples.photo_voice_chat cat files/example.json > example_report.json
```

The items run in order. Each has an `"id"` and one of:

- `"photo": "<example id>"`: an example photo (`fruit_bowl`, `coffee`, `umbrella`; a new conversation)
- `"photo_file": "<path in the app's files dirs>"`: a photo file
- `"ask": "<example question id>"`: an example question (`umbrella_q1` and so on), its recording played first
- `"ask_wav": "<path>"`: a question recording from a WAV file
- `"ask_text": "<question>"`: a typed question
- `"action": "record"`: Hold to ask, held for `"hold_ms"`
- `"action": "camera"`: the camera app (its shutter or BACK pressed by hand or by a script)
- `"action": "new_conversation"`
- `"action": "engine"`: the engine opened again, with other CPU thread counts (`"lm_threads": 6`, …) or `"lm": "gpu"`

An item can also set `"rest_ms"` (the pause after it), and an ask can set `"instruction"`, `"seed"` and, for checks, `"audio_temperature"`, `"fixed_order"` and `"dump"`. The report `files/<tag>.json` has every item's screen lines and each answer's record (text, the time from the end of the question to the first sound out of the speaker, the length of the speech, the timings of each graph, AudioTrack's underruns), and `files/out_<tag>.f32` the speech of every answer (24 kHz float32). Every answer, tapped or planned, also leaves its record in `files/runs/`.

## The example photos and questions

The photos are CC0 photos from Wikimedia Commons, made square and 512 x 512 for the app. The questions were written for this sample and spoken by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0, revision `f3ff3571`), stored as 16 kHz mono WAV files.

| Example | Photo (Wikimedia Commons, CC0) | Voice | Questions |
|---|---|---|---|
| Fruit bowl | [Fruit bowl - Massachusetts.jpg](https://commons.wikimedia.org/wiki/File:Fruit_bowl_-_Massachusetts.jpg), Daderot | `af_heart` | What fruit is in this bowl? / Which one is red? / What color are the other apples? |
| Coffee | [Breakfast on a table (Unsplash).jpg](https://commons.wikimedia.org/wiki/File:Breakfast_on_a_table_(Unsplash).jpg), Katerina Pavlickova | `am_michael` | What's in the cup? / What is next to it? / What color is that? |
| Umbrella | [Folded Umbrella.jpg](https://commons.wikimedia.org/wiki/File:Folded_Umbrella.jpg), Swethasakthivel | `bf_emma` | What is this thing? / What color is it? / Is it open or closed? |

## License

The sample code is Apache-2.0. LFM3-Omni-2B's weights are under the LFM Open License v1.0 (LiquidAI); the converted files keep that license. The example photos are CC0 (Wikimedia Commons, authors above); the example questions were synthesized with Kokoro-82M (Apache-2.0) from text written for this sample.
