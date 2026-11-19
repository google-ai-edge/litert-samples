# Photo voice chat: LFM3-Omni-2B on LiteRT (CompiledModel API)

This sample lets you show an Android phone a photo, ask about it out loud, and hear the answer, then ask again about the same photo, all with no network connection: the photo and your voice never leave the phone. It takes a short video too: the model sees the video's pictures and hears its sound together, so you can ask, for example, what was on the screen when someone said something. The model is [LiquidAI/LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B) (LFM Open License v1.0), one model that sees the photo, hears the question and speaks the answer (no separate speech recognizer, vision model or voice), run with [LiteRT](https://github.com/google-ai-edge/litert)'s Compiled Model API.

People who cannot see a label well already photograph things to ask about them. [Be My Eyes](https://www.bemyeyes.com/about/) has over a million blind and low-vision users, and Google says the average TalkBack user comes across 90 unlabeled images a day ([Android Developers Blog, 2024](https://android-developers.googleblog.com/2024/09/talkback-uses-gemini-nano-to-increase-low-vision-accessibility.html)). Sending the photo to a person has a cost: in a survey of 155 visually impaired people, the authors note that people "can (inadvertently) reveal sensitive information" to the volunteers who answer ([Akter et al., USENIX Security 2020](https://www.usenix.org/conference/usenixsecurity20/presentation/akter)). Here the photo, the question and the answer stay on the phone.

## What you can give it

1. **A photo**: Camera takes one with the phone's camera app, Choose photo picks one from the phone, and the app also appears under "Open with" and "Share" for pictures from other apps. Three example photos are in the app (Fruit bowl, Coffee, Umbrella). The app looks at the center square of the photo (the square on the screen is what the model sees).
2. **A short video**: Choose video picks one from the phone, and the app also appears under "Open with" and "Share" for videos. Two example videos are in the app (Video 1, Video 2; below), each with a typed example question. The app reads the first 13 s of a video: a picture about every second (How it runs) and its sound; a longer video is read up to there, and the screen says so. A video without sound is read from its pictures only, and the screen says so too. The square plays the video (tap it to pause); the strip under it shows the pictures the model sees, each stretched to a square, with their times. The screen says "Reading Video 1…" (or the file's name), then each step ("Decoding the video: frame 3 of 12…", "Reading the video: picture 3 of 12…", "Reading the video: the prompt…"), until the video is ready for questions.
3. **A question, spoken**: hold Hold to ask while you speak and let go (or tap once to start and tap again to finish), up to 30 s. Each example photo has three example questions; tapping one plays its recording as if you asked it.
4. **A question, typed**: type it and tap Ask. An example video's question has no recording: tapping it asks it as a typed question, and the answer is spoken as for any typed question.
5. **New conversation**: starts over about the same photo or video. A new photo or video also starts a new conversation.

The answer's text appears as the model writes it and its speech plays as it is generated. The next question goes on from the answers before it ("What color is it?" after "What is this thing?"). A conversation is limited by the model's 4,096-token context; each three-question example conversation about a photo used 731–892 tokens of it, and a 12 s video takes 3,348 tokens before the first question, so a video's conversation fills up after an answer or two: the app then says so, and New conversation starts over about the same video. The answers can be wrong in the details: asked "What fruit is in this bowl?" about the example Fruit bowl, the model answered on a Galaxy S26 that there were five apples, where the photo shows seven. About a video, the model can name what was on the screen at another moment: of three such test videos (the two examples among them), the same prompt run on a Mac with the same model files named the right photo for the two examples, and for the third named the photo shown just before the line was spoken. The app asks for one permission, the microphone (for Hold to ask); it has no internet permission and does not ask for the camera (the phone's camera app takes the photo).

## How it runs

The prompt is the model card's for a spoken question about an image: the system prompt `Answer questions about the provided image. Base every answer only on what can be seen. Respond with interleaved text and audio.`, the user turn = the photo, then the question (as in the card's code example: nothing else is written), and the answer opened with `<think></think><|mixed_start|>` (the template's empty thinking, then text and speech). The model then writes six text tokens, then twelve speech frames (80 ms each), and so on until the text ends and the speech finishes. A conversation goes on in the language model's state, as the reference implementation's `generate` continues from its cache: the next question is added after the last answer, whose text and speech frames are already in the state, so a follow-up question costs only its own tokens.

A video is written into the prompt the way the model's processor writes a video with its sound (`use_audio_in_video`): pictures at two a second (at least 4, at most 384, at evenly spaced frames) in groups of two, so the model sees one picture a second; each group is its time (`<0.3 seconds>`, the middle of its two frames), the 256 rows of its first picture (the bundle's vision encoder takes one picture), then the soundtrack's rows from that group's first frame up to the next group's. The system prompt is `Answer questions about the video using what you see and hear. Respond with interleaved text and audio.`; the question follows the video as it follows a photo.

The speech path is a host loop around LiteRT graphs:

1. **Vision encoder and adapter**: the photo at 512 x 512 becomes 256 rows at the `<|image|>` positions (the bundle's two vision sections), on the GPU at float32 on a phone with a Qualcomm chip, at float16 on a phone with a Google Tensor G6, and on the CPU on other phones (Performance). This runs when you choose the photo, together with the system prompt, so a question only waits for itself. A video's pictures go the same way, one per group, at the `<|video|>` positions: each is the group's first frame (decoded at its own time with Android's MediaMetadataRetriever) stretched to 512 x 512 as the model's processor stretches it (Pillow's bicubic resize); `Video.kt` repeats that resize in Kotlin and gets the same bytes.
2. **Audio encoder**: the spoken question (16 kHz, up to 30 s) becomes the rows at the `<|audio|>` positions (the bundle's audio section). A video's soundtrack goes through it once, when the video is read: decoded with Android's MediaCodec (the encoder's start-up samples that the file's edit list skips are cut), its channels averaged, and resampled to 16 kHz with the filter of ffmpeg's resampler, which decoded the soundtrack for the Mac answers this README compares with.
3. **Language model**: the decoder and its token embedder, read from the sections of the `.litertlm` bundle (the file LiteRT-LM runs for text answers). The app copies the sections it needs into files of their own once, because LiteRT's Kotlin API opens a model by path; a graph's weights section goes into a file beside the graph. Each step returns the last hidden state and, before a text slot, the text logits; a speech step uses the bundle's `prefill_1` signature, which returns the hidden state without the text logits. The KV and conv state live in one set of buffers that each call updates in place.
4. **Depth decoder**: the hidden state of a speech step goes through `depth_frame2` once per frame, on the GPU at float16 on a phone with a Qualcomm chip and on the CPU on other phones (Performance). Inside the graph, three passes each reveal four of the twelve codebooks (MaskGIT order, schedule 4 + 4 + 4) and sample their codes; the app gives it the order, one random number per codebook and the codes used in the last 64 frames. Two guards keep a number close to 1 from landing on the last code, which ends the speech: the app keeps the numbers at or below 1 − 2^-11 (at float16 a larger number rounds to 1, and an earlier version of this graph then ended some speech early), and the graph compares the cumulative probabilities with the number times their total rather than with the number alone, so the total's rounding at float16 cannot push the pick past the last code.
5. **Frame embedding**: the twelve codes of a frame become the next input of the language model (the sum of their codebook embeddings).
6. **Codec decoder**: every twelve frames (0.96 s) become 24 kHz audio on a thread of its own while the model writes the next frames; the app overlap-adds the decoder's windowed frames and plays them as they come.

Everything between the graphs is Kotlin: the byte-level BPE tokenizer (inflated from the bundle), the chat prompt, the sampling settings of the model card (audio temperature 0.4, repetition penalty 1.2 over the last 64 frames, applied inside `depth_frame2`; text greedy), the text / speech turns of the reference implementation and the overlap-add.

| File | Role |
|---|---|
| `MainActivity.kt` | the screen: the photo or the video, the questions, the answers |
| `Conversation.kt` | a conversation about one photo or video: the photo or video prefilled, each question asked after the answers before it, the timings |
| `SpeechGenerator.kt` | the text / speech state machine, sampling, MaskGIT order |
| `LmRunner.kt` | prefill in chunks that the prefill signatures cover exactly, from the current position; decode; the state in place |
| `Parts.kt` | depth decoder, frame embedding, codec decoder + overlap-add, audio encoder, vision encoder |
| `Lfm3Tokenizer.kt` | `tokenizer.json` (byte-level BPE, the file's split regex, `ignore_merges`) and the chat prompt |
| `Lfm3OmniEngine.kt` | the files, the bundle's sections, the graphs and the memory check before each stage |
| `LitertlmHeader.kt` | what a `.litertlm` file's header lists |
| `Photo.kt` | a photo from a file, the camera or the assets: upright, the center square, 512 x 512 |
| `Video.kt` | a video from a file: the frames the model sees and their times, the soundtrack at 16 kHz, the resize and the resampler |
| `AudioIO.kt` | recording, checking (length, silence) and playing audio |
| `Examples.kt` | the example photos, videos and questions in the assets |
| `AutoRun.kt` | the measuring tool (below) |

## Files

The model files are on Hugging Face at [litert-community/LFM3-Omni-2B-LiteRT](https://huggingface.co/litert-community/LFM3-Omni-2B-LiteRT) (the four files below, 4.0 GB):

| File | Bytes | SHA-256 |
|---|---|---|
| `LFM3-Omni-2B_int8.litertlm` | 3,226,812,416 | `2bb3ecc360523afc420895f1d5fd1b490dbe93608add532bae9e004b880f6c56` |
| `speech/LFM3-Omni-2B_depth_frame2_f32.tflite` | 435,046,636 | `19646b570f453711a60add2523febb4884e3214e8c3b7f5ca898dc29dae3d60a` |
| `speech/LFM3-Omni-2B_frame_embed_f32.tflite` | 198,281,972 | `71b68529124ef770a495554ea6928f42a66c8c0f2756df2452b15b22bdbc707d` |
| `speech/LFM3-Omni-2B_codec_decoder_f12_f32.tflite` | 152,370,912 | `7134523fb54c2fd9b31697373b37e1a5bd1669c3a432ab7896a8a7d070f38ed3` |

```bash
pip install -U huggingface_hub
hf download litert-community/LFM3-Omni-2B-LiteRT --revision dd97115ec5ecaebe4e9aa9eeccd7a3b3a37f4fae --include "*.litertlm" --include "speech/*depth_frame2*" \
  --include "speech/*frame_embed*" --include "speech/*f12*" --local-dir lfm3_omni
```

(`hf` 1.x and later take one `--include` per pattern.) The bundle holds the language model (its weights rescaled per channel and quantized to int8) with a 4,096-token context, its embedder, the audio encoder, the vision encoder and adapter, and the repo's `tokenizer.json`. The language model's and the embedder's weights are in sections of their own. The app reads where each section is from the file's own header, so it runs either form of the bundle: this one, or the earlier one, which keeps them inside the graphs (`--revision 6d95b0222f6724130b7cfdff9e63ed65982bcb77`, 3,226,750,192 bytes, SHA-256 `95961797…`). The three graphs are float32.

## Build and install

Android Studio, or JDK 17 and the Android SDK (platform 35) on the command line; the Gradle wrapper fetches Gradle 8.11.1. minSdk 26, arm64-v8a only. Download the model files first (Files).

```bash
cd kotlin_cpu_gpu/android
./gradlew :app:installDebug
./install_to_device.sh <path to lfm3_omni>
```

With more than one phone on adb, set `ANDROID_SERIAL` to the phone's serial first: `./gradlew` and `adb` then use that phone only. `install_to_device.sh` pushes the four files into the app's external files dir (`/sdcard/Android/data/com.google.ai.edge.examples.photo_voice_chat/files`) with `adb push`. Open the app: the first start copies the bundle's sections it needs into files of their own, then says Ready (13 to 14 s in two first starts on a Galaxy S26; 40 s on a Pixel 8a with the earlier form of the bundle). Without the files, the app names the missing ones; push them, come back to the app, and it loads them.

Every part runs on the CPU (4 threads), as the model card of the LiteRT files does (its speech loop runs on the CPU, and it keeps the codec decoder there), except two on Android 12 or later (the version that names the chip's maker): the depth decoder runs on the GPU at float16 on a phone with a Qualcomm chip, and the vision encoder and adapter run on the GPU at float32 on a phone with a Qualcomm chip and at float16 on a phone with a Google Tensor G6. A GPU that cannot open a graph leaves it on the CPU, and the app's load record says why. The weights are packed into XNNPACK's cache files on the first start and mapped from them. With the earlier form of the bundle, the app's resident size after loading was 1.1 GB on a Galaxy S26 and on a Pixel 8a with every part on the CPU; with the depth decoder and the vision encoder on the S26's GPU it was 2.8–3.1 GB (three starts), and the GPU's allocations grew by 1.5 GB; with the vision encoder on the Pixel 11 Pro's GPU it was 1.3–1.8 GB (three starts). The vision encoder's compiled GPU programs are kept in the app's files as well (0.76 GB at float32, 0.38 GB at float16), so only the first start compiles them (Performance). Before each loading stage the app checks the free memory and stops with a message instead of starting a stage that does not fit.

## Performance

Galaxy S26 (Qualcomm SM8850, Android 16) and Pixel 11 Pro (Google Tensor G6, Android 17) in airplane mode, the language model and the other parts on the CPU with 4 threads. Umbrella's first example question, two answers with each depth decoder in one session (the second ones in the reverse order), the phone cooled to thermal status 0 and a battery temperature of 36 °C or below before each answer (October 31, 2026). Time to make one second of speech, in bold where this app runs the depth decoder:

| Depth decoder | Galaxy S26 | Pixel 11 Pro |
|---|---|---|
| `depth_frame2` on the GPU, float16 | **0.75 s, 0.75 s** | 1.76 s, 1.85 s |
| `depth_frame2` on the CPU | 0.99 s, 0.92 s | **1.62 s, 1.58 s** |
| `depth_frame2` on the GPU, float16 with float32 accumulation | 0.82 s, 0.77 s | 1.89 s, 1.95 s |
| `depth_frame2` on the GPU, float32 | 0.91 s, 0.87 s | 2.13 s, 2.10 s |
| `depth_frame` on the CPU | 0.90 s, 0.91 s | 1.58 s, 1.60 s |

`depth_frame` is an earlier version of `depth_frame2` that also returns the logits (not on Hugging Face). In another session the same day, `depth_step` three times per frame (the first version of this sample, and the form for other speech settings) took 0.98 s and 0.91 s against `depth_frame`'s 0.88 s and 0.91 s on the S26, and 1.67 s and 1.80 s against 1.56 s and 1.54 s on the Pixel 11 Pro.

On the S26 the GPU at float16 ran the depth decoder in 10.4 ms per frame, against 22–27 ms on the CPU; no answer paused, and the first sound came 2.0–2.2 s after the end of the question. Here and below, the first sound (when an answer started) is when the answer's audio starts to play; the model's speech starts with a near-silent lead (0.7–1.3 s in eight answers on the phones, 0.3–1.1 s in nine from the original model on a Mac), so its first word is heard that much later. Three answers in a row at the end of the session, with no cooling between them, took 0.81, 0.79 and 0.80 s per second of speech, with no pause. Opening the depth decoder on the GPU took 3.0–3.3 s when the app started (0.3–0.4 s on the CPU). The S26's NPU, with the depth decoder split into two graphs (not on Hugging Face) and an APK that carries LiteRT's Qualcomm NPU libraries (48 MB larger), took 0.77 s and 0.75 s: no faster than the GPU.

On the Pixel 11 Pro the GPU made the depth decoder only a little faster (32 ms per frame at float16, against 33–36 ms on the CPU), and while it ran there the language model's step on the CPU took 84–89 ms instead of 71–72 ms. The language model on the Pixel 11 Pro's GPU was slower too: 1.91 s per second of speech in two answers, against 1.58–1.60 s on the CPU (`depth_frame` on the CPU in both). So the app runs the depth decoder on the GPU at float16 on a phone with a Qualcomm chip (measured on one, the S26) and on the CPU on other phones. On the Pixel 11 Pro the speech is made slower than it plays: the language model's step alone takes 71–72 ms of each 80 ms frame, and each answer paused 5 to 10 times.

With the GPU at float16 the S26 gave a shorter answer to the same question (7.1 s of speech, against 12.4 s with the CPU and with the GPU at float32): the codes of each frame go back into the language model, float16 changes some of them, and the words after them can change. On the Pixel 11 Pro the words also changed between answers with the same depth decoder (five different answers among the ten). On the S26 the speech with `depth_frame2` on the GPU at float16 passed the same checks as with `depth_frame` on the CPU, over 114 answers made on the phone through the measuring tool's `prompt_ids`: 90 sentences read aloud (30 sentences, three seeds each) and 24 answers to 8 typed questions (three seeds each). Whisper large-v3 heard no word error in the 852 words read aloud, with either; UTMOS (a predicted listening score from 1 to 5) gave the read-aloud speech 4.48 against 4.47; and 21 of the 24 answers were right, with either.

With a video (November 1, 2026), reading a 12 s video, from choosing it until it is ready for a question, with the vision encoder and adapter on each phone's GPU and on its CPU: each example video read twice with each, in one session (the order CPU, float32, float16, float16, float32, CPU; the phone cooled before each reading). The mean of the four readings, with the vision encoder's time for the 12 pictures in parentheses, in bold where this app runs them:

| Vision encoder and adapter | Galaxy S26 | Pixel 11 Pro |
|---|---|---|
| GPU, float32 | **11.0 s** (2.60 s) | 26.0 s (5.41 s) |
| GPU, float16 | 10.5 s (1.35 s) | **23.6 s** (3.36 s) |
| CPU | 16.2 s (6.38 s) | 34.5 s (14.2 s) |

The rest of a reading is mostly the prompt's first 3,348 tokens, on the CPU in every row (6.2–9.2 s on the S26, 14.9–16.1 s on the Pixel 11 Pro). A photo is ready sooner too: its vision encoder and the prompt's start took 1.14 s against 1.68 s on the S26 and 1.95 s against 2.92 s on the Pixel 11 Pro. On the S26 float16 was faster still, but with it the spoken answers about Video 2 named the cat shown after the line instead of the microwave oven on the screen when it was spoken (all three), while its text-only answer named the oven; its vision rows were further from the CPU's than on the Pixel 11 Pro (the lowest cosine over the rows 0.70 against 0.81). So the app uses float32 on the S26 and float16 on the Pixel 11 Pro. Opening the vision encoder on the GPU compiles its programs: without the program cache that took 1.6–4.0 s on the S26 at float32 and 13.3–24.9 s on the Pixel 11 Pro at float16 at every open (0.06–0.13 s on the CPU). The app keeps them in its files, so only the first start compiles them (2.9–3.1 s and 14.2–14.3 s, writing them included), and later opens read them in 1.2–2.5 s and 1.0–1.7 s.

A spoken answer about a video then started 1.9–2.7 s after the question on the S26 and took 0.78–0.85 s to make each second of speech, with no pause, in twelve answers (each example video's question three times in each of two sessions, the phone cooled before each). On the Pixel 11 Pro seven spoken answers started 3.8–4.2 s after the question, took 1.54–2.38 s per second of speech, and paused 8 to 20 times. By Whisper's transcript, every one of them named the photo on the screen when the line was spoken.

The first version of this sample (`depth_step` three times) on the Galaxy S26 with the three example photos and their three example questions each (one conversation per photo, after one warm-up answer, the nine answers in a row, October 30, 2026):

| | Galaxy S26 |
|---|---|
| First sound after the end of the question | 3.1 s on average (2.5–3.7 s) |
| Time to make one second of speech | 1.15 s on average (0.88–1.30 s) |
| Length of a spoken answer | 2.6–14.1 s |
| Answers whose speech paused | 4 of 9 |

In that run the phone warmed up and the speech paused: all four answers with a pause came late, in the second and third conversations. More threads did not help: one answer took 1.25 s per second of speech with 6 threads and 1.21 s with 8, against 1.02–1.09 s with 4, and 2.0 s with the language model on the GPU; so the app keeps 4 threads on the CPU. On a Pixel 8a (Android 16, airplane mode) that version runs, but slowly: in 11 answers the model took 2.2–3.5 s to make each second of speech (2.8 s on average), the first sound came 5.1–10.3 s after the question, and the speech of 8 of the 11 answers paused.

These runs used the earlier form of the bundle. On a Galaxy S26, the two forms gave word-for-word the same answer to Umbrella's first example question.

## What was run on a phone

On a Galaxy S26, with the app built from this directory, once each: `./install_to_device.sh`, then the first example question with the bundle above and with its earlier form (the same answer); the app killed while it copied a section (the next start copied it again); a `.litertlm` file cut short, one that does not begin like a `.litertlm` file, and a whole file whose header no longer lists the audio encoder (each time, the app says the bundle is missing); one byte changed in a weights section (the app stops and names the section).

On a Galaxy S26 and on a Pixel 11 Pro, with the build before the app read weights sections, once each:

- `./gradlew :app:installDebug`, then `./install_to_device.sh` on a folder laid out as `hf download` leaves it, before the first start; the app opened from the launcher
- the measuring tool's example below (an example question, then a typed one); the load record: the depth decoder on the GPU at float16 and the vision encoder on the GPU at float32 on the S26, the depth decoder on the CPU and the vision encoder on the GPU at float16 on the Pixel 11 Pro
- the app force-stopped and started again from the launcher, twice (the vision encoder's GPU programs read back from the app's files)
- the measuring tool's failure checks (on the Pixel 11 Pro with the same build; on the S26 with the build before the vision encoder moved to the GPU): an answer asked of `depth_frame2` with audio temperature 0 (it stops and names the form that takes other settings), an answer asked of the `step` form, which the app had not opened (it stops and names the one it opened), and the app started without `depth_frame2` on the phone (it names the missing file)

With videos, on the Pixel 11 Pro, with the build before the vision encoder moved to the GPU (the same video code): the two example videos, each with its example question answered in text only (the measuring tool's `"answer": "text"`) and spoken; Choose video (the system photo picker), Open with and Share, each with a copy of Video 2 from the phone's videos, and a typed question about the one from Choose video. Each answer named the photo on the screen when the line was spoken (some also named the photos before and after it, in the video's order), and each prompt's token ids were the same as the Mac's rendering of the same prompt. With the build before the app read weights sections: on the Pixel 11 Pro, Video 2's example question, tapped (by Whisper's transcript, the answer named the microwave oven); on the Galaxy S26, each example video's question three times, spoken, with the same prompt token ids as on the Mac; by Whisper's transcript of the speech, all six answers named the right photo, and none paused.

On the Galaxy S26 and the Pixel 11 Pro, with a build of this sample from October 31 (the same speech code; the video input was added after it): the timed answers of the first table above, the S26's NPU answers with an APK that also carries the NPU libraries. On the Galaxy S26, with that build: the 114 answers of the checks above with each of the two depth decoders.

On a Pixel 8a, with the first version of this sample (October 30), once each:

- the steps above as written: `hf download`, `./gradlew :app:installDebug`, `./install_to_device.sh` before the first start, the app opened from the launcher
- an example question; a typed question; Stop; New conversation
- Hold to ask, and tap then tap again, with a question played into the microphone from a computer's speaker (both answers fit the question, also when transcribed)
- a question too short to use (0.45 s, through the measuring tool's `record` item)
- Camera (a photo taken with the phone's camera app); Choose photo (the system photo picker); Open with and Share (a JPEG from the phone's photos)
- a model file pushed after the app had named it missing
- the measuring tool's example below

On the Galaxy S26, with that version: the nine example questions of the second table above and three Umbrella answers of a later run on a cool phone (the first sound 2.8–3.1 s after the question, no pause).

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
- `"video": "<example video id>"`: an example video (`video_1`, `video_2`; a new conversation), or `"video_file": "<path in the app's files dirs>"`: a video file. With `"answer": "text"` its questions are answered in text only (greedy, the system prompt without "Respond with interleaved text and audio.", at most 200 tokens), the form of the text checks on the phone; `"play": false` does not play it in the square; `"dump": true` writes what was read to `files/dumps/` (each picture as decoded and as the model sees it, RGB bytes; the 16 kHz soundtrack, float32; how it was read, JSON)
- `"ask": "<example question id>"`: an example question (`umbrella_q1` and so on), its recording played first; a video's (`video_1_q1`, `video_2_q1`) is asked as typed
- `"ask_wav": "<path>"`: a question recording from a WAV file
- `"ask_text": "<question>"`: a typed question
- `"prompt_ids": [<token ids>]`: an answer to a prompt given as token ids (rendered off the phone with the model's chat template), with no photo and the speech not played; `"label"` is its line on the screen
- `"action": "record"`: Hold to ask, held for `"hold_ms"`
- `"action": "camera"`: the camera app (its shutter or BACK pressed by hand or by a script)
- `"action": "new_conversation"`
- `"action": "engine"`: the engine opened again, with other CPU thread counts (`"lm_threads": 6`, …), `"lm": "gpu"`, or another depth decoder: `"depth_form"` (the one answers use) and `"depth_forms"` (more opened beside it, comma-separated), each a form (`frame2`, `step`) on the CPU or a form and an accelerator (`frame2@gpu16`; `cpu`, `gpu32`, `gpu16a` = float16 with float32 accumulation, `gpu16`, `npu`). `"vision"` places the vision encoder and adapter by the same names (`cpu`, `gpu32`, `gpu16a`, `gpu16`). The `step` form's graph (`speech/*depth_step*` on Hugging Face) goes into the same folder with `adb push`. An `npu` decoder needs an APK built with LiteRT's Qualcomm NPU libraries in `app/src/main/jniLibs/arm64-v8a/`; without them, or on another phone, it opens on the CPU and the load record says why

An item can also set `"rest_ms"` (the pause after it) and `"snapshot": false` (no picture of the screen after it), and an ask can set `"instruction"`, `"seed"`, `"max_new"` (the most tokens of the answer) and, for checks, `"depth"` (an opened depth decoder for this answer), `"audio_temperature"` (other values than 0.4 need the `step` form), `"fixed_order"` and `"dump"` (the vision rows of the photo or of the video's pictures, the question's audio rows, a video's soundtrack rows and the token ids so far, float32 and JSON in `files/dumps/`). The report `files/<tag>.json` has every item's screen lines, the phone's free memory after it (`MemAvailable`), and each answer's record (text, the whole prompt's token ids, the time from the end of the question to the first sound out of the speaker, the length of the speech, the timings of each graph, AudioTrack's underruns); a photo item's record has the time of the vision encoder and of the prompt's start, a video item's record has how the video was read (the tracks, the frame times and the pictures' indices, the sound's decoding, the time of each step), and `files/out_<tag>.f32` the speech of every answer (24 kHz float32). Every answer, tapped or planned, also leaves its record in `files/runs/`.

## The example photos, videos and questions

The photos are CC0 photos from Wikimedia Commons, made square and 512 x 512 for the app. The questions were written for this sample and spoken by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0, revision `f3ff3571`), stored as 16 kHz mono WAV files.

| Example | Photo (Wikimedia Commons, CC0) | Voice | Questions |
|---|---|---|---|
| Fruit bowl | [Fruit bowl - Massachusetts.jpg](https://commons.wikimedia.org/wiki/File:Fruit_bowl_-_Massachusetts.jpg), Daderot | `af_heart` | What fruit is in this bowl? / Which one is red? / What color are the other apples? |
| Coffee | [Breakfast on a table (Unsplash).jpg](https://commons.wikimedia.org/wiki/File:Breakfast_on_a_table_(Unsplash).jpg), Katerina Pavlickova | `am_michael` | What's in the cup? / What is next to it? / What color is that? |
| Umbrella | [Folded Umbrella.jpg](https://commons.wikimedia.org/wiki/File:Folded_Umbrella.jpg), Swethasakthivel | `bf_emma` | What is this thing? / What color is it? / Is it open or closed? |

The two example videos were made for this sample: 12 s each (640 x 360, 30 frames a second, H.264 and AAC), three CC0 photos from Wikimedia Commons shown for 4 s each, one line spoken by Kokoro-82M on each photo (heard from 1 s into it), and a quiet noise under them (-55 dBFS). Each question asks about a line that is about something shown at another moment of the video (the laundry: the washing machine at 8–12 s; sleeping in the sun: the cat at 8–12 s), so the answer has to come from when the line was spoken, not from its words.

| Example | Photos, 0–4 s / 4–8 s / 8–12 s (Wikimedia Commons, CC0) | Voice | Lines | Question |
|---|---|---|---|---|
| Video 1 (`video_1.mp4`) | [Folded Umbrella.jpg](https://commons.wikimedia.org/wiki/File:Folded_Umbrella.jpg), Swethasakthivel / [Breakfast on a table (Unsplash).jpg](https://commons.wikimedia.org/wiki/File:Breakfast_on_a_table_(Unsplash).jpg), Katerina Pavlickova / [Domestic washing machine Germany.jpg](https://commons.wikimedia.org/wiki/File:Domestic_washing_machine_Germany.jpg), Spacekid | `am_michael` | "I should do the laundry tonight." / "Let me think." / "That's about it." | What was on the screen when the speaker said "I should do the laundry tonight"? |
| Video 2 (`video_2.mp4`) | [Fruit bowl - Massachusetts.jpg](https://commons.wikimedia.org/wiki/File:Fruit_bowl_-_Massachusetts.jpg), Daderot / [Microwave oven "Erisson".jpg](https://commons.wikimedia.org/wiki/File:Microwave_oven_%22Erisson%22.jpg), Karamanskaya / [Cat with an eye on the breakfast table, Pou des Lleó. Ibiza.jpg](https://commons.wikimedia.org/wiki/File:Cat_with_an_eye_on_the_breakfast_table,_Pou_des_Lle%C3%B3._Ibiza.jpg), Gerda Arendt | `af_bella` | "So, here we are." / "She loves sleeping in the sun." / "Anyway, moving on." | What was on the screen when the speaker said "She loves sleeping in the sun"? |

## License

The sample code is Apache-2.0. LFM3-Omni-2B's weights are under the LFM Open License v1.0 (LiquidAI); the converted files keep that license. The example photos and the photos in the example videos are CC0 (Wikimedia Commons, authors above); the example photo questions and the videos' lines were synthesized with Kokoro-82M (Apache-2.0) from text written for this sample.
