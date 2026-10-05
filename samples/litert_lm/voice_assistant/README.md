# Voice assistant on Android with LiteRT and LiteRT-LM

## What this sample demonstrates

- A phone that hears a spoken request, acts on it with its own tools and answers aloud. Every model runs on the phone: once the models are downloaded, no request needs the network.
- Three models in one app: Zipformer CTC speech recognition on [LiteRT](https://github.com/google-ai-edge/litert) `CompiledModel` (GPU), Gemma 4 E2B with tool calling on [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) (GPU), and KittenTTS speech synthesis on LiteRT (CPU).
- "Set an alarm for six forty five tomorrow morning." sets a real alarm in the Clock app, checks it against Android's next alarm, and says the result.

## The models

| Role | Hugging Face repo | Files | Size | License | Revision |
|---|---|---|---|---|---|
| Speech recognition | [litert-community/Zipformer-medium-CR-CTC-LiteRT](https://huggingface.co/litert-community/Zipformer-medium-CR-CTC-LiteRT) | `zipformer_ctc_fp16.tflite`, `tokens.txt` | 131 MB | Apache-2.0 | `7732ad6c` |
| Chat model with tools | [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) | `gemma-4-E2B-it.litertlm` | 2.59 GB | Apache-2.0 | `b3ca0d2f` |
| Speech synthesis | [litert-community/kitten-tts-nano-0.8](https://huggingface.co/litert-community/kitten-tts-nano-0.8) | `kitten_predictor.tflite`, `kitten_prosody.tflite`, `kitten_vocoder.tflite`, `voices.npz` | 67 MB | Apache-2.0 | `d4662d89` |
| Speech synthesis: text to phonemes (G2P) | [litert-community/Matcha-TTS](https://huggingface.co/litert-community/Matcha-TTS) | `dp_g2p_matcha_fp16.tflite`, `g2p_dict.txt.gz`, `g2p_meta.json` | 28 MB | DeepPhonemizer: MIT; OpenPhonemizer dictionary: BSD-3-Clause-Clear | `8d650e79` |

The app downloads the files on first use into its private storage (2.81 GB in all) and checks each one's size and SHA-256 against [`app/src/main/assets/models.json`](android/app/src/main/assets/models.json). No model file is in the APK. KittenTTS's 178-symbol table (`symbols.json`, 1.6 KB, the table of `samples/litert/text_to_speech_streaming`) ships in the app's assets.

## Requirements

JDK 17 and Android SDK platform 36 to build. The app builds for arm64-v8a on Android 12+ (minSdk 31) and runs the speech recognition and the chat model on the phone's GPU. Before each model's download the app checks that twice its missing bytes are free in its storage (5.18 GB for the chat model on the first start).

## Architecture

```
microphone (16 kHz mono, 20 ms chunks)
  -> Endpointer: frame energy; 800 ms of silence ends an utterance, 16 s at most
  -> Zipformer CTC: host fbank, the graph on CompiledModel (GPU), greedy CTC on the host
  -> Gemma 4 E2B: one LiteRT-LM Conversation per request, tools declared in ConversationConfig,
     automaticToolCalling = false; the app runs each call in Message.toolCalls and sends the
     result back with Message.tool(...)
  -> KittenTTS: dictionary G2P plus a G2P graph (CompiledModel, CPU), three synthesis graphs
     (Interpreter API, CPU), sentence by sentence while the reply streams
  -> AudioTrack: each sentence plays while the next one is synthesized
```

A turn streams these events: `Heard`, `Thinking`, `ToolCalled`, `Speaking`, `Error`, `Done` (and `Listening` before each utterance). The tools are `get_current_datetime`, `set_alarm`, `set_timer` (the Clock app), `get_calendar_events` and `add_calendar_event` (a local "Phone Agent" calendar the app creates; account calendars are never touched). After a tool that changes something on the phone has run, the app says the tool's own result ("Alarm set for 07:30 (Morning Alarm)") instead of the model's words about it.

| Package | What it holds |
|---|---|
| `asr/` | Zipformer CTC on `CompiledModel`: fbank, the graph contract, greedy CTC |
| `tts/` | KittenTTS: G2P, the three synthesis graphs, the voices |
| `llm/` | `ChatEngine`: the LiteRT-LM `Engine` and its conversations, the streaming bridge |
| `loop/` | the voice loop: endpointer, sentence splitter, tool runner, phone tools, microphone, player |
| `data/` | `models.json`, the free-space rule and the download store (resume, SHA-256, HTTPS only) |
| `ui/` | the Compose screen and its state |

## Quickstart

```sh
cd samples/litert_lm/voice_assistant/android
./gradlew :app:installDebug
P=com.google.ai.edge.examples.voice_assistant
adb shell pm grant $P android.permission.RECORD_AUDIO
adb shell pm grant $P android.permission.READ_CALENDAR
adb shell pm grant $P android.permission.WRITE_CALENDAR
```

Without the grants the app asks on its first start; `SET_ALARM` is granted at install. Open the app, tap **Download (2.81 GB) and load** (on a metered network the app asks first), then the microphone button, and say "Set an alarm for six forty five tomorrow morning." The same request typed instead of spoken, from the command line (a debug build):

```sh
adb shell am start -n $P/.MainActivity --ez autoload true --es say "'Set an alarm for six forty five tomorrow morning.'"
adb logcat -d -s VoiceAssistant
adb shell dumpsys alarm | grep -A1 "Next alarm clock information"
```

The log has one line per event and a `TURN` line per request (what was heard, the calls and their results, the milliseconds, what was said, Android's next alarm, the network).

## Measured on Galaxy S26

Galaxy S26 SM-S942Q, Android 16 (BP4A.251205.006), LiteRT 2.2.0, LiteRT-LM 0.16.1. The first table is from this sample, a debug build in airplane mode: the first two rows on 2026-10-04 with thermal status 1 before each run, the third on 2026-10-05 with thermal status 0.

| Input | Succeeded | Transcript ready (ms) | Model's first token (ms) | First sentence ready (ms) | First sound (ms) |
|---|---|---|---|---|---|
| One typed command, "Set an alarm for six forty five tomorrow morning." | 1/1; Android then reported the next alarm at 06:45 | 0 | 714 | 1,402 | 1,682 |
| The device check's command, "Set an alarm for seven thirty tomorrow morning.", as a WAV through the endpointer | 1/1 (heard, calls, network none) | 87 | 848 | not recorded | 1,852 |
| One spoken request, "What time is it?", from a loudspeaker through the microphone | 1/1 (heard as "What time is it.", `get_current_datetime` called, answered aloud) | 117 | 1,008 | 1,623 | 2,143 |

Times are in ms from the end of the utterance; for the typed command, from the moment the loaded app took the text. First sound is the start of the player's first write; the output latency after it was not measured. For the WAV and the microphone the utterance ends with the endpointer's 800 ms of silence, so counted from the last spoken word the first sound comes 800 ms later than the column says. The WAV and the loudspeaker request are a synthetic voice (macOS `say`, voice Samantha), not a person.

The parts were measured on 2026-10-03 with the library this sample was adapted from (the `// Adapted from` lines name it), not with this sample. In the chat model's run the vision encoder was also on the GPU; this sample does not load it.

| Part | Measured |
|---|---|
| Zipformer `medium_fp16`, GPU, the ten commands as synthetic WAVs | graph run 37.1 ms (median of 10 warm calls, 36.3 to 40.8), whole call (fbank, graph, CTC) 60.6 ms, load 2.50 s; transcript equal to the command after lower case and without punctuation 4/10 (a strict comparison: "TO MORROW" for "tomorrow" and "B IS" for "What is" count as misses) |
| Kitten `fp32`, CPU, 4 threads, the predictor without XNNPACK, ten replies of 34 to 77 characters | median synthesis 301.2 ms (253.6 to 559.6), real-time factor 0.086; peak RSS (VmHWM) 334,360 kB after the load and 614,304 kB after the ten replies (with XNNPACK on the predictor too: 624,404 kB and 869,020 kB) |
| Gemma 4 E2B, GPU, the ten commands as text, with tools that record the call and do nothing | 10/10 (a command succeeds when the expected calls are made with the expected arguments); median first token / reply 650 / 1,728 ms; the first command again on a new conversation gave the same calls and reply |

One phone, one run each: not a promise for other devices, other voices or a real room.

## Known limits

- A GPU is needed: the speech recognition and the chat model have no CPU fallback.
- Under the keyguard the app has no visible activity, and Android drops the Clock app's `SET_ALARM` start (`BAL_BLOCK`, result code 102, on the Galaxy S26); the alarm tool then says the Clock app did not take the alarm. A debug-build launch with any of the extras below shows the screen over the keyguard.
- The alarm and the timer are real: turn them off or delete them in the Clock app afterwards. If an alarm with the same time and label is already there and off, the Samsung Clock turns that one on instead of adding a new one (Galaxy S26, 2026-10-04). Android has no public API to read the Clock app's timers, so the timer tool says the timer was requested, not started.
- Android reports one next alarm: when an alarm at the same minute or an earlier one is already set, the alarm tool says the alarm was requested and names the alarm Android reports, instead of confirming it.
- One utterance is at most 16 s, the window of the Zipformer graph; the endpointer cuts a longer one there.
- Each request runs in its own conversation, so the assistant does not remember the previous request.
- If a generation does not confirm its stop within 10 s, the chat engine is left unusable: later requests fail until the app is restarted.
- The three KittenTTS graphs run on the Interpreter API, not `CompiledModel`: the predictor and prosody graphs keep their LSTM state in variable tensors, which the `CompiledModel` loader does not accept yet (b/365299994), as [`samples/litert/text_to_speech_streaming`](../../litert/text_to_speech_streaming/) explains; that sample runs the vocoder on `CompiledModel` with its own JNI resize, this one keeps it on the Interpreter too.
- English only: the Zipformer and Kitten variants are English.

## Development

**Side-load (development only).** Copies of the model files in the app's external files dir are hashed and, when size and SHA-256 match `models.json`, imported instead of downloaded; the next load logs `side-loaded <file> ... (sha256 verified)` under `adb logcat -d -s VoiceAssistantDownload`. The copies must be readable by the app: the block below makes them as the app (`run-as`, a debug build).

```sh
P=com.google.ai.edge.examples.voice_assistant
adb shell mkdir -p /data/local/tmp/voice
adb push zipformer_ctc_fp16.tflite tokens.txt kitten_predictor.tflite kitten_prosody.tflite kitten_vocoder.tflite voices.npz \
  dp_g2p_matcha_fp16.tflite g2p_dict.txt.gz g2p_meta.json gemma-4-E2B-it.litertlm /data/local/tmp/voice/
adb shell "run-as $P sh -c 'mkdir -p /sdcard/Android/data/$P/files && cp /data/local/tmp/voice/* /sdcard/Android/data/$P/files/'"
```

The copies can be deleted after the import.

**Launch extras (debug builds only; a release build ignores them).** A running screen takes them again (`singleTop`). A download they start does not ask first on a metered network. With any of them the screen shows over the keyguard and stays on; a normal launch keeps the screen on only while the models load, the microphone is open or a request runs. While the screen is not visible, the microphone and any request in progress stop.

| Extra | What it does |
|---|---|
| `--ez autoload true` | download what is missing and load the three models |
| `--ez autolisten true` | open the microphone once they are loaded |
| `--es say "<text>"` | one request from this text once they are loaded (no microphone) |
| `--ef start_rms 0.01` | the endpointer's start level (default 0.02, a voice toward the phone; sound from a speaker needs less) |
| `--es record <name>` | each turn under `<external files>/record/<name>/<turn>/`: `utterance.wav` (microphone turns), `reply.wav` (the sentences said, synthesized again after the turn) and `events.json` (every event with its times) |

**Device check.** [`VoiceDeviceCheck`](android/app/src/androidTest/java/com/google/ai/edge/examples/voice_assistant/check/VoiceDeviceCheck.kt) loads the three models from the app's store, runs one command's WAV (16 kHz mono 16-bit) through the endpointer and the loop with the real phone tools and the loudspeaker, checks the alarm against `AlarmManager.getNextAlarmClock()`, reports the network, and releases everything. Keep the APKs installed: uninstalling the app deletes its model store. Android reports one next alarm, so no alarm may be set at or before 07:30 when it starts (the Quickstart's 06:45 included); otherwise it stops with `RESULT step=precondition ok=false` and names the alarm Android reports. It sets a real 07:30 alarm and then asks the Clock app to dismiss it by its label; on the Samsung Clock a dismiss by label did not remove it, and the cleanup line names the label to turn off or delete by hand.

```sh
adb push c01.wav /data/local/tmp/voice-assistant/commands/c01.wav
./gradlew :app:connectedDebugAndroidTest -Pandroid.injected.androidTest.leaveApksInstalledAfterRun=true \
  -Pandroid.testInstrumentationRunnerArguments.class=com.google.ai.edge.examples.voice_assistant.check.VoiceDeviceCheck
adb logcat -d -s voice-assistant-check | grep RESULT
```

The last line is `RESULT ok=true ...` when every step passed. The WAV says "Set an alarm for seven thirty tomorrow morning." by default (`-Pandroid.testInstrumentationRunnerArguments.expect=<text>` for another wording of a 07:30 alarm: the check still looks for `set_alarm` at 07:30). One way to make it, on a Mac: `say -v Samantha -r 175 -o c01.aiff "<text>"`, then `ffmpeg -i c01.aiff -ar 16000 -ac 1 -c:a pcm_s16le c01.wav`.

The JVM tests (`./gradlew :app:testDebugUnitTest`) cover the catalog and the symbol table, the download store and its free-space rule, the endpointer, the sentence splitter, the alarm check, what the tools read from the model, the microphone button against the loop, and the screen state.
