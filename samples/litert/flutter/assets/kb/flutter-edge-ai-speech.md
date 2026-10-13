---
title: On-device speech with flutter_edge_ai_speech
source: https://pub.dev/packages/flutter_edge_ai/versions/2.1.1 (skills/flutter-edge-ai-speech/SKILL.md) ; https://pub.dev/packages/flutter_edge_ai_speech/versions/0.5.4 (README.md, CHANGELOG.md, lib/src/voice/voice_session.dart, lib/src/voice/voice_event.dart, lib/src/voice/voice_responder.dart, lib/src/voice/clause_splitter.dart) ; https://github.com/DenisovAV/flutter_edge_ai/tree/main/packages/flutter_edge_ai_speech
license: MIT
---

# On-device speech with flutter_edge_ai_speech

This document explains flutter_edge_ai_speech (formerly flutter_gemma_speech), the opt-in package for on-device speech-to-text, text-to-speech and the VoiceSession voice loop. It covers the STT models (moonshine, Whisper, Parakeet), the 16 kHz PCM input contract, the Whisper output language, the TTS models (Matcha, Qwen3-TTS, Inflect-Nano-v2), streaming reply audio, barge-in and platform support.

## What flutter_edge_ai_speech provides and where it runs

flutter_edge_ai_speech is on-device speech for flutter_edge_ai — STT, TTS, and a VoiceSession voice loop — via the LiteRT C API and dart:ffi. It is an opt-in package: add it only if your app needs speech-to-text, text-to-speech, or a push-to-talk voice loop. It depends on flutter_edge_ai_litertlm, which owns the shared libLiteRtLm native bundle and exposes the LiteRT interpreter FFI used here; the package has no build hook of its own, and speech runs on the same native libraries as the .litertlm engine.

STT works end-to-end for moonshine-tiny (raw-PCM seq2seq), Whisper and Parakeet (log-mel) via LiteRtSttBackend. TTS works end-to-end for Matcha, Qwen3-TTS and Inflect-Nano-v2 via LiteRtTtsBackend; kokoro and supertonic are follow-ons. Both backends are pure factories — the model is selected per install via SttModelType or TtsModelType, not by the backend.

STT and TTS run through FFI on Android, iOS, macOS, Linux and Windows. Web is a stub: both throw UnsupportedError. Every flutter_edge_ai_litertlm release contains the fix for Windows STT and TTS, which older engine releases failed with CreateTensorBufferFromHostMemory status=3.

## Rules for using the speech package

1. Depend on flutter_edge_ai and flutter_edge_ai_speech, and import both. The speech package does not re-export core.
2. transcribe takes raw PCM — 16 kHz, mono, 16-bit little-endian, as a Uint8List — and returns the text. Not a WAV file, not 44.1 or 48 kHz: nothing resamples or converts it.
3. Play synthesized audio at synth.sampleRate. It differs per model.
4. Only Whisper has a selectable output language. moonshine-tiny and Parakeet are English-only, and passing a language to them throws ArgumentError.
5. STT language: set a default with getActiveStt(language:) or override one call with transcribe(pcm, language:). Nothing reloads.
6. TTS language: close() the synthesizer first. Asking a live synthesizer for another language throws StateError.
7. Android needs minSdk 30. There is no web support — the web backends throw UnsupportedError.
8. Close recognizers and synthesizers.

## Registering backends and installing a speech-to-text model

Speech backends are registered in FlutterEdgeAi.initialize, the same call that registers inference engines: pass sttBackends: [LiteRtSttBackend()] and ttsBackends: [LiteRtTtsBackend()]. The build setup — Android minSdk 30 and the Apple entries — is the same as for the .litertlm inference engine.

An STT model is two files — the model and its tokenizer — usually from different repos. install() skips files already on disk. For Whisper tiny, the model is whisper_tiny_30s_f32.tflite from litert-community/whisper-tiny and the tokenizer is tokenizer.json from openai/whisper-tiny:

```dart
await FlutterEdgeAi.installStt()
    .modelFromNetwork('https://huggingface.co/litert-community/whisper-tiny/resolve/main/whisper_tiny_30s_f32.tflite')
    .tokenizerFromNetwork('https://huggingface.co/openai/whisper-tiny/resolve/main/tokenizer.json')
    .ofType(SttModelType.whisper)
    .install();
final SpeechRecognizer recognizer = await FlutterEdgeAi.getActiveStt(language: 'de');
```

Then call recognizer.transcribe(pcm) and close the recognizer when done.

## Speech-to-text models, their windows and truncation

| SttModelType | Languages | Window |
|---|---|---|
| moonshine | English | 5 s |
| whisper | 99, selectable, default 'en' | 30 s |
| parakeet | English | 5 s; 2.35 GB, so desktop in practice — nothing refuses it on a phone |

Whisper tiny is weak outside English. Whisper base int8 is the next size up in the catalog; install it the same way from litert-community/whisper-base, file whisper_base_30s_i8.tflite, with the tokenizer.json from openai/whisper-base.

Audio longer than the window is silently truncated, not rejected: it is zero-padded when shorter and cut when longer, so a 40-second clip on Whisper returns the first 30 seconds with no error. Split long recordings yourself. STT can also hallucinate text on near-silence; both the truncation beyond the fixed window (about 5 seconds for moonshine) and the hallucination on near-silence are properties of the STT model.

## The 16 kHz mono PCM input contract

The package has no resampler and no WAV reader. transcribe takes 16 kHz, mono, 16-bit little-endian PCM with no header.

Record in the right format from the start. With the record package, use a RecordConfig with encoder AudioEncoder.wav, sampleRate 16000 and numChannels 1. That produces a WAV file, and its header is not always 44 bytes, so take the samples from the WAV file's data chunk: walk the RIFF chunks after the 12-byte RIFF/WAVE header, read each chunk's id and little-endian size, and return the bytes of the chunk whose id is "data" (chunks are padded to an even size).

A file recorded at another rate or channel count — 44.1 kHz stereo, say — has to be converted first: average the channels to mono, then resample with a low-pass filter. Dropping samples instead aliases and costs accuracy.

## Microphone permissions for speech recording

Recording needs microphone access. On Android, the record package declares android.permission.RECORD_AUDIO in its own manifest and the manifest merger adds it to the app. iOS and macOS need entries of your own: an NSMicrophoneUsageDescription string in ios/Runner/Info.plist and macos/Runner/Info.plist, and the com.apple.security.device.audio-input entitlement set to true in both macos/Runner/DebugProfile.entitlements and macos/Runner/Release.entitlements. The macOS sandbox withholds the microphone without the entitlement, and it has to be in both entitlements files.

On Android, iOS and macOS the user must also allow the microphone at run time. AudioRecorder.hasPermission() asks, so call it before recording.

## Choosing the Whisper output language and why transcripts come back in English

Whisper's shipped checkpoints are multilingual. The output language is one token in the decoder's seed prompt, rebuilt per transcription — so it is a per-call knob, and switching it never reloads the model. Set a default with getActiveStt(language: 'de') and override it for one call with recognizer.transcribe(frenchPcm, language: 'fr').

Codes are Whisper's own, bare and lowercase: 'en', 'de', 'uk' — not 'de-DE', 'DE' or 'german'. The default is 'en'. Malformed codes throw ArgumentError from getActiveStt, and a well-formed code the installed checkpoint lacks, such as 'zz', throws ArgumentError from transcribe. moonshine and Parakeet have no language token and throw ArgumentError rather than ignoring the value.

A common trap: German audio produces fluent English text, with no error. The cause is that Whisper's language token decides the output language, not what it understands — the setting changes what the model writes, not what it hears, so with 'en' on German audio it returns an English translation. moonshine only ever produces English. The fix is to use Whisper and pass language:.

## Text-to-speech models

| TtsModelType | Languages |
|---|---|
| matcha | English, fixed by the installed bundle — it ignores language: |
| qwen3 | chinese, english, german, italian, portuguese, spanish, japanese, korean, french, russian, or auto |
| inflect | English |

TtsModelType.supertonic and TtsModelType.kokoro are in the enum but throw UnimplementedError — do not use them.

Matcha comes from litert-community/Matcha-TTS and runs at 22050 Hz. It is a 3-graph LiteRT pipeline: encoder, then a CFM decoder, then a HiFi-GAN vocoder, producing 16-bit PCM. Matcha ships a config.json in its bundle, and the numeric synthesis parameters are read from it at load time.

## How the Qwen3-TTS and Inflect pipelines work

Qwen3-TTS is an autoregressive codec-token language model (a talker with prefill and decode over a threaded KV cache), followed by a 15-step residual-codebook inner loop and a windowed codec decoder that outputs 24 kHz PCM.

Inflect-Nano-v2 is VITS-style: a text encoder emits latents and log-durations, a host-side length regulator repeats each frame by its duration and adds a fixed-seed Gaussian noise sample, and the decoder emits a 24 kHz waveform directly, with no separate vocoder. Phoneme ids go in and PCM comes out; Inflect reuses Matcha's grapheme-to-phoneme bundle.

## Synthesizing speech and playing it at the right sample rate

Install a TTS bundle once with FlutterEdgeAi.installTts().fromNetwork(<bundle URL>).ofType(TtsModelType.matcha).install(), then get the synthesizer with FlutterEdgeAi.getActiveTts(). synth.synthesize('Hello world.') returns a Uint8List of 16-bit PCM, and synth.sampleRate gives its rate — 22050 for Matcha. Close the synthesizer when done.

Play synthesized audio at synth.sampleRate, because it differs per model; playing it at the wrong rate gives the wrong pitch. VoiceSession owns no microphone or player — the app captures PCM with package:record and plays the reply, for example with pcmToWav and package:just_audio.

## Shipping a text-to-speech voice inside the app

A voice can ship with the app instead of being downloaded: swap fromNetwork for fromAsset, fromFile or fromBundled on the same installTts() builder. fromAsset('assets/tts/matcha/') takes a Flutter asset directory laid out like the Hugging Face repo — declare every subdirectory in pubspec.yaml, for example Qwen3's tables/ and voices/ — and copies it into app storage. fromFile('/abs/dir') takes a directory on disk with the same layout and uses it in place, so uninstallTts() deletes those files. fromBundled() takes native resources named <type>__<file>, for example matcha__config.json, from Android's assets/models/ or the iOS Runner target, on Android and iOS only.

Inflect's four Matcha grapheme-to-phoneme files go next to its own two model files. A missing file fails install() with the full list before the current voice is replaced; a switch that fails partway, for example on a lost connection, leaves no voice active, so getActiveTts throws until install() is run again. install() throws UnsupportedError on web. To ship a changed voice in an app update, use a new asset directory (a bundled voice refreshes itself); with fromFile on iOS, call install() at launch, because the app's data directory moves on update.

## Switching the Qwen3-TTS language

Qwen3-TTS languages use full lowercase names, not ISO codes, and only work with the Qwen3 bundle installed; with Matcha the same StateError is still thrown but the language changes nothing. To switch, close the current synthesizer before asking for another language: get getActiveTts(language: 'english'), close it, then call getActiveTts(language: 'german'). Without the close(), the second call throws "StateError: Active TTS synthesizer was created for language 'english'; call close() before requesting 'german'."

## VoiceSession, the push-to-talk voice loop

VoiceSession chains STT, LLM and TTS into one push-to-talk turn with barge-in: transcribe, generate, speak. It uses the recognizer's current language. It is pure orchestration — it owns no microphone, no player, and none of the injected components' lifecycles: the caller creates and closes the recognizer, chat and synthesizer.

runTurn takes one recorded utterance as 16 kHz mono 16-bit little-endian PCM — the same contract as transcribe — and streams back VoiceEvents. The sequence is: transcribe the PCM, emit a final VoiceTranscriptEvent, stream the responder's reply as VoiceReplyTextEvents, synthesize the reply audio, and finish with VoiceTurnCompleteEvent. Calling runTurn while a turn is in flight throws StateError.

## VoiceSession.fromChat and chats with tools

VoiceSession.fromChat(recognizer:, chat:, synthesizer:) wraps an InferenceChat and is the recommended default for multi-turn conversational voice. Create the chat via getActiveModel().createChat(...) with a short maxOutputTokens and a "reply concisely, this will be spoken aloud" system instruction; the README example uses tokenBuffer 256 and maxOutputTokens 128 for short replies.

When the chat has no tools, this is the plain text-only path. A chat with tools is supported: pass onToolCall, which is the tool implementation, and the turn runs through core's InferenceChat.generateChatResponseWithTools function-calling loop. Without onToolCall there is nothing to run a tool call with, so fromChat throws for a tools-enabled chat — a real throw, not an assert.

## VoiceSession.custom and the VoiceResponder

VoiceSession.custom is the general constructor — any LLM behind a VoiceResponder. Use it for an AgentSession or MCP responder. A VoiceResponder is the LLM step of a voice turn: it streams the reply's tokens for a user utterance through respond, and offers a portable stop for barge-in, for example chat.stopGeneration. Its contract is that stop() must cause the respond stream to terminate promptly. VoiceSession.interrupt drains with a timeout, so a stop() that does not end the stream will not hang the session, but it defers the interrupt and leaves generation running in the background.

## Streaming reply audio sentence by sentence

By default the reply is synthesized in one shot after the LLM finishes, as a single VoiceReplyAudioEvent with isFinal true. With streamAudio: true, VoiceSession synthesizes the reply clause by clause as the LLM streams, for lower time-to-first-audio: it emits a series of VoiceReplyAudioEvent chunks with isFinal false, then a zero-byte isFinal true marker. It is opt-in because consumers must play the chunks in sequence, and because per-clause synthesis has slightly flatter prosody at the clause joins than synthesizing the whole reply at once.

A clause splitter decides the clauses: a clause ends at one or more of '.', '!' or '?' followed by whitespace, or at a newline, and must reach a minimum length. Requiring trailing whitespace means 3.14 never splits, and short fragments such as "Hi." or "Dr." merge forward into the next clause, which also mitigates false splits on abbreviations.

## Voice events in a turn

VoiceEvent is sealed. VoiceTranscriptEvent carries the recognized user speech. VoiceReplyTextEvent carries a chunk of the LLM's streamed text reply. VoiceReplyAudioEvent carries synthesized reply audio as 16-bit little-endian mono PCM at its sampleRate. VoiceTurnCompleteEvent is terminal: the turn finished normally. VoiceTurnInterruptedEvent is terminal: the turn was cut short by interrupt, and it carries what was produced so the caller can reconcile app state and chat history. A successful non-empty turn ends with exactly one isFinal audio event; an empty-reply turn emits no audio; an errored or interrupted turn emits no isFinal audio, though an interrupted streaming turn may already have emitted chunks. Turns are serialized: one turn's events fully precede the next turn's.

## Barge-in and errors in a voice turn

To barge in, call await voice.interrupt() — cancelling the stream subscription is not a portable stop. interrupt is idempotent and a no-op when idle. It sets the interrupt flag, calls the responder's stop(), drains the reply stream with a bounded timeout, and the turn ends with VoiceTurnInterruptedEvent; stop the audio player when that event arrives.

Wrap the loop in try and catch: a failed stage — transcribe, generate or synthesize — arrives as a stream error, not as an event. VoiceErrorEvent is reserved in this release and never emitted; the case exists only because a switch over the sealed events must be exhaustive.
