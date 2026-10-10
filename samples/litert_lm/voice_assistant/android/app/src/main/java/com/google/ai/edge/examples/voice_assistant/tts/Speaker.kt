/*
 * Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Adapted from john-rocky/hfmodels-android (commit 3086d647):
// core/src/main/kotlin/io/github/johnrocky/hfmodels/speech/Speaker.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.tts

/**
 * A speech synthesizer: text in, mono PCM out. One sentence or short chunk per call; splitting a
 * longer text is the caller's (`SentenceSplitter`).
 *
 * One `synthesize` at a time per model; a second concurrent call fails with `MODEL_BUSY`.
 */
interface Speaker : AutoCloseable {
  /** The voices this load offers; `voices[0]` is the default. */
  val voices: List<String>

  /** Samples per second of [SpeechAudio.samples] (24000 for the kitten family). */
  val sampleRate: Int

  /**
   * One chunk's limit in characters (Unicode code points; 400 for the kitten family). Longer text
   * is `INVALID_INPUT`, and so is text that is empty or has no symbol the model knows.
   */
  val maxChars: Int

  /**
   * [voice]: one of [voices], null = `voices[0]`. [speed]: 1 = the publisher's default pace for
   * the voice, 2 = twice that (the voice's speed prior multiplies it before the model, as the
   * publisher's `say.py` does; 0.8 for most kitten voices).
   */
  suspend fun synthesize(text: String, voice: String? = null, speed: Float = 1f): SpeechAudio

  /**
   * The symbol ids the synthesizer receives for this text, including the 0 at each end; for tests
   * and for showing what was said. Blocks while the out-of-dictionary graph runs.
   */
  fun phonemeIds(text: String): IntArray

  /** Releases the graphs and waits until they are released. Idempotent. */
  suspend fun closeAndJoin()
}

/** [samples]: mono, in [-1, 1], at [sampleRate]. */
data class SpeechAudio(val samples: FloatArray, val sampleRate: Int, val timing: SpeechTiming)

/**
 * One call's wall clock on the device, not a benchmark: text to symbols, the graphs with the host
 * glue, and the whole call. [frames]: the number of 40 Hz acoustic frames the model gave the text
 * (600 samples each at 24 kHz, before the end of the audio is trimmed).
 */
data class SpeechTiming(
  val g2pMs: Double,
  val synthMs: Double,
  val totalMs: Double,
  val frames: Int,
)
