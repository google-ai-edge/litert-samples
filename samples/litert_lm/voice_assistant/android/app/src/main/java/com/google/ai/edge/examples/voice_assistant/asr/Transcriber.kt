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
// core/src/main/kotlin/io/github/johnrocky/hfmodels/speech/Transcriber.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.asr

/**
 * A speech recognizer: mono PCM in, text out. One fixed window per call; audio longer than
 * [TranscriberLimits.windowSeconds] is `INVALID_INPUT` (consecutive windows are not handled here),
 * shorter audio is padded to the window.
 *
 * One call at a time per model; a second concurrent call fails with `MODEL_BUSY`.
 */
interface Transcriber : AutoCloseable {
  val limits: TranscriberLimits

  /** [pcm]: mono samples at [TranscriberLimits.sampleRate], in [-1, 1]. */
  suspend fun transcribe(pcm: FloatArray): Transcript

  /** Releases the graph and waits until it is released. Idempotent. */
  suspend fun closeAndJoin()
}

/** What this load can take. */
data class TranscriberLimits(
  /** Samples per second the model expects (16000 for the zipformer family). */
  val sampleRate: Int,
  /** The longest audio one call accepts, in seconds (16 for the zipformer family). */
  val windowSeconds: Double,
  /** Languages the publisher declares for this variant (BCP-47); informational. */
  val languages: List<String>,
)

data class Transcript(val text: String, val timing: TranscriptTiming)

/**
 * One call's wall clock on the device, not a benchmark: the host front-end, the graph run with its
 * readback, and the whole call.
 */
data class TranscriptTiming(val featureMs: Double, val inferenceMs: Double, val totalMs: Double)
