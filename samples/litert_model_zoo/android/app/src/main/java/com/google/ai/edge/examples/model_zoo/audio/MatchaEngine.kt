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

package com.google.ai.edge.examples.model_zoo.audio

import com.google.ai.edge.examples.model_zoo.models.matcha.MatchaG2P
import com.google.ai.edge.examples.model_zoo.models.matcha.MatchaSynthesizer
import java.io.File

/** Result time includes phonemization, the ODE loop, and all graph readbacks. */
data class AudioSpeechResult(
  val samples: FloatArray,
  val sampleRate: Int,
  val inferenceMs: Double,
  val backend: String,
  val fallbackReason: String?,
  val backendDetails: String,
)

/** Construct, synthesize, and close on the same confined worker dispatcher. */
class MatchaEngine(modelDir: File, backend: String = "gpu") : AutoCloseable {
  private val g2p = MatchaG2P(modelDir)
  private val synth =
    try {
      MatchaSynthesizer(modelDir, backend)
    } catch (e: Exception) {
      g2p.close()
      throw e
    }

  fun synthesize(text: String): AudioSpeechResult {
    require(text.isNotBlank()) { "Enter English text to speak." }
    val start = System.nanoTime()
    val ids = g2p.phonemize(text)
    require(ids.isNotEmpty()) { "The text did not contain supported English phonemes." }
    require(ids.size * 2 + 1 <= MatchaSynthesizer.MAX_TEXT) {
      "This model supports short phrases. Shorten the text and try again."
    }
    val result = synth.synthesize(ids)
    require(result.audio.isNotEmpty() && result.audio.all { it.isFinite() }) {
      "The model returned invalid audio."
    }
    return AudioSpeechResult(
      result.audio,
      MatchaSynthesizer.SAMPLE_RATE,
      (System.nanoTime() - start) / 1_000_000.0,
      synth.backend,
      synth.fallbackReason,
      synth.backendDetails,
    )
  }

  override fun close() {
    try {
      synth.close()
    } finally {
      g2p.close()
    }
  }
}
