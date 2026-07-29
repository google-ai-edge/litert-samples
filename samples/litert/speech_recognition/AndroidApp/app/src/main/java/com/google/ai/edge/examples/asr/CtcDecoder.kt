/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.examples.asr

import androidx.annotation.VisibleForTesting
import com.google.ai.edge.litert.TensorBuffer

/**
 * Helper class to decode CTC model logits into token IDs adhering to the Decoder interface. Samples
 * maximum logit values and deduplicates tokens.
 */
class CtcDecoder(private val vocabSize: Int) : LiteRtRunner.Decoder {
  /**
   * Decodes the model output (FloatArray logits) into an array of token IDs. Performs CTC decoding
   * (collapse repeats -> remove blanks) and text cleanup.
   */
  override fun decode(encodeOutputBuffers: List<TensorBuffer>): Sequence<Pair<Int, Int>> =
    sequence {
      val logits = encodeOutputBuffers[0].readFloat()
      if (logits.isEmpty()) {
        yield(Pair(SpeechRecognizer.END_OF_SEQUENCE, SpeechRecognizer.NO_TIMESTAMP))
        return@sequence
      }

      val tokenIds = sampleMaximum(logits)
      yieldAll(collapseDuplicates(tokenIds))
    }

  @VisibleForTesting
  internal fun collapseDuplicates(tokenIds: IntArray): Sequence<Pair<Int, Int>> = sequence {
    var previousId = -1
    // Timestamp is the index of the token in the original list of logits as the input of CTC has
    // logits for every sample in the input audio.
    var timestamp = 0
    for (id in tokenIds) {
      if (id != previousId) {
        yield(Pair(id, timestamp))
        previousId = id
      }
      timestamp++
    }
    yield(Pair(SpeechRecognizer.END_OF_SEQUENCE, timestamp))
  }

  @VisibleForTesting
  internal fun sampleMaximum(logits: FloatArray): IntArray {
    val numFrames = logits.size / vocabSize
    val tokenIds = IntArray(numFrames)

    for (t in 0 until numFrames) {
      val start = t * vocabSize
      val end = start + vocabSize
      tokenIds[t] = (start until end).maxByOrNull { logits[it] }?.minus(start) ?: 0
    }
    return tokenIds
  }
}
