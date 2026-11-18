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

import kotlin.time.Duration

/** Interface to output raw audio data from a source. */
interface AudioSource : AutoCloseable {
  val samplingRate: Int
  val interval: Duration
  val overlap: Duration

  /**
   * Outputs chunks of audio data of size intervalMs overlapping by overlapMs with previous chunk.
   *
   * @return Sequence of audio data chunks in float array.
   */
  fun getAudioData(): Sequence<FloatArray>
}

/**
 * Returns the number of samples of [audioChunk] up to and including its last non-zero one. Audio
 * sources fill the part of a chunk that has no audio with zeros (e.g. the end of the last chunk of
 * a file), which a speech recognizer need not decode.
 */
fun getNumOfValidSamples(audioChunk: FloatArray): Int {
  var numOfValidSamples = audioChunk.size
  while (numOfValidSamples > 0 && audioChunk[numOfValidSamples - 1] == 0.0f) {
    numOfValidSamples--
  }
  return numOfValidSamples
}
