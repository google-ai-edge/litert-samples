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

/** Interface to recognize speech from preprocessed features. */
interface SpeechRecognizer : AutoCloseable {
  /**
   * Recognizes speech from preprocessed features and returns the token IDs and their relative
   * timestamps optionally.
   *
   * @param features Preprocessed features in float array.
   * @return Token IDs and their timestamps (if available) as a sequence.
   */
  fun recognize(features: FloatArray): Sequence<Pair<Int, Int>>

  companion object {
    /** Special token ID that indicates the end of the sequence. */
    const val END_OF_SEQUENCE = -1

    /** Special timestamp that indicates the timestamp is not available. */
    const val NO_TIMESTAMP = -1
  }
}
