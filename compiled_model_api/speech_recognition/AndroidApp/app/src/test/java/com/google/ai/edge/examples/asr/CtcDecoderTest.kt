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

import org.junit.Assert.assertArrayEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class CtcDecoderTest {

  @Test
  fun sampleMaximum_basicArgmax() {
    val vocabSize = 4
    val decoder = CtcDecoder(vocabSize)

    val logits =
      floatArrayOf(
        0.1f,
        0.9f,
        0.0f,
        0.0f, // Frame 0 -> 1
        0.2f,
        0.8f,
        0.0f,
        0.0f, // Frame 1 -> 1
        0.1f,
        0.1f,
        0.7f,
        0.1f, // Frame 2 -> 2
      )

    val result = decoder.sampleMaximum(logits)
    assertArrayEquals(intArrayOf(1, 1, 2), result)
  }

  @Test
  fun collapseDuplicates_removesConsecutiveDuplicatesAndYieldsEos() {
    val vocabSize = 4
    val decoder = CtcDecoder(vocabSize)
    val inputIds = intArrayOf(1, 1, 2, 2, 2, 3, 1)

    val outputSequence = decoder.collapseDuplicates(inputIds).toList()
    assertArrayEquals(
      intArrayOf(1, 2, 3, 1, SpeechRecognizer.END_OF_SEQUENCE),
      outputSequence.map { it.first }.toIntArray(),
    )
  }
}
