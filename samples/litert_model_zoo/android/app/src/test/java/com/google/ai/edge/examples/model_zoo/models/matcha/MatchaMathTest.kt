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

package com.google.ai.edge.examples.model_zoo.models.matcha

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MatchaMathTest {
  @Test
  fun timeEmbeddingUsesSineThenCosineAtZero() {
    val embedding = MatchaSynthesizer.sinPosEmb(0f)
    assertEquals(160, embedding.size)
    assertArrayEquals(FloatArray(80), embedding.copyOfRange(0, 80), 0f)
    assertArrayEquals(FloatArray(80) { 1f }, embedding.copyOfRange(80, 160), 0f)
  }

  @Test
  fun timeEmbeddingCoversExpectedFrequencyRange() {
    val embedding = MatchaSynthesizer.sinPosEmb(0.001f)
    assertEquals(0.84147096f, embedding[0], 0.000001f)
    assertEquals(0.5403023f, embedding[80], 0.000001f)
    assertEquals(0.0001f, embedding[79], 0.0000001f)
    assertTrue(embedding.all { it.isFinite() && it in -1f..1f })
  }

  @Test
  fun embeddingTableUsesLittleEndianFloat32() {
    val bytes = byteArrayOf(0, 0, 0x80.toByte(), 0x3f, 0, 0, 0, 0xc0.toByte())
    assertArrayEquals(floatArrayOf(1f, -2f), MatchaSynthesizer.readFloats(bytes), 0f)
  }

  @Test
  fun normalizesGroupedDecimalAndZero() {
    assertEquals(
      listOf("one", "thousand", "two", "hundred", "thirty", "four", "point", "five"),
      MatchaG2P.numToWords("1,234.5"),
    )
    assertEquals(listOf("zero"), MatchaG2P.numToWords("0"))
  }

  @Test
  fun preservesGapsBetweenThousandsAndSpellsLargeIntegers() {
    assertEquals(listOf("one", "million", "one"), MatchaG2P.intToWords(1_000_001))
    assertEquals(listOf("minus", "twenty", "one"), MatchaG2P.intToWords(-21))
    assertEquals(
      "1234567890123456"
        .map {
          listOf("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")[
            it - '0']
        },
      MatchaG2P.intToWords(1_234_567_890_123_456),
    )
  }
}
