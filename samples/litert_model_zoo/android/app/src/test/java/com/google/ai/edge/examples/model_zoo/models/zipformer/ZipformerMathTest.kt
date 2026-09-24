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

package com.google.ai.edge.examples.model_zoo.models.zipformer

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ln
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ZipformerMathTest {
  @Test
  fun silenceUsesKaldiEnergyFloorWithoutCmn() {
    val fbank = frontend()
    val features = fbank.compute(FloatArray(400))
    assertEquals(3, features.size)
    val expected = ln(1.1920928955078125e-07f)
    features.forEach { row ->
      assertEquals(80, row.size)
      row.forEach { value ->
        assertTrue(value.isFinite())
        assertEquals(expected, value, 0f)
      }
    }
    assertTrue("No CMN: the log floor must not be zero", expected != 0f)
    assertEquals(0, fbank.compute(FloatArray(0)).size)
  }

  @Test
  fun constantFrameIsRemovedBeforeWindowAndFft() {
    val floor = ln(1.1920928955078125e-07f)
    frontend().compute(FloatArray(400) { 0.25f }).forEach { row ->
      row.forEach { assertEquals(floor, it, 0f) }
    }
  }

  @Test
  fun shortSyntheticToneHasFiniteNonConstantLogMelFeatures() {
    val audio =
      FloatArray(640) { (0.25 * kotlin.math.sin(2.0 * Math.PI * 440 * it / 16000)).toFloat() }
    val features = frontend().compute(audio)
    assertEquals(4, features.size)
    assertTrue(features.all { row -> row.size == 80 && row.all { it.isFinite() } })
    assertTrue(features.all { row -> row.maxOrNull()!! > row.minOrNull()!! })
    assertTrue("No CMN applied to each row", features.any { row -> row.sum() != 0f })
  }

  @Test
  fun frameCountPreservesSnipEdgesFalseAndFixedWindowCap() {
    val fbank = frontend()
    assertEquals(0, fbank.frames(79))
    assertEquals(1, fbank.frames(80))
    assertEquals(3, fbank.frames(400))
    assertEquals(1600, fbank.frames(ZipformerAsr.MAX_SAMPLES))
    assertEquals(255920, ZipformerAsr.MAX_SAMPLES)
  }

  @Test
  fun ctcDropsRepeatedTokensAndBlanksThenDetokenizesSpaces() {
    val pieces = mapOf(1 to "▁HEL", 2 to "LO", 3 to "▁WORLD")
    assertEquals("HELLO WORLD", ZipformerAsr.decode(logits(1, 1, 0, 2, 2, 3, 0), 7, pieces))
  }

  @Test
  fun ctcKeepsRepeatedPieceSeparatedByBlankAndIgnoresPadding() {
    val pieces = mapOf(1 to "▁A", 2 to "▁PAD")
    val scores = logits(1, 1, 0, 1, 2)
    assertEquals("A A", ZipformerAsr.decode(scores, 4, pieces))
    assertEquals("", ZipformerAsr.decode(scores, 0, pieces))
    assertEquals("", ZipformerAsr.decode(logits(0, 0), 2, pieces))
  }

  private fun frontend() = ZipformerFbank(readFloats("mel80_257.bin"), readFloats("povey400.bin"))

  private fun readFloats(name: String): FloatArray {
    val bytes =
      requireNotNull(javaClass.classLoader!!.getResourceAsStream(name)).use { it.readBytes() }
    val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
    return FloatArray(buffer.remaining()).also { buffer.get(it) }
  }

  private fun logits(vararg ids: Int): FloatArray =
    FloatArray(ids.size * ZipformerAsr.NCLASS) { -10f }
      .also { scores ->
        ids.forEachIndexed { index, token -> scores[index * ZipformerAsr.NCLASS + token] = 10f }
      }
}
