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

package com.google.ai.edge.examples.model_zoo.models.ppocr

import org.junit.Assert.assertEquals
import org.junit.Test

class PpocrMathTest {
  @Test
  fun ctcDropsBlankAndConsecutiveRepeatsButKeepsSeparatedRepeats() {
    val chars = arrayOf("", "A", "B", " ")
    val path = intArrayOf(1, 1, 0, 1, 2, 2, 3, 0)
    val logits = FloatArray(path.size * chars.size) { -5f }
    path.forEachIndexed { time, token -> logits[time * chars.size + token] = 5f }
    assertEquals("AAB ", PpocrRecognizer.decode(logits, chars))
  }

  @Test
  fun textBoxFloodFillExpandsAcceptedRegionAndRejectsLowConfidence() {
    val p = FloatArray(PpocrDetector.SIZE * PpocrDetector.SIZE)
    for (y in 20..27) for (x in 30..37) p[y * PpocrDetector.SIZE + x] = 0.9f
    for (y in 50..57) for (x in 50..57) p[y * PpocrDetector.SIZE + x] = 0.4f
    assertEquals(listOf(PpocrDetector.Box(28, 18, 39, 29)), PpocrDetector.boxes(p))
  }
}
