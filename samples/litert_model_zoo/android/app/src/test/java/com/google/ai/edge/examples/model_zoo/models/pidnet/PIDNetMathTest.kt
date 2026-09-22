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

package com.google.ai.edge.examples.model_zoo.models.pidnet

import org.junit.Assert.assertEquals
import org.junit.Test

class PIDNetMathTest {
  @Test
  fun planarArgmaxMapsToCityscapesColorsAndKeepsFirstClassOnTies() {
    val hw = 3
    val logits = FloatArray(Segmenter.N_CLASS * hw) { -10f }
    logits[0] = 2f
    logits[13 * hw] = 2f // Tied car must not replace road (strict >).
    logits[11 * hw + 1] = 3f
    logits[18 * hw + 2] = 4f
    assertEquals(0xff804080.toInt(), Segmenter.colorPixel(logits, 0, hw))
    assertEquals(0xffdc143c.toInt(), Segmenter.colorPixel(logits, 1, hw))
    assertEquals(0xff770b20.toInt(), Segmenter.colorPixel(logits, 2, hw))
    assertEquals("person", CityscapesPalette.NAMES[11])
    assertEquals(19, CityscapesPalette.COLORS.size)
  }
}
