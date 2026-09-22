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

package com.google.ai.edge.examples.model_zoo.models.ormbg

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class BgRemoverMathTest {
  @Test
  fun normalizesRangeThenUsesNearestNeighborDownsampling() {
    val full = FloatArray(16) { 2f }
    full[2] = 6f
    full[8] = 4f
    full[1] = 3f
    val matte = BgRemover.normalizeMatte(full, SIZE = 4, OUT = 2)
    assertEquals(4, matte.size)
    assertEquals(0f, matte[0], 0f)
    assertEquals(4f / (4f + 1e-6f), matte[1], 0f)
    assertEquals(2f / (4f + 1e-6f), matte[2], 0f)
    assertTrue(BgRemover.normalizeMatte(FloatArray(16) { 5f }, SIZE = 4, OUT = 2).all { it == 0f })
  }
}
