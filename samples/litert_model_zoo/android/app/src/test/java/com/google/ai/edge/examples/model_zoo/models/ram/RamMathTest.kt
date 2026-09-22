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

package com.google.ai.edge.examples.model_zoo.models.ram

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RamMathTest {
  @Test
  fun sigmoidThresholdAndTopKPreserveTagOrderAndStrictBoundary() {
    val logits = FloatArray(RamTagger.NCLASS) { -100f }
    val thresholds = FloatArray(RamTagger.NCLASS) { 0.5f }
    val tags = List(RamTagger.NCLASS) { "tag-$it" }
    logits[0] = 0f // sigmoid is exactly 0.5: strict threshold must reject it.
    logits[1] = 1f
    logits[2] = 2f
    val result = RamTagger.selectTags(logits, tags, thresholds, 1)
    assertEquals(listOf("tag-2"), result.map { it.name })
    assertEquals(0.880797f, result.single().prob, 0.000001f)
    assertTrue(RamTagger.selectTags(logits, tags, thresholds, 5).none { it.name == "tag-0" })
  }
}
