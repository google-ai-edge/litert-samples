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

package com.google.ai.edge.examples.model_zoo.models.movinet

import org.junit.Assert.*
import org.junit.Test

class ActionRecognizerMathTest {
  @Test
  fun softmaxTopKPreservesClassOrderAndHandlesLargeCommonOffset() {
    val logits = floatArrayOf(10000f, 10002f, 10001f)
    val result = ActionRecognizer.topK(logits, 3)
    assertEquals(listOf(1, 2, 0), result.map { it.index })
    assertEquals(Kinetics600Labels.NAMES[1], result.first().label)
    assertEquals(0.66524096f, result[0].score, 1e-6f)
    assertEquals(1f, result.sumOf { it.score.toDouble() }.toFloat(), 1e-6f)
    assertEquals(600, Kinetics600Labels.NAMES.size)
  }
}
