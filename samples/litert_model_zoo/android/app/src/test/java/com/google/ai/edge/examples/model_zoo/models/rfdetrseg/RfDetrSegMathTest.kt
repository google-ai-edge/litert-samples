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

package com.google.ai.edge.examples.model_zoo.models.rfdetrseg

import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Test

class RfDetrSegMathTest {
  private fun detection(cls: Int, score: Float, cx: Float = .5f) =
    RfDetrSeg.Detection(cls, score, cx, .5f, .4f, .4f, floatArrayOf(-1f, 1f))

  @Test
  fun nmsSuppressesSameClassOverlapAndKeepsDifferentClassesAndMaskIdentity() {
    val high = detection(1, .9f)
    val low = detection(1, .7f, .51f)
    val otherClass = detection(2, .8f)
    val separate = detection(1, .6f, 1.5f)
    val result = RfDetrSeg.nms(listOf(low, separate, otherClass, high))
    assertEquals(listOf(high, otherClass, separate), result)
    assertSame(high.mask, result[0].mask)
    assertEquals(1f, RfDetrSeg.iou(high, otherClass), 0.000001f)
    assertEquals(0f, RfDetrSeg.iou(high, separate), 0f)
  }
}
