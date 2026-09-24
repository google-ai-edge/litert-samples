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

package com.google.ai.edge.examples.model_zoo.models.rfdetr

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Exercises the same extracted host functions called by the two-graph runtime wrapper. */
class RfDetrMathTest {
  @Test
  fun imagenetNormalizationPreservesRgbChannelPlanesAndScale() {
    val rgb = FloatArray(RfDetr.SIZE * RfDetr.SIZE * 3)
    floatArrayOf(255f, 0f, 127.5f, 0f, 255f, 255f).copyInto(rgb)
    val actual = RfDetr.normalizeRgb(rgb)
    val plane = RfDetr.SIZE * RfDetr.SIZE
    assertArrayEquals(
      floatArrayOf(2.2489083f, -2.0357143f, 0.4177778f),
      floatArrayOf(actual[0], actual[plane], actual[2 * plane]),
      0.00001f,
    )
    assertArrayEquals(
      floatArrayOf(-2.1179039f, 2.4285714f, 2.64f),
      floatArrayOf(actual[1], actual[plane + 1], actual[2 * plane + 1]),
      0.00001f,
    )
    assertTrue(actual.all { it.isFinite() })
    assertEquals(3 * plane, actual.size)
  }

  @Test
  fun proposalsUseEveryClassAndGatherFourCoordinatesInDescendingOrder() {
    val classes = FloatArray(RfDetr.NPROP * RfDetr.NCLS) { -20f }
    classes[575 * RfDetr.NCLS + 90] = 9f
    classes[2 * RfDetr.NCLS + 3] = 8f
    classes[17 * RfDetr.NCLS] = 7f
    val coords = FloatArray(RfDetr.NPROP * 4) { it.toFloat() }
    val selected = RfDetr.selectReferencePoints(classes, coords)
    assertEquals(RfDetr.NQ * 4, selected.size)
    assertArrayEquals(
      floatArrayOf(2300f, 2301f, 2302f, 2303f, 8f, 9f, 10f, 11f, 68f, 69f, 70f, 71f),
      selected.copyOfRange(0, 12),
      0f,
    )
    // Equal-score proposals retain ascending source order in the zoo's stable sort.
    assertArrayEquals(floatArrayOf(0f, 1f, 2f, 3f), selected.copyOfRange(12, 16), 0f)
  }

  @Test
  fun decodeDropsBackgroundAndLowScoresSuppressingOverlapsOnlyWithinClass() {
    val boxes = FloatArray(RfDetr.NQ * 4)
    val logits = FloatArray(RfDetr.NQ * RfDetr.NCLS) { -20f }
    candidate(boxes, logits, 0, 1, 4f, 0.5f, 0.5f, 0.4f, 0.4f)
    candidate(boxes, logits, 1, 1, 3f, 0.5f, 0.5f, 0.4f, 0.4f)
    candidate(boxes, logits, 2, 3, 2f, 0.5f, 0.5f, 0.4f, 0.4f)
    candidate(boxes, logits, 3, 0, 10f, 0.5f, 0.5f, 0.4f, 0.4f)
    candidate(boxes, logits, 4, 7, -1f, 0.5f, 0.5f, 0.4f, 0.4f)
    candidate(boxes, logits, 5, 1, 1f, 0.1f, 0.1f, 0.1f, 0.1f)

    val detections = RfDetr.decode(boxes, logits)
    assertEquals(listOf(1, 3, 1), detections.map { it.cls })
    assertEquals(0.98201376f, detections[0].score, 0.000001f)
    assertEquals(0.8807971f, detections[1].score, 0.000001f)
    assertEquals(0.7310586f, detections[2].score, 0.000001f)
    assertEquals(0.1f, detections[2].cx, 0f)
    assertEquals(0.4f, detections[0].w, 0f)
    assertTrue(detections.all { it.score.isFinite() })
  }

  @Test
  fun zeroAreaBoxesHaveFiniteScoresAndDoNotSuppressEachOther() {
    val boxes = FloatArray(RfDetr.NQ * 4)
    val logits = FloatArray(RfDetr.NQ * RfDetr.NCLS) { -20f }
    candidate(boxes, logits, 0, 1, 1f, 0.5f, 0.5f, 0f, 0f)
    candidate(boxes, logits, 1, 1, 2f, 0.5f, 0.5f, 0f, 0f)
    val detections = RfDetr.decode(boxes, logits)
    assertEquals(2, detections.size)
    assertTrue(detections.all { it.score.isFinite() })
  }

  @Test
  fun uniformlyLowLogitsProduceNoDetections() {
    assertTrue(
      RfDetr.decode(FloatArray(RfDetr.NQ * 4), FloatArray(RfDetr.NQ * RfDetr.NCLS) { -20f })
        .isEmpty()
    )
  }

  private fun candidate(
    boxes: FloatArray,
    logits: FloatArray,
    query: Int,
    cls: Int,
    logit: Float,
    cx: Float,
    cy: Float,
    width: Float,
    height: Float,
  ) {
    floatArrayOf(cx, cy, width, height).copyInto(boxes, query * 4)
    logits[query * RfDetr.NCLS + cls] = logit
  }
}
