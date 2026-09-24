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

package com.google.ai.edge.examples.model_zoo.image

import org.junit.Assert.assertEquals
import org.junit.Test

class OcrReadingOrderTest {
  @Test
  fun groupsVerticalCentersAndSortsLeftToRight() {
    val boxes =
      listOf(
        OcrDisplayBox(90, 42, 140, 62, "morning."),
        OcrDisplayBox(90, 1, 130, 21, "TODAY"),
        OcrDisplayBox(10, 40, 50, 60, "Fresh"),
        OcrDisplayBox(10, 0, 80, 20, "OPEN"),
        OcrDisplayBox(55, 41, 85, 61, "breadlb"),
      )
    assertEquals(listOf("OPEN TODAY", "Fresh breadlb morning."), OcrReadingOrder.lines(boxes))
    assertEquals("breadlb", boxes.last().text)
  }

  @Test
  fun keepsSeparateLinesAndEmptyInput() {
    assertEquals(emptyList<String>(), OcrReadingOrder.lines(emptyList()))
    assertEquals(
      listOf("top", "bottom"),
      OcrReadingOrder.lines(
        listOf(OcrDisplayBox(0, 30, 20, 40, "bottom"), OcrDisplayBox(0, 0, 20, 10, "top"))
      ),
    )
  }

  @Test
  fun overlaySizesScaleWithImageWidth() {
    assertEquals(2f * OverlaySizing.text(540), OverlaySizing.text(1080), 0.0001f)
    assertEquals(2f * OverlaySizing.stroke(540), OverlaySizing.stroke(1080), 0.0001f)
  }
}
