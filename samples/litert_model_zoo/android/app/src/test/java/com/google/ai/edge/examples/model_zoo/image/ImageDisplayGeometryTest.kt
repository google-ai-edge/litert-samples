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

class ImageDisplayGeometryTest {
  @Test
  fun squareSceneResultsUseTheFullSourcePhotoAspectRatio() {
    for (task in
      listOf(
        "background-removal",
        "portrait-matting",
        "super-resolution",
        "dense-feature-visualization",
        "crowd-counting",
      )) {
      assertEquals(4f / 3, ImageDisplayGeometry.resultAspectRatio(task, 960, 720, 512, 512), 0f)
      assertEquals(3f / 4, ImageDisplayGeometry.resultAspectRatio(task, 720, 960, 512, 512), 0f)
    }
  }

  @Test
  fun letterboxedOcrAndCombinedMatchingKeepTheirOwnCanvasGeometry() {
    assertEquals(1f, ImageDisplayGeometry.resultAspectRatio("ocr", 960, 720, 960, 960), 0f)
    assertEquals(
      8f / 3,
      ImageDisplayGeometry.resultAspectRatio("image-matching", 960, 720, 1280, 480),
      0f,
    )
  }

  @Test
  fun missingInputUsesActualOutputGeometry() {
    assertEquals(
      2f,
      ImageDisplayGeometry.resultAspectRatio("super-resolution", null, null, 800, 400),
      0f,
    )
  }

  @Test
  fun matchingPanelsPreserveBothSourceAspectRatios() {
    assertEquals(640, ImageDisplayGeometry.matchingPanelWidth(960, 720, 480))
    assertEquals(360, ImageDisplayGeometry.matchingPanelWidth(720, 960, 480))
    val width = ImageDisplayGeometry.matchingPanelWidth(960, 611, 480)
    assertEquals(960.0 / 611, width / 480.0, 1.0 / 480)
  }
}
