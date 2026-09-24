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

package com.google.ai.edge.examples.model_zoo.models.da3

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class DA3MathTest {
  @Test
  fun depthVisualizationCropsBeforePercentilesAndInvertsDisparity() {
    // Outside values are deliberately extreme; only the middle two columns enter normalization.
    val result =
      DA3Result(
        floatArrayOf(0.001f, 1f, 2f, 999f, 0.001f, 4f, 0f, 999f),
        4,
        2,
        1,
        0,
        3,
        2,
        0,
        "CPU",
      )
    // 4 samples: 2% index=0, 98% index=2, lo=0, hi=.5. Disparities 1,.5,.25,0.
    // Spectral indices are 0,0,127,255; endpoints also verify inversion.
    assertArrayEquals(
      intArrayOf(0xff9e0142.toInt(), 0xff9e0142.toInt(), 0xfffffebe.toInt(), 0xff5e4fa2.toInt()),
      result.depthPixels(),
    )
  }

  @Test
  fun constantDepthHasStableOpaqueVisualization() {
    val result = DA3Result(floatArrayOf(2f, 2f), 2, 1, 0, 0, 2, 1, 0, "CPU")
    assertArrayEquals(intArrayOf(0xff5e4fa2.toInt(), 0xff5e4fa2.toInt()), result.depthPixels())
    assertEquals(2, result.depthPixels().size)
  }
}
