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

package com.google.ai.edge.examples.model_zoo.models.realesrgan

import org.junit.Assert.assertArrayEquals
import org.junit.Test

class RealEsrganMathTest {
  @Test
  fun catalogNchwOutputIsReorderedIntoOriginalNhwcDecode() {
    val planar = floatArrayOf(1f, 2f, 10f, 20f, 100f, 200f)
    val output = RealEsrganUpscaler.outputFromNchw(planar)
    assertArrayEquals(floatArrayOf(1f, 10f, 100f, 2f, 20f, 200f), output, 0f)
    assertArrayEquals(floatArrayOf(1f, 2f, 10f, 20f, 100f, 200f), planar, 0f)
  }

  @Test
  fun interleavedRgbClampsChannelsAndTruncatesToOpaquePixels() {
    val pixels = IntArray(2)
    RealEsrganUpscaler.fillNhwcPixels(floatArrayOf(-0.1f, 0.5f, 1f, 1.2f, 0.25f, 0f), pixels)
    assertArrayEquals(intArrayOf(0xFF007FFF.toInt(), 0xFFFF3F00.toInt()), pixels)
  }
}
