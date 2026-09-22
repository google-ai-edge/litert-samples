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

package com.google.ai.edge.examples.model_zoo.models.tiger

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TigerMathTest {
  @Test
  fun conjugateSpectrumAndCenterTrimProduceImpulseAtExpectedSample() {
    val frames = 3
    // A constant all-bin spectrum is a unit impulse at each frame start.
    val real = FloatArray(Istft.ENC * frames) { 1f }
    val imaginary = FloatArray(real.size)
    val output = Istft.run(real, imaginary, frames, 1024)
    // Periodic Hann is zero at every impulse's frame start, so overlap-add is zero.
    assertEquals(1024, output.size)
    assertTrue(output.all { it.isFinite() && kotlin.math.abs(it) < 1e-6f })
  }
}
