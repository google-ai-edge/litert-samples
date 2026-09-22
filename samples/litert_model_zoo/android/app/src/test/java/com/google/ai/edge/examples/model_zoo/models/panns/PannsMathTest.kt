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

package com.google.ai.edge.examples.model_zoo.models.panns

import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertEquals
import org.junit.Test

class PannsMathTest {
  @Test
  fun periodicHannCenteredImpulseHasUnitDcPowerAndSilentMelsFloorAtMinus100() {
    val weights =
      ByteBuffer.allocate(MelSpectrogram.N_MELS * MelSpectrogram.N_FREQS * 4)
        .order(ByteOrder.LITTLE_ENDIAN)
    weights.putFloat(0, 1f)
    val output = MelSpectrogram(weights.array()).compute(floatArrayOf(1f))
    assertEquals(1001 * 64, output.size)
    assertEquals(0f, output[0], 1e-6f)
    assertEquals(-100f, output[1], 0f)
    assertEquals(-100f, output.last(), 0f)
  }
}
