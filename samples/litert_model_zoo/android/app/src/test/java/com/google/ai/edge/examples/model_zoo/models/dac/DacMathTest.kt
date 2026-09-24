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

package com.google.ai.edge.examples.model_zoo.models.dac

import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DacMathTest {
  @Test
  fun residualQuantizerUsesNormalizedCosineArgmaxAndRawCodebookForDecode() {
    val bookFloats =
      DacRVQ.SIZE * DacRVQ.DIM +
        DacRVQ.DIM * DacRVQ.HID +
        DacRVQ.DIM +
        DacRVQ.HID * DacRVQ.DIM +
        DacRVQ.HID
    val weights = ByteBuffer.allocate(DacRVQ.NQ * bookFloats * 4).order(ByteOrder.LITTLE_ENDIAN)
    weights.putFloat(0, 1f)
    weights.putFloat(DacRVQ.DIM * 4, -2f)
    val inProj = DacRVQ.SIZE * DacRVQ.DIM
    weights.putFloat(inProj * 4, 1f)
    val outProj = inProj + DacRVQ.DIM * DacRVQ.HID + DacRVQ.DIM
    weights.putFloat(outProj * 4, 1f)
    val rvq = DacRVQ(weights.array())
    val latent = FloatArray(DacRVQ.HID).also { it[0] = -5f }
    val codes = rvq.encode(latent, 1)
    assertEquals(12, codes.size)
    assertEquals(1, codes[0])
    assertTrue(codes.drop(1).all { it == 0 })
    val decoded = rvq.decode(codes, 1)
    assertEquals(-2f, decoded[0], 0f)
    assertTrue(decoded.drop(1).all { it == 0f })
  }
}
