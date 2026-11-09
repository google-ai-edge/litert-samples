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

package com.google.aiedge.examples.digit_classifier

import com.google.ai.edge.litert.Accelerator
import com.google.aiedge.examples.digit_classifier.DigitClassificationHelper.AcceleratorEnum
import org.junit.Assert.assertEquals
import org.junit.Test

class DigitClassificationHelperTest {
    @Test
    fun toAccelerator_mapsAllSupportedBackends() {
        assertEquals(Accelerator.CPU, DigitClassificationHelper.toAccelerator(AcceleratorEnum.CPU))
        assertEquals(Accelerator.GPU, DigitClassificationHelper.toAccelerator(AcceleratorEnum.GPU))
        assertEquals(Accelerator.NPU, DigitClassificationHelper.toAccelerator(AcceleratorEnum.NPU))
        assertEquals(3, AcceleratorEnum.entries.size)
    }

    @Test
    fun findResult_returnsHighestProbabilityDigitAndScore() {
        val scores = floatArrayOf(0.01f, 0.02f, 0.05f, 0.78f, 0.04f, 0.03f, 0.02f, 0.02f, 0.02f, 0.01f)
        val (digit, score) = DigitClassificationHelper.findResult(scores)
        assertEquals(3, digit)
        assertEquals(0.78f, score, 1e-6f)
    }

    @Test
    fun findResult_handlesEmptyArray() {
        val (digit, score) = DigitClassificationHelper.findResult(floatArrayOf())
        assertEquals(-1, digit)
        assertEquals(0f, score, 1e-6f)
    }
}
