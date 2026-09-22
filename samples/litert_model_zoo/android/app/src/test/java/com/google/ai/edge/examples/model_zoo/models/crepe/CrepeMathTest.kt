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

package com.google.ai.edge.examples.model_zoo.models.crepe

import kotlin.math.PI
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CrepeMathTest {
  @Test
  fun normalizationAndWeightedPitchDecodeRetainZooConventions() {
    val frame = FloatArray(1024) { (2.0 + 3.0 * sin(2 * PI * it / 1024)).toFloat() }
    val normalized = PitchDetector.normalizeFrame(frame)
    assertEquals(0.0, normalized.sumOf { it.toDouble() } / 1024, 1e-5)
    assertEquals(1.0, normalized.sumOf { it.toDouble() * it } / 1024, 1e-5)
    val activations = FloatArray(360)
    activations[227] = 0.25f
    activations[228] = 0.75f
    val pitch = PitchDetector.decodeActivations(activations)
    assertEquals("A", pitch.note)
    assertEquals(4, pitch.octave)
    assertEquals(0.75f, pitch.confidence, 0f)
    assertTrue(pitch.hz in 439f..442f)
  }
}
