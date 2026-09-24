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

import com.google.ai.edge.examples.model_zoo.models.crowdcount.CrowdCounter
import com.google.ai.edge.examples.model_zoo.models.dehaze.Dehazer
import com.google.ai.edge.examples.model_zoo.models.dinov2.Dinov2Features
import com.google.ai.edge.examples.model_zoo.models.modnet.Matter
import com.google.ai.edge.examples.model_zoo.models.nima.NimaScorer
import com.google.ai.edge.examples.model_zoo.models.plantnet.PlantClassifier
import com.google.ai.edge.examples.model_zoo.models.vrwkv.VrwkvClassifier
import com.google.ai.edge.examples.model_zoo.models.xfeat.XFeatMatcher
import org.junit.Assert.*
import org.junit.Test

/** Calls the production pure Kotlin methods without loading Android bitmaps or a native model. */
class RemainingImageMathTest {
  private fun <T : Any> pureInstance(type: Class<T>): T {
    val allocatorClass = Class.forName("sun.misc.Unsafe")
    val field = allocatorClass.getDeclaredField("theUnsafe").apply { isAccessible = true }
    val instance =
      allocatorClass.getMethod("allocateInstance", Class::class.java).invoke(field.get(null), type)
    return requireNotNull(type.cast(instance))
  }

  @Test
  fun dehazeClampsAndRoundsModelOutputRange() {
    val runner = pureInstance(Dehazer::class.java)
    assertEquals(0, runner.toByteRange(-2f))
    assertEquals(128, runner.toByteRange(0f))
    assertEquals(255, runner.toByteRange(2f))
  }

  @Test
  fun portraitCompositePreservesOpaqueForegroundAndTransparentBackground() {
    val output = IntArray(3)
    Matter.compositePixels(
      floatArrayOf(0f, 0.5f, 1f),
      IntArray(3) { 0xffff0000.toInt() },
      0xff0000ff.toInt(),
      output,
      3,
    )
    assertArrayEquals(
      intArrayOf(0xff0000ff.toInt(), 0xff7f007f.toInt(), 0xffff0000.toInt()),
      output,
    )
  }

  @Test
  fun nimaMeanUsesOneBasedQualityBins() {
    val runner = pureInstance(NimaScorer::class.java)
    assertEquals(5.5f, runner.meanScore(FloatArray(10) { 0.1f }), 1e-6f)
    assertEquals(10f, runner.meanScore(FloatArray(10) { if (it == 9) 1f else 0f }), 0f)
  }

  @Test
  fun plantNetStableSoftmaxKeepsSpeciesOrdering() {
    val result =
      PlantClassifier.decode(floatArrayOf(999f, 1000f, 998f), arrayOf("oak", "pine", "grass"), 3)
    assertEquals(listOf("pine", "oak", "grass"), result.map { it.first })
    assertEquals(1f, result.sumOf { it.second.toDouble() }.toFloat(), 1e-6f)
    assertEquals(0.66524094f, result[0].second, 1e-6f)
  }

  @Test
  fun visionRwkvSoftmaxRanksLabelsAndRemainsFiniteForLargeLogits() {
    val runner = pureInstance(VrwkvClassifier::class.java)
    VrwkvClassifier::class
      .java
      .getDeclaredField("labels")
      .apply { isAccessible = true }
      .set(runner, listOf("dog", "cat", "truck"))
    val result = runner.topK(floatArrayOf(1000f, 999f, 1001f), 3)
    assertEquals(listOf("truck", "dog", "cat"), result.map { it.label })
    assertEquals(1f, result.sumOf { it.probability.toDouble() }.toFloat(), 1e-6f)
  }

  @Test
  fun crowdCountSumsRawDensityWithoutDisplayNormalization() {
    assertEquals(2f, CrowdCounter.sumDensity(floatArrayOf(0f, 0.25f, 0.5f, 1.25f)), 0f)
    assertEquals(0f, CrowdCounter.sumDensity(FloatArray(4)), 0f)
  }

  @Test
  fun dinov2PcaVectorNormalizationHandlesZeroAndNonzeroVectors() {
    val runner = pureInstance(Dinov2Features::class.java)
    assertArrayEquals(floatArrayOf(0.6f, 0.8f), runner.normalize(floatArrayOf(3f, 4f)), 1e-6f)
    assertArrayEquals(floatArrayOf(0f, 0f), runner.normalize(FloatArray(2)), 0f)
  }

  @Test
  fun xfeatMatchingRequiresMutualNearestNeighborsAndCosineFloor() {
    val runner = pureInstance(XFeatMatcher::class.java)
    fun features(desc: Array<FloatArray>) =
      XFeatMatcher.Features(
        FloatArray(desc.size) { it * 10f },
        FloatArray(desc.size) { it * 20f },
        FloatArray(desc.size) { 1f },
        desc,
      )
    val x = FloatArray(64) { if (it == 0) 1f else 0f }
    val y = FloatArray(64) { if (it == 1) 1f else 0f }
    val matches = runner.match(features(arrayOf(x, y)), features(arrayOf(y, x)))
    assertEquals(2, matches.size)
    assertEquals(10f, matches[0].x1, 0f)
    assertEquals(1f, matches[0].sim, 0f)
    assertTrue(runner.match(features(arrayOf(x)), features(arrayOf(y))).isEmpty())
    assertTrue(runner.match(features(emptyArray()), features(arrayOf(x))).isEmpty())
  }
}
