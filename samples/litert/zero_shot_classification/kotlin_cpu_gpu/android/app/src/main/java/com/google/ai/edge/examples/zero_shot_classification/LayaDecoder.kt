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

// NumPy float32 host decoder port; source attribution is retained in NOTICE.
package com.google.ai.edge.examples.zero_shot_classification

import java.math.BigDecimal
import java.math.RoundingMode
import kotlin.math.exp
import kotlin.math.ln

/** Float32 host decoding with captured NumPy reduction order and Python rounding. */
object LayaDecoder {
  /** Reads option logits in the builder's marker order. */
  fun gather(tokenLogits: FloatArray, markers: IntArray): FloatArray =
    FloatArray(markers.size) { tokenLogits[markers[it]] }

  /** Stable softmax using the reference's float32 exponentiation and reduction. */
  fun softmax(values: FloatArray): FloatArray {
    require(values.isNotEmpty()) { "At least one logit is required" }
    val max = values.maxOrNull()!!
    val exponents = FloatArray(values.size) { exp((values[it] - max).toDouble()).toFloat() }
    val sum = numpySum(exponents)
    return FloatArray(values.size) { exponents[it] / sum }
  }

  /** Mirrors NumPy's float32 pairwise reduction (including the 8 accumulator order). */
  private fun numpySum(values: FloatArray, from: Int = 0, size: Int = values.size): Float {
    if (size < 8) {
      var sum = -0.0f
      for (i in from until from + size) sum += values[i]
      return sum
    }
    if (size <= 128) {
      val sums = FloatArray(8) { values[from + it] }
      var i = 8
      while (i < size - size % 8) {
        for (j in 0..7) sums[j] += values[from + i + j]
        i += 8
      }
      var sum =
        ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]))
      while (i < size) {
        sum += values[from + i]
        i++
      }
      return sum
    }
    val left = (size / 2) - (size / 2) % 8
    return numpySum(values, from, left) + numpySum(values, from + left, size - left)
  }

  /** Raw, uncalibrated probabilities supply the act graph's four features. */
  fun actFeatures(rawLogits: FloatArray): FloatArray {
    val p = softmax(rawLogits)
    val k = maxOf(p.size, 2)
    val sorted = p.sortedDescending()
    val top1 = sorted[0]
    val top2 = sorted.getOrElse(1) { 0.0f }
    val terms = FloatArray(p.size) { p[it] * ln(maxOf(p[it], 1e-9f).toDouble()).toFloat() }
    val entropy = -numpySum(terms) / ln(k.toFloat().toDouble()).toFloat()
    return floatArrayOf(top1, top1 - top2, entropy, k.toFloat() / 255.0f)
  }

  /** Applies option-bucket calibration before the raw softmax; temperature is floored at 1e-3. */
  fun probabilities(
    rawLogits: FloatArray,
    question: LayaQuestion,
    calibration: LayaCalibration = LayaCalibration.identity(),
  ): FloatArray {
    // NumPy 2.x weak scalar promotion keeps this division in float32.
    val temperature = maxOf(1e-3, calibration.temperature(question.type, rawLogits.size)).toFloat()
    return softmax(FloatArray(rawLogits.size) { rawLogits[it] / temperature })
  }

  /** Returns normalized entropy confidence, separate from the action-head probability. */
  fun confidence(p: FloatArray): Double {
    if (p.size < 2) return 1.0
    val terms = FloatArray(p.size) { p[it] * ln(p[it].coerceIn(1e-12f, 1.0f).toDouble()).toFloat() }
    val entropy = -numpySum(terms)
    // np.float32 / Python float and Python float - np.float32 remain float32 in NumPy 2.x.
    return (1.0f - entropy / ln(p.size.toDouble()).toFloat()).coerceIn(0.0f, 1.0f).toDouble()
  }

  /**
   * BigDecimal(double), not valueOf(double), rounds the represented binary value as Python does.
   */
  fun round4(value: Double): Double {
    if (!value.isFinite()) return value
    val rounded = BigDecimal(value).setScale(4, RoundingMode.HALF_EVEN).toDouble()
    return if (rounded == 0.0 && java.lang.Double.doubleToRawLongBits(value) < 0) -0.0 else rounded
  }

  /** Produces the official four-decimal answer schema without changing the question order. */
  fun decode(
    rawLogits: FloatArray,
    actLogits: FloatArray,
    question: Map<String, Any?>,
    calibration: LayaCalibration = LayaCalibration.identity(),
  ): Map<String, Any?> =
    decode(rawLogits, actLogits, LayaPromptBuilder.normalize(question), calibration)

  /** Produces the official four-decimal answer schema without changing the question order. */
  fun decode(
    rawLogits: FloatArray,
    actLogits: FloatArray,
    question: LayaQuestion,
    calibration: LayaCalibration = LayaCalibration.identity(),
  ): Map<String, Any?> {
    require(rawLogits.size == question.optionCount) { "Logits/criteria option count mismatch" }
    require(actLogits.size == 2) { "Act head must return two logits" }
    val p = probabilities(rawLogits, question, calibration)
    val action = linkedMapOf("act_probability" to round4(softmax(actLogits)[0].toDouble()))
    return when (question.type) {
      "choice" -> {
        val labels = LayaJson.asObject(question.criteria).keys.toList()
        var argmax = 0
        for (index in 1 until p.size) if (p[index] > p[argmax]) argmax = index
        linkedMapOf(
          "type" to "choice",
          "choice" to labels[argmax],
          "probabilities" to
            labels.indices.associateTo(LinkedHashMap()) { labels[it] to round4(p[it].toDouble()) },
          "confidence" to round4(confidence(p)),
          "action" to action,
        )
      }
      "score" -> {
        val criteria = LayaJson.asArray(question.criteria)
        // NumPy arange(int64) * float32 promotes each multiplication and reduction to float64.
        val expectation = p.indices.sumOf { it.toDouble() * p[it].toDouble() }
        linkedMapOf(
          "type" to "score",
          "score" to round4(expectation),
          "legend" to
            criteria.indices.associateTo(LinkedHashMap()) { it.toString() to criteria[it] },
          "probabilities" to
            p.indices.associateTo(LinkedHashMap()) { it.toString() to round4(p[it].toDouble()) },
          "confidence" to round4(confidence(p)),
          "action" to action,
        )
      }
      "noul" -> {
        val trueProbability = p[1].toDouble()
        linkedMapOf(
          "type" to "noul",
          "noul" to round4(trueProbability),
          "confidence" to round4(maxOf(trueProbability, 1.0 - trueProbability)),
          "action" to action,
        )
      }
      else -> error("Unknown question type: ${question.type}")
    }
  }
}
