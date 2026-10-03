// Vendored from https://huggingface.co/litert-community/Julia-1-LiteRT/blob/8f36857c56e891c023060586759c6cdc8baf6b3e/android/app/src/main/java/com/julia1/JuliaDecoder.kt (Apache-2.0)
// SPDX-License-Identifier: Apache-2.0
// Port of julia/typed.py answer arithmetic: float64 softmax at temperature 1, no calibration.
package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.julia

import kotlin.math.exp

/** Turns the marker logits of one request into the author's named-question answer. */
object JuliaDecoder {
  /** One answer in the author's schema; `probabilities` follow the order of `question.keys`. */
  class Answer(
    val question: JuliaQuestion,
    val probabilities: DoubleArray,
    val choice: String?,
    val score: Double?,
    val noul: Double?,
    val maxProbability: Double?,
  ) {
    /** Key of the highest probability, first maximum on ties, as the author's `max` picks it. */
    val argmax: Int
      get() {
        var best = 0
        for (index in 1 until probabilities.size) {
          if (probabilities[index] > probabilities[best]) {
            best = index
          }
        }
        return best
      }
  }

  /** Reads the option logits at the builder's marker positions. */
  fun gather(tokenLogits: FloatArray, markers: IntArray): DoubleArray =
    DoubleArray(markers.size) { tokenLogits[markers[it]].toDouble() }

  /** `p_i = exp(z_i - max) / sum`, computed in float64 with a sequential sum like typed.py. */
  fun softmax(logits: DoubleArray): DoubleArray {
    require(logits.isNotEmpty()) { "At least one logit is required" }
    val maximum = logits.max()
    val exponents = DoubleArray(logits.size) { exp(logits[it] - maximum) }
    var total = 0.0
    for (value in exponents) {
      total += value
    }
    return DoubleArray(logits.size) { exponents[it] / total }
  }

  /** Choice = winning ID, score = expected zero-based rubric index, not rounded, noul = P(true). */
  fun decode(markerLogits: DoubleArray, question: JuliaQuestion): Answer {
    require(markerLogits.size == question.keys.size) { "Logits/criteria option count mismatch" }
    check(markerLogits.all { it.isFinite() }) { "Nonfinite model scores" }
    val p = softmax(markerLogits)
    return when (question.type) {
      "choice" -> {
        var best = 0
        for (index in 1 until p.size) {
          if (p[index] > p[best]) {
            best = index
          }
        }
        Answer(question, p, question.keys[best], null, null, p.max())
      }
      "score" -> {
        var expectation = 0.0
        for (index in p.indices) {
          expectation += index * p[index]
        }
        Answer(question, p, null, expectation, null, p.max())
      }
      else -> Answer(question, p, null, null, p[1], null)
    }
  }
}
