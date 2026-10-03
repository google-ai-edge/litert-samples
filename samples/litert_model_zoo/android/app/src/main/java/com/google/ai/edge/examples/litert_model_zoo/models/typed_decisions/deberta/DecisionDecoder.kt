// Vendored from https://huggingface.co/litert-community/Open-Decision-DeBERTa-v3-Large-LiteRT/blob/7a276235b795e8ad3ae7ac6a9f237daa2098863a/android/sample/app/src/main/java/com/opendecision/DecisionDecoder.kt (Apache-2.0)
// formatted for this repository's 100-column and brace rules; no logic change
// one comment no longer names the source model
package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.deberta

import kotlin.math.exp

/**
 * Host read-out of the graph's `logits [1,1,1,128]` leaf, as the author's `decide()` and
 * `schema.readout`: the slots are consumed in request order, each question's options get a softmax
 * at the shipped temperature 1.05 (the author's config), then `choice` = the best option with its
 * probability, `score` = the expected zero-based level with the best level's probability, `noul` =
 * p(yes).
 */
object DecisionDecoder {
  const val TEMPERATURE = 1.05

  /** One answered question. [probabilities] has one entry per option in request order. */
  data class Answer(
    val question: Question,
    val probabilities: DoubleArray,
    val logits: FloatArray,
  ) {
    val best: Int
      get() = argmax(probabilities)

    /** `choice`: the winning option; `score`: the best level; `noul`: "yes" or "no". */
    val label: String
      get() = question.options[best]

    val confidence: Double
      get() = probabilities[best]

    /** `score` only: Σ i·pᵢ, the expected zero-based level, which may fall between levels. */
    val expectedLevel: Double
      get() = probabilities.withIndex().sumOf { (i, p) -> i * p }

    /** `noul` only: p(yes). */
    val yes: Double
      get() = probabilities[1]
  }

  fun decode(logits: FloatArray, questions: List<Question>): List<Answer> {
    val count = questions.sumOf { it.options.size }
    require(logits.size >= count) { "Graph returned ${logits.size} logits for $count options" }
    var offset = 0
    return questions.map { question ->
      val part = logits.copyOfRange(offset, offset + question.options.size)
      offset += question.options.size
      Answer(question, softmax(part, TEMPERATURE), part)
    }
  }

  /** Softmax of [logits] / [temperature] in double precision (max-shifted). */
  fun softmax(logits: FloatArray, temperature: Double): DoubleArray {
    val scaled = DoubleArray(logits.size) { logits[it].toDouble() / temperature }
    val max = scaled.max()
    val out = DoubleArray(scaled.size) { exp(scaled[it] - max) }
    val sum = out.sum()
    for (i in out.indices) {
      out[i] /= sum
    }
    return out
  }

  /** First index of the maximum, as `max(range(n), key=...)` in the author's read-out. */
  fun argmax(values: DoubleArray): Int {
    var best = 0
    for (i in 1 until values.size) {
      if (values[i] > values[best]) {
        best = i
      }
    }
    return best
  }
}
