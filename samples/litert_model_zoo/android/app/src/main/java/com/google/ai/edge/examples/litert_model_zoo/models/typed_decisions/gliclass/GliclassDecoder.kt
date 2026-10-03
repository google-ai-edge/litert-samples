// Vendored from https://huggingface.co/litert-community/GLiClass-Edge-v3.0-LiteRT/blob/88c90950587eb951974c094eef91afa0fe3552c0/android/sample/app/src/main/java/com/gliclass/GliclassDecoder.kt (Apache-2.0)
package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.gliclass

import kotlin.math.exp

/**
 * Host post-processing of the graph's `logits [1,1,1,25]`, ported from `gliclass` 0.1.20
 * `_postprocess_logits`: the first n slots belong to the n labels (later slots are never read).
 * Single-label = softmax, the top label with its probability (first maximum on ties, as
 * `torch.argmax`). Multi-label = sigmoid, every label whose probability is at least the threshold,
 * in input order, possibly none. Probabilities are float32 like torch's; the threshold comparison
 * widens the float32 probability to double like Python's `float(score) >= threshold`.
 */
object GliclassDecoder {
  /** Classification mode, named as the pipeline's `classification_type`. */
  enum class Mode(val key: String) {
    SINGLE_LABEL("single-label"),
    MULTI_LABEL("multi-label"),
  }

  /** One returned label and its probability (the pipeline's `{"label", "score"}`). */
  data class Prediction(val label: String, val score: Float)

  /**
   * The decision for one request. [predictions] is the pipeline's output; [probabilities] holds
   * every label's softmax (single) or sigmoid (multi) value and [chosen] the indices of the
   * selected labels, for display.
   */
  data class Decision(
    val mode: Mode,
    val threshold: Double,
    val labels: List<String>,
    val logits: FloatArray,
    val probabilities: FloatArray,
    val chosen: List<Int>,
    val predictions: List<Prediction>,
  )

  /** Decides one request from at least `labels.size` logits. */
  fun decide(
    logits: FloatArray,
    labels: List<String>,
    mode: Mode,
    threshold: Double = DEFAULT_THRESHOLD,
  ): Decision {
    require(labels.isNotEmpty() && logits.size >= labels.size) {
      "Graph returned ${logits.size} logits for ${labels.size} labels"
    }
    val values = logits.copyOf(labels.size)
    if (mode == Mode.SINGLE_LABEL) {
      val probabilities = softmax(values)
      val best = argmax(probabilities)
      return Decision(
        mode,
        threshold,
        labels,
        values,
        probabilities,
        listOf(best),
        listOf(Prediction(labels[best], probabilities[best])),
      )
    }
    val probabilities = sigmoid(values)
    val chosen = labels.indices.filter { probabilities[it].toDouble() >= threshold }
    // The pipeline builds {label: score}: a repeated label keeps its first position, last score.
    val byLabel = LinkedHashMap<String, Float>()
    labels.forEachIndexed { index, label -> byLabel[label] = probabilities[index] }
    val predictions =
      byLabel.entries
        .filter { it.value.toDouble() >= threshold }
        .map { Prediction(it.key, it.value) }
    return Decision(mode, threshold, labels, values, probabilities, chosen, predictions)
  }

  /** torch's CPU float32 softmax shape: max-shift, exp, sum, multiply by the reciprocal. */
  fun softmax(logits: FloatArray): FloatArray {
    var max = logits[0]
    for (value in logits) {
      if (value > max) {
        max = value
      }
    }
    val out = FloatArray(logits.size) { exp((logits[it] - max).toDouble()).toFloat() }
    var sum = 0f
    for (value in out) {
      sum += value
    }
    val inverse = 1f / sum
    for (index in out.indices) {
      out[index] *= inverse
    }
    return out
  }

  /** torch's CPU float32 sigmoid shape: 1 / (1 + exp(-x)). */
  fun sigmoid(logits: FloatArray): FloatArray =
    FloatArray(logits.size) { 1f / (1f + exp((-logits[it]).toDouble()).toFloat()) }

  /** First index of the maximum, as `torch.argmax`. */
  fun argmax(values: FloatArray): Int {
    var best = 0
    for (index in 1 until values.size) {
      if (values[index] > values[best]) {
        best = index
      }
    }
    return best
  }

  /** The pipeline's default multi-label threshold. */
  const val DEFAULT_THRESHOLD = 0.5
}
