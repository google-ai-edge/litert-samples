package com.example.mobilenetlitert

import kotlin.math.exp

object ClassificationPostprocessor {
  fun topK(logits: FloatArray, labels: List<String>, k: Int = 5): List<Classification> {
    if (logits.isEmpty()) return emptyList()

    val maxLogit = logits.max()
    val probabilities = FloatArray(logits.size)
    var probabilitySum = 0.0

    logits.forEachIndexed { index, logit ->
      val value = exp((logit - maxLogit).toDouble())
      probabilities[index] = value.toFloat()
      probabilitySum += value
    }

    return probabilities
      .indices
      .asSequence()
      .sortedByDescending { probabilities[it] }
      .take(k.coerceAtMost(probabilities.size))
      .map { index ->
        Classification(
          label = labels.getOrNull(index) ?: "class_$index",
          score = (probabilities[index] / probabilitySum).toFloat(),
          index = index,
        )
      }
      .toList()
  }
}
