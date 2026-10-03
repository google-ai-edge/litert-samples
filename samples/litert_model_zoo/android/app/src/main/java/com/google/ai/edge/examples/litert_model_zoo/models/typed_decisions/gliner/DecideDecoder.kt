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

// Vendored from https://huggingface.co/litert-community/GLiNER2.5-Decide-LiteRT/blob/db80197282d11373df084c0ceed67a54544cfa84/android/app/src/main/java/com/gliner25decide/DecideDecoder.kt (Apache-2.0)
package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.gliner

import kotlin.math.exp

/**
 * Host post-processing for the graph's `logits [1,1,1,32]` leaf, ported from
 * `host_assets/runtime/host_decide.py` in the model repository = gliner2 2.0.0
 * `inference/runtime.py:_extract_classification_result` at temperature 1.0 (dividing by 1.0 is the
 * identity in float32, so it is omitted).
 *
 * Slots are consumed in request order (task order, label order inside a task); slots past the last
 * label hold `classifier(0)` and are never read. Per task: sigmoid for multi-label or softmax
 * otherwise (unless `class_act` forces one), then argmax, or every label whose probability is at
 * least `cls_threshold` with the argmax label alone when none is. Probabilities are float32 like
 * torch's; ties keep the first index like `torch.argmax`.
 */
object DecideDecoder {
  /** One head's decision. Single-label heads always choose exactly one label. */
  data class TaskDecision(
    val task: String,
    val multiLabel: Boolean,
    val labels: List<String>,
    val chosenProbabilities: List<Float>,
    val probabilities: FloatArray,
    val logits: FloatArray,
  )

  /** Splits [logits] by task in request order and decides every task. */
  fun decode(logits: FloatArray, tasks: List<Task>): List<TaskDecision> {
    val labelCount = tasks.sumOf { it.labels.size }
    require(logits.size >= labelCount) {
      "Graph returned ${logits.size} logits for $labelCount labels"
    }
    var offset = 0
    return tasks.map { task ->
      val part = logits.copyOfRange(offset, offset + task.labels.size)
      offset += task.labels.size
      decide(part, task)
    }
  }

  /** `decide_task` in host_decide.py for one head's logits. */
  fun decide(logits: FloatArray, task: Task): TaskDecision {
    require(logits.size == task.labels.size && logits.isNotEmpty())
    val probabilities = activate(logits, task)
    if (task.multiLabel) {
      // Python compares probs[j].item() (a float32 widened to a double) with the double threshold.
      var chosen = task.labels.indices.filter { probabilities[it].toDouble() >= task.clsThreshold }
      if (chosen.isEmpty()) {
        chosen = listOf(argmax(probabilities))
      }
      return TaskDecision(
        task.name,
        true,
        chosen.map { task.labels[it] },
        chosen.map { probabilities[it] },
        probabilities,
        logits,
      )
    }
    val best = argmax(probabilities)
    return TaskDecision(
      task.name,
      false,
      listOf(task.labels[best]),
      listOf(probabilities[best]),
      probabilities,
      logits,
    )
  }

  /** The activation gliner2 applies: `class_act` if forced, else sigmoid iff multi-label. */
  fun activate(logits: FloatArray, task: Task): FloatArray =
    when (task.activation) {
      Activation.SIGMOID -> sigmoid(logits)
      Activation.SOFTMAX -> softmax(logits)
      Activation.AUTO ->
        if (task.multiLabel) {
          sigmoid(logits)
        } else {
          softmax(logits)
        }
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
}
