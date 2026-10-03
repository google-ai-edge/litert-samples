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

package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions

import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.deberta.DebertaDecider
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.gliclass.GliclassDecider
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.gliner.GlinerDecider
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.julia.JuliaDecider
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.laya.LayaDecider
import java.io.File

/** One multiple-choice question about [text]; [options] are (key, description) in display order. */
data class TextDecisionRequest(
  val text: String,
  val question: String,
  val options: List<Pair<String, String>>,
) {
  init {
    require(text.isNotBlank()) { "Enter the text to ask about." }
    require(question.isNotBlank()) { "Enter a question." }
    require(options.size >= 2) { "Write at least two options as key: description." }
    require(options.map { it.first }.distinct().size == options.size) {
      "Each option needs its own key."
    }
  }

  /** The string a label or option model reads: the description, or the key when it has none. */
  fun optionText(index: Int): String = options[index].second.ifBlank { options[index].first }

  companion object {
    /**
     * Parses `key: description` rows. A row without a colon is a key alone; blank rows are
     * skipped.
     */
    fun parseOptions(rows: List<String>): List<Pair<String, String>> =
      rows
        .filter { it.isNotBlank() }
        .map { row ->
          val colon = row.indexOf(':')
          if (colon < 0) {
            row.trim() to ""
          } else {
            row.substring(0, colon).trim() to row.substring(colon + 1).trim()
          }
        }
        .also { parsed ->
          require(parsed.all { it.first.isNotEmpty() }) {
            "Each option needs a key before the colon."
          }
        }
  }
}

/**
 * The answer to one [TextDecisionRequest]: every option's probability in request order, the
 * chosen key, and the wall time of the call from tokenization to the decoded answer.
 */
data class TextDecisionResult(
  val probabilities: List<Pair<String, Float>>,
  val answerKey: String,
  val ms: Double,
  val backend: String,
  val window: Int,
  val fallbackReason: String? = null,
  val details: String = "",
)

/** Construct, decide and close on the same confined worker dispatcher. */
interface TextDecisionEngine : AutoCloseable {
  fun decide(request: TextDecisionRequest): TextDecisionResult

  override fun close()
}

object TextDecisionTasks {
  val ids =
    setOf(
      "text-decision-laya-multilingual",
      "text-decision-julia-1",
      "text-decision-gliclass-edge",
      "text-decision-open-decision",
      "text-decision-gliner-decide",
    )

  fun create(taskId: String, directory: File, backend: String): TextDecisionEngine =
    when (taskId) {
      "text-decision-laya-multilingual" -> LayaDecider(directory, backend)
      "text-decision-julia-1" -> JuliaDecider(directory, backend)
      "text-decision-gliclass-edge" -> GliclassDecider(directory, backend)
      "text-decision-open-decision" -> DebertaDecider(directory, backend)
      "text-decision-gliner-decide" -> GlinerDecider(directory, backend)
      else -> error("Unknown text task: $taskId")
    }
}
