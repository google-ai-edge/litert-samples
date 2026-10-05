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

package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.deberta

import com.google.ai.edge.examples.litert_model_zoo.image.compileImageBackend
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.DecisionGraph
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.TextDecisionEngine
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.TextDecisionRequest
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.TextDecisionResult
import java.io.File

/**
 * Open-Decision DeBERTa-v3-large (S256) through the model card's vendored host: one choice question
 * whose options are the descriptions (a key stands for itself when it has none), the text as the
 * state (cut at 256 tokens by the author's collator), and a softmax at the author's temperature
 * 1.05.
 */
class DebertaDecider(modelDir: File, backend: String = "gpu") : TextDecisionEngine {
  private val inputs = DecisionInputs(DecisionTokenizer(File(modelDir, "tokenizer.json")))
  private val table = DecisionInputs.EmbeddingTable(File(modelDir, "word_embeddings_fp16.bin"))
  private val loaded =
    try {
      compileImageBackend(backend, TAG) { accelerator ->
        DecisionGraph.create(
          File(modelDir, GRAPH),
          accelerator,
          listOf("inputs_embeds", "attention_mask", "q_routing", "o_routing"),
          listOf("logits"),
        )
      }
    } catch (failure: Throwable) {
      table.close()
      throw failure
    }

  override fun decide(request: TextDecisionRequest): TextDecisionResult {
    val started = System.nanoTime()
    val question =
      Question.choice(request.question, request.options.indices.map { request.optionText(it) })
    val prepared = inputs.prepare(request.text, listOf(question), WINDOW)
    val graph = loaded.runner
    graph.write("inputs_embeds", table.lookup(prepared.inputIds))
    graph.write("attention_mask", prepared.attentionMask)
    graph.write("q_routing", prepared.qRouting)
    graph.write("o_routing", prepared.oRouting)
    graph.run()
    val logits = graph.read("logits")
    check(logits.all { it.isFinite() }) {
      "The model returned non-finite scores on ${loaded.backend}."
    }
    val answer = DecisionDecoder.decode(logits, listOf(question)).single()
    val ms = DecisionGraph.milliseconds(started)
    return TextDecisionResult(
      request.options.mapIndexed { index, (key, _) ->
        key to answer.probabilities[index].toFloat()
      },
      request.options[answer.best].first,
      ms,
      loaded.backend,
      WINDOW,
      loaded.fallbackReason,
      "${prepared.encoded.encodedLength} of $WINDOW tokens. The model reads each option as " +
        question.options.joinToString("; ") { "\"$it\"" } +
        ". Softmax at the author's temperature ${DecisionDecoder.TEMPERATURE}.",
    )
  }

  override fun close() {
    try {
      loaded.runner.close()
    } finally {
      table.close()
    }
  }

  private companion object {
    const val TAG = "ModelZooDeberta"
    const val WINDOW = 256
    const val GRAPH = "deberta_v3_large_decision_s256_wfp16.tflite"
  }
}
