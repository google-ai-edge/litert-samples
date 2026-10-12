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

package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.julia

import com.google.ai.edge.examples.litert_model_zoo.image.compileImageBackend
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.DecisionGraph
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.TextDecisionEngine
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.TextDecisionRequest
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.TextDecisionResult
import java.io.File

/**
 * Julia-1 (S512, host token lookup) through the model card's vendored host: each option is scored
 * as its description (a key stands for itself when it has none), strict encoding rejects a request
 * that does not fit the window, and the answer is the plain softmax of the marker logits.
 */
class JuliaDecider(modelDir: File, backend: String = "gpu") : TextDecisionEngine {
  private val tokenizer = JuliaTokenizer(File(modelDir, "tokenizer.json"))
  private val embeddings = JuliaEmbeddings(File(modelDir, "julia1_token_table_fp16.bin"))
  private val loaded =
    try {
      compileImageBackend(backend, TAG) { accelerator ->
        DecisionGraph.create(
          File(modelDir, GRAPH),
          accelerator,
          listOf("attention_mask", "inputs_embeds", "qtype_onehot"),
          listOf("token_logits"),
        )
      }
    } catch (failure: Throwable) {
      embeddings.close()
      throw failure
    }
  private val embeds = FloatArray(WINDOW * JuliaEmbeddings.WIDTH)

  override fun decide(request: TextDecisionRequest): TextDecisionResult {
    val started = System.nanoTime()
    val criteria = LinkedHashMap<String, Any?>()
    request.options.forEachIndexed { index, (key, _) -> criteria[key] = request.optionText(index) }
    val question = JuliaQuestion("choice", request.question, criteria)
    val sequence = JuliaSequenceBuilder(tokenizer, WINDOW).build(request.text, question)
    embeddings.gather(sequence.ids, WINDOW, embeds)
    val graph = loaded.runner
    graph.write("inputs_embeds", embeds)
    graph.write("attention_mask", FloatArray(WINDOW) { if (it < sequence.ids.size) 1f else 0f })
    graph.write("qtype_onehot", FloatArray(3).also { it[sequence.qtype] = 1f })
    graph.run()
    val tokens = graph.read("token_logits")
    check(tokens.all { it.isFinite() }) {
      "The model returned non-finite scores on ${loaded.backend}."
    }
    val answer = JuliaDecoder.decode(JuliaDecoder.gather(tokens, sequence.markers), question)
    val ms = DecisionGraph.milliseconds(started)
    return TextDecisionResult(
      question.keys.mapIndexed { index, key -> key to answer.probabilities[index].toFloat() },
      question.keys[answer.argmax],
      ms,
      loaded.backend,
      WINDOW,
      loaded.fallbackReason,
      "${sequence.ids.size} of $WINDOW tokens. The model reads each option as " +
        question.options.joinToString("; ") { "\"$it\"" } +
        ". Plain softmax of the option scores; Julia-1 ships no calibration.",
    )
  }

  override fun close() {
    try {
      loaded.runner.close()
    } finally {
      embeddings.close()
    }
  }

  private companion object {
    const val TAG = "ModelZooJulia"
    const val WINDOW = 512
    const val GRAPH = "julia1_s512_fp32.tflite"
  }
}
