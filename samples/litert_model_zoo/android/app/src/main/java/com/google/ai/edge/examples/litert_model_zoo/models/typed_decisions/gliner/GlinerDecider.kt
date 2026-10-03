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

package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.gliner

import com.google.ai.edge.examples.litert_model_zoo.image.compileImageBackend
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.DecisionGraph
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.TextDecisionEngine
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.TextDecisionRequest
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.TextDecisionResult
import java.io.File

/**
 * GLiNER2.5-Decide (S128) through the card's vendored host: the question is one gliner2 task
 * named `answer` with the question as its prompt, the options are its labels (the description, or
 * the key when there is none), and the labels get a single-label softmax at temperature 1.
 */
class GlinerDecider(modelDir: File, backend: String = "gpu") : TextDecisionEngine {
  private val inputs = DecideInputs(GlinerTokenizer(File(modelDir, "tokenizer.json")))
  private val table = DecideInputs.EmbeddingTable(File(modelDir, "word_embeddings_fp16.bin"))
  private val loaded =
    try {
      compileImageBackend(backend, TAG) { accelerator ->
        // The converter named the inputs args_0..args_2; their meaning follows from the shapes.
        DecisionGraph.create(
          File(modelDir, GRAPH),
          accelerator,
          { dimensions ->
            listOf(
              listOf(1, WINDOW, DecideInputs.HIDDEN_SIZE),
              listOf(1, WINDOW),
              listOf(1, DecideInputs.LABEL_SLOTS, WINDOW),
            ).map { shape -> INPUT_NAMES.single { dimensions(it) == shape } }
          },
          listOf("output_0"),
        )
      }
    } catch (failure: Throwable) {
      table.close()
      throw failure
    }

  override fun decide(request: TextDecisionRequest): TextDecisionResult {
    val started = System.nanoTime()
    val labels = request.options.indices.map { request.optionText(it) }
    val task = Task(TASK_NAME, labels, prompt = request.question)
    val prepared = inputs.prepare(request.text, listOf(task), WINDOW)
    val graph = loaded.runner
    val (embedsName, attentionName, routingName) = graph.inputNames
    graph.write(embedsName, table.lookup(prepared.inputIds))
    graph.write(attentionName, prepared.attentionMask)
    graph.write(routingName, prepared.labelRouting)
    graph.run()
    val logits = graph.read("output_0")
    check(logits.all { it.isFinite() }) {
      "The model returned non-finite scores on ${loaded.backend}."
    }
    val decision = DecideDecoder.decode(logits, listOf(task)).single()
    val ms = DecisionGraph.milliseconds(started)
    return TextDecisionResult(
      request.options.mapIndexed { index, (key, _) -> key to decision.probabilities[index] },
      request.options[DecideDecoder.argmax(decision.probabilities)].first,
      ms,
      loaded.backend,
      WINDOW,
      loaded.fallbackReason,
      "${prepared.encoded.encodedLength} of $WINDOW tokens. The model reads each option as " +
        labels.joinToString("; ") { "\"$it\"" } +
        ". Single-label softmax at temperature 1.",
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
    const val TAG = "ModelZooGliner"
    const val WINDOW = 128
    const val GRAPH = "gliner25_decide_s128_wfp16.tflite"
    const val TASK_NAME = "answer"
    val INPUT_NAMES = listOf("args_0", "args_1", "args_2")
  }
}
