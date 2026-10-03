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

package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.gliclass

import com.google.ai.edge.examples.litert_model_zoo.image.compileImageBackend
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.DecisionGraph
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.TextDecisionEngine
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.TextDecisionRequest
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.TextDecisionResult
import java.io.File

/**
 * GLiClass-Edge v3.0 (S128) through the card's vendored host: the options become labels (the
 * description, or the key when there is none), the question plus one space is the prompt so the
 * text starts a word, and the labels get a single-label softmax.
 */
class GliclassDecider(modelDir: File, backend: String = "gpu") : TextDecisionEngine {
  private val table = GliclassInputs.EmbeddingTable(File(modelDir, "tok_embeddings_fp16.bin"))
  private val inputs =
    try {
      GliclassInputs(GliclassTokenizer(File(modelDir, "tokenizer.json")), table)
    } catch (failure: Throwable) {
      table.close()
      throw failure
    }
  private val loaded =
    try {
      compileImageBackend(backend, TAG) { accelerator ->
        DecisionGraph.create(
          File(modelDir, GRAPH),
          accelerator,
          listOf("inputs_embeds", "attention_mask", "label_routing"),
          listOf("logits"),
        )
      }
    } catch (failure: Throwable) {
      table.close()
      throw failure
    }

  override fun decide(request: TextDecisionRequest): TextDecisionResult {
    val started = System.nanoTime()
    val labels = request.options.indices.map { request.optionText(it) }
    val prepared = inputs.prepare(request.text, labels, request.question + " ", WINDOW)
    val graph = loaded.runner
    graph.write("inputs_embeds", prepared.embeds)
    graph.write("attention_mask", prepared.attentionMask)
    graph.write("label_routing", prepared.labelRouting)
    graph.run()
    val logits = graph.read("logits")
    check(logits.all { it.isFinite() }) {
      "The model returned non-finite scores on ${loaded.backend}."
    }
    val decision = GliclassDecoder.decide(logits, labels, GliclassDecoder.Mode.SINGLE_LABEL)
    val ms = DecisionGraph.milliseconds(started)
    return TextDecisionResult(
      request.options.mapIndexed { index, (key, _) -> key to decision.probabilities[index] },
      request.options[decision.chosen.single()].first,
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
    const val TAG = "ModelZooGliclass"
    const val WINDOW = 128
    const val GRAPH = "gliclass_edge_v3_s128_fp32.tflite"
  }
}
