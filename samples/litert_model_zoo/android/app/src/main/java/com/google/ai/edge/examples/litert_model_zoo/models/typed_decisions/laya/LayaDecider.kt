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

package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.laya

import com.google.ai.edge.examples.litert_model_zoo.image.compileImageBackend
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.DecisionGraph
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.TextDecisionEngine
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.TextDecisionRequest
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.TextDecisionResult
import java.io.File

/**
 * Laya multilingual (S256, host token lookup) through the model card's vendored host: the prompt
 * builder renders each option as `key: description`, the main graph scores one marker per option,
 * the act head reads the raw probabilities, and the decoder applies the checkpoint's option-count
 * temperature and rounds to four decimals.
 */
class LayaDecider(modelDir: File, backend: String = "gpu") : TextDecisionEngine {
  private class Graphs(val main: DecisionGraph, val act: DecisionGraph)

  private val tokenizer = LayaTokenizer(File(modelDir, "tokenizer.json"))
  private val calibration = LayaCalibration.load(File(modelDir, "laya_ml_calibration.json"))
  private val embeddings =
    LayaEmbeddings(
      File(modelDir, "token_embeddings_fp16.bin"),
      File(modelDir, "token_embeddings.json"),
    )
  private val loaded =
    try {
      compileImageBackend(backend, TAG) { accelerator ->
        val main =
          DecisionGraph.create(
            File(modelDir, MAIN_GRAPH),
            accelerator,
            listOf("attention_mask", "inputs_embeds", "qtype_onehot"),
            listOf("pooled_cls", "token_logits"),
          )
        try {
          Graphs(
            main,
            DecisionGraph.create(
              File(modelDir, ACT_GRAPH),
              accelerator,
              listOf("feats", "pooled_cls"),
              listOf("act_logits"),
            ),
          )
        } catch (failure: Throwable) {
          main.close()
          throw failure
        }
      }
    } catch (failure: Throwable) {
      embeddings.close()
      throw failure
    }
  private val embeds = FloatArray(WINDOW * LayaEmbeddings.WIDTH)

  override fun decide(request: TextDecisionRequest): TextDecisionResult {
    val started = System.nanoTime()
    val criteria = LinkedHashMap<String, Any?>()
    request.options.forEach { (key, description) -> criteria[key] = description }
    val question =
      mapOf("type" to "choice", "instructions" to request.question, "criteria" to criteria)
    val builder = LayaPromptBuilder(tokenizer, maxLen = WINDOW, headMaxLen = WINDOW)
    val sequence = builder.build(request.text, question)
    embeddings.gather(sequence.ids, WINDOW, embeds)
    val graphs = loaded.runner
    graphs.main.write("inputs_embeds", embeds)
    val attention = FloatArray(WINDOW) { if (it < sequence.ids.size) 1f else 0f }
    graphs.main.write("attention_mask", attention)
    graphs.main.write("qtype_onehot", FloatArray(3).also { it[sequence.question.qtype] = 1f })
    graphs.main.run()
    val tokens = graphs.main.read("token_logits")
    val pooled = graphs.main.read("pooled_cls")
    check(tokens.all { it.isFinite() } && pooled.all { it.isFinite() }) {
      "The model returned non-finite scores on ${loaded.backend}."
    }
    val markers = LayaDecoder.gather(tokens, sequence.markers)
    graphs.act.write("pooled_cls", pooled)
    graphs.act.write("feats", LayaDecoder.actFeatures(markers))
    graphs.act.run()
    val action = graphs.act.read("act_logits")
    val answer = LayaDecoder.decode(markers, action, sequence.question, calibration)
    val ms = DecisionGraph.milliseconds(started)
    val probabilities = LayaJson.asObject(answer["probabilities"])
    return TextDecisionResult(
      request.options.map { (key, _) -> key to (probabilities.getValue(key) as Number).toFloat() },
      answer["choice"] as String,
      ms,
      loaded.backend,
      WINDOW,
      loaded.fallbackReason,
      "${sequence.ids.size} of $WINDOW tokens. The model reads each option as " +
        sequence.options.joinToString("; ") { "\"$it\"" } +
        ". Probabilities use the model card's calibrated temperature, rounded to four decimals.",
    )
  }

  override fun close() {
    try {
      loaded.runner.main.close()
    } finally {
      try {
        loaded.runner.act.close()
      } finally {
        embeddings.close()
      }
    }
  }

  private companion object {
    const val TAG = "ModelZooLaya"
    const val WINDOW = 256
    const val MAIN_GRAPH = "laya_ml_s256_embeds_wfp16.tflite"
    const val ACT_GRAPH = "laya_ml_act_head_fp32.tflite"
  }
}
