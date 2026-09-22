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

package com.google.ai.edge.examples.model_zoo.image

import android.content.Context
import com.google.ai.edge.examples.model_zoo.models.ram.RamTagger
import java.io.File
import java.util.Locale

/** Keeps RAM's stage-3 and reweight graphs on CPU, as in the verified RAM++ pipeline. */
class RamImageEngine(context: Context, modelDir: File, preferredBackend: String) :
  SingleImageEngine {
  private val compiled =
    compileImageBackend(preferredBackend, "ModelZooRam") { accelerator ->
      listOf(
          "ram_swin_s012_fp16.tflite",
          "ram_stage3_tail_fp16.tflite",
          "ram_reweight_fp16.tflite",
          "ram_taghead_fp16.tflite",
          "ram_tag_list.txt",
          "ram_tag_threshold.bin",
        )
        .forEach { require(File(modelDir, it).isFile) { "Download $it in Models first." } }
      RamTagger(modelDir, accelerator)
    }

  override fun run(request: ImageTaskRequest): ImageTaskOutput {
    val started = System.nanoTime()
    val tags = compiled.runner.tag(request.bitmap)
    val ms = (System.nanoTime() - started) / 1_000_000.0
    return ImageTaskOutput(
      text =
        com.google.ai.edge.examples.model_zoo.ResultCounts.tags(tags.size) +
          " above the model thresholds." +
          if (tags.isEmpty()) ""
          else
            "\n" +
              tags.joinToString("\n") {
                String.format(Locale.US, "%s — %.1f%%", it.name, it.prob * 100)
              },
      inferenceMs = ms,
      backend = compiled.backend,
      fallbackReason = compiled.fallbackReason,
      backendDetails =
        if (compiled.backend == "CPU") "All four graphs: CPU."
        else "Swin stages 0–2 and tag head: GPU; stage 3 and reweight: CPU.",
    )
  }

  override fun close() = compiled.runner.close()
}
