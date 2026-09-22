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

package com.google.ai.edge.examples.model_zoo.audio

import android.content.Context
import android.util.Log
import com.google.ai.edge.examples.model_zoo.image.ImageBackend
import com.google.ai.edge.examples.model_zoo.image.compileImageBackend
import com.google.ai.edge.examples.model_zoo.models.zipformer.ZipformerAsr
import com.google.ai.edge.examples.model_zoo.models.zipformer.ZipformerFbank
import java.io.File

data class AudioTextResult(
  val text: String,
  val inferenceMs: Double,
  val fbankMs: Long,
  val backend: String,
  val fallbackReason: String?,
)

/** App I/O adapter; inference and CTC decode remain in the Zipformer wrapper. */
class ZipformerEngine(context: Context, modelDir: File, preferredBackend: String = "gpu") :
  AutoCloseable {
  private val loaded: ImageBackend<ZipformerAsr>

  init {
    require(preferredBackend == "gpu" || preferredBackend == "cpu")
    for (name in listOf(ZipformerAsr.MODEL, "tokens.txt")) {
      check(File(modelDir, name).isFile) {
        "Model file not found: $name. Download Speech Recognition first."
      }
    }
    loaded =
      compileImageBackend(preferredBackend, "ModelZooZipformer") { accelerator ->
        ZipformerAsr(context, modelDir, accelerator)
      }
  }

  fun transcribe(audio: FloatArray): AudioTextResult {
    // Very short PCM cannot cover the source frontend's fixed reflect margin.
    require(audio.size >= ZipformerFbank.WIN) {
      "Record at least a short word before transcribing."
    }
    require(audio.all { it.isFinite() }) { "Audio contains non-finite samples." }
    val start = System.nanoTime()
    val result = loaded.runner.transcribe(audio)
    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.i(
      "ModelZooZipformer",
      "transcript=${result.text} fbank=${result.fbankMs}ms model=${result.gpuMs}ms backend=${loaded.backend}",
    )
    return AudioTextResult(result.text, ms, result.fbankMs, loaded.backend, loaded.fallbackReason)
  }

  override fun close() = loaded.runner.close()
}
