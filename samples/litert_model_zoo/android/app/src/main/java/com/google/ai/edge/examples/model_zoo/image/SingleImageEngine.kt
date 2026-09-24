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

import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator

/** UI input and output only; each engine delegates numerical work to the model wrappers. */
data class ImageTaskRequest(
  val bitmap: Bitmap,
  val secondaryBitmap: Bitmap? = null,
)

data class ImageTaskOutput(
  val bitmap: Bitmap? = null,
  val text: String = "",
  val inferenceMs: Double,
  val backend: String,
  val fallbackReason: String? = null,
  val backendDetails: String = "",
  val metrics: Map<String, Any?> = emptyMap(),
  val details: String = "",
)

interface SingleImageEngine : AutoCloseable {
  fun run(request: ImageTaskRequest): ImageTaskOutput
}

data class ImageBackend<T>(val runner: T, val backend: String, val fallbackReason: String?)

/** A mixed wrapper retains its CPU-only graphs when the eligible graphs compile for GPU. */
fun <T> compileImageBackend(
  preferredBackend: String,
  tag: String,
  logFailure: (String, Exception) -> Unit = { reason, failure ->
    Log.w(tag, "GPU compilation failed; using CPU: $reason", failure)
  },
  create: (Accelerator) -> T,
): ImageBackend<T> {
  require(preferredBackend in setOf("gpu", "cpu", "mixed"))
  if (preferredBackend == "cpu") return ImageBackend(create(Accelerator.CPU), "CPU", null)
  return try {
    ImageBackend(
      create(Accelerator.GPU),
      if (preferredBackend == "mixed") "GPU + CPU" else "GPU",
      null,
    )
  } catch (failure: Exception) {
    val reason = failure.message ?: failure.javaClass.simpleName
    logFailure(reason, failure)
    ImageBackend(create(Accelerator.CPU), "CPU", reason)
  }
}
