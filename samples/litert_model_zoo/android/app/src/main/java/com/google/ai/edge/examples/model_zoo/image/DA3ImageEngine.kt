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
import com.google.ai.edge.examples.model_zoo.models.da3.DA3Predictor
import java.io.File

class DA3ImageEngine(context: Context, modelDir: File, preferredBackend: String) :
  SingleImageEngine {
  private val compiled =
    compileImageBackend(preferredBackend, "DA3ImageEngine") { DA3Predictor(context, modelDir, it) }

  override fun run(request: ImageTaskRequest): ImageTaskOutput {
    val result = compiled.runner.predict(request.bitmap)
    return ImageTaskOutput(
      bitmap = result.depthBitmap(),
      text = "Relative depth, cropped to image content",
      inferenceMs = result.inferenceMs.toDouble(),
      backend = compiled.backend,
      fallbackReason = compiled.fallbackReason,
    )
  }

  override fun close() = compiled.runner.close()
}
