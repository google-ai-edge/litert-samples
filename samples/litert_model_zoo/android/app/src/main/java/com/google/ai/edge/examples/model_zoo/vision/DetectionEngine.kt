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

package com.google.ai.edge.examples.model_zoo.vision

import android.graphics.Bitmap
import com.google.ai.edge.examples.model_zoo.image.ImageBackend
import com.google.ai.edge.examples.model_zoo.image.compileImageBackend
import com.google.ai.edge.examples.model_zoo.models.rfdetr.RfDetr
import java.io.File

data class DetectionResult(
  val boxes: List<RfDetr.Detection>,
  val inferenceMs: Float,
  val backend: String,
  val fallbackReason: String?,
)

/** Confine construction, detect, and close to the same serial inference dispatcher. */
class DetectionEngine(modelDir: File, preferredBackend: String = "gpu") : AutoCloseable {
  private val loaded = load(modelDir, preferredBackend)
  val backend: String
    get() = loaded.backend

  val fallbackReason: String?
    get() = loaded.fallbackReason

  val labels: List<String> = COCO_LABELS

  private fun load(modelDir: File, preferredBackend: String): ImageBackend<RfDetr> {
    require(preferredBackend == "gpu" || preferredBackend == "cpu") {
      "Unsupported backend: $preferredBackend"
    }
    // Validate both before compilation so an absent download is not reported as GPU failure.
    for (name in listOf(RfDetr.MODEL_A, RfDetr.MODEL_B)) {
      check(File(modelDir, name).isFile) {
        "Model not found: $name. Download this task from Models first."
      }
    }
    return compileImageBackend(preferredBackend, "ModelZooDetection") { accelerator ->
      RfDetr(modelDir, accelerator)
    }
  }

  fun detect(bitmap: Bitmap): DetectionResult {
    val square = Bitmap.createScaledBitmap(bitmap, RfDetr.SIZE, RfDetr.SIZE, true)
    try {
      val rgb = bitmapToRgb(square)
      val t0 = System.nanoTime()
      val dets = loaded.runner.detect(rgb)
      val ms = (System.nanoTime() - t0) / 1e6f
      return DetectionResult(dets, ms, backend, fallbackReason)
    } finally {
      if (square !== bitmap) square.recycle()
    }
  }

  private fun bitmapToRgb(bm: Bitmap): FloatArray {
    val n = bm.width * bm.height
    val px = IntArray(n)
    bm.getPixels(px, 0, bm.width, 0, 0, bm.width, bm.height)
    val out = FloatArray(n * 3)
    for (i in 0 until n) {
      val p = px[i]
      out[i * 3] = ((p shr 16) and 0xFF).toFloat()
      out[i * 3 + 1] = ((p shr 8) and 0xFF).toFloat()
      out[i * 3 + 2] = (p and 0xFF).toFloat()
    }
    return out
  }

  override fun close() = loaded.runner.close()
}
