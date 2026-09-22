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
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import com.google.ai.edge.examples.model_zoo.models.edsr.Upscaler
import com.google.ai.edge.examples.model_zoo.models.ormbg.BgRemover
import com.google.ai.edge.examples.model_zoo.models.realesrgan.RealEsrganUpscaler
import java.io.File

class OrmbgEngine(context: Context, modelDir: File, preferredBackend: String) : SingleImageEngine {
  private val loaded =
    compileImageBackend(preferredBackend, "ModelZooOrmbg") {
      BgRemover(context, modelDir.requiredImageModel("ormbg.tflite"), it)
    }

  override fun run(request: ImageTaskRequest): ImageTaskOutput {
    val bmp = request.bitmap
    val (alpha, ms) = loaded.runner.matte(bmp)
    val O = BgRemover.OUT
    val fgScaled = Bitmap.createBitmap(O, O, Bitmap.Config.ARGB_8888)
    val fgPixels = IntArray(O * O)
    val compPixels = IntArray(O * O)
    val compBitmap = Bitmap.createBitmap(O, O, Bitmap.Config.ARGB_8888)
    val BG_R = 30
    val BG_G = 190
    val BG_B = 120
    Canvas(fgScaled)
      .drawBitmap(
        bmp,
        Matrix().apply { setScale(O.toFloat() / bmp.width, O.toFloat() / bmp.height) },
        null,
      )
    fgScaled.getPixels(fgPixels, 0, O, 0, 0, O, O)
    for (i in 0 until O * O) {
      val a = alpha[i]
      val p = fgPixels[i]
      val fr = (p shr 16) and 0xFF
      val fg = (p shr 8) and 0xFF
      val fb = p and 0xFF
      val rr = (fr * a + BG_R * (1 - a)).toInt()
      val gg = (fg * a + BG_G * (1 - a)).toInt()
      val bb = (fb * a + BG_B * (1 - a)).toInt()
      compPixels[i] = (0xFF shl 24) or (rr shl 16) or (gg shl 8) or bb
    }
    compBitmap.setPixels(compPixels, 0, O, 0, 0, O, O)
    fgScaled.recycle()
    // Presentation only: retain the original green-composite calculations above, but let the UI
    // choose a background behind the same foreground pixels and unchanged model alpha.
    val transparent = Bitmap.createBitmap(O, O, Bitmap.Config.ARGB_8888)
    val displayPixels =
      IntArray(O * O) { i ->
        val a = (alpha[i].coerceIn(0f, 1f) * 255f).toInt()
        (a shl 24) or (fgPixels[i] and 0x00FFFFFF)
      }
    transparent.setPixels(displayPixels, 0, O, 0, 0, O, O)
    compBitmap.recycle()
    return ImageTaskOutput(
      transparent,
      "Foreground cut-out",
      ms.toDouble(),
      loaded.backend,
      loaded.fallbackReason,
    )
  }

  override fun close() = loaded.runner.close()
}

class EdsrEngine(context: Context, modelDir: File, preferredBackend: String) : SingleImageEngine {
  private val loaded =
    compileImageBackend(preferredBackend, "ModelZooEdsr") {
      Upscaler(context, modelDir.requiredImageModel("edsr.tflite"), it)
    }

  override fun run(request: ImageTaskRequest): ImageTaskOutput {
    val (bitmap, ms) = loaded.runner.upscale(request.bitmap)
    // The model wrapper reuses its output bitmap; publish an independently owned UI result.
    val output = checkNotNull(bitmap.copy(Bitmap.Config.ARGB_8888, false))
    return ImageTaskOutput(
      output,
      "Model 128 × 128 → 512 × 512. The photo is resized to the model’s square input; display restores its original aspect ratio.",
      ms.toDouble(),
      loaded.backend,
      loaded.fallbackReason,
    )
  }

  override fun close() = loaded.runner.close()
}

class RealEsrganEngine(context: Context, modelDir: File, preferredBackend: String) :
  SingleImageEngine {
  private val loaded =
    compileImageBackend(preferredBackend, "ModelZooRealEsrgan") {
      RealEsrganUpscaler(context, modelDir.requiredImageModel("realesr_general_x4v3.tflite"), it)
    }

  override fun run(request: ImageTaskRequest): ImageTaskOutput {
    val started = System.nanoTime()
    val output = loaded.runner.upscale(request.bitmap)
    val ms = (System.nanoTime() - started) / 1e6
    return ImageTaskOutput(
      output,
      "${request.bitmap.width} × ${request.bitmap.height} → ${output.width} × ${output.height}",
      ms,
      loaded.backend,
      loaded.fallbackReason,
    )
  }

  override fun close() = loaded.runner.close()
}

private fun File.requiredImageModel(name: String): File =
  File(this, name).also { check(it.isFile) { "Download this task's model first: $name" } }
