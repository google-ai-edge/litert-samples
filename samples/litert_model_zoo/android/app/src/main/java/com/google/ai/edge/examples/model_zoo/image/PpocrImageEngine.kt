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
import android.graphics.Color
import android.graphics.Paint
import com.google.ai.edge.examples.model_zoo.models.ppocr.PpocrDetector
import com.google.ai.edge.examples.model_zoo.models.ppocr.PpocrRecognizer
import java.io.File

/** Model files and dictionary are all downloaded into this task's private model directory. */
class PpocrImageEngine(context: Context, modelDir: File, preferredBackend: String) :
  SingleImageEngine {
  private val compiled =
    compileImageBackend(preferredBackend, "ModelZooPpocr") { accelerator ->
      listOf(PpocrDetector.MODEL, PpocrRecognizer.MODEL, PpocrRecognizer.DICT).forEach {
        require(File(modelDir, it).isFile) { "Download $it in Models first." }
      }
      val detector = PpocrDetector(modelDir, accelerator)
      try {
        detector to PpocrRecognizer(modelDir, accelerator)
      } catch (failure: Exception) {
        detector.close()
        throw failure
      }
    }

  override fun run(request: ImageTaskRequest): ImageTaskOutput {
    val img = letterbox(request.bitmap)
    val d = compiled.runner.first
    val r = compiled.runner.second
    val rgb = bitmapToRgb(img)
    // The source's warm-up branch only ran on its bundled sample, which this app does not ship.
    val t0 = System.nanoTime()
    val boxes = PpocrDetector.boxes(d.probMap(rgb))
    val lines = ArrayList<Pair<PpocrDetector.Box, String>>()
    for (b in boxes) {
      val text = r.recognize(cropResize(img, b))
      if (text.isNotBlank()) lines.add(b to text)
    }
    val ms = (System.nanoTime() - t0) / 1_000_000.0
    val annotated = img.copy(Bitmap.Config.ARGB_8888, true)
    val stroke =
      Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = OverlaySizing.stroke(annotated.width)
      }
    val canvas = Canvas(annotated)
    for ((box, _) in lines) {
      canvas.drawRect(
        box.x0.toFloat(),
        box.y0.toFloat(),
        box.x1.toFloat(),
        box.y1.toFloat(),
        stroke,
      )
    }
    img.recycle()
    return ImageTaskOutput(
      bitmap = annotated,
      text =
        com.google.ai.edge.examples.model_zoo.ResultCounts.boxes(lines.size) +
          " with text." +
          if (lines.isEmpty()) ""
          else
            "\n" +
              OcrReadingOrder.lines(
                  lines.map { (box, text) -> OcrDisplayBox(box.x0, box.y0, box.x1, box.y1, text) }
                )
                .joinToString("\n"),
      inferenceMs = ms,
      backend = compiled.backend,
      fallbackReason = compiled.fallbackReason,
      backendDetails =
        "Detector and recognizer: ${compiled.backend}; box extraction and CTC decode: CPU.",
      details =
        "Raw recognizer boxes (detector order):\n" +
          lines.joinToString("\n") { (box, text) ->
            "[${box.x0}, ${box.y0}, ${box.x1}, ${box.y1}] $text"
          },
    )
  }

  private fun letterbox(src: Bitmap): Bitmap {
    val s =
      minOf(PpocrDetector.SIZE.toFloat() / src.width, PpocrDetector.SIZE.toFloat() / src.height)
    val nw = (src.width * s).toInt().coerceAtLeast(1)
    val nh = (src.height * s).toInt().coerceAtLeast(1)
    val out = Bitmap.createBitmap(PpocrDetector.SIZE, PpocrDetector.SIZE, Bitmap.Config.ARGB_8888)
    Canvas(out).apply {
      drawColor(Color.WHITE)
      drawBitmap(
        Bitmap.createScaledBitmap(src, nw, nh, true),
        (PpocrDetector.SIZE - nw) / 2f,
        (PpocrDetector.SIZE - nh) / 2f,
        null,
      )
    }
    return out
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

  private fun cropResize(img: Bitmap, b: PpocrDetector.Box): FloatArray {
    val bw = b.x1 - b.x0 + 1
    val bh = b.y1 - b.y0 + 1
    val crop = Bitmap.createBitmap(img, b.x0, b.y0, bw, bh)
    val nw =
      minOf((PpocrRecognizer.H.toFloat() * bw / bh).toInt(), PpocrRecognizer.W).coerceAtLeast(1)
    val rz = Bitmap.createScaledBitmap(crop, nw, PpocrRecognizer.H, true)
    val px = IntArray(nw * PpocrRecognizer.H)
    rz.getPixels(px, 0, nw, 0, 0, nw, PpocrRecognizer.H)
    val out = FloatArray(PpocrRecognizer.H * PpocrRecognizer.W * 3)
    for (y in 0 until PpocrRecognizer.H) for (x in 0 until nw) {
      val p = px[y * nw + x]
      val o = (y * PpocrRecognizer.W + x) * 3
      out[o] = ((p shr 16) and 0xFF).toFloat()
      out[o + 1] = ((p shr 8) and 0xFF).toFloat()
      out[o + 2] = (p and 0xFF).toFloat()
    }
    return out
  }

  override fun close() {
    compiled.runner.first.close()
    compiled.runner.second.close()
  }
}
