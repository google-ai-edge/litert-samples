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
import android.graphics.RectF
import com.google.ai.edge.examples.model_zoo.models.rfdetrseg.RfDetrSeg
import java.io.File

class RfDetrSegImageEngine(context: Context, modelDir: File, preferredBackend: String) :
  SingleImageEngine {
  private val labels =
    context.assets.open("rfdetrseg_coco_labels.txt").bufferedReader().use { it.readLines() }
  private val compiled =
    compileImageBackend(preferredBackend, "RfDetrSegImageEngine") {
      RfDetrSeg(context, modelDir, it)
    }

  companion object {
    val PALETTE =
      intArrayOf(
        0xFF00C853.toInt(),
        0xFFFF6D00.toInt(),
        0xFF2962FF.toInt(),
        0xFFD50000.toInt(),
        0xFFAA00FF.toInt(),
        0xFF00B8D4.toInt(),
        0xFFFFD600.toInt(),
        0xFFC51162.toInt(),
      )
  }

  override fun run(request: ImageTaskRequest): ImageTaskOutput {
    val frame = request.bitmap
    val square = Bitmap.createScaledBitmap(frame, RfDetrSeg.SIZE, RfDetrSeg.SIZE, true)
    val rgb =
      try {
        bitmapToRgb(square)
      } finally {
        if (square !== frame) square.recycle()
      }
    val t0 = System.nanoTime()
    val dets = compiled.runner.detect(rgb)
    val ms = (System.nanoTime() - t0) / 1e6f
    val masks = dets.mapIndexed { i, d -> maskBitmap(d, i) }
    val result = frame.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(result)
    val bm = frame
    val width = result.width
    val height = result.height
    val box =
      Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = OverlaySizing.stroke(width)
      }
    val txt =
      Paint().apply {
        color = Color.WHITE
        textSize = OverlaySizing.text(width)
        isFakeBoldText = true
      }
    val bgp = Paint().apply { style = Paint.Style.FILL }
    val maskPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    val s = minOf(width.toFloat() / bm.width, height.toFloat() / bm.height)
    val dw = bm.width * s
    val dh = bm.height * s
    val ox = (width - dw) / 2f
    val oy = (height - dh) / 2f
    val imageRect = RectF(ox, oy, ox + dw, oy + dh)
    canvas.drawBitmap(bm, null, imageRect, null)
    for (mb in masks) canvas.drawBitmap(mb, null, imageRect, maskPaint)
    for ((i, d) in dets.withIndex()) {
      val color = PALETTE[i % PALETTE.size] // per-instance color, matching the mask tint
      box.color = color
      bgp.color = color
      val x0 = ox + (d.cx - d.w / 2) * dw
      val y0 = oy + (d.cy - d.h / 2) * dh
      val x1 = ox + (d.cx + d.w / 2) * dw
      val y1 = oy + (d.cy + d.h / 2) * dh
      canvas.drawRect(x0, y0, x1, y1, box)
      val name = labels.getOrNull(d.cls)?.ifBlank { "id ${d.cls}" } ?: "id ${d.cls}"
      val label = "$name ${(d.score * 100).toInt()}%"
      val tw = txt.measureText(label)
      val pad = 4f * OverlaySizing.unit(width)
      val labelHeight = txt.textSize + pad * 2
      val labelLeft = x0.coerceIn(0f, (width - tw - pad * 2).coerceAtLeast(0f))
      val labelTop = (y0 - labelHeight).coerceIn(0f, (height - labelHeight).coerceAtLeast(0f))
      canvas.drawRect(labelLeft, labelTop, labelLeft + tw + pad * 2, labelTop + labelHeight, bgp)
      canvas.drawText(label, labelLeft + pad, labelTop + pad - txt.ascent(), txt)
    }
    masks.forEach { it.recycle() }
    val summary =
      if (dets.isEmpty()) "No objects found"
      else
        dets.joinToString("\n") { d ->
          val name = labels.getOrNull(d.cls)?.ifBlank { "id ${d.cls}" } ?: "id ${d.cls}"
          "$name ${(d.score * 100).toInt()}%"
        }
    return ImageTaskOutput(
      bitmap = result,
      text = summary,
      inferenceMs = ms.toDouble(),
      backend = compiled.backend,
      fallbackReason = compiled.fallbackReason,
    )
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

  private fun maskBitmap(d: RfDetrSeg.Detection, i: Int): Bitmap {
    val tint = (PALETTE[i % PALETTE.size] and 0x00FFFFFF) or (110 shl 24)
    val px = IntArray(RfDetrSeg.MASK * RfDetrSeg.MASK)
    for (i in px.indices) if (d.mask[i] > 0f) px[i] = tint
    return Bitmap.createBitmap(px, RfDetrSeg.MASK, RfDetrSeg.MASK, Bitmap.Config.ARGB_8888)
  }

  override fun close() = compiled.runner.close()
}
