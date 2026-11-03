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

// The colours are WeVisOne-VL's visualization.py COLORS (Tencent, Apache-2.0; see NOTICE).
package com.google.ai.edge.examples.image_cutout

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/**
 * The photo with what the model found: boxes, points and labels in the package's colours, the
 * cut-outs tinted over the photo; tap to show the cut-out alone on a checkerboard (the transparent
 * PNG the Save button writes).
 */
class ResultView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) :
  View(context, attrs) {
  private var photo: Bitmap? = null
  private var tint: Bitmap? = null
  private var cutout: Bitmap? = null
  private var shapes: List<Shape> = emptyList()
  private var cutoutBoxes: List<IntArray> = emptyList()
  private var showCutout = false

  private val box = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
  private val dot = Paint(Paint.ANTI_ALIAS_FLAG)
  private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE }
  private val textBack = Paint()
  private val bitmapPaint = Paint(Paint.FILTER_BITMAP_FLAG)
  private val checkA = Paint().apply { color = Color.rgb(204, 204, 204) }
  private val checkB = Paint().apply { color = Color.WHITE }

  init {
    setOnClickListener {
      if (cutout != null) {
        showCutout = !showCutout
        invalidate()
      }
    }
  }

  fun setPhoto(bitmap: Bitmap?) {
    photo = bitmap
    tint = null
    cutout = null
    shapes = emptyList()
    cutoutBoxes = emptyList()
    showCutout = false
    invalidate()
  }

  /** [tintBitmap] and [cutoutBitmap] are at the photo bitmap's size. */
  fun setResult(
    found: List<Shape>,
    boxes: List<IntArray>,
    tintBitmap: Bitmap?,
    cutoutBitmap: Bitmap?,
  ) {
    shapes = found
    cutoutBoxes = boxes
    tint = tintBitmap
    cutout = cutoutBitmap
    showCutout = false
    invalidate()
  }

  val isShowingCutout: Boolean
    get() = showCutout

  override fun onDraw(canvas: Canvas) {
    super.onDraw(canvas)
    val p = photo ?: return
    val scale = minOf(width.toFloat() / p.width, height.toFloat() / p.height)
    val w = p.width * scale
    val h = p.height * scale
    val left = (width - w) / 2
    val top = (height - h) / 2
    val dst = RectF(left, top, left + w, top + h)
    if (showCutout && cutout != null) {
      val cell = 16f * resources.displayMetrics.density
      var y = top
      var row = 0
      while (y < top + h) {
        var x = left
        var col = 0
        while (x < left + w) {
          canvas.drawRect(
            x,
            y,
            minOf(x + cell, left + w),
            minOf(y + cell, top + h),
            if ((row + col) % 2 == 0) checkA else checkB,
          )
          x += cell
          col++
        }
        y += cell
        row++
      }
      canvas.drawBitmap(cutout!!, null, dst, bitmapPaint)
      return
    }
    canvas.drawBitmap(p, null, dst, bitmapPaint)
    tint?.let { canvas.drawBitmap(it, null, dst, bitmapPaint) }
    val density = resources.displayMetrics.density
    box.strokeWidth = 3f * density
    text.textSize = 14f * density
    val labels = shapes.map { it.label }.distinct()
    fun color(label: String) = COLORS[labels.indexOf(label).coerceAtLeast(0) % COLORS.size]
    fun px(bin: Int) = left + bin / 1000f * w
    fun py(bin: Int) = top + bin / 1000f * h
    for (s in shapes) {
      val c = color(s.label)
      if (s.isPoint) {
        dot.color = c
        canvas.drawCircle(px(s.bins[0]), py(s.bins[1]), 7f * density, dot)
        drawLabel(canvas, s.label, px(s.bins[0]) + 9f * density, py(s.bins[1]), c)
      } else {
        box.color = c
        canvas.drawRect(px(s.bins[0]), py(s.bins[1]), px(s.bins[2]), py(s.bins[3]), box)
        drawLabel(canvas, s.label, px(s.bins[0]), py(s.bins[1]), c)
      }
    }
    if (shapes.isEmpty()) {
      box.color = COLORS[0]
      for (b in cutoutBoxes) {
        canvas.drawRect(px(b[0]), py(b[1]), px(b[2]), py(b[3]), box)
      }
    }
  }

  private fun drawLabel(canvas: Canvas, label: String, x: Float, y: Float, color: Int) {
    val pad = 3f * resources.displayMetrics.density
    val tw = text.measureText(label)
    val th = text.textSize
    val top = maxOf(0f, y - th - 2 * pad)
    textBack.color = color
    canvas.drawRect(x, top, x + tw + 2 * pad, top + th + 2 * pad, textBack)
    canvas.drawText(label, x + pad, top + th + pad / 2, text)
  }

  companion object {
    /** wevisone_vl/visualization.py COLORS. */
    val COLORS =
      intArrayOf(
        Color.rgb(255, 92, 124),
        Color.rgb(47, 212, 123),
        Color.rgb(64, 156, 255),
        Color.rgb(255, 184, 48),
        Color.rgb(176, 92, 255),
        Color.rgb(0, 204, 196),
      )
  }
}
