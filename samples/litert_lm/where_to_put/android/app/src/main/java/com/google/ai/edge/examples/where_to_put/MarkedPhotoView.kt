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

package com.google.ai.edge.examples.where_to_put

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.TypedValue
import android.view.View
import kotlin.math.min

/**
 * The photo at its own aspect ratio, centred in the view, and the model's point as a ring with a dot. The point is in
 * 0-1000 of the whole picture: x / 1000 of the photo's drawn width, y / 1000 of its drawn height. There is no text
 * label on the photo: next to a free spot a label would cover the very objects the request names.
 */
class MarkedPhotoView(context: Context) : View(context) {
  private var bitmap: Bitmap? = null
  private var spot: Spot? = null
  private val drawn = RectF()
  private fun dp(v: Float) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, resources.displayMetrics)

  private val photoPaint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
  private val ringOuter = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    style = Paint.Style.STROKE; strokeWidth = dp(7f); color = Color.WHITE
  }
  private val ringInner = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    style = Paint.Style.STROKE; strokeWidth = dp(4f); color = Color.rgb(230, 40, 60)
  }
  private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(230, 40, 60) }

  fun setPhoto(b: Bitmap?) {
    bitmap = b
    spot = null
    invalidate()
  }

  /** The point to mark, or null for none (a point outside the picture is never drawn). */
  fun setSpot(s: Spot?) {
    spot = s?.takeIf { it.inPicture }
    invalidate()
  }

  override fun onDraw(canvas: Canvas) {
    super.onDraw(canvas)
    val b = bitmap ?: return
    val scale = min(width.toFloat() / b.width, height.toFloat() / b.height)
    val w = b.width * scale
    val h = b.height * scale
    drawn.set((width - w) / 2, (height - h) / 2, (width + w) / 2, (height + h) / 2)
    canvas.drawBitmap(b, null, drawn, photoPaint)
    val s = spot ?: return
    val cx = drawn.left + (s.x / 1000.0 * drawn.width()).toFloat()
    val cy = drawn.top + (s.y / 1000.0 * drawn.height()).toFloat()
    val r = dp(16f)
    canvas.drawCircle(cx, cy, r, ringOuter)
    canvas.drawCircle(cx, cy, r, ringInner)
    canvas.drawCircle(cx, cy, dp(3.5f), dotPaint)
  }
}
