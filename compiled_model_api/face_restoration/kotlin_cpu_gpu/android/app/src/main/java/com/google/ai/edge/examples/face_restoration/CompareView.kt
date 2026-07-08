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

package com.google.ai.edge.examples.face_restoration

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.view.MotionEvent
import android.view.View

/**
 * Before/after comparison slider. Left side = original (degraded) face, right side = restored.
 * Drag the divider to compare.
 */
class CompareView(context: Context) : View(context) {

  private var beforeBitmap: Bitmap? = null
  private var afterBitmap: Bitmap? = null
  private var dividerRatio = 0.5f

  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
  private val dividerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = 0xFFFFFFFF.toInt()
    strokeWidth = 4f
    style = Paint.Style.STROKE
  }
  private val handlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = 0xFFFFFFFF.toInt()
    style = Paint.Style.FILL
    setShadowLayer(8f, 0f, 0f, 0xFF000000.toInt())
  }
  private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = 0xFFFFFFFF.toInt()
    textSize = 32f
    setShadowLayer(4f, 0f, 0f, 0xFF000000.toInt())
  }

  private val srcRect = Rect()
  private val dstRect = Rect()
  private val clipRect = RectF()

  fun setImages(before: Bitmap, after: Bitmap) {
    beforeBitmap = before
    afterBitmap = after
    dividerRatio = 0.5f
    invalidate()
  }

  private fun getImageRect(): RectF {
    val bmp = afterBitmap ?: beforeBitmap ?: return RectF()
    val scale = minOf(width.toFloat() / bmp.width, height.toFloat() / bmp.height)
    val imgW = bmp.width * scale
    val imgH = bmp.height * scale
    val left = (width - imgW) / 2f
    val top = (height - imgH) / 2f
    return RectF(left, top, left + imgW, top + imgH)
  }

  override fun onDraw(canvas: Canvas) {
    super.onDraw(canvas)
    val after = afterBitmap ?: return
    val before = beforeBitmap ?: return
    val imgRect = getImageRect()
    val dividerX = imgRect.left + imgRect.width() * dividerRatio

    srcRect.set(0, 0, after.width, after.height)
    dstRect.set(
      imgRect.left.toInt(), imgRect.top.toInt(),
      imgRect.right.toInt(), imgRect.bottom.toInt())
    canvas.drawBitmap(after, srcRect, dstRect, paint)

    canvas.save()
    clipRect.set(imgRect.left, imgRect.top, dividerX, imgRect.bottom)
    canvas.clipRect(clipRect)
    srcRect.set(0, 0, before.width, before.height)
    canvas.drawBitmap(before, srcRect, dstRect, paint)
    canvas.restore()

    canvas.drawLine(dividerX, imgRect.top, dividerX, imgRect.bottom, dividerPaint)
    val cy = imgRect.centerY()
    canvas.drawCircle(dividerX, cy, 20f, handlePaint)
    val arrowPaint = Paint(handlePaint).apply {
        textSize = 28f
        textAlign = Paint.Align.CENTER
    }
    canvas.drawText("◀", dividerX - 6f, cy + 10f, arrowPaint)
    canvas.drawText("▶", dividerX + 6f, cy + 10f, arrowPaint)

    labelPaint.textAlign = Paint.Align.LEFT
    canvas.drawText("Input", imgRect.left + 12f, imgRect.top + 40f, labelPaint)
    labelPaint.textAlign = Paint.Align.RIGHT
    canvas.drawText("GFPGAN", imgRect.right - 12f, imgRect.top + 40f, labelPaint)
  }

  @SuppressLint("ClickableViewAccessibility")
  override fun onTouchEvent(event: MotionEvent): Boolean {
    val imgRect = getImageRect()
    when (event.action) {
      MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> {
        dividerRatio = ((event.x - imgRect.left) / imgRect.width()).coerceIn(0f, 1f)
        invalidate()
        return true
      }
    }
    return super.onTouchEvent(event)
  }
}
