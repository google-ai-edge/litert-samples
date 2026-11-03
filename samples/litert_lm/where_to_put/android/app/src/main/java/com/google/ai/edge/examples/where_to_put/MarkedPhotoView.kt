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

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.TypedValue
import android.view.View
import android.view.animation.DecelerateInterpolator
import kotlin.math.min

/** How long the ring shows, with the item held just above it, before the item drops onto it. */
private const val RING_MS = 400L

/** How long the item takes to land and the ring to fade after it. */
private const val LAND_MS = 600L

/** The share of the landing spent falling; the ring fades in the rest. */
private const val FALL_SHARE = 0.6f

/**
 * The photo at its own aspect ratio, centred in the view, and the model's point. The point is in 0-1000 of the whole
 * picture: x / 1000 of the photo's drawn width, y / 1000 of its drawn height. It shows as a ring with a dot; when the
 * request names an item the app can draw (Items.kt), the item then drops onto the point, standing with the middle of
 * its foot on it, and the ring fades. There is no text label on the photo: next to a free spot a label would cover the
 * very objects the request names.
 */
class MarkedPhotoView(context: Context) : View(context) {
  private var bitmap: Bitmap? = null
  private var spot: Spot? = null
  private var item: ItemDrawing? = null
  private var itemWidth = 0
  /** 0 when the item starts to drop, 1 once it has landed and the ring has gone (1 when there is no item). */
  private var landing = 1f
  private var animator: ValueAnimator? = null
  private val drawn = RectF()
  private val oval = RectF()
  private val fall = DecelerateInterpolator(1.6f)
  private fun dp(v: Float) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, resources.displayMetrics)

  private val photoPaint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
  private val ringOuter = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    style = Paint.Style.STROKE; strokeWidth = dp(7f); color = Color.WHITE
  }
  private val ringInner = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    style = Paint.Style.STROKE; strokeWidth = dp(4f); color = Color.rgb(230, 40, 60)
  }
  private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(230, 40, 60) }
  private val shadowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.BLACK }
  private val outlinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = Color.WHITE; strokeJoin = Paint.Join.ROUND; strokeCap = Paint.Cap.ROUND
  }
  private val layerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    strokeJoin = Paint.Join.ROUND; strokeCap = Paint.Cap.ROUND
  }

  /**
   * False from the mark until the item has landed (the measuring tool waits before it keeps the view); isStarted, not
   * isRunning, so that the ring's 0.4 s before the drop counts too.
   */
  val landed: Boolean get() = animator?.isStarted != true

  fun setPhoto(b: Bitmap?) {
    bitmap = b
    setSpot(null)
  }

  /**
   * The point to mark, or null for none (a point outside the picture is never drawn). With [drawing], that item drops
   * onto the point, [width] thousandths of the photo's width wide.
   */
  fun setSpot(s: Spot?, drawing: ItemDrawing? = null, width: Int = drawing?.width ?: 0) {
    animator?.cancel()
    animator = null
    spot = s?.takeIf { it.inPicture }
    item = drawing?.takeIf { spot != null && width > 0 }
    itemWidth = width
    landing = 1f
    if (item != null) {
      landing = 0f
      animator = ValueAnimator.ofFloat(0f, 1f).apply {
        startDelay = RING_MS
        duration = LAND_MS
        addUpdateListener { landing = it.animatedValue as Float; invalidate() }
        start()
      }
    }
    invalidate()
  }

  override fun onDetachedFromWindow() {
    animator?.cancel()
    super.onDetachedFromWindow()
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
    val drawing = item
    val ringAlpha = if (drawing == null) 1f else 1f - ((landing - FALL_SHARE) / (1f - FALL_SHARE)).coerceIn(0f, 1f)
    if (ringAlpha > 0f) drawRing(canvas, cx, cy, ringAlpha)
    if (drawing != null) drawItem(canvas, drawing, cx, cy)
  }

  private fun drawRing(canvas: Canvas, cx: Float, cy: Float, alpha: Float) {
    val a = (255 * alpha).toInt()
    ringOuter.alpha = a
    ringInner.alpha = a
    dotPaint.alpha = a
    val r = dp(16f)
    canvas.drawCircle(cx, cy, r, ringOuter)
    canvas.drawCircle(cx, cy, r, ringInner)
    canvas.drawCircle(cx, cy, dp(3.5f), dotPaint)
  }

  /** The item standing on the point, falling from a little above it while [landing] runs, kept inside the photo. */
  private fun drawItem(canvas: Canvas, d: ItemDrawing, cx: Float, cy: Float) {
    val k = itemWidth / 1000f * drawn.width() / d.w
    val t = (landing / FALL_SHARE).coerceIn(0f, 1f)
    val drop = (1f - fall.getInterpolation(t)) * dp(28f)
    canvas.save()
    canvas.clipRect(drawn)
    // The shadow on the surface darkens as the item comes down.
    val sx = d.footW * k * 0.525f
    val sy = sx * 0.3f
    shadowPaint.alpha = (0x55 * t).toInt()
    oval.set(cx - sx, cy - sy, cx + sx, cy + sy)
    canvas.drawOval(oval, shadowPaint)
    canvas.translate(cx - d.spotX * k, cy - d.spotY * k - drop)
    canvas.scale(k, k)
    // A white outline around the whole drawing keeps it apart from a busy or a dark photo.
    val halo = 2f * dp(2.5f) / k
    for ((i, p) in d.paths.withIndex()) {
      val l = d.layers[i]
      outlinePaint.style = if (l.stroke > 0f) Paint.Style.STROKE else Paint.Style.FILL_AND_STROKE
      outlinePaint.strokeWidth = l.stroke + halo
      canvas.drawPath(p, outlinePaint)
    }
    for ((i, p) in d.paths.withIndex()) {
      val l = d.layers[i]
      layerPaint.color = l.color
      layerPaint.style = if (l.stroke > 0f) Paint.Style.STROKE else Paint.Style.FILL
      layerPaint.strokeWidth = l.stroke
      canvas.drawPath(p, layerPaint)
    }
    canvas.restore()
  }
}
