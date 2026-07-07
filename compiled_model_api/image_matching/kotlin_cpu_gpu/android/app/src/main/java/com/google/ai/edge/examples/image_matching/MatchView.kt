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

package com.google.ai.edge.examples.image_matching

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.view.View

/** Side-by-side image pair with match lines colored by similarity. */
class MatchView(ctx: Context) : View(ctx) {

    private var bmA: Bitmap? = null
    private var bmB: Bitmap? = null
    private var matches: List<XFeatMatcher.Match> = emptyList()
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)

    fun set(a: Bitmap, b: Bitmap, m: List<XFeatMatcher.Match>) {
        bmA = a
        bmB = b
        matches = m
        requestLayout()
        invalidate()
    }

    override fun onMeasure(w: Int, h: Int) {
        val width = MeasureSpec.getSize(w)
        setMeasuredDimension(width, (width / 2f * 480f / 640f).toInt().coerceAtLeast(1))
    }

    override fun onDraw(c: Canvas) {
        val a = bmA ?: return
        val b = bmB ?: return
        val half = width / 2f
        val hgt = height.toFloat()
        c.drawBitmap(a, null, RectF(0f, 0f, half, hgt), null)
        c.drawBitmap(b, null, RectF(half, 0f, width.toFloat(), hgt), null)
        val sx = half / XFeatMatcher.W
        val sy = hgt / XFeatMatcher.H
        paint.strokeWidth = 2.5f
        for (m in matches) {
            // green (high sim) -> yellow (borderline)
            val t = ((m.sim - XFeatMatcher.MIN_COSSIM) / (1f - XFeatMatcher.MIN_COSSIM)).coerceIn(0f, 1f)
            paint.color = Color.argb(200, (255 * (1 - t)).toInt(), 220, 40)
            c.drawLine(m.x0 * sx, m.y0 * sy, half + m.x1 * sx, m.y1 * sy, paint)
            c.drawCircle(m.x0 * sx, m.y0 * sy, 3.5f, paint)
            c.drawCircle(half + m.x1 * sx, m.y1 * sy, 3.5f, paint)
        }
    }
}
