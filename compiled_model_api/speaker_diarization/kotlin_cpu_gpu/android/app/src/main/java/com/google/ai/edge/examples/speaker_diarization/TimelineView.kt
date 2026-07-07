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

package com.google.ai.edge.examples.speaker_diarization

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.View

/** Colored per-speaker timeline: one row per speaker, bars where they talk. */
class TimelineView(ctx: Context) : View(ctx) {

    companion object {
        val COLORS = intArrayOf(
            Color.rgb(0x42, 0x85, 0xF4), Color.rgb(0xEA, 0x43, 0x35),
            Color.rgb(0xFB, 0xBC, 0x05), Color.rgb(0x34, 0xA8, 0x53),
            Color.rgb(0xAB, 0x47, 0xBC), Color.rgb(0x00, 0xAC, 0xC1),
        )
    }

    private var segments: List<Diarizer.Segment> = emptyList()
    private var duration = 1.0
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.DKGRAY
        textSize = 28f
    }

    fun set(segs: List<Diarizer.Segment>, durationSec: Double) {
        segments = segs
        duration = maxOf(durationSec, 0.001)
        requestLayout()
        invalidate()
    }

    override fun onMeasure(w: Int, h: Int) {
        val speakers = segments.map { it.speaker }.distinct().size
        val rowH = 64
        setMeasuredDimension(MeasureSpec.getSize(w), maxOf(1, speakers) * rowH + 40)
    }

    override fun onDraw(c: Canvas) {
        val speakers = segments.map { it.speaker }.distinct().sorted()
        if (speakers.isEmpty()) return
        val rowH = 64f
        val labelW = 130f
        val w = width - labelW
        for ((row, spk) in speakers.withIndex()) {
            val y = row * rowH + 20
            textPaint.color = COLORS[spk % COLORS.size]
            c.drawText("SPK ${spk + 1}", 8f, y + rowH / 2 + 10, textPaint)
            paint.color = Color.rgb(0xEE, 0xEE, 0xEE)
            c.drawRect(labelW, y, labelW + w, y + rowH - 12, paint)
            paint.color = COLORS[spk % COLORS.size]
            for (s in segments) {
                if (s.speaker != spk) continue
                val x0 = labelW + (s.start / duration * w).toFloat()
                val x1 = labelW + (s.end / duration * w).toFloat()
                c.drawRect(x0, y, x1, y + rowH - 12, paint)
            }
        }
    }
}
