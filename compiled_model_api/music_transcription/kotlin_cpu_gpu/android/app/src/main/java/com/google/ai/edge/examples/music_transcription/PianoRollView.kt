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

package com.google.ai.edge.examples.music_transcription

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.View

/** Piano-roll: time -> x, pitch -> y, brightness -> confidence. */
class PianoRollView(ctx: Context) : View(ctx) {

    private var notes: List<Transcriber.Note> = emptyList()
    private var duration = 1.0
    private var loNote = 21
    private var hiNote = 108
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.GRAY; textSize = 24f }
    private val names = arrayOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

    fun set(events: List<Transcriber.Note>, durationSec: Double) {
        notes = events
        duration = maxOf(durationSec, 0.001)
        if (events.isNotEmpty()) {
            loNote = maxOf(21, events.minOf { it.midi } - 2)
            hiNote = minOf(108, events.maxOf { it.midi } + 2)
        }
        invalidate()
    }

    override fun onMeasure(w: Int, h: Int) {
        setMeasuredDimension(MeasureSpec.getSize(w), 560)
    }

    override fun onDraw(c: Canvas) {
        paint.color = Color.rgb(0xF5, 0xF5, 0xF5)
        c.drawRect(0f, 0f, width.toFloat(), height.toFloat(), paint)
        val span = (hiNote - loNote + 1).coerceAtLeast(12)
        val rowH = height.toFloat() / span
        // octave guides (C rows)
        for (m in loNote..hiNote) {
            if (m % 12 == 0) {
                val y = height - (m - loNote + 1) * rowH
                paint.color = Color.rgb(0xE0, 0xE0, 0xE0)
                c.drawRect(0f, y, width.toFloat(), y + rowH, paint)
                c.drawText("C${m / 12 - 1}", 4f, y + rowH, text)
            }
        }
        for (n in notes) {
            val x0 = (n.startSec / duration * width).toFloat()
            val x1 = (n.endSec / duration * width).toFloat().coerceAtLeast(x0 + 4f)
            val y = height - (n.midi - loNote + 1) * rowH
            val a = (80 + 175 * n.amplitude.coerceIn(0f, 1f)).toInt()
            paint.color = Color.argb(a, 0x1E, 0x88, 0xE5)
            c.drawRoundRect(x0, y + 1, x1, y + rowH - 1, 4f, 4f, paint)
        }
    }

    fun noteName(midi: Int) = "${names[midi % 12]}${midi / 12 - 1}"
}
