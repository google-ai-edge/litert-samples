/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.ai.edge.examples.ocr

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.util.concurrent.Executors

/**
 * PP-OCRv5 demo: detects text in a bundled image and recognizes each line, fully on the GPU
 * (detector + recognizer both on CompiledModel GPU; DB box extraction + CTC decode on CPU).
 */
class MainActivity : Activity() {

    private val tag = "PPOCR"
    private val bg = Executors.newSingleThreadExecutor()
    private var det: PpocrDetector? = null
    private var rec: PpocrRecognizer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 80, 24, 24) }
        val status = TextView(this).apply { textSize = 15f; text = "Loading PP-OCRv5 on GPU…" }
        val overlay = OverlayView(this)
        val results = TextView(this).apply { textSize = 14f; setPadding(0, 24, 0, 0) }
        root.addView(status); root.addView(overlay,
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 900))
        val scroll = ScrollView(this).apply { addView(results) }
        root.addView(scroll)
        setContentView(root)

        val src = BitmapFactory.decodeStream(assets.open("test_image.png"))
        val img = Bitmap.createScaledBitmap(src, PpocrDetector.SIZE, PpocrDetector.SIZE, true)
        overlay.bitmap = img

        bg.execute {
            try {
                val d = PpocrDetector(this); det = d
                val r = PpocrRecognizer(this); rec = r
                val rgb = bitmapToRgb(img)
                val t0 = System.nanoTime()
                val prob = d.probMap(rgb)
                val boxes = d.boxes(prob)
                val lines = ArrayList<Pair<PpocrDetector.Box, String>>()
                for (b in boxes) {
                    val crop = cropResize(img, b)
                    val text = r.recognize(crop)
                    if (text.isNotBlank()) lines.add(b to text)
                }
                val ms = (System.nanoTime() - t0) / 1_000_000
                Log.i(tag, "boxes=${boxes.size} lines=${lines.size} ${ms}ms")
                runOnUiThread {
                    status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
                    status.text = "On-device GPU OCR ✓  ${lines.size} lines · ${ms} ms\n" +
                        "detector + recognizer both on CompiledModel GPU"
                    overlay.boxes = lines.map { it.first }
                    overlay.invalidate()
                    results.text = lines.joinToString("\n") { "• ${it.second}" }
                }
            } catch (e: Throwable) {
                Log.e(tag, "ocr failed", e)
                runOnUiThread {
                    status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2))
                    status.text = "FAIL: ${e.message}"
                }
            }
        }
    }

    private fun bitmapToRgb(bm: Bitmap): FloatArray {
        val n = bm.width * bm.height
        val px = IntArray(n); bm.getPixels(px, 0, bm.width, 0, 0, bm.width, bm.height)
        val out = FloatArray(n * 3)
        for (i in 0 until n) {
            val p = px[i]
            out[i * 3] = ((p shr 16) and 0xFF).toFloat()
            out[i * 3 + 1] = ((p shr 8) and 0xFF).toFloat()
            out[i * 3 + 2] = (p and 0xFF).toFloat()
        }
        return out
    }

    /** Crop the box from img, resize to H keeping aspect, pad to W. Returns H*W*3 [0,255]. */
    private fun cropResize(img: Bitmap, b: PpocrDetector.Box): FloatArray {
        val bw = b.x1 - b.x0 + 1; val bh = b.y1 - b.y0 + 1
        val crop = Bitmap.createBitmap(img, b.x0, b.y0, bw, bh)
        val nw = minOf((PpocrRecognizer.H.toFloat() * bw / bh).toInt(), PpocrRecognizer.W).coerceAtLeast(1)
        val rz = Bitmap.createScaledBitmap(crop, nw, PpocrRecognizer.H, true)
        val px = IntArray(nw * PpocrRecognizer.H)
        rz.getPixels(px, 0, nw, 0, 0, nw, PpocrRecognizer.H)
        val out = FloatArray(PpocrRecognizer.H * PpocrRecognizer.W * 3)   // zero-padded
        for (y in 0 until PpocrRecognizer.H) for (x in 0 until nw) {
            val p = px[y * nw + x]; val o = (y * PpocrRecognizer.W + x) * 3
            out[o] = ((p shr 16) and 0xFF).toFloat()
            out[o + 1] = ((p shr 8) and 0xFF).toFloat()
            out[o + 2] = (p and 0xFF).toFloat()
        }
        return out
    }

    override fun onDestroy() { super.onDestroy(); bg.shutdown(); det?.close(); rec?.close() }

    /** Draws the image with detected boxes overlaid. */
    class OverlayView(ctx: android.content.Context) : View(ctx) {
        var bitmap: Bitmap? = null
        var boxes: List<PpocrDetector.Box> = emptyList()
        private val stroke = Paint().apply { color = Color.RED; style = Paint.Style.STROKE; strokeWidth = 3f }

        override fun onDraw(canvas: Canvas) {
            val bm = bitmap ?: return
            val s = minOf(width.toFloat() / bm.width, height.toFloat() / bm.height)
            val dw = bm.width * s; val dh = bm.height * s
            val dst = android.graphics.RectF(0f, 0f, dw, dh)
            canvas.drawBitmap(bm, null, dst, null)
            for (b in boxes) canvas.drawRect(b.x0 * s, b.y0 * s, b.x1 * s, b.y1 * s, stroke)
        }
    }
}
