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
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.util.concurrent.Executors

/**
 * PP-OCRv5 demo: detects + recognizes text fully on the GPU (detector + recognizer both on
 * CompiledModel GPU; DB box extraction + CTC decode on CPU). Runs on a bundled image at launch and
 * on any image picked from the gallery.
 */
class MainActivity : Activity() {

    private val tag = "PPOCR"
    private val bg = Executors.newSingleThreadExecutor()
    private var det: PpocrDetector? = null
    private var rec: PpocrRecognizer? = null
    private val pickReq = 100

    private lateinit var status: TextView
    private lateinit var overlay: OverlayView
    private lateinit var results: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 80, 24, 24)
        }
        status = TextView(this).apply {
            textSize = 15f
            text = "Loading PP-OCRv5 on GPU…"
        }
        val pick = Button(this).apply {
            text = "🖼  Pick image"
            isEnabled = false
            setOnClickListener {
                startActivityForResult(
                    Intent(Intent.ACTION_GET_CONTENT).apply { type = "image/*" }, pickReq)
            }
        }
        overlay = OverlayView(this)
        results = TextView(this).apply {
            textSize = 14f
            setPadding(0, 24, 0, 0)
        }
        root.addView(status)
        root.addView(pick)
        root.addView(overlay, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 820))
        root.addView(ScrollView(this).apply { addView(results) })
        setContentView(root)

        bg.execute {
            try {
                det = PpocrDetector(this)
                rec = PpocrRecognizer(this)
                val bundled = letterbox(BitmapFactory.decodeStream(assets.open("test_image.png")))
                runOcr(bundled, warm = true)
                runOnUiThread { pick.isEnabled = true }
            } catch (e: Throwable) {
                Log.e(tag, "load failed", e)
                runOnUiThread {
                    status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2))
                    status.text = "FAIL: ${e.message}"
                }
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != pickReq || resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        runOnUiThread { status.text = "Running OCR…" }
        bg.execute {
            try {
                runOcr(letterbox(loadOriented(uri)), warm = false)
            } catch (e: Throwable) {
                Log.e(tag, "ocr failed", e)
                runOnUiThread { status.text = "Failed: ${e.message}" }
            }
        }
    }

    /** Detect + recognize on a SIZE×SIZE bitmap and update the UI. */
    private fun runOcr(img: Bitmap, warm: Boolean) {
        val d = det!!
        val r = rec!!
        val rgb = bitmapToRgb(img)
        if (warm) d.probMap(rgb)                       // warm up GPU shaders once
        val t0 = System.nanoTime()
        val boxes = d.boxes(d.probMap(rgb))
        val lines = ArrayList<Pair<PpocrDetector.Box, String>>()
        for (b in boxes) {
            val text = r.recognize(cropResize(img, b))
            if (text.isNotBlank()) lines.add(b to text)
        }
        val ms = (System.nanoTime() - t0) / 1_000_000
        Log.i(tag, "boxes=${boxes.size} lines=${lines.size} ${ms}ms")
        runOnUiThread {
            status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
            status.text = "On-device GPU OCR ✓  ${lines.size} lines · ${ms} ms\n" +
                "detector + recognizer both on CompiledModel GPU"
            overlay.bitmap = img
            overlay.boxes = lines.map { it.first }
            overlay.invalidate()
            results.text = if (lines.isEmpty()) "(no text found)" else lines.joinToString("\n") { "• ${it.second}" }
        }
    }

    /** Decode a picked image and apply its EXIF orientation. */
    private fun loadOriented(uri: Uri): Bitmap {
        val bm = contentResolver.openInputStream(uri).use { BitmapFactory.decodeStream(it) }
            ?: error("cannot decode image")
        val rot = contentResolver.openInputStream(uri).use {
            when (ExifInterface(it!!).getAttributeInt(ExifInterface.TAG_ORIENTATION, 1)) {
                ExifInterface.ORIENTATION_ROTATE_90 -> 90f
                ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                else -> 0f
            }
        }
        if (rot == 0f) return bm
        return Bitmap.createBitmap(bm, 0, 0, bm.width, bm.height, Matrix().apply { postRotate(rot) }, true)
    }

    /** Resize keeping aspect to fit SIZE, center on a white SIZE×SIZE canvas (letterbox). */
    private fun letterbox(src: Bitmap): Bitmap {
        val s = minOf(PpocrDetector.SIZE.toFloat() / src.width, PpocrDetector.SIZE.toFloat() / src.height)
        val nw = (src.width * s).toInt().coerceAtLeast(1)
        val nh = (src.height * s).toInt().coerceAtLeast(1)
        val out = Bitmap.createBitmap(PpocrDetector.SIZE, PpocrDetector.SIZE, Bitmap.Config.ARGB_8888)
        Canvas(out).apply {
            drawColor(Color.WHITE)
            drawBitmap(Bitmap.createScaledBitmap(src, nw, nh, true),
                (PpocrDetector.SIZE - nw) / 2f, (PpocrDetector.SIZE - nh) / 2f, null)
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
        val nw = minOf((PpocrRecognizer.H.toFloat() * bw / bh).toInt(), PpocrRecognizer.W).coerceAtLeast(1)
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

    override fun onDestroy() {
        super.onDestroy()
        bg.shutdown()
        det?.close()
        rec?.close()
    }

    class OverlayView(ctx: android.content.Context) : View(ctx) {
        var bitmap: Bitmap? = null
        var boxes: List<PpocrDetector.Box> = emptyList()
        private val stroke = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 3f
        }
        override fun onDraw(canvas: Canvas) {
            val bm = bitmap ?: return
            val s = minOf(width.toFloat() / bm.width, height.toFloat() / bm.height)
            canvas.drawBitmap(bm, null, android.graphics.RectF(0f, 0f, bm.width * s, bm.height * s), null)
            for (b in boxes) canvas.drawRect(b.x0 * s, b.y0 * s, b.x1 * s, b.y1 * s, stroke)
        }
    }
}
