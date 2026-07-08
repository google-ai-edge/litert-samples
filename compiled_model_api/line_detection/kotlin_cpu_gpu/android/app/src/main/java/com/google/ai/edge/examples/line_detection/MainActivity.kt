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


package com.google.ai.edge.examples.line_detection
import android.app.Activity
import android.content.Context
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
import android.widget.TextView
import java.util.concurrent.Executors
import kotlin.math.min

/**
 * M-LSD line segment detection demo, fully on the CompiledModel GPU. Detects line segments
 * (building edges, document borders, wireframes) and draws them. Works on a bundled image and
 * any image picked from the gallery.
 */
class MainActivity : Activity() {

    private val tag = "MLSD"
    private val bg = Executors.newSingleThreadExecutor()
    private var net: MlsdDetector? = null
    private val pickReq = 100

    private lateinit var status: TextView
    private lateinit var lineView: LineView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 80, 24, 24)
        }
        status = TextView(this).apply {
            textSize = 15f
            text = "Loading M-LSD on GPU…"
        }
        val pick = Button(this).apply {
            text = "🖼  Pick image"
            isEnabled = false
            setOnClickListener {
                val intent = Intent(Intent.ACTION_GET_CONTENT).apply { type = "image/*" }
                startActivityForResult(intent, pickReq)
            }
        }
        lineView = LineView(this)
        root.addView(status)
        root.addView(pick)
        root.addView(
            lineView, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 1000))
        setContentView(root)

        bg.execute {
            try {
                net = MlsdDetector(this)
                try {
                    val bundled = BitmapFactory.decodeStream(assets.open("test_image.jpg"))
                    run(squareResize(bundled), warm = true)
                } catch (_: java.io.IOException) {
                    runOnUiThread { status.text = "Ready — pick an image to detect line segments." }
                }
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
        runOnUiThread { status.text = "Detecting lines…" }
        bg.execute {
            try { run(squareResize(loadOriented(uri)), warm = false) }
            catch (e: Throwable) {
                Log.e(tag, "detect failed", e)
                runOnUiThread { status.text = "Failed: ${e.message}" }
            }
        }
    }

    private fun run(img: Bitmap, warm: Boolean) {
        val n = net!!
        val rgb = bitmapToRgb(img)
        if (warm) {
            n.detect(rgb, img.width.toFloat(), img.height.toFloat())
        }
        val t0 = System.nanoTime()
        val lines = n.detect(rgb, img.width.toFloat(), img.height.toFloat())
        val ms = (System.nanoTime() - t0) / 1_000_000
        Log.i(tag, "detect ${ms}ms lines=${lines.size}")
        runOnUiThread {
            status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
            status.text = "On-device GPU line detection ✓  ${ms} ms  ·  ${lines.size} segments" +
                "  ·  M-LSD-tiny, CompiledModel GPU"
            lineView.set(img, lines)
            lineView.invalidate()
        }
    }

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
        return Bitmap.createBitmap(
            bm, 0, 0, bm.width, bm.height, Matrix().apply { postRotate(rot) }, true)
    }

    private fun squareResize(src: Bitmap): Bitmap {
        val s = min(src.width, src.height)
        val crop = Bitmap.createBitmap(src, (src.width - s) / 2, (src.height - s) / 2, s, s)
        return Bitmap.createScaledBitmap(crop, MlsdDetector.SIZE, MlsdDetector.SIZE, true)
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

    override fun onDestroy() {
        super.onDestroy()
        bg.shutdown()
        net?.close()
    }

    class LineView(ctx: Context) : View(ctx) {
        private var bm: Bitmap? = null
        private var lines: List<MlsdDetector.Line> = emptyList()
        private val paint = Paint().apply {
            color = Color.rgb(255, 40, 40)
            strokeWidth = 3f
            isAntiAlias = true
        }
        private val imgPaint = Paint().apply { isFilterBitmap = true }

        fun set(b: Bitmap, l: List<MlsdDetector.Line>) {
            bm = b
            lines = l
        }

        override fun onDraw(canvas: Canvas) {
            val b = bm ?: return
            val s = min(width.toFloat() / b.width, height.toFloat() / b.height)
            val w = b.width * s
            val h = b.height * s
            val ox = (width - w) / 2
            val oy = (height - h) / 2
            canvas.drawBitmap(b, null, android.graphics.RectF(ox, oy, ox + w, oy + h), imgPaint)
            for (l in lines) {
                canvas.drawLine(ox + l.x1 * s, oy + l.y1 * s, ox + l.x2 * s, oy + l.y2 * s, paint)
            }
        }
    }
}
