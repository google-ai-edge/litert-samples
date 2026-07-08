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

package com.google.ai.edge.examples.gaze_estimation

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
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

/**
 * L2CS-Net gaze estimation demo, fully on the CompiledModel GPU. Estimates where a centered
 * face is looking and draws the gaze direction arrow. Works on a bundled image and any image
 * picked from the gallery.
 */
class MainActivity : Activity() {

    private val tag = "GAZE"
    private val bg = Executors.newSingleThreadExecutor()
    private var net: GazeEstimator? = null
    private val pickReq = 100

    private lateinit var status: TextView
    private lateinit var gazeView: GazeView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 80, 24, 24)
        }
        status = TextView(this).apply {
            textSize = 15f
            text = "Loading gaze model on GPU…"
        }
        val pick = Button(this).apply {
            text = "🖼  Pick image"
            isEnabled = false
            setOnClickListener {
                val intent = Intent(Intent.ACTION_GET_CONTENT).apply { type = "image/*" }
                startActivityForResult(intent, pickReq)
            }
        }
        gazeView = GazeView(this)
        root.addView(status)
        root.addView(pick)
        root.addView(
            gazeView, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 980))
        setContentView(root)

        bg.execute {
            try {
                net = GazeEstimator(this)
                try {
                    val bundled = BitmapFactory.decodeStream(assets.open("test_image.jpg"))
                    run(squareResize(bundled), warm = true)
                } catch (_: java.io.IOException) {
                    runOnUiThread { status.text = "Ready — pick a face image to estimate gaze." }
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
        runOnUiThread { status.text = "Estimating gaze…" }
        bg.execute {
            try { run(squareResize(loadOriented(uri)), warm = false) }
            catch (e: Throwable) {
                Log.e(tag, "estimate failed", e)
                runOnUiThread { status.text = "Failed: ${e.message}" }
            }
        }
    }

    private fun run(face: Bitmap, warm: Boolean) {
        val n = net!!
        val rgb = bitmapToRgb(face)
        if (warm) {
            n.estimate(rgb)
        }
        val t0 = System.nanoTime()
        val g = n.estimate(rgb)
        val ms = (System.nanoTime() - t0) / 1_000_000
        Log.i(tag, "gaze ${ms}ms yaw=${g.yawDeg} pitch=${g.pitchDeg}")
        runOnUiThread {
            status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
            status.text = ("On-device GPU gaze ✓  ${ms} ms  ·  yaw %.0f° pitch %.0f°" +
                "  ·  L2CS-Net, CompiledModel GPU").format(g.yawDeg, g.pitchDeg)
            gazeView.set(face, g)
            gazeView.invalidate()
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
        return Bitmap.createScaledBitmap(crop, GazeEstimator.SIZE, GazeEstimator.SIZE, true)
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

    class GazeView(ctx: Context) : View(ctx) {
        private var bm: Bitmap? = null
        private var gaze: GazeEstimator.Gaze? = null
        private val arrow = Paint().apply {
            color = Color.rgb(255, 40, 40)
            strokeWidth = 9f
            isAntiAlias = true
        }
        private val dot = Paint().apply {
            color = Color.rgb(0, 200, 0)
            isAntiAlias = true
        }
        private val imgPaint = Paint().apply { isFilterBitmap = true }

        fun set(b: Bitmap, g: GazeEstimator.Gaze) {
            bm = b
            gaze = g
        }

        override fun onDraw(canvas: Canvas) {
            val b = bm ?: return
            val g = gaze ?: return
            val s = min(width.toFloat() / b.width, height.toFloat() / b.height)
            val w = b.width * s
            val h = b.height * s
            val ox = (width - w) / 2
            val oy = (height - h) / 2
            canvas.drawBitmap(b, null, android.graphics.RectF(ox, oy, ox + w, oy + h), imgPaint)
            // gaze arrow from the upper-center (~face) of the image
            val cx = ox + w / 2
            val cy = oy + h * 0.4f
            val len = w * 0.32f
            val yaw = Math.toRadians(g.yawDeg.toDouble())
            val pit = Math.toRadians(g.pitchDeg.toDouble())
            val dx = (-len * sin(yaw) * cos(pit)).toFloat()
            val dy = (-len * sin(pit)).toFloat()
            canvas.drawLine(cx, cy, cx + dx, cy + dy, arrow)
            canvas.drawCircle(cx, cy, 9f, dot)
        }
    }
}
