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

package com.google.ai.edge.examples.face_detection

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
 * YuNet face-detection demo, fully on the CompiledModel GPU. Detects faces + 5 landmarks in a bundled image
 * and any image picked from the gallery, drawing boxes and landmark points.
 */
class MainActivity : Activity() {

    private val tag = "YUNET"
    private val bg = Executors.newSingleThreadExecutor()
    private var net: FaceDetector? = null
    private val pickReq = 100

    private lateinit var status: TextView
    private lateinit var faceView: FaceView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 80, 24, 24) }
        status = TextView(this).apply { textSize = 15f; text = "Loading YuNet on GPU…" }
        val pick = Button(this).apply {
            text = "🖼  Pick image"; isEnabled = false
            setOnClickListener {
                startActivityForResult(Intent(Intent.ACTION_GET_CONTENT).apply { type = "image/*" }, pickReq)
            }
        }
        faceView = FaceView(this)
        root.addView(status); root.addView(pick)
        root.addView(faceView, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 980))
        setContentView(root)

        bg.execute {
            try {
                net = FaceDetector(this)
                try {
                    run(squareResize(BitmapFactory.decodeStream(assets.open("test_image.jpg"))), warm = true)
                } catch (_: java.io.IOException) {
                    runOnUiThread { status.text = "Ready — pick an image to detect faces." }
                }
                runOnUiThread { pick.isEnabled = true }
            } catch (e: Throwable) {
                Log.e(tag, "load failed", e)
                runOnUiThread { status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2)); status.text = "FAIL: ${e.message}" }
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != pickReq || resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        runOnUiThread { status.text = "Detecting…" }
        bg.execute {
            try { run(squareResize(loadOriented(uri)), warm = false) }
            catch (e: Throwable) { Log.e(tag, "detect failed", e); runOnUiThread { status.text = "Failed: ${e.message}" } }
        }
    }

    private fun run(bm: Bitmap, warm: Boolean) {
        val n = net!!
        val rgb = bitmapToRgb(bm)
        if (warm) n.detect(rgb)
        val t0 = System.nanoTime()
        val faces = n.detect(rgb)
        val ms = (System.nanoTime() - t0) / 1_000_000
        Log.i(tag, "detect ${ms}ms faces=${faces.size}")
        runOnUiThread {
            status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
            status.text = "On-device GPU faces ✓  ${ms} ms  ·  ${faces.size} face(s)  ·  YuNet, CompiledModel GPU"
            faceView.set(bm, faces); faceView.invalidate()
        }
    }

    private fun loadOriented(uri: Uri): Bitmap {
        val bm = contentResolver.openInputStream(uri).use { BitmapFactory.decodeStream(it) } ?: error("cannot decode image")
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

    /** Letterbox into a SIZE×SIZE square (keep aspect, pad) so no face is cropped out. */
    private fun squareResize(src: Bitmap): Bitmap {
        val sz = FaceDetector.SIZE
        val s = min(sz.toFloat() / src.width, sz.toFloat() / src.height)
        val nw = (src.width * s).toInt(); val nh = (src.height * s).toInt()
        val scaled = Bitmap.createScaledBitmap(src, nw, nh, true)
        val out = Bitmap.createBitmap(sz, sz, Bitmap.Config.ARGB_8888)
        Canvas(out).apply { drawColor(Color.BLACK); drawBitmap(scaled, ((sz - nw) / 2).toFloat(), ((sz - nh) / 2).toFloat(), null) }
        return out
    }

    private fun bitmapToRgb(bm: Bitmap): FloatArray {
        val n = bm.width * bm.height; val px = IntArray(n)
        bm.getPixels(px, 0, bm.width, 0, 0, bm.width, bm.height)
        val out = FloatArray(n * 3)
        for (i in 0 until n) {
            val p = px[i]
            out[i * 3] = ((p shr 16) and 0xFF).toFloat(); out[i * 3 + 1] = ((p shr 8) and 0xFF).toFloat()
            out[i * 3 + 2] = (p and 0xFF).toFloat()
        }
        return out
    }

    override fun onDestroy() { super.onDestroy(); bg.shutdown(); net?.close() }

    class FaceView(ctx: Context) : View(ctx) {
        private var bm: Bitmap? = null
        private var faces: List<FaceDetector.Face> = emptyList()
        private val box = Paint().apply { color = Color.rgb(0, 230, 0); style = Paint.Style.STROKE; strokeWidth = 4f; isAntiAlias = true }
        private val lm = Paint().apply { color = Color.rgb(255, 40, 40); isAntiAlias = true }
        private val imgPaint = Paint().apply { isFilterBitmap = true }

        fun set(b: Bitmap, f: List<FaceDetector.Face>) { bm = b; faces = f }

        override fun onDraw(canvas: Canvas) {
            val b = bm ?: return
            val s = min(width.toFloat() / b.width, height.toFloat() / b.height)
            val w = b.width * s; val h = b.height * s
            val ox = (width - w) / 2; val oy = (height - h) / 2
            canvas.drawBitmap(b, null, android.graphics.RectF(ox, oy, ox + w, oy + h), imgPaint)
            for (f in faces) {
                canvas.drawRect(ox + f.x1 * s, oy + f.y1 * s, ox + f.x2 * s, oy + f.y2 * s, box)
                for (j in 0 until 5) canvas.drawCircle(ox + f.landmarks[2 * j] * s, oy + f.landmarks[2 * j + 1] * s, 4f, lm)
            }
        }
    }
}
