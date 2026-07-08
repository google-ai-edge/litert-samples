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

package com.google.ai.edge.examples.low_light_enhancement

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
import android.view.MotionEvent
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import java.util.concurrent.Executors
import kotlin.math.min

/**
 * CPGA-Net low-light enhancement demo, fully on the CompiledModel GPU.
 * Brightens a dark image; press-and-hold the image to compare with the
 * original. Works on a bundled image and any image picked from the
 * gallery.
 */
class MainActivity : Activity() {

    private val tag = "LOWLIGHT"
    private val bg = Executors.newSingleThreadExecutor()
    private var net: LowLightEnhancer? = null
    private val pickReq = 100

    private lateinit var status: TextView
    private lateinit var view: CompareView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 80, 24, 24)
        }
        status = TextView(this).apply {
            textSize = 15f
            text = "Loading CPGA-Net on GPU…"
        }
        val pick = Button(this).apply {
            text = "🖼  Pick image"
            isEnabled = false
            setOnClickListener {
                val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
                    type = "image/*"
                }
                startActivityForResult(intent, pickReq)
            }
        }
        view = CompareView(this)
        root.addView(status)
        root.addView(pick)
        root.addView(view, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 980))
        setContentView(root)

        bg.execute {
            try {
                net = LowLightEnhancer(this)
                try {
                    val bundled = squareResize(
                        BitmapFactory.decodeStream(
                            assets.open("test_image.jpg")
                        )
                    )
                    run(bundled, warm = true)
                } catch (_: java.io.IOException) {
                    runOnUiThread { status.text = "Ready — pick a dark image." }
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
        runOnUiThread { status.text = "Enhancing…" }
        bg.execute {
            try { run(squareResize(loadOriented(uri)), warm = false) }
            catch (e: Throwable) {
                Log.e(tag, "enhance failed", e)
                runOnUiThread { status.text = "Failed: ${e.message}" }
            }
        }
    }

    private fun run(src: Bitmap, warm: Boolean) {
        val n = net!!
        val rgb = bitmapToRgb(src)
        if (warm) {
            n.enhance(rgb)
        }
        val t0 = System.nanoTime()
        val enh = n.enhance(rgb)
        val ms = (System.nanoTime() - t0) / 1_000_000
        Log.i(tag, "enhance ${ms}ms")
        runOnUiThread {
            status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
            status.text =
                "On-device GPU low-light ✓  ${ms} ms  ·  hold to compare  ·  " +
                    "CPGA-Net, CompiledModel GPU"
            view.set(src, enh)
            view.invalidate()
        }
    }

    private fun loadOriented(uri: Uri): Bitmap {
        val bm = contentResolver.openInputStream(uri).use {
            BitmapFactory.decodeStream(it)
        } ?: error("cannot decode image")
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
            bm, 0, 0, bm.width, bm.height,
            Matrix().apply { postRotate(rot) }, true
        )
    }

    private fun squareResize(src: Bitmap): Bitmap {
        val s = min(src.width, src.height)
        val crop = Bitmap.createBitmap(src, (src.width - s) / 2, (src.height - s) / 2, s, s)
        return Bitmap.createScaledBitmap(crop, LowLightEnhancer.SIZE, LowLightEnhancer.SIZE, true)
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

    /** Shows the enhanced image; while pressed, shows the original (before/after compare). */
    class CompareView(ctx: Context) : View(ctx) {
        private var original: Bitmap? = null
        private var enhanced: Bitmap? = null
        private var showOriginal = false
        private val paint = Paint().apply { isFilterBitmap = true }

        fun set(orig: Bitmap, enh: Bitmap) {
            original = orig
            enhanced = enh
            showOriginal = false
        }

        override fun onTouchEvent(e: MotionEvent): Boolean {
            showOriginal = when (e.action) {
                MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> true
                else -> false
            }
            invalidate()
            return true
        }

        override fun onDraw(canvas: Canvas) {
            val bm = (if (showOriginal) original else enhanced) ?: return
            val s = min(width.toFloat() / bm.width, height.toFloat() / bm.height)
            val w = bm.width * s
            val h = bm.height * s
            val dst = android.graphics.RectF(
                (width - w) / 2, (height - h) / 2, (width + w) / 2, (height + h) / 2
            )
            canvas.drawBitmap(bm, null, dst, paint)
        }
    }
}
