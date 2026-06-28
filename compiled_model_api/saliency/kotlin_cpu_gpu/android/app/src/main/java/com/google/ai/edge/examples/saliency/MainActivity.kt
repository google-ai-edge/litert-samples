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

package com.google.ai.edge.examples.saliency

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
 * UniSal visual-saliency demo, fully on the CompiledModel GPU. Predicts where humans look and overlays a
 * jet heatmap on a bundled image and any image picked from the gallery.
 */
class MainActivity : Activity() {

    private val tag = "SALIENCY"
    private val bg = Executors.newSingleThreadExecutor()
    private var net: SaliencyPredictor? = null
    private val pickReq = 100

    private lateinit var status: TextView
    private lateinit var view: SaliencyView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 80, 24, 24) }
        status = TextView(this).apply { textSize = 15f; text = "Loading UniSal on GPU…" }
        val pick = Button(this).apply {
            text = "🖼  Pick image"; isEnabled = false
            setOnClickListener {
                startActivityForResult(Intent(Intent.ACTION_GET_CONTENT).apply { type = "image/*" }, pickReq)
            }
        }
        view = SaliencyView(this)
        root.addView(status); root.addView(pick)
        root.addView(view, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 980))
        setContentView(root)

        bg.execute {
            try {
                net = SaliencyPredictor(this)
                try {
                    run(squareResize(BitmapFactory.decodeStream(assets.open("test_image.jpg"))), warm = true)
                } catch (_: java.io.IOException) {
                    runOnUiThread { status.text = "Ready — pick an image." }
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
        runOnUiThread { status.text = "Predicting saliency…" }
        bg.execute {
            try { run(squareResize(loadOriented(uri)), warm = false) }
            catch (e: Throwable) { Log.e(tag, "predict failed", e); runOnUiThread { status.text = "Failed: ${e.message}" } }
        }
    }

    private fun run(bm: Bitmap, warm: Boolean) {
        val n = net!!
        val rgb = bitmapToRgb(bm)
        if (warm) n.predict(rgb)
        val t0 = System.nanoTime()
        val sal = n.predict(rgb)
        val ms = (System.nanoTime() - t0) / 1_000_000
        Log.i(tag, "predict ${ms}ms")
        val heat = overlay(bm, sal)
        runOnUiThread {
            status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
            status.text = "On-device GPU saliency ✓  ${ms} ms  ·  UniSal, CompiledModel GPU"
            view.bitmap = heat; view.invalidate()
        }
    }

    /** Blend a jet heatmap of the saliency over the image. */
    private fun overlay(bm: Bitmap, sal: FloatArray): Bitmap {
        val s = SaliencyPredictor.SIZE
        val src = if (bm.width == s && bm.height == s) bm else Bitmap.createScaledBitmap(bm, s, s, true)
        val px = IntArray(s * s); src.getPixels(px, 0, s, 0, 0, s, s)
        val out = IntArray(s * s)
        for (i in px.indices) {
            val v = sal[i]
            val (hr, hg, hb) = jet(v)
            val a = 0.55f * v + 0.15f          // more opaque where salient
            val p = px[i]
            val r = ((p shr 16) and 0xFF) * (1 - a) + hr * a
            val g = ((p shr 8) and 0xFF) * (1 - a) + hg * a
            val b2 = (p and 0xFF) * (1 - a) + hb * a
            out[i] = Color.rgb(r.toInt().coerceIn(0, 255), g.toInt().coerceIn(0, 255), b2.toInt().coerceIn(0, 255))
        }
        return Bitmap.createBitmap(out, s, s, Bitmap.Config.ARGB_8888)
    }

    /** Jet colormap: [0,1] -> (r,g,b) in [0,255]. */
    private fun jet(x: Float): Triple<Float, Float, Float> {
        val v = x.coerceIn(0f, 1f)
        fun clamp(a: Float) = (a.coerceIn(0f, 1f)) * 255f
        val r = clamp(1.5f - kotlin.math.abs(4f * v - 3f))
        val g = clamp(1.5f - kotlin.math.abs(4f * v - 2f))
        val b = clamp(1.5f - kotlin.math.abs(4f * v - 1f))
        return Triple(r, g, b)
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

    private fun squareResize(src: Bitmap): Bitmap {
        val s = min(src.width, src.height)
        val crop = Bitmap.createBitmap(src, (src.width - s) / 2, (src.height - s) / 2, s, s)
        return Bitmap.createScaledBitmap(crop, SaliencyPredictor.SIZE, SaliencyPredictor.SIZE, true)
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

    class SaliencyView(ctx: Context) : View(ctx) {
        var bitmap: Bitmap? = null
        private val paint = Paint().apply { isFilterBitmap = true }
        override fun onDraw(canvas: Canvas) {
            val bm = bitmap ?: return
            val s = min(width.toFloat() / bm.width, height.toFloat() / bm.height)
            val w = bm.width * s; val h = bm.height * s
            canvas.drawBitmap(bm, null, android.graphics.RectF((width - w) / 2, (height - h) / 2, (width + w) / 2, (height + h) / 2), paint)
        }
    }
}
