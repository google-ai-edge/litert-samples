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

package com.google.ai.edge.examples.inpainting

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
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
 * MI-GAN object-removal ("magic eraser") demo, fully on the CompiledModel GPU. Paint over what you want to
 * remove, tap Erase, and the masked region is inpainted on device. Works on a bundled image and any image
 * picked from the gallery.
 */
class MainActivity : Activity() {

    private val tag = "MIGAN"
    private val bg = Executors.newSingleThreadExecutor()
    private var net: MiganInpainter? = null
    private val pickReq = 100

    private lateinit var status: TextView
    private lateinit var canvasView: DrawView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 80, 24, 24)
        }
        status = TextView(this).apply {
            textSize = 15f
            text = "Loading MI-GAN on GPU…"
        }
        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        val pick = Button(this).apply {
            text = "🖼 Pick"
            isEnabled = false
            setOnClickListener { startActivityForResult(Intent(Intent.ACTION_GET_CONTENT).apply { type = "image/*" }, pickReq) }
        }
        val erase = Button(this).apply {
            text = "✨ Erase"
            setOnClickListener { runErase() }
        }
        val reset = Button(this).apply {
            text = "↺ Reset"
            setOnClickListener {
                canvasView.clearStrokes()
                canvasView.restoreOriginal()
            }
        }
        listOf(pick, erase, reset).forEach { row.addView(it, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)) }
        canvasView = DrawView(this)
        root.addView(status)
        root.addView(row)
        root.addView(canvasView, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 980))
        setContentView(root)

        bg.execute {
            try {
                net = MiganInpainter(this)
                try {
                    val bm = squareResize(BitmapFactory.decodeStream(assets.open("test_image.jpg")))
                    runOnUiThread {
                        canvasView.setImage(bm)
                        status.text = "Paint over an object, then tap ✨ Erase."
                    }
                } catch (_: java.io.IOException) {
                    runOnUiThread { status.text = "Ready — pick an image, paint a region, tap Erase." }
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
        bg.execute {
            try {
                val bm = squareResize(loadOriented(uri))
                runOnUiThread {
                    canvasView.setImage(bm)
                    status.text = "Paint over an object, then tap ✨ Erase."
                }
            } catch (e: Throwable) { Log.e(tag, "load failed", e)
            runOnUiThread { status.text = "Failed: ${e.message}" } }
        }
    }

    private fun runErase() {
        val n = net ?: return
        val mask = canvasView.buildMask() ?: run {
            runOnUiThread { status.text = "Paint a region first." }
            return
        }
        val rgb = canvasView.imageRgb() ?: return
        runOnUiThread { status.text = "Erasing on GPU…" }
        bg.execute {
            try {
                val t0 = System.nanoTime()
                val out = n.inpaint(rgb, mask)
                val ms = (System.nanoTime() - t0) / 1_000_000
                Log.i(tag, "inpaint ${ms}ms")
                runOnUiThread {
                    status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
                    status.text = "On-device GPU inpaint ✓  ${ms} ms  ·  MI-GAN, CompiledModel GPU"
                    canvasView.setImage(out, isOriginal = false)
                    canvasView.clearStrokes()
                }
            } catch (e: Throwable) { Log.e(tag, "inpaint failed", e)
            runOnUiThread { status.text = "Failed: ${e.message}" } }
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

    private fun squareResize(src: Bitmap): Bitmap {
        val s = min(src.width, src.height)
        val crop = Bitmap.createBitmap(src, (src.width - s) / 2, (src.height - s) / 2, s, s)
        return Bitmap.createScaledBitmap(crop, MiganInpainter.SIZE, MiganInpainter.SIZE, true)
    }

    override fun onDestroy() {
        super.onDestroy()
        bg.shutdown()
        net?.close()
    }

    /** Shows a SIZE×SIZE image and lets the user paint a mask (the region to erase) with a finger. */
    class DrawView(ctx: Context) : View(ctx) {
        private val S = MiganInpainter.SIZE
        private var image: Bitmap? = null      // current working image (original or inpainted result)
        private var original: Bitmap? = null   // the last-loaded image (for Reset)
        private val strokes = ArrayList<Path>()
        private var cur: Path? = null
        private val brush = S * 0.06f          // brush radius in image space

        private val imgPaint = Paint().apply { isFilterBitmap = true }
        private val overlay = Paint().apply {
            color = Color.argb(140, 255, 40, 40)
            style = Paint.Style.STROKE
            strokeWidth = brush * 2
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
            isAntiAlias = true
        }

        fun setImage(bm: Bitmap, isOriginal: Boolean = true) {
            image = bm
            if (isOriginal) original = bm
            postInvalidate()
        }
        fun restoreOriginal() {
            original?.let { image = it }
            postInvalidate()
        }
        fun clearStrokes() {
            strokes.clear()
            cur = null
            postInvalidate()
        }

        // image-fit transform
        private fun scale() = min(width.toFloat() / S, height.toFloat() / S)
        private fun ox() = (width - S * scale()) / 2
        private fun oy() = (height - S * scale()) / 2

        override fun onTouchEvent(e: MotionEvent): Boolean {
            val s = scale()
            val ix = (e.x - ox()) / s
            val iy = (e.y - oy()) / s
            when (e.action) {
                MotionEvent.ACTION_DOWN -> { cur = Path().apply { moveTo(ix, iy) }.also { strokes.add(it) } }
                MotionEvent.ACTION_MOVE -> cur?.lineTo(ix, iy)
                MotionEvent.ACTION_UP -> cur = null
            }
            postInvalidate()
            return true
        }

        override fun onDraw(canvas: Canvas) {
            val bm = image ?: return
            val s = scale()
            canvas.save()
            canvas.translate(ox(), oy())
            canvas.scale(s, s)
            canvas.drawBitmap(bm, 0f, 0f, imgPaint)
            for (p in strokes) canvas.drawPath(p, overlay)
            canvas.restore()
        }

        /** Rasterize the painted strokes to a SIZE×SIZE mask: 1 = keep, 0 = erase. null if nothing painted. */
        fun buildMask(): FloatArray? {
            if (strokes.isEmpty()) return null
            val mb = Bitmap.createBitmap(S, S, Bitmap.Config.ARGB_8888)
            val c = Canvas(mb)
            c.drawColor(Color.BLACK)
            val p = Paint().apply {
                color = Color.WHITE
                style = Paint.Style.STROKE
                strokeWidth = brush * 2
                strokeCap = Paint.Cap.ROUND
                strokeJoin = Paint.Join.ROUND
                isAntiAlias = false
            }
            for (st in strokes) c.drawPath(st, p)
            val px = IntArray(S * S)
            mb.getPixels(px, 0, S, 0, 0, S, S)
            val mask = FloatArray(S * S) { if ((px[it] and 0xFF) > 127) 0f else 1f }  // white(painted)->0 erase
            mb.recycle()
            return mask
        }

        fun imageRgb(): FloatArray? {
            val bm = image ?: return null
            val px = IntArray(S * S)
            bm.getPixels(px, 0, S, 0, 0, S, S)
            val out = FloatArray(S * S * 3)
            for (i in px.indices) {
                val v = px[i]
                out[i * 3] = ((v shr 16) and 0xFF).toFloat()
                out[i * 3 + 1] = ((v shr 8) and 0xFF).toFloat()
                out[i * 3 + 2] = (v and 0xFF).toFloat()
            }
            return out
        }
    }
}
