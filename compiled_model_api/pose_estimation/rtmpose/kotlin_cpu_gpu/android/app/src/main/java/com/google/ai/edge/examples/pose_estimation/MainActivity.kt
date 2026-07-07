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


package com.google.ai.edge.examples.pose_estimation
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

/**
 * RTMPose-s 2D human pose demo, fully on the CompiledModel GPU. Top-down: center-crops the image to a
 * 192x256 (3:4) person box, estimates 17 COCO keypoints, and draws the skeleton. Works on a bundled image
 * at launch and any image picked from the gallery.
 */
class MainActivity : Activity() {

    private val tag = "RTMPOSE"
    private val bg = Executors.newSingleThreadExecutor()
    private var net: RtmPoseEstimator? = null
    private val pickReq = 100

    private lateinit var status: TextView
    private lateinit var poseView: PoseView

    // COCO 17-keypoint skeleton edges.
    private val skeleton = arrayOf(
        5 to 7, 7 to 9, 6 to 8, 8 to 10, 11 to 13, 13 to 15, 12 to 14, 14 to 16,
        5 to 6, 11 to 12, 5 to 11, 6 to 12, 0 to 5, 0 to 6, 0 to 1, 0 to 2, 1 to 3, 2 to 4,
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 80, 24, 24)
        }
        status = TextView(this).apply {
            textSize = 15f
            text = "Loading RTMPose on GPU…"
        }
        val pick = Button(this).apply {
            text = "🖼  Pick image"
            isEnabled = false
            setOnClickListener {
                startActivityForResult(Intent(Intent.ACTION_GET_CONTENT).apply { type = "image/*" }, pickReq)
            }
        }
        poseView = PoseView(this)
        root.addView(status)
        root.addView(pick)
        root.addView(poseView, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 960))
        setContentView(root)

        bg.execute {
            try {
                net = RtmPoseEstimator(this)
                // Optional bundled demo image; if absent just wait for a picked image.
                try {
                    val bundled = cropPerson(BitmapFactory.decodeStream(assets.open("test_image.jpg")))
                    run(bundled, warm = true)
                } catch (_: java.io.IOException) {
                    runOnUiThread { status.text = "Ready — pick an image to estimate pose." }
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
        runOnUiThread { status.text = "Estimating pose…" }
        bg.execute {
            try { run(cropPerson(loadOriented(uri)), warm = false) }
            catch (e: Throwable) {
                Log.e(tag, "estimate failed", e)
                runOnUiThread { status.text = "Failed: ${e.message}" }
            }
        }
    }

    private fun run(crop: Bitmap, warm: Boolean) {
        val n = net!!
        val rgb = bitmapToRgb(crop)
        if (warm) n.estimate(rgb)
        val t0 = System.nanoTime()
        val kpts = n.estimate(rgb)
        val ms = (System.nanoTime() - t0) / 1_000_000
        val visible = kpts.count { it.score > 0.3f }
        Log.i(tag, "estimate ${ms}ms visible=$visible/17")
        runOnUiThread {
            status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
            status.text = "On-device GPU pose ✓  ${ms} ms  ·  $visible/17 keypoints  ·  RTMPose-s, CompiledModel GPU"
            poseView.set(crop, kpts, skeleton)
            poseView.invalidate()
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

    /** Center-crop to the model's 3:4 (192x256) aspect, then resize. */
    private fun cropPerson(src: Bitmap): Bitmap {
        val ar = RtmPoseEstimator.W.toFloat() / RtmPoseEstimator.H
        val w = src.width
        val h = src.height
        val crop = if (w.toFloat() / h > ar) {
            val nw = (h * ar).toInt()
            Bitmap.createBitmap(src, (w - nw) / 2, 0, nw, h)
        } else {
            val nh = (w / ar).toInt()
            Bitmap.createBitmap(src, 0, (h - nh) / 2, w, nh)
        }
        return Bitmap.createScaledBitmap(crop, RtmPoseEstimator.W, RtmPoseEstimator.H, true)
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

    class PoseView(ctx: Context) : View(ctx) {
        private var bm: Bitmap? = null
        private var kpts: List<RtmPoseEstimator.Keypoint> = emptyList()
        private var edges: Array<Pair<Int, Int>> = emptyArray()
        private val bone = Paint().apply {
            color = Color.rgb(0, 230, 0)
            strokeWidth = 6f
            isAntiAlias = true
        }
        private val joint = Paint().apply {
            color = Color.rgb(255, 40, 40)
            isAntiAlias = true
        }
        private val imgPaint = Paint().apply { isFilterBitmap = true }

        fun set(b: Bitmap, k: List<RtmPoseEstimator.Keypoint>, e: Array<Pair<Int, Int>>) {
            bm = b
            kpts = k
            edges = e
        }

        override fun onDraw(canvas: Canvas) {
            val b = bm ?: return
            val s = minOf(width.toFloat() / b.width, height.toFloat() / b.height)
            val w = b.width * s
            val h = b.height * s
            val ox = (width - w) / 2
            val oy = (height - h) / 2
            canvas.drawBitmap(b, null, android.graphics.RectF(ox, oy, ox + w, oy + h), imgPaint)
            fun px(i: Int) = ox + kpts[i].x * s
            fun py(i: Int) = oy + kpts[i].y * s
            for ((a, c) in edges) {
                if (kpts[a].score > 0.3f && kpts[c].score > 0.3f) canvas.drawLine(px(a), py(a), px(c), py(c), bone)
            }
            for (i in kpts.indices) if (kpts[i].score > 0.3f) canvas.drawCircle(px(i), py(i), 7f, joint)
        }
    }
}
