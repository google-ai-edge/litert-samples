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

package com.google.ai.edge.examples.metric_depth

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
import kotlin.math.max
import kotlin.math.min

/**
 * Metric3D v2 demo: monocular METRIC depth (absolute meters) fully on the CompiledModel GPU.
 * Runs on a bundled image at launch and on any image picked from the gallery; renders a depth
 * colormap and the near/far metric range.
 */
class MainActivity : Activity() {

  private val tag = "METRIC3D"
  private val bg = Executors.newSingleThreadExecutor()
  private var net: MetricDepth? = null
  private val pickReq = 100

  private lateinit var status: TextView
  private lateinit var inputView: DepthView
  private lateinit var depthView: DepthView

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val root = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(24, 80, 24, 24)
    }
    status = TextView(this).apply {
        textSize = 15f
        text = "Loading Metric3D v2 on GPU…"
    }
    val pick = Button(this).apply {
      text = "🖼  Pick image"
      isEnabled = false
      setOnClickListener {
        startActivityForResult(
          Intent(Intent.ACTION_GET_CONTENT).apply { type = "image/*" }, pickReq)
      }
    }
    inputView = DepthView(this)
    depthView = DepthView(this)
    val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
    row.addView(inputView, LinearLayout.LayoutParams(0, 760, 1f))
    row.addView(depthView, LinearLayout.LayoutParams(0, 760, 1f))
    root.addView(status)
    root.addView(pick)
    root.addView(row)
    setContentView(root)

    bg.execute {
      try {
        net = MetricDepth(this)
        val bundled = squareResize(BitmapFactory.decodeStream(assets.open("test_image.jpg")))
        run(bundled, warm = true)
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
    runOnUiThread { status.text = "Estimating depth…" }
    bg.execute {
      try {
        run(squareResize(loadOriented(uri)), warm = false)
      } catch (e: Throwable) {
        Log.e(tag, "depth failed", e)
        runOnUiThread { status.text = "Failed: ${e.message}" }
      }
    }
  }

  /** Estimate depth on a SIZE×SIZE bitmap and update the UI. */
  private fun run(img: Bitmap, warm: Boolean) {
    val n = net!!
    val rgb = bitmapToRgb(img)
    if (warm) {
      n.depth(rgb)                      // warm up GPU shaders once
    }
    val t0 = System.nanoTime()
    val depth = n.depth(rgb)
    val ms = (System.nanoTime() - t0) / 1_000_000
    // robust min/max over the 2nd..98th percentile for the colormap + labels
    val sorted = depth.clone().also { it.sort() }
    val lo = sorted[(sorted.size * 0.02f).toInt()]
    val hi = sorted[(sorted.size * 0.98f).toInt()]
    val dmap = colorize(depth, MetricDepth.SIZE, lo, hi)
    Log.i(tag, "depth ${ms}ms near=${"%.2f".format(lo)} far=${"%.2f".format(hi)}")
    runOnUiThread {
      status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
      status.text = "On-device GPU metric depth ✓  ${ms} ms\n" +
        "near ${"%.1f".format(lo)} m  ·  far ${"%.1f".format(hi)} m   " +
        "(DINOv2 ViT-S + RAFT, all on CompiledModel GPU)"
      inputView.bitmap = img
      inputView.invalidate()
      depthView.bitmap = dmap
      depthView.invalidate()
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

  /**
   * Center-crop to square, then resize to SIZE×SIZE (preserves local geometry;
   * no letterbox padding).
   */
  private fun squareResize(src: Bitmap): Bitmap {
    val s = min(src.width, src.height)
    val crop = Bitmap.createBitmap(src, (src.width - s) / 2, (src.height - s) / 2, s, s)
    return Bitmap.createScaledBitmap(crop, MetricDepth.SIZE, MetricDepth.SIZE, true)
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

  /** Map depth (meters) -> Turbo colormap bitmap. Near = warm, far = cool. */
  private fun colorize(depth: FloatArray, size: Int, lo: Float, hi: Float): Bitmap {
    val px = IntArray(size * size)
    val span = max(hi - lo, 1e-3f)
    for (i in depth.indices) {
      val t = ((depth[i] - lo) / span).coerceIn(0f, 1f)
      px[i] = turbo(1f - t)                 // invert so near objects are warm/bright
    }
    return Bitmap.createBitmap(px, size, size, Bitmap.Config.ARGB_8888)
  }

  /** Google "Turbo" colormap approximation, t in [0,1]. */
  private fun turbo(t: Float): Int {
    val r = (34.61 + t * (1172.33 + t * (-10793.56 + t * (33300.12 +
      t * (-38394.49 + t * 14825.05))))).toInt()
    val g = (23.31 + t * (557.33 + t * (1225.33 + t * (-3574.96 +
      t * (4520.0 + t * -1894.0))))).toInt()
    val b = (27.2 + t * (3211.1 + t * (-15327.97 + t * (27814.0 +
      t * (-22569.18 + t * 6838.66))))).toInt()
    return Color.rgb(r.coerceIn(0, 255), g.coerceIn(0, 255), b.coerceIn(0, 255))
  }

  override fun onDestroy() {
      super.onDestroy()
      bg.shutdown()
      net?.close()
  }

  class DepthView(ctx: Context) : View(ctx) {
    var bitmap: Bitmap? = null
    private val paint = Paint().apply { isFilterBitmap = true }
    override fun onDraw(canvas: Canvas) {
      val bm = bitmap ?: return
      val s = min(width.toFloat() / bm.width, height.toFloat() / bm.height)
      val w = bm.width * s
      val h = bm.height * s
      canvas.drawBitmap(
        bm, null,
        android.graphics.RectF(
          (width - w) / 2, (height - h) / 2, (width + w) / 2, (height + h) / 2),
        paint,
      )
    }
  }
}
