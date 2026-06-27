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

package com.google.ai.edge.examples.semantic_segmentation

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
 * LR-ASPP semantic segmentation demo (COCO-VOC 21 classes), fully on the CompiledModel GPU.
 * Segments a bundled image at launch and any image picked from the gallery; shows input | colored overlay.
 */
class MainActivity : Activity() {

  private val tag = "LRASPP"
  private val bg = Executors.newSingleThreadExecutor()
  private var net: LrasppSegmenter? = null
  private val pickReq = 100

  private lateinit var status: TextView
  private lateinit var inputView: ImgView
  private lateinit var segView: ImgView

  private val voc = arrayOf("background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor")
  private val palette = IntArray(256).also { p ->
    for (i in 0 until 256) {
      var r = 0; var g = 0; var b = 0; var c = i
      for (j in 0 until 8) {
        r = r or (((c shr 0) and 1) shl (7 - j)); g = g or (((c shr 1) and 1) shl (7 - j))
        b = b or (((c shr 2) and 1) shl (7 - j)); c = c shr 3
      }
      p[i] = Color.rgb(r, g, b)
    }
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 80, 24, 24) }
    status = TextView(this).apply { textSize = 15f; text = "Loading LR-ASPP on GPU…" }
    val pick = Button(this).apply {
      text = "🖼  Pick image"; isEnabled = false
      setOnClickListener {
        startActivityForResult(Intent(Intent.ACTION_GET_CONTENT).apply { type = "image/*" }, pickReq)
      }
    }
    inputView = ImgView(this); segView = ImgView(this)
    val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
    row.addView(inputView, LinearLayout.LayoutParams(0, 760, 1f))
    row.addView(segView, LinearLayout.LayoutParams(0, 760, 1f))
    root.addView(status); root.addView(pick); root.addView(row)
    setContentView(root)

    bg.execute {
      try {
        net = LrasppSegmenter(this)
        val bundled = squareResize(BitmapFactory.decodeStream(assets.open("test_image.jpg")))
        run(bundled, warm = true)
        runOnUiThread { pick.isEnabled = true }
      } catch (e: Throwable) {
        Log.e(tag, "load failed", e)
        runOnUiThread {
          status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2)); status.text = "FAIL: ${e.message}"
        }
      }
    }
  }

  override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
    super.onActivityResult(requestCode, resultCode, data)
    if (requestCode != pickReq || resultCode != RESULT_OK) return
    val uri = data?.data ?: return
    runOnUiThread { status.text = "Segmenting…" }
    bg.execute {
      try { run(squareResize(loadOriented(uri)), warm = false) }
      catch (e: Throwable) { Log.e(tag, "segment failed", e); runOnUiThread { status.text = "Failed: ${e.message}" } }
    }
  }

  private fun run(img: Bitmap, warm: Boolean) {
    val n = net!!
    val rgb = bitmapToRgb(img)
    if (warm) n.segment(rgb)
    val t0 = System.nanoTime()
    val cls = n.segment(rgb)
    val ms = (System.nanoTime() - t0) / 1_000_000
    val seg = overlay(img, cls, LrasppSegmenter.SIZE)
    val names = cls.toHashSet().filter { it != 0 }.map { voc[it] }.distinct().sorted()
    Log.i(tag, "segment ${ms}ms classes=$names")
    runOnUiThread {
      status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
      status.text = "On-device GPU segmentation ✓  ${ms} ms\n" +
        (if (names.isEmpty()) "(background only)" else names.joinToString(", ")) +
        "  ·  LR-ASPP MobileNetV3, all on CompiledModel GPU"
      inputView.bitmap = img; inputView.invalidate()
      segView.bitmap = seg; segView.invalidate()
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
    return Bitmap.createScaledBitmap(crop, LrasppSegmenter.SIZE, LrasppSegmenter.SIZE, true)
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

  /** Blend the class colormap over the input (background kept as-is). */
  private fun overlay(img: Bitmap, cls: IntArray, size: Int): Bitmap {
    val src = IntArray(size * size); img.getPixels(src, 0, size, 0, 0, size, size)
    val out = IntArray(size * size)
    for (i in 0 until size * size) {
      val c = cls[i]
      if (c == 0) { out[i] = src[i]; continue }
      val col = palette[c]
      val a = 0.55f
      val r = ((1 - a) * Color.red(src[i]) + a * Color.red(col)).toInt()
      val g = ((1 - a) * Color.green(src[i]) + a * Color.green(col)).toInt()
      val b = ((1 - a) * Color.blue(src[i]) + a * Color.blue(col)).toInt()
      out[i] = Color.rgb(r, g, b)
    }
    return Bitmap.createBitmap(out, size, size, Bitmap.Config.ARGB_8888)
  }

  override fun onDestroy() { super.onDestroy(); bg.shutdown(); net?.close() }

  class ImgView(ctx: Context) : View(ctx) {
    var bitmap: Bitmap? = null
    private val paint = Paint().apply { isFilterBitmap = true }
    override fun onDraw(canvas: Canvas) {
      val bm = bitmap ?: return
      val s = min(width.toFloat() / bm.width, height.toFloat() / bm.height)
      val w = bm.width * s; val h = bm.height * s
      canvas.drawBitmap(
        bm, null,
        android.graphics.RectF((width - w) / 2, (height - h) / 2, (width + w) / 2, (height + h) / 2), paint,
      )
    }
  }
}
