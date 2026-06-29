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

package com.google.ai.edge.examples.d_fine

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
import android.widget.ScrollView
import android.widget.TextView
import java.util.concurrent.Executors

/**
 * D-FINE-S demo: object detection with both transformer graphs on CompiledModel GPU; only topk +
 * the per-token tail (enc_output / bbox_head) + decode + NMS run on the CPU. Runs on a bundled image at
 * launch and on any gallery image.
 */
class MainActivity : Activity() {

  private val tag = "DFINE"
  private val bg = Executors.newSingleThreadExecutor()
  private var model: DFine? = null
  private var labels: List<String> = emptyList()
  private val pickReq = 100

  private lateinit var status: TextView
  private lateinit var overlay: OverlayView
  private lateinit var results: TextView

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 80, 24, 24) }
    status = TextView(this).apply { textSize = 15f; text = "Loading D-FINE on GPU…" }
    val pick = Button(this).apply {
      text = "🖼  Pick image"; isEnabled = false
      setOnClickListener {
        startActivityForResult(Intent(Intent.ACTION_GET_CONTENT).apply { type = "image/*" }, pickReq)
      }
    }
    overlay = OverlayView(this)
    results = TextView(this).apply { textSize = 14f; setPadding(0, 24, 0, 0) }
    root.addView(status); root.addView(pick)
    root.addView(overlay, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 900))
    root.addView(ScrollView(this).apply { addView(results) })
    setContentView(root)

    bg.execute {
      try {
        labels = assets.open("coco_labels.txt").bufferedReader().readLines()
        model = DFine(this)
        val bundled = squareResize(BitmapFactory.decodeStream(assets.open("test_image.jpg")))
        runDetect(bundled, warm = true)
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
    runOnUiThread { status.text = "Detecting…" }
    bg.execute {
      try {
        runDetect(squareResize(loadOriented(uri)), warm = false)
      } catch (e: Throwable) {
        Log.e(tag, "detect failed", e)
        runOnUiThread { status.text = "Failed: ${e.message}" }
      }
    }
  }

  private fun runDetect(img: Bitmap, warm: Boolean) {
    val m = model!!
    val rgb = bitmapToRgb(img)
    if (warm) m.detect(rgb)                          // warm up GPU shaders once
    val t0 = System.nanoTime()
    val dets = m.detect(rgb)
    val ms = (System.nanoTime() - t0) / 1_000_000
    Log.i(tag, "detections=${dets.size} ${ms}ms")
    runOnUiThread {
      status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
      status.text = "On-device GPU detection ✓  ${dets.size} objects · ${ms} ms\n" +
        "D-FINE-S — both transformer graphs on CompiledModel GPU"
      overlay.bitmap = img; overlay.dets = dets; overlay.labels = labels; overlay.invalidate()
      results.text = if (dets.isEmpty()) "(no objects found)" else dets.joinToString("\n") {
        "• ${labelOf(it.cls)}  ${(it.score * 100).toInt()}%"
      }
    }
  }

  private fun labelOf(cls: Int) = labels.getOrNull(cls)?.ifBlank { "id $cls" } ?: "id $cls"

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

  /** Square resize to SIZE×SIZE (squash, matches DFineImageProcessor's resize to {640,640}). */
  private fun squareResize(src: Bitmap): Bitmap =
    Bitmap.createScaledBitmap(src, DFine.SIZE, DFine.SIZE, true)

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

  override fun onDestroy() { super.onDestroy(); bg.shutdown(); model?.close() }

  /** Draws the squashed image and the detection boxes (coords are normalized [0,1] in SIZE space). */
  class OverlayView(ctx: Context) : View(ctx) {
    var bitmap: Bitmap? = null
    var dets: List<DFine.Detection> = emptyList()
    var labels: List<String> = emptyList()
    private val box = Paint().apply { color = Color.rgb(0x00, 0xC8, 0x53); style = Paint.Style.STROKE; strokeWidth = 4f }
    private val txt = Paint().apply { color = Color.WHITE; textSize = 30f; isFakeBoldText = true }
    private val bgp = Paint().apply { color = Color.rgb(0x00, 0xC8, 0x53); style = Paint.Style.FILL }
    private val palette = intArrayOf(
      0xFF00C853.toInt(), 0xFFFF6D00.toInt(), 0xFF2962FF.toInt(), 0xFFD50000.toInt(),
      0xFFAA00FF.toInt(), 0xFF00B8D4.toInt(), 0xFFFFD600.toInt(), 0xFFC51162.toInt())

    override fun onDraw(canvas: Canvas) {
      val bm = bitmap ?: return
      val s = minOf(width.toFloat() / bm.width, height.toFloat() / bm.height)
      canvas.drawBitmap(bm, null, android.graphics.RectF(0f, 0f, bm.width * s, bm.height * s), null)
      for (d in dets) {
        val color = palette[d.cls % palette.size]
        box.color = color; bgp.color = color
        val x0 = (d.cx - d.w / 2) * DFine.SIZE * s; val y0 = (d.cy - d.h / 2) * DFine.SIZE * s
        val x1 = (d.cx + d.w / 2) * DFine.SIZE * s; val y1 = (d.cy + d.h / 2) * DFine.SIZE * s
        canvas.drawRect(x0, y0, x1, y1, box)
        val name = labels.getOrNull(d.cls)?.ifBlank { "id ${d.cls}" } ?: "id ${d.cls}"
        val label = "$name ${(d.score * 100).toInt()}%"
        val tw = txt.measureText(label)
        canvas.drawRect(x0, y0 - 36f, x0 + tw + 12f, y0, bgp)
        canvas.drawText(label, x0 + 6f, y0 - 8f, txt)
      }
    }
  }
}
