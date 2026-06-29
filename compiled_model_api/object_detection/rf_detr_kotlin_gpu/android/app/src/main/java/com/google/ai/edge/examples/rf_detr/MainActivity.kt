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

package com.google.ai.edge.examples.rf_detr

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

/**
 * RF-DETR Nano real-time demo: live camera object detection with both transformer graphs on CompiledModel
 * GPU (only topk/gather + decode + NMS on CPU). Each camera frame is squashed to 384×384, detected, and
 * drawn with its boxes + COCO labels. ~9 fps on a Pixel 8a (a transformer detector, fully on the GPU).
 */
class MainActivity : ComponentActivity() {

  private val tag = "RFDETR"
  private val analysisExec = Executors.newSingleThreadExecutor()
  private var model: RfDetr? = null
  private var labels: List<String> = emptyList()

  private lateinit var status: TextView
  private lateinit var overlay: OverlayView
  private var emaMs = 0f

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val root = FrameLayout(this)
    overlay = OverlayView(this)
    status = TextView(this).apply {
      textSize = 14f; setTextColor(Color.WHITE); setBackgroundColor(0xAA000000.toInt())
      setPadding(24, 70, 24, 16); text = "Loading RF-DETR on GPU…"
    }
    root.addView(overlay, FrameLayout.LayoutParams(-1, -1))
    root.addView(status, FrameLayout.LayoutParams(-1, -2, Gravity.TOP))
    setContentView(root)

    labels = assets.open("coco_labels.txt").bufferedReader().readLines()
    if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
      initModelAndCamera()
    else
      requestPermissions(arrayOf(Manifest.permission.CAMERA), 1)
  }

  override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) initModelAndCamera()
    else runOnUiThread { status.text = "Camera permission denied" }
  }

  private fun initModelAndCamera() {
    analysisExec.execute {
      try {
        model = RfDetr(this)                       // load + compile GPU off the main thread
        runOnUiThread { startCamera() }
      } catch (e: Throwable) {
        Log.e(tag, "model load failed", e)
        runOnUiThread { status.text = "FAIL: ${e.message}" }
      }
    }
  }

  private fun startCamera() {
    val future = ProcessCameraProvider.getInstance(this)
    future.addListener({
      val provider = future.get()
      val analysis = ImageAnalysis.Builder()
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
        .setTargetResolution(Size(RfDetr.SIZE, RfDetr.SIZE))
        .build()
      analysis.setAnalyzer(analysisExec) { proxy -> processFrame(proxy) }
      provider.unbindAll()
      provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
    }, ContextCompat.getMainExecutor(this))
  }

  private fun processFrame(proxy: ImageProxy) {
    val m = model ?: run { proxy.close(); return }
    try {
      val frame = proxyToBitmap(proxy)
      val square = Bitmap.createScaledBitmap(frame, RfDetr.SIZE, RfDetr.SIZE, true)
      val rgb = bitmapToRgb(square)
      val t0 = System.nanoTime()
      val dets = m.detect(rgb)
      val ms = (System.nanoTime() - t0) / 1e6f
      emaMs = if (emaMs == 0f) ms else emaMs * 0.8f + ms * 0.2f
      val fps = 1000f / emaMs
      runOnUiThread {
        overlay.bitmap = frame; overlay.dets = dets; overlay.labels = labels; overlay.invalidate()
        status.text = "RF-DETR Nano on CompiledModel GPU ✓  ${dets.size} objects · " +
          "${"%.1f".format(fps)} fps (${ms.toInt()} ms)"
      }
    } catch (e: Throwable) {
      Log.e(tag, "frame failed", e)
    } finally {
      proxy.close()
    }
  }

  /** ImageProxy (RGBA_8888) -> upright Bitmap (handles row padding + sensor rotation). */
  private fun proxyToBitmap(proxy: ImageProxy): Bitmap {
    val plane = proxy.planes[0]
    val buf = plane.buffer.apply { rewind() }
    val pixelStride = plane.pixelStride
    val rowPadPx = (plane.rowStride - pixelStride * proxy.width) / pixelStride
    val bmp = Bitmap.createBitmap(proxy.width + rowPadPx, proxy.height, Bitmap.Config.ARGB_8888)
    bmp.copyPixelsFromBuffer(buf)
    val cropped = if (rowPadPx > 0) Bitmap.createBitmap(bmp, 0, 0, proxy.width, proxy.height) else bmp
    val rot = proxy.imageInfo.rotationDegrees
    return if (rot == 0) cropped
    else Bitmap.createBitmap(cropped, 0, 0, cropped.width, cropped.height,
      Matrix().apply { postRotate(rot.toFloat()) }, true)
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

  override fun onDestroy() { super.onDestroy(); analysisExec.shutdown(); model?.close() }

  /** Draws the latest frame (FIT_CENTER) and the detection boxes (coords normalized [0,1]). */
  class OverlayView(ctx: Context) : View(ctx) {
    var bitmap: Bitmap? = null
    var dets: List<RfDetr.Detection> = emptyList()
    var labels: List<String> = emptyList()
    private val box = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 5f }
    private val txt = Paint().apply { color = Color.WHITE; textSize = 34f; isFakeBoldText = true }
    private val bgp = Paint().apply { style = Paint.Style.FILL }
    private val palette = intArrayOf(
      0xFF00C853.toInt(), 0xFFFF6D00.toInt(), 0xFF2962FF.toInt(), 0xFFD50000.toInt(),
      0xFFAA00FF.toInt(), 0xFF00B8D4.toInt(), 0xFFFFD600.toInt(), 0xFFC51162.toInt())

    override fun onDraw(canvas: Canvas) {
      val bm = bitmap ?: return
      val s = minOf(width.toFloat() / bm.width, height.toFloat() / bm.height)
      val dw = bm.width * s; val dh = bm.height * s
      val ox = (width - dw) / 2f; val oy = (height - dh) / 2f
      canvas.drawBitmap(bm, null, RectF(ox, oy, ox + dw, oy + dh), null)
      for (d in dets) {
        val color = palette[d.cls % palette.size]
        box.color = color; bgp.color = color
        val x0 = ox + (d.cx - d.w / 2) * dw; val y0 = oy + (d.cy - d.h / 2) * dh
        val x1 = ox + (d.cx + d.w / 2) * dw; val y1 = oy + (d.cy + d.h / 2) * dh
        canvas.drawRect(x0, y0, x1, y1, box)
        val name = labels.getOrNull(d.cls)?.ifBlank { "id ${d.cls}" } ?: "id ${d.cls}"
        val label = "$name ${(d.score * 100).toInt()}%"
        val tw = txt.measureText(label)
        canvas.drawRect(x0, y0 - 40f, x0 + tw + 14f, y0, bgp)
        canvas.drawText(label, x0 + 7f, y0 - 9f, txt)
      }
    }
  }
}
