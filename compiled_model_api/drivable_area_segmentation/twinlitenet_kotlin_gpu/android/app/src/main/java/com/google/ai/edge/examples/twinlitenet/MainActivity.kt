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

package com.google.ai.edge.examples.twinlitenet

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Bundle
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.util.concurrent.Executors

/**
 * Runs TwinLiteNet on a bundled dashcam frame and overlays the drivable area (green) and
 * lane lines (red) — a deterministic, self-contained demo. The 3.1 MB model is loaded from
 * filesDir; push it there first with install_to_device.sh.
 */
class MainActivity : AppCompatActivity() {

  private val executor = Executors.newSingleThreadExecutor()

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val status = TextView(this).apply {
        textSize = 15f
        setPadding(28, 40, 28, 20)
    }
    val imageView = ImageView(this).apply { adjustViewBounds = true }
    setContentView(LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      addView(status)
      addView(imageView)
    })

    executor.execute {
      val modelFile = File(filesDir, "twinlite.tflite")
      if (!modelFile.exists()) {
        runOnUiThread {
          status.text = "Model not found at:\n${modelFile.absolutePath}\n\n" +
            "Push it first:  ./install_to_device.sh <dir-with-twinlite.tflite>\n" +
            "(build with ../conversion or download from\n litert-community/TwinLiteNet-LiteRT)"
        }
        return@execute
      }
      val input = assets.open("test_image.jpg").use { BitmapFactory.decodeStream(it) }
      TwinLiteSegmenter(modelFile.absolutePath).use { seg ->
        val (da, ll, ms) = seg.segment(input)
        val out = render(input, da, ll)
        runOnUiThread {
          status.text = "TwinLiteNet  ·  drivable area + lanes  ·  CompiledModel GPU  ·  ${ms} ms"
          imageView.setImageBitmap(out)
        }
      }
    }
  }

  /** Green = drivable area, red = lane lines, scaled from the 640×360 masks. */
  private fun render(image: Bitmap, da: ByteArray, ll: ByteArray): Bitmap {
    val W = TwinLiteSegmenter.W
    val H = TwinLiteSegmenter.H
    val out = image.copy(Bitmap.Config.ARGB_8888, true)
    val px = IntArray(out.width * out.height)
    out.getPixels(px, 0, out.width, 0, 0, out.width, out.height)
    for (y in 0 until out.height) {
      val my = y * H / out.height
      for (x in 0 until out.width) {
        val mi = my * W + (x * W / out.width)
        val p = px[y * out.width + x]
        px[y * out.width + x] = when {
          ll[mi].toInt() == 1 -> Color.rgb(255, 48, 48)
          da[mi].toInt() == 1 -> {
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            Color.rgb(
              (r * 0.45f + 40 * 0.55f).toInt(),
              (g * 0.45f + 220 * 0.55f).toInt(),
              (b * 0.45f + 90 * 0.55f).toInt())
          }
          else -> p
        }
      }
    }
    return Bitmap.createBitmap(px, out.width, out.height, Bitmap.Config.ARGB_8888)
  }

  override fun onDestroy() {
      super.onDestroy()
      executor.shutdown()
  }
}
