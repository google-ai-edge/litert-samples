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

package com.google.ai.edge.examples.clothseg

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.util.concurrent.Executors

/**
 * Runs cloth segmentation on a bundled photo and overlays the clothing classes (upper /
 * lower / full body) — a deterministic, self-contained demo. The 176 MB model is loaded
 * from filesDir; push it there first with install_to_device.sh.
 */
class MainActivity : AppCompatActivity() {

  private val executor = Executors.newSingleThreadExecutor()
  // 0 bg (keep), 1 upper (cyan), 2 lower (orange), 3 full (magenta)
  private val colors = intArrayOf(0, Color.rgb(0, 200, 255), Color.rgb(255, 150, 0), Color.rgb(230, 0, 200))

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val status = TextView(this).apply { textSize = 15f; setPadding(28, 40, 28, 20) }
    val imageView = ImageView(this).apply { adjustViewBounds = true }
    setContentView(LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL; addView(status); addView(imageView)
    })

    executor.execute {
      val modelFile = File(filesDir, "clothseg.tflite")
      if (!modelFile.exists()) {
        runOnUiThread {
          status.text = "Model not found at:\n${modelFile.absolutePath}\n\n" +
            "Push it first:  ./install_to_device.sh <dir-with-clothseg.tflite>\n" +
            "(build with ../conversion or download from\n litert-community/Cloth-Segmentation-U2Net-LiteRT)"
        }
        return@execute
      }
      val input = assets.open("test_image.jpg").use { BitmapFactory.decodeStream(it) }
      ClothSegmenter(modelFile.absolutePath).use { seg ->
        val (cls, ms) = seg.segment(input)
        val out = overlay(input, cls)
        runOnUiThread {
          status.text = "Cloth Segmentation  ·  U²-Net  ·  CompiledModel GPU  ·  ${ms} ms"
          imageView.setImageBitmap(out)
        }
      }
    }
  }

  private fun overlay(image: Bitmap, cls: ByteArray): Bitmap {
    val O = ClothSegmenter.OUT
    val out = image.copy(Bitmap.Config.ARGB_8888, true)
    val px = IntArray(out.width * out.height)
    out.getPixels(px, 0, out.width, 0, 0, out.width, out.height)
    for (y in 0 until out.height) {
      val my = y * O / out.height
      for (x in 0 until out.width) {
        val c = cls[my * O + (x * O / out.width)].toInt()
        if (c != 0) {
          val p = px[y * out.width + x]; val col = colors[c]
          val r = (((p shr 16) and 0xFF) * 0.4f + ((col shr 16) and 0xFF) * 0.6f).toInt()
          val g = (((p shr 8) and 0xFF) * 0.4f + ((col shr 8) and 0xFF) * 0.6f).toInt()
          val b = ((p and 0xFF) * 0.4f + (col and 0xFF) * 0.6f).toInt()
          px[y * out.width + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
      }
    }
    return Bitmap.createBitmap(px, out.width, out.height, Bitmap.Config.ARGB_8888)
  }

  override fun onDestroy() { super.onDestroy(); executor.shutdown() }
}
