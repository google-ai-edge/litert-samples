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

package com.google.ai.edge.examples.crowdcount

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.os.Bundle
import android.view.Gravity
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.roundToInt

/**
 * Runs DM-Count on a bundled crowd photo and shows the density heatmap plus the estimated
 * person count — a deterministic, self-contained demo. The 86 MB model is loaded from
 * filesDir; push it there first with install_to_device.sh.
 */
class MainActivity : AppCompatActivity() {

  private val executor = Executors.newSingleThreadExecutor()

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val status = TextView(this).apply {
      textSize = 15f
      setPadding(28, 40, 28, 16)
    }
    val inputView = ImageView(this).apply { adjustViewBounds = true }
    val heatView = ImageView(this).apply { adjustViewBounds = true }
    setContentView(
      LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        addView(status)
        addView(
          TextView(this@MainActivity).apply {
            text = "Input"
            setPadding(28, 8, 28, 4)
          })
        addView(
          inputView,
          LinearLayout.LayoutParams(420, 420).apply { gravity = Gravity.CENTER_HORIZONTAL })
        addView(
          TextView(this@MainActivity).apply {
            text = "Density heatmap (GPU)"
            setPadding(28, 16, 28, 4)
          })
        addView(
          heatView,
          LinearLayout.LayoutParams(420, 420).apply { gravity = Gravity.CENTER_HORIZONTAL })
      })

    executor.execute {
      val modelFile = File(filesDir, "dmcount.tflite")
      if (!modelFile.exists()) {
        runOnUiThread {
          status.text = "Model not found at:\n${modelFile.absolutePath}\n\n" +
            "Push it first:  ./install_to_device.sh <dir-with-dmcount.tflite>\n" +
            "(build with ../conversion or download from\n" +
            " litert-community/DM-Count-Crowd-LiteRT)"
        }
        return@execute
      }
      val input = assets.open("test_image.jpg").use { BitmapFactory.decodeStream(it) }
      CrowdCounter(modelFile.absolutePath).use { counter ->
        val result = counter.count(input)
        val overlay = renderOverlay(input, result.density)
        val people = result.count.roundToInt()
        runOnUiThread {
          status.text =
            "DM-Count  ·  ~$people people  ·  CompiledModel GPU  ·  ${result.inferenceMs} ms"
          inputView.setImageBitmap(input)
          heatView.setImageBitmap(overlay)
        }
      }
    }
  }

  /** Composites the per-frame-normalized density map over [input] as a red heatmap. */
  private fun renderOverlay(input: Bitmap, density: FloatArray): Bitmap {
    val side = CrowdCounter.OUT
    var maxV = 1e-5f
    for (v in density) {
      if (v > maxV) {
        maxV = v
      }
    }
    val heatPixels = IntArray(side * side)
    for (i in heatPixels.indices) {
      val v = (density[i] / maxV).coerceIn(0f, 1f)
      val alpha = (v * 220).toInt()
      val green = ((1f - v) * 160).toInt() // faint orange -> strong red
      heatPixels[i] = (alpha shl 24) or (0xFF shl 16) or (green shl 8)
    }
    val heat = Bitmap.createBitmap(heatPixels, side, side, Bitmap.Config.ARGB_8888)
    val out = input.copy(Bitmap.Config.ARGB_8888, true)
    Canvas(out).drawBitmap(
      heat,
      null,
      Rect(0, 0, out.width, out.height),
      Paint(Paint.FILTER_BITMAP_FLAG),
    )
    return out
  }

  override fun onDestroy() {
    super.onDestroy()
    executor.shutdown()
  }
}
