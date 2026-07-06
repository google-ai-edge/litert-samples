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

package com.google.ai.edge.examples.pidnet

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.util.concurrent.Executors

/**
 * Runs PIDNet-S semantic segmentation on a bundled Cityscapes road scene and shows
 * the input with the 19-class colored segmentation overlaid — a deterministic,
 * self-contained demo. The 30 MB model is loaded from the app's filesDir; push it
 * there first with install_to_device.sh (it is not bundled in the APK).
 */
class MainActivity : AppCompatActivity() {

  private val executor = Executors.newSingleThreadExecutor()

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val status = TextView(this).apply { textSize = 15f; setPadding(28, 40, 28, 20) }
    val imageView = ImageView(this).apply { adjustViewBounds = true }
    setContentView(LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      addView(status); addView(imageView)
    })

    executor.execute {
      val modelFile = File(filesDir, "pidnet_s.tflite")
      if (!modelFile.exists()) {
        runOnUiThread {
          status.text = "Model not found at:\n${modelFile.absolutePath}\n\n" +
            "Push it first:  ./install_to_device.sh <dir-with-pidnet_s.tflite>\n" +
            "(build with ../conversion or download from\n litert-community/PIDNet-S-Cityscapes-LiteRT)"
        }
        return@execute
      }
      val input = assets.open("road.jpg").use { BitmapFactory.decodeStream(it) }
      PIDNet(modelFile.absolutePath).use { seg ->
        val (labels, ms) = seg.segment(input)
        val overlay = blend(input, labels)
        runOnUiThread {
          status.text = "PIDNet-S  ·  Cityscapes 19-class  ·  CompiledModel GPU  ·  ${ms} ms"
          imageView.setImageBitmap(overlay)
        }
      }
    }
  }

  /** Scale the label map to the input size and alpha-blend it over the image. */
  private fun blend(image: Bitmap, labels: Bitmap): Bitmap {
    val out = image.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(out)
    val paint = Paint(Paint.FILTER_BITMAP_FLAG).apply { alpha = 135 }
    canvas.drawBitmap(labels, Rect(0, 0, labels.width, labels.height),
      Rect(0, 0, out.width, out.height), paint)
    return out
  }

  override fun onDestroy() {
    super.onDestroy()
    executor.shutdown()
  }
}
