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

package com.google.ai.edge.examples.ufld

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
 * Runs Ultra-Fast-Lane-Detection on a bundled dashcam frame and draws the detected lane
 * points, colored per lane — a deterministic, self-contained demo. The 178 MB model is
 * loaded from filesDir; push it there first with install_to_device.sh.
 */
class MainActivity : AppCompatActivity() {

  private val executor = Executors.newSingleThreadExecutor()
  private val laneColors = intArrayOf(
    Color.rgb(255, 60, 60), Color.rgb(60, 255, 60),
    Color.rgb(60, 120, 255), Color.rgb(255, 220, 40))

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val status = TextView(this).apply { textSize = 15f; setPadding(28, 40, 28, 20) }
    val imageView = ImageView(this).apply { adjustViewBounds = true }
    setContentView(LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL; addView(status); addView(imageView)
    })

    executor.execute {
      val modelFile = File(filesDir, "ufld.tflite")
      if (!modelFile.exists()) {
        runOnUiThread {
          status.text = "Model not found at:\n${modelFile.absolutePath}\n\n" +
            "Push it first:  ./install_to_device.sh <dir-with-ufld.tflite>\n" +
            "(build with ../conversion or download from\n litert-community/Ultra-Fast-Lane-Detection-LiteRT)"
        }
        return@execute
      }
      val input = assets.open("test_image.jpg").use { BitmapFactory.decodeStream(it) }
      LaneDetector(modelFile.absolutePath).use { det ->
        val (points, ms) = det.detect(input)
        val out = draw(input, points)
        runOnUiThread {
          status.text = "UFLD  ·  lane detection  ·  CompiledModel GPU  ·  ${ms} ms  ·  ${points.size} points"
          imageView.setImageBitmap(out)
        }
      }
    }
  }

  private fun draw(image: Bitmap, points: List<LanePoint>): Bitmap {
    val out = image.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(out)
    val r = out.width / 90f
    val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    for (pt in points) {
      paint.color = laneColors[pt.lane % laneColors.size]
      canvas.drawCircle(pt.x * out.width, pt.y * out.height, r, paint)
    }
    return out
  }

  override fun onDestroy() { super.onDestroy(); executor.shutdown() }
}
