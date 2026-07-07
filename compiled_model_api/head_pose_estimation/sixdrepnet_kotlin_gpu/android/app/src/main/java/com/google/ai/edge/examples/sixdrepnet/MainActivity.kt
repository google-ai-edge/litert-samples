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

package com.google.ai.edge.examples.sixdrepnet

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
import kotlin.math.cos
import kotlin.math.sin

/**
 * Runs 6DRepNet on a bundled face photo and draws the 3D head-pose axes — a deterministic,
 * self-contained demo. The 157 MB model is loaded from filesDir; push it there first with
 * install_to_device.sh.
 */
class MainActivity : AppCompatActivity() {

  private val executor = Executors.newSingleThreadExecutor()

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val status = TextView(this).apply { textSize = 15f; setPadding(28, 40, 28, 20) }
    val imageView = ImageView(this).apply { adjustViewBounds = true }
    setContentView(LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL; addView(status); addView(imageView)
    })

    executor.execute {
      val modelFile = File(filesDir, "6drepnet.tflite")
      if (!modelFile.exists()) {
        runOnUiThread {
          status.text = "Model not found at:\n${modelFile.absolutePath}\n\n" +
            "Push it first:  ./install_to_device.sh <dir-with-6drepnet.tflite>\n" +
            "(build with ../conversion or download from\n litert-community/6DRepNet-HeadPose-LiteRT)"
        }
        return@execute
      }
      val input = assets.open("test_image.jpg").use { BitmapFactory.decodeStream(it) }
      // centered square crop (assume the face is centered)
      val s = minOf(input.width, input.height)
      val crop = Bitmap.createBitmap(input, (input.width - s) / 2, (input.height - s) / 2, s, s)
      HeadPoseEstimator(modelFile.absolutePath).use { est ->
        val (pose, ms) = est.estimate(crop)
        val out = drawAxis(crop, pose)
        runOnUiThread {
          status.text = "6DRepNet  ·  head pose  ·  CompiledModel GPU  ·  ${ms} ms  ·  " +
            "yaw ${pose.yaw.toInt()} pitch ${pose.pitch.toInt()} roll ${pose.roll.toInt()}"
          imageView.setImageBitmap(out)
        }
      }
    }
  }

  /** Draw the 3D head-pose axes centered on the face crop. */
  private fun drawAxis(face: Bitmap, hp: HeadPose): Bitmap {
    val out = face.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(out)
    val cx = out.width / 2f; val cy = out.height / 2f; val size = out.width * 0.3f
    val p = Math.toRadians(hp.pitch.toDouble())
    val ya = Math.toRadians(-hp.yaw.toDouble())
    val r = Math.toRadians(hp.roll.toDouble())
    val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { strokeWidth = out.width / 60f }
    paint.color = Color.rgb(255, 60, 60)
    canvas.drawLine(cx, cy, size * (cos(ya) * cos(r)).toFloat() + cx,
      size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(ya)).toFloat() + cy, paint)
    paint.color = Color.rgb(60, 220, 90)
    canvas.drawLine(cx, cy, size * (-cos(ya) * sin(r)).toFloat() + cx,
      size * (cos(p) * cos(r) - sin(p) * sin(ya) * sin(r)).toFloat() + cy, paint)
    paint.color = Color.rgb(70, 130, 255)
    canvas.drawLine(cx, cy, size * sin(ya).toFloat() + cx, size * (-cos(ya) * sin(p)).toFloat() + cy, paint)
    return out
  }

  override fun onDestroy() { super.onDestroy(); executor.shutdown() }
}
