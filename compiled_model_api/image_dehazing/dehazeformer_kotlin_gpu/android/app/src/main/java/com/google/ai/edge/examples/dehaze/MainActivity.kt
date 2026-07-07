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

package com.google.ai.edge.examples.dehaze

import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.Gravity
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.util.concurrent.Executors

/**
 * Runs DehazeFormer-MCT on a bundled hazy photo and shows the dehazed result — a
 * deterministic, self-contained demo. The 17 MB model is loaded from filesDir; push it
 * there first with install_to_device.sh.
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
    val outputView = ImageView(this).apply { adjustViewBounds = true }
    setContentView(
      LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        addView(status)
        addView(
          TextView(this@MainActivity).apply {
            text = "Hazy input"
            setPadding(28, 8, 28, 4)
          })
        addView(
          inputView,
          LinearLayout.LayoutParams(480, 480).apply { gravity = Gravity.CENTER_HORIZONTAL })
        addView(
          TextView(this@MainActivity).apply {
            text = "Dehazed (GPU)"
            setPadding(28, 16, 28, 4)
          })
        addView(
          outputView,
          LinearLayout.LayoutParams(480, 480).apply { gravity = Gravity.CENTER_HORIZONTAL })
      })

    executor.execute {
      val modelFile = File(filesDir, "dehazeformer_base.tflite")
      if (!modelFile.exists()) {
        runOnUiThread {
          status.text = "Model not found at:\n${modelFile.absolutePath}\n\n" +
            "Push it first:  ./install_to_device.sh <dir-with-dehazeformer_base.tflite>\n" +
            "(build with ../conversion or download from\n" +
            " litert-community/DehazeFormer-MCT-LiteRT)"
        }
        return@execute
      }
      val input = assets.open("test_image.jpg").use { BitmapFactory.decodeStream(it) }
      Dehazer(modelFile.absolutePath).use { dehazer ->
        val (dehazed, ms) = dehazer.dehaze(input)
        runOnUiThread {
          status.text =
            "DehazeFormer-MCT  ·  haze removal  ·  CompiledModel GPU  ·  ${ms} ms"
          inputView.setImageBitmap(input)
          outputView.setImageBitmap(dehazed)
        }
      }
    }
  }

  override fun onDestroy() {
    super.onDestroy()
    executor.shutdown()
  }
}
