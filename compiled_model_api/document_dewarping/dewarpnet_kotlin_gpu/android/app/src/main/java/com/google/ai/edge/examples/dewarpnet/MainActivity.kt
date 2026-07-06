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

package com.google.ai.edge.examples.dewarpnet

import android.graphics.Bitmap
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
 * Runs DewarpNet on a bundled photo of a curved document and shows the flattened result
 * below the input — a deterministic, self-contained demo. The 189 MB model is loaded from
 * filesDir; push it there first with install_to_device.sh.
 */
class MainActivity : AppCompatActivity() {

  private val executor = Executors.newSingleThreadExecutor()

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val status = TextView(this).apply { textSize = 15f; setPadding(28, 40, 28, 16) }
    val inputView = ImageView(this).apply { adjustViewBounds = true }
    val outputView = ImageView(this).apply { adjustViewBounds = true }
    val col = LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      addView(status)
      addView(TextView(this@MainActivity).apply { text = "Input (warped)"; setPadding(28, 8, 28, 4) })
      addView(inputView, LinearLayout.LayoutParams(600, 450).apply { gravity = Gravity.CENTER_HORIZONTAL })
      addView(TextView(this@MainActivity).apply { text = "Dewarped (GPU)"; setPadding(28, 16, 28, 4) })
      addView(outputView, LinearLayout.LayoutParams(600, 600).apply { gravity = Gravity.CENTER_HORIZONTAL })
    }
    setContentView(col)

    executor.execute {
      val modelFile = File(filesDir, "dewarp.tflite")
      if (!modelFile.exists()) {
        runOnUiThread {
          status.text = "Model not found at:\n${modelFile.absolutePath}\n\n" +
            "Push it first:  ./install_to_device.sh <dir-with-dewarp.tflite>\n" +
            "(build with ../conversion or download from\n litert-community/DewarpNet-LiteRT)"
        }
        return@execute
      }
      val input = assets.open("test_image.jpg").use { BitmapFactory.decodeStream(it) }
      DocumentDewarper(modelFile.absolutePath).use { d ->
        val (flat, ms) = d.dewarp(input)
        runOnUiThread {
          status.text = "DewarpNet  ·  document dewarping  ·  CompiledModel GPU  ·  ${ms} ms"
          inputView.setImageBitmap(input)
          outputView.setImageBitmap(flat)
        }
      }
    }
  }

  override fun onDestroy() { super.onDestroy(); executor.shutdown() }
}
