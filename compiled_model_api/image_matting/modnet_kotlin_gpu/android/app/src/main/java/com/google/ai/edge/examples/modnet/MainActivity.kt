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

package com.google.ai.edge.examples.modnet

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
 * Runs MODNet portrait matting on a bundled photo and shows the foreground
 * composited over a replaced background — a deterministic, self-contained demo.
 * The 26 MB model is loaded from the app's filesDir; push it there first with
 * install_to_device.sh (it is not bundled in the APK).
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
      val modelFile = File(filesDir, "modnet.tflite")
      if (!modelFile.exists()) {
        runOnUiThread {
          status.text = "Model not found at:\n${modelFile.absolutePath}\n\n" +
            "Push it first:  ./install_to_device.sh <dir-with-modnet.tflite>\n" +
            "(build with ../conversion or download from\n litert-community/MODNet-LiteRT)"
        }
        return@execute
      }
      val input = assets.open("portrait.jpg").use { BitmapFactory.decodeStream(it) }
      Matter(modelFile.absolutePath).use { matter ->
        val (composite, ms) = matter.matte(input, Color.rgb(0, 177, 64))  // green-screen bg
        runOnUiThread {
          status.text = "MODNet  ·  trimap-free portrait matting  ·  CompiledModel GPU  ·  ${ms} ms"
          imageView.setImageBitmap(composite.copy(android.graphics.Bitmap.Config.ARGB_8888, false))
        }
      }
    }
  }

  override fun onDestroy() {
    super.onDestroy()
    executor.shutdown()
  }
}
