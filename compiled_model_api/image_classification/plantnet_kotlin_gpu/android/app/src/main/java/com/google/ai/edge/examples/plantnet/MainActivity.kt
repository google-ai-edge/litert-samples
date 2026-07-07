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

package com.google.ai.edge.examples.plantnet

import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.util.concurrent.Executors

/**
 * Runs PlantNet-300K on a bundled plant photo and prints the top-5 species — a
 * deterministic, self-contained demo. The 47 MB model is loaded from the app's
 * filesDir; push it there first with install_to_device.sh (not bundled in the APK).
 */
class MainActivity : AppCompatActivity() {

  private val executor = Executors.newSingleThreadExecutor()

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val status = TextView(this).apply {
        textSize = 16f
        setPadding(28, 40, 28, 20)
    }
    val imageView = ImageView(this).apply { adjustViewBounds = true }
    setContentView(LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      addView(status)
      addView(imageView)
    })

    executor.execute {
      val modelFile = File(filesDir, "plantnet.tflite")
      if (!modelFile.exists()) {
        runOnUiThread {
          status.text = "Model not found at:\n${modelFile.absolutePath}\n\n" +
            "Push it first:  ./install_to_device.sh <dir-with-plantnet.tflite>\n" +
            "(build with ../conversion or download from\n" +
            " litert-community/PlantNet-300K-ResNet18-LiteRT)"
        }
        return@execute
      }
      val input = assets.open("plant.jpg").use { BitmapFactory.decodeStream(it) }
      PlantClassifier(modelFile.absolutePath).use { clf ->
        val (preds, ms) = clf.classify(input)
        val txt = "PlantNet-300K  ·  CompiledModel GPU  ·  ${ms} ms\n\n" +
          preds.joinToString("\n") { (n, p) -> "%s   %d%%".format(n, (p * 100).toInt()) }
        runOnUiThread {
            status.text = txt
            imageView.setImageBitmap(input)
        }
      }
    }
  }

  override fun onDestroy() {
    super.onDestroy()
    executor.shutdown()
  }
}
