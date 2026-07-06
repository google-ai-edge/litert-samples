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

package com.google.ai.edge.examples.ormbg

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.util.concurrent.Executors

/**
 * Runs ormbg background removal on a bundled photo and shows the subject cut out onto a
 * transparency checkerboard — a deterministic, self-contained demo. The 176 MB model is
 * loaded from filesDir; push it there first with install_to_device.sh.
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
      val modelFile = File(filesDir, "ormbg.tflite")
      if (!modelFile.exists()) {
        runOnUiThread {
          status.text = "Model not found at:\n${modelFile.absolutePath}\n\n" +
            "Push it first:  ./install_to_device.sh <dir-with-ormbg.tflite>\n" +
            "(build with ../conversion or download from\n litert-community/ormbg-LiteRT)"
        }
        return@execute
      }
      val input = assets.open("test_image.jpg").use { BitmapFactory.decodeStream(it) }
      BgRemover(modelFile.absolutePath).use { remover ->
        val (alpha, ms) = remover.matte(input)
        val out = cutout(input, alpha)
        runOnUiThread {
          status.text = "ormbg  ·  background removal  ·  CompiledModel GPU  ·  ${ms} ms"
          imageView.setImageBitmap(out)
        }
      }
    }
  }

  /** Composite the foreground onto a transparency checkerboard using the alpha matte. */
  private fun cutout(image: Bitmap, alpha: FloatArray): Bitmap {
    val S = BgRemover.SIZE
    val w = image.width; val h = image.height
    val scaled = Bitmap.createScaledBitmap(image, w, h, true)
    val px = IntArray(w * h); scaled.getPixels(px, 0, w, 0, 0, w, h)
    val out = IntArray(w * h)
    for (y in 0 until h) {
      val ay = y * S / h
      for (x in 0 until w) {
        val a = alpha[ay * S + x * S / w]
        val p = px[y * w + x]
        val fr = (p shr 16) and 0xFF; val fg = (p shr 8) and 0xFF; val fb = p and 0xFF
        val ck = if (((x / 24) + (y / 24)) % 2 == 0) 255 else 205
        val rr = (fr * a + ck * (1 - a)).toInt()
        val gg = (fg * a + ck * (1 - a)).toInt()
        val bb = (fb * a + ck * (1 - a)).toInt()
        out[y * w + x] = (0xFF shl 24) or (rr shl 16) or (gg shl 8) or bb
      }
    }
    return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
  }

  override fun onDestroy() { super.onDestroy(); executor.shutdown() }
}
