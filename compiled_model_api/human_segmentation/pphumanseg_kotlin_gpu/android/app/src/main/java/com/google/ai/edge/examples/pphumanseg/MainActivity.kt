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

package com.google.ai.edge.examples.pphumanseg

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.util.concurrent.Executors

/**
 * Runs PP-HumanSeg on a bundled portrait and replaces the background with a studio color —
 * a deterministic, self-contained demo. The 6 MB model is bundled in assets.
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
      val input = assets.open("test_image.jpg").use { BitmapFactory.decodeStream(it) }
      PortraitSegmenter(this).use { seg ->
        val (mask, ms) = seg.segment(input)
        val out = replaceBackground(input, mask)
        runOnUiThread {
          status.text = "PP-HumanSeg  ·  human segmentation  ·  CompiledModel GPU  ·  ${ms} ms"
          imageView.setImageBitmap(out)
        }
      }
    }
  }

  /** Keep the person, replace the background with studio green (mask is S×S). */
  private fun replaceBackground(image: Bitmap, mask: ByteArray): Bitmap {
    val S = PortraitSegmenter.S
    val out = image.copy(Bitmap.Config.ARGB_8888, true)
    val px = IntArray(out.width * out.height)
    out.getPixels(px, 0, out.width, 0, 0, out.width, out.height)
    val bg = Color.rgb(30, 190, 120)
    for (y in 0 until out.height) {
      val my = y * S / out.height
      for (x in 0 until out.width) {
        val mi = my * S + (x * S / out.width)
        if (mask[mi].toInt() != 1) px[y * out.width + x] = bg
      }
    }
    return Bitmap.createBitmap(px, out.width, out.height, Bitmap.Config.ARGB_8888)
  }

  override fun onDestroy() { super.onDestroy(); executor.shutdown() }
}
