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

package com.google.ai.edge.examples.edsr

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.Gravity
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.util.concurrent.Executors

/**
 * Runs EDSR ×4 on a bundled low-res image and shows bicubic ×4 vs EDSR ×4 — a
 * deterministic, self-contained demo. The 7.7 MB model is bundled in assets.
 */
class MainActivity : AppCompatActivity() {

  private val executor = Executors.newSingleThreadExecutor()

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val status = TextView(this).apply { textSize = 15f; setPadding(28, 40, 28, 16) }
    val bicubicView = ImageView(this).apply { adjustViewBounds = true }
    val srView = ImageView(this).apply { adjustViewBounds = true }
    setContentView(LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      addView(status)
      addView(TextView(this@MainActivity).apply { text = "Bicubic ×4"; setPadding(28, 8, 28, 4) })
      addView(bicubicView, LinearLayout.LayoutParams(512, 512).apply { gravity = Gravity.CENTER_HORIZONTAL })
      addView(TextView(this@MainActivity).apply { text = "EDSR ×4 (GPU)"; setPadding(28, 16, 28, 4) })
      addView(srView, LinearLayout.LayoutParams(512, 512).apply { gravity = Gravity.CENTER_HORIZONTAL })
    })

    executor.execute {
      val lr = assets.open("test_image.jpg").use { BitmapFactory.decodeStream(it) }
      Upscaler(this).use { u ->
        val (hr, ms) = u.upscale(lr)
        val bicubic = Bitmap.createScaledBitmap(lr, Upscaler.HR, Upscaler.HR, true)
        runOnUiThread {
          status.text = "EDSR ×4 super-resolution  ·  CompiledModel GPU  ·  ${ms} ms  ·  128 → 512"
          bicubicView.setImageBitmap(bicubic)
          srView.setImageBitmap(hr)
        }
      }
    }
  }

  override fun onDestroy() { super.onDestroy(); executor.shutdown() }
}
