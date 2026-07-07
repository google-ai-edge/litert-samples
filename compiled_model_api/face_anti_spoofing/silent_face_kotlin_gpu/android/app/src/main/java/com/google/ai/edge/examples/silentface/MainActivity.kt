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

package com.google.ai.edge.examples.silentface

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
import java.util.concurrent.Executors

/**
 * Runs Silent-Face liveness on a bundled face photo and shows the verdict. A photographed
 * face is a presentation attack, so it is correctly flagged as spoof (replay); a live
 * camera capture would score live. Deterministic, self-contained; the 1.85 MB model is bundled.
 */
class MainActivity : AppCompatActivity() {

  private val executor = Executors.newSingleThreadExecutor()

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val status = TextView(this).apply {
        textSize = 15f
        setPadding(28, 40, 28, 20)
    }
    val imageView = ImageView(this).apply { adjustViewBounds = true }
    setContentView(LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      addView(status)
      addView(imageView)
    })

    executor.execute {
      val input = assets.open("test_image.jpg").use { BitmapFactory.decodeStream(it) }
      LivenessDetector(this).use { d ->
        val (p, ms) = d.detect(input)
        val live = p[1] >= p[0] && p[1] >= p[2]
        val out = annotate(input, p, live)
        runOnUiThread {
          status.text = "Silent-Face liveness  ·  CompiledModel GPU  ·  ${ms} ms  ·  " +
            "${if (live) "LIVE" else "SPOOF"} (live ${(p[1] * 100).toInt()}%)"
          imageView.setImageBitmap(out)
        }
      }
    }
  }

  private fun annotate(image: Bitmap, p: FloatArray, live: Boolean): Bitmap {
    val out = image.copy(Bitmap.Config.ARGB_8888, true)
    val c = Canvas(out)
    val col = if (live) Color.rgb(50, 220, 100) else Color.rgb(240, 70, 70)
    val box = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = out.width / 90f
        color = col
    }
    c.drawRect(4f, 4f, out.width - 4f, out.height - 4f, box)
    val tp = Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = col
      textSize = out.width / 8f
      setShadowLayer(6f, 0f, 0f, Color.BLACK)
    }
    c.drawText(if (live) "LIVE" else "SPOOF", 20f, tp.textSize + 20f, tp)
    return out
  }

  override fun onDestroy() {
      super.onDestroy()
      executor.shutdown()
  }
}
