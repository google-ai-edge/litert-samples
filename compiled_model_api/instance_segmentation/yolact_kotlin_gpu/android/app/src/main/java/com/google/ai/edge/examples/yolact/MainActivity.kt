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

package com.google.ai.edge.examples.yolact

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
 * Runs YOLACT instance segmentation on a bundled photo and draws colored per-instance
 * masks + boxes + COCO labels — a deterministic, self-contained demo. The 125 MB model
 * is loaded from filesDir; push it (and priors.bin) there first with install_to_device.sh.
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
      val modelFile = File(filesDir, "yolact.tflite")
      if (!modelFile.exists()) {
        runOnUiThread {
          status.text = "Model not found at:\n${modelFile.absolutePath}\n\n" +
            "Push it first:  ./install_to_device.sh <dir-with-yolact.tflite-and-priors.bin>\n" +
            "(build with ../conversion or download from\n litert-community/YOLACT-ResNet50-LiteRT)"
        }
        return@execute
      }
      val input = assets.open("test_image.jpg").use { BitmapFactory.decodeStream(it) }
      YolactSegmenter(this, modelFile.absolutePath).use { seg ->
        val (instances, ms) = seg.segment(input)
        val out = render(input, instances)
        runOnUiThread {
          status.text = "YOLACT  ·  COCO instance segmentation  ·  CompiledModel GPU  ·  " +
            "${ms} ms  ·  ${instances.size} instances"
          imageView.setImageBitmap(out)
        }
      }
    }
  }

  /** Draw masks (scaled from 550) + boxes + labels onto a copy of the input. */
  private fun render(image: Bitmap, insts: List<Instance>): Bitmap {
    val out = image.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(out)
    val S = YolactSegmenter.SIZE
    val sx = out.width.toFloat() / S
    val sy = out.height.toFloat() / S
    val mp = Paint()
    val bp = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = out.width / 200f
    }
    val tp = Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = Color.WHITE
      textSize = out.width / 28f
      setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }
    // masks: composite a translucent color per instance
    val row = IntArray(out.width)
    for (ins in insts) {
      val c = (0x88 shl 24) or (Palette.color(ins.cls) and 0x00FFFFFF)
      mp.color = c
      for (yy in 0 until out.height) {
        val my = (yy / sy).toInt().coerceIn(0, S - 1)
        var started = -1
        for (xx in 0 until out.width) {
          val on = ins.mask[my * S + (xx / sx).toInt().coerceIn(0, S - 1)]
          if (on && started < 0) {
            started = xx
          }
          if ((!on || xx == out.width - 1) && started >= 0) {
            canvas.drawRect(started.toFloat(), yy.toFloat(), xx.toFloat(), (yy + 1).toFloat(), mp)
            started = -1
          }
        }
      }
    }
    for (ins in insts) {
      bp.color = Palette.color(ins.cls)
      canvas.drawRect(ins.x1 * S * sx, ins.y1 * S * sy, ins.x2 * S * sx, ins.y2 * S * sy, bp)
      canvas.drawText("${CocoLabels.NAMES[ins.cls]} ${(ins.score * 100).toInt()}%",
        ins.x1 * S * sx + 6, ins.y1 * S * sy + tp.textSize, tp)
    }
    return out
  }

  override fun onDestroy() {
      super.onDestroy()
      executor.shutdown()
  }
}
