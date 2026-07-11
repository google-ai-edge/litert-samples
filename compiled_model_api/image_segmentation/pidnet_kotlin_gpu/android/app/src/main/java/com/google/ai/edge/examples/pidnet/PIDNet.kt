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

package com.google.ai.edge.examples.pidnet

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

/**
 * PIDNet-S real-time semantic segmentation on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 1024, 1024]  NCHW, RGB, ImageNet-normalized.
 * Output: [1, 19, 128, 128]   Cityscapes class logits at 1/8 resolution.
 *
 * PIDNet-S is a pure CNN — the whole graph runs on the GPU delegate (no CPU
 * fallback). We argmax over the 19 classes per pixel and return a 128x128
 * Cityscapes-colored label bitmap.
 */
class PIDNet(modelPath: String) : AutoCloseable {

  companion object {
    private const val TAG = "PIDNet"
    const val INPUT = 1024
    const val OUT = 128
    const val N_CLASS = 19
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
  }

  private val model = CompiledModel.create(modelPath, CompiledModel.Options(Accelerator.GPU), null)
  private val inBufs: List<TensorBuffer> = model.createInputBuffers()
  private val outBufs: List<TensorBuffer> = model.createOutputBuffers()

  private val inputFloats = FloatArray(3 * INPUT * INPUT)
  private val pixels = IntArray(INPUT * INPUT)
  private val resized = Bitmap.createBitmap(INPUT, INPUT, Bitmap.Config.ARGB_8888)
  private val labelBitmap = Bitmap.createBitmap(OUT, OUT, Bitmap.Config.ARGB_8888)
  private val labelPixels = IntArray(OUT * OUT)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  init {
    Log.i(TAG, "GPU compiled OK — ${inBufs.size} in / ${outBufs.size} out")
  }

  /** Segment one image. Returns a 128x128 Cityscapes-colored label bitmap + time (ms). */
  fun segment(bitmap: Bitmap): Pair<Bitmap, Long> {
    val t = System.nanoTime()
    val canvas = Canvas(resized)
    matrix.setScale(INPUT.toFloat() / bitmap.width, INPUT.toFloat() / bitmap.height)
    canvas.drawBitmap(bitmap, matrix, paint)
    resized.getPixels(pixels, 0, INPUT, 0, 0, INPUT, INPUT)
    val plane = INPUT * INPUT
    for (i in 0 until plane) {
      val p = pixels[i]
      inputFloats[i] = (((p shr 16) and 0xFF) / 255f - MEAN[0]) / STD[0]
      inputFloats[plane + i] = (((p shr 8) and 0xFF) / 255f - MEAN[1]) / STD[1]
      inputFloats[2 * plane + i] = ((p and 0xFF) / 255f - MEAN[2]) / STD[2]
    }
    inBufs[0].writeFloat(inputFloats)

    model.run(inBufs, outBufs)
    val logits = outBufs[0].readFloat()   // [19, 128, 128]

    val hw = OUT * OUT
    for (i in 0 until hw) {
      var best = 0
      var bestV = logits[i]
      for (c in 1 until N_CLASS) {
        val v = logits[c * hw + i]
        if (v > bestV) {
            bestV = v
            best = c
        }
      }
      labelPixels[i] = CityscapesPalette.COLORS[best]
    }
    labelBitmap.setPixels(labelPixels, 0, OUT, 0, 0, OUT, OUT)
    return labelBitmap to ((System.nanoTime() - t) / 1_000_000)
  }

  override fun close() = model.close()
}
