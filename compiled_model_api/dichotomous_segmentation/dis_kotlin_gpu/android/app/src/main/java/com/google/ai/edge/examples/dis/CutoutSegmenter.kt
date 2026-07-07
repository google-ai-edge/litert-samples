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

package com.google.ai.edge.examples.dis

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

/**
 * DIS (IS-Net, general-use) high-precision object cutout on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 1024, 1024] NCHW, RGB, x/255 - 0.5.
 * Output: [1, 1, 1024, 1024] sigmoid alpha (0..1).
 *
 * IS-Net (RSU blocks) — pure CNN, fully on the GPU delegate (defensive align_corners=False).
 */
class CutoutSegmenter(modelPath: String) : AutoCloseable {

  companion object {
    private const val TAG = "DIS"
    const val SIZE = 1024
  }

  private val model = CompiledModel.create(modelPath, CompiledModel.Options(Accelerator.GPU), null)
  private val inBufs: List<TensorBuffer> = model.createInputBuffers()
  private val outBufs: List<TensorBuffer> = model.createOutputBuffers()

  private val inputFloats = FloatArray(3 * SIZE * SIZE)
  private val pixels = IntArray(SIZE * SIZE)
  private val resized = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  init { Log.i(TAG, "GPU compiled OK — ${inBufs.size} in / ${outBufs.size} out") }

  /** Returns a SIZE×SIZE alpha matte (0..1) + time (ms). */
  fun matte(bitmap: Bitmap): Pair<FloatArray, Long> {
    val t = System.nanoTime()
    Canvas(resized).drawBitmap(
      bitmap, matrix.apply { setScale(SIZE.toFloat() / bitmap.width, SIZE.toFloat() / bitmap.height) }, paint)
    resized.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
    val plane = SIZE * SIZE
    for (i in 0 until plane) {
      val p = pixels[i]
      inputFloats[i] = ((p shr 16) and 0xFF) / 255f - 0.5f
      inputFloats[plane + i] = ((p shr 8) and 0xFF) / 255f - 0.5f
      inputFloats[2 * plane + i] = (p and 0xFF) / 255f - 0.5f
    }
    inBufs[0].writeFloat(inputFloats)
    model.run(inBufs, outBufs)
    return outBufs[0].readFloat() to ((System.nanoTime() - t) / 1_000_000)
  }

  override fun close() {
    model.close()
    if (!resized.isRecycled) resized.recycle()
  }
}
