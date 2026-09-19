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

package com.google.ai.edge.examples.portrait

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

/**
 * U²-Net portrait: photo -> pencil line drawing, on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 512, 512] NCHW, RGB, x/255 then ImageNet-normalized.
 * Output: [1, 1, 512, 512] in [0,1]. Min-max normalize, then invert (1-x) = dark strokes on white.
 *
 * U²-Net (RSU blocks) — pure CNN, fully on the GPU delegate (defensive align_corners=False).
 */
class PortraitSketcher(modelPath: String) : AutoCloseable {

  companion object {
    private const val TAG = "Portrait"
    const val SIZE = 512
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
  }

  private val model = CompiledModel.create(modelPath, CompiledModel.Options(Accelerator.GPU), null)
  private val inBufs: List<TensorBuffer> = model.createInputBuffers()
  private val outBufs: List<TensorBuffer> = model.createOutputBuffers()
  private val inputFloats = FloatArray(3 * SIZE * SIZE)
  private val pixels = IntArray(SIZE * SIZE)
  private val sketchPixels = IntArray(SIZE * SIZE)
  private val resized = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  init { Log.i(TAG, "GPU compiled OK — ${inBufs.size} in / ${outBufs.size} out") }

  fun sketch(bitmap: Bitmap): Pair<Bitmap, Long> {
    val t = System.nanoTime()
    Canvas(resized).drawBitmap(bitmap, matrix.apply {
      setScale(SIZE.toFloat() / bitmap.width, SIZE.toFloat() / bitmap.height) }, paint)
    resized.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
    val plane = SIZE * SIZE
    for (i in 0 until plane) {
      val p = pixels[i]
      inputFloats[i] = (((p shr 16) and 0xFF) / 255f - MEAN[0]) / STD[0]
      inputFloats[plane + i] = (((p shr 8) and 0xFF) / 255f - MEAN[1]) / STD[1]
      inputFloats[2 * plane + i] = ((p and 0xFF) / 255f - MEAN[2]) / STD[2]
    }
    inBufs[0].writeFloat(inputFloats)
    model.run(inBufs, outBufs)
    val d = outBufs[0].readFloat()
    var mn = Float.MAX_VALUE
    var mx = -Float.MAX_VALUE
    for (v in d) {
        if (v < mn) {
          mn = v
        }
        if (v > mx) {
          mx = v
        }
    }
    val inv = 1f / (mx - mn + 1e-6f)
    for (i in 0 until plane) {
      val g = (255f * (1f - (d[i] - mn) * inv)).toInt().coerceIn(0, 255)
      sketchPixels[i] = (0xFF shl 24) or (g shl 16) or (g shl 8) or g
    }
    return Bitmap.createBitmap(sketchPixels, SIZE, SIZE, Bitmap.Config.ARGB_8888) to
      ((System.nanoTime() - t) / 1_000_000)
  }

  override fun close() {
    model.close()
    if (!resized.isRecycled) {
      resized.recycle()
    }
  }
}
