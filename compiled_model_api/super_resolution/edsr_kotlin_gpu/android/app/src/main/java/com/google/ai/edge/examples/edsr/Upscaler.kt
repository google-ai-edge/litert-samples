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

package com.google.ai.edge.examples.edsr

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

/**
 * EDSR ×4 single-image super-resolution on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 128, 128] NCHW, RGB, x/255.
 * Output: [1, 3, 512, 512] NCHW, RGB in 0..1 (clamp, ×255).
 *
 * EDSR-baseline (pure CNN); the PixelShuffle upsampler is re-authored as a fixed
 * ConvTranspose2d → ZeroStuffConvT2d so the whole graph runs on the GPU delegate. 7.7 MB.
 */
class Upscaler(context: Context, modelFileName: String = "edsr.tflite") : AutoCloseable {

  companion object {
    private const val TAG = "EDSR"
    const val LR = 128
    const val HR = 512
  }

  private val model = CompiledModel.create(
    context.assets, modelFileName, CompiledModel.Options(Accelerator.GPU), null)
  private val inBufs: List<TensorBuffer> = model.createInputBuffers()
  private val outBufs: List<TensorBuffer> = model.createOutputBuffers()

  private val inputFloats = FloatArray(3 * LR * LR)
  private val lrPixels = IntArray(LR * LR)
  private val hrPixels = IntArray(HR * HR)
  private val lrBitmap = Bitmap.createBitmap(LR, LR, Bitmap.Config.ARGB_8888)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  init { Log.i(TAG, "GPU compiled OK — ${inBufs.size} in / ${outBufs.size} out") }

  fun upscale(bitmap: Bitmap): Pair<Bitmap, Long> {
    val t = System.nanoTime()
    Canvas(lrBitmap).drawBitmap(
      bitmap,
      matrix.apply { setScale(LR.toFloat() / bitmap.width, LR.toFloat() / bitmap.height) },
      paint)
    lrBitmap.getPixels(lrPixels, 0, LR, 0, 0, LR, LR)
    val plane = LR * LR
    for (i in 0 until plane) {
      val p = lrPixels[i]
      inputFloats[i] = ((p shr 16) and 0xFF) / 255f
      inputFloats[plane + i] = ((p shr 8) and 0xFF) / 255f
      inputFloats[2 * plane + i] = (p and 0xFF) / 255f
    }
    inBufs[0].writeFloat(inputFloats)
    model.run(inBufs, outBufs)
    val o = outBufs[0].readFloat()   // [3*512*512] planar RGB 0..1
    val hp = HR * HR
    for (i in 0 until hp) {
      val r = (o[i] * 255f).toInt().coerceIn(0, 255)
      val g = (o[hp + i] * 255f).toInt().coerceIn(0, 255)
      val b = (o[2 * hp + i] * 255f).toInt().coerceIn(0, 255)
      hrPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }
    val hr = Bitmap.createBitmap(hrPixels, HR, HR, Bitmap.Config.ARGB_8888)
    return hr to ((System.nanoTime() - t) / 1_000_000)
  }

  override fun close() {
    model.close()
    if (!lrBitmap.isRecycled) {
      lrBitmap.recycle()
    }
  }
}
