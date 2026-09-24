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

package com.google.ai.edge.examples.model_zoo.models.edsr

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File

/**
 * EDSR ×4 single-image super-resolution on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 128, 128] NCHW, RGB, x/255. Output: [1, 3, 512, 512] NCHW, RGB in 0..1 (clamp,
 * ×255).
 *
 * EDSR-baseline (pure CNN). The PixelShuffle upsampler is re-authored as a fixed ConvTranspose2d →
 * ZeroStuffConvT2d to run on the GPU. ~23 ms/frame, 7.7 MB.
 */
class Upscaler(context: Context, modelFile: File, accelerator: Accelerator = Accelerator.GPU) :
  AutoCloseable {

  companion object {
    private const val TAG = "EDSR"
    const val LR = 128
    const val HR = 512

    internal fun fillNchwPixels(o: FloatArray, hp: Int, hrPixels: IntArray) {
      for (i in 0 until hp) {
        val r = (o[i] * 255f).toInt().coerceIn(0, 255)
        val g = (o[hp + i] * 255f).toInt().coerceIn(0, 255)
        val b = (o[2 * hp + i] * 255f).toInt().coerceIn(0, 255)
        hrPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
      }
    }
  }

  private val model: CompiledModel
  private val inBufs: List<TensorBuffer>
  private val outBufs: List<TensorBuffer>

  private val inputFloats = FloatArray(3 * LR * LR)
  private val lrPixels = IntArray(LR * LR)
  private val hrPixels = IntArray(HR * HR)
  private val lrBitmap = Bitmap.createBitmap(LR, LR, Bitmap.Config.ARGB_8888)
  private val hrBitmap = Bitmap.createBitmap(HR, HR, Bitmap.Config.ARGB_8888)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  init {
    var createdModel: CompiledModel? = null
    var createdInputs: List<TensorBuffer>? = null
    var createdOutputs: List<TensorBuffer>? = null
    try {
      val options = CompiledModel.Options(accelerator)
      model = CompiledModel.create(modelFile.absolutePath, options, null).also { createdModel = it }
      inBufs = model.createInputBuffers().also { createdInputs = it }
      outBufs = model.createOutputBuffers().also { createdOutputs = it }
      Log.i(TAG, "$accelerator compiled OK — ${inBufs.size} in / ${outBufs.size} out")
    } catch (failure: Throwable) {
      createdOutputs?.forEach { runCatching { it.close() } }
      createdInputs?.forEach { runCatching { it.close() } }
      runCatching { createdModel?.close() }
      runCatching { if (!lrBitmap.isRecycled) lrBitmap.recycle() }
      runCatching { if (!hrBitmap.isRecycled) hrBitmap.recycle() }
      throw failure
    }
  }

  /** Upscale a low-res bitmap 4× (input is resized to 128×128). Returns 512×512 HR + time (ms). */
  fun upscale(bitmap: Bitmap): Pair<Bitmap, Long> {
    val t = System.nanoTime()
    Canvas(lrBitmap)
      .drawBitmap(
        bitmap,
        matrix.apply { setScale(LR.toFloat() / bitmap.width, LR.toFloat() / bitmap.height) },
        paint,
      )
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
    val o = outBufs[0].readFloat() // [3*512*512] planar RGB 0..1
    val hp = HR * HR
    fillNchwPixels(o, hp, hrPixels)

    hrBitmap.setPixels(hrPixels, 0, HR, 0, 0, HR, HR)
    return hrBitmap to ((System.nanoTime() - t) / 1_000_000)
  }

  override fun close() {
    inBufs.forEach { it.close() }
    outBufs.forEach { it.close() }
    model.close()
    if (!lrBitmap.isRecycled) lrBitmap.recycle()
    if (!hrBitmap.isRecycled) hrBitmap.recycle()
  }
}
