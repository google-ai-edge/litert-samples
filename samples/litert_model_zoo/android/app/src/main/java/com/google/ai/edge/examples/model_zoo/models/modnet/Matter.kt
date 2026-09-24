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

package com.google.ai.edge.examples.model_zoo.models.modnet

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
 * MODNet trimap-free portrait matting on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 512, 512] NCHW, RGB, normalized to [-1, 1] ((x/255 - 0.5) / 0.5). Output:
 * [1, 1, 512, 512] soft alpha matte in [0, 1].
 *
 * MODNet is a pure CNN (MobileNetV2 backbone + LR/HR/Fusion branches). Two GPU re-authoring patches
 * are baked into the converted graph (see scripts/): the SE block's Linear channel-attention as 1x1
 * convs, and a fp16-safe hierarchical-mean InstanceNorm (the stock instance-norm variance overflows
 * fp16 over large spatial maps on the Mali delegate). Everything runs on the GPU delegate.
 */
class Matter(context: Context, modelFile: File, accelerator: Accelerator = Accelerator.GPU) :
  AutoCloseable {

  companion object {
    internal fun compositePixels(
      matte: FloatArray,
      pixels: IntArray,
      bgColor: Int,
      outPixels: IntArray,
      plane: Int,
    ) {
      val bgR = (bgColor shr 16) and 0xFF
      val bgG = (bgColor shr 8) and 0xFF
      val bgB = bgColor and 0xFF
      for (i in 0 until plane) {
        val a = matte[i].coerceIn(0f, 1f)
        val ia = 1f - a
        val p = pixels[i]
        val r = (((p shr 16) and 0xFF) * a + bgR * ia).toInt()
        val g = (((p shr 8) and 0xFF) * a + bgG * ia).toInt()
        val b = ((p and 0xFF) * a + bgB * ia).toInt()
        outPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
      }
    }

    private const val TAG = "MODNet"
    const val SIZE = 512
  }

  private val model: CompiledModel
  private val inBufs: List<TensorBuffer>
  private val outBufs: List<TensorBuffer>

  private val inputFloats = FloatArray(3 * SIZE * SIZE)
  private val pixels = IntArray(SIZE * SIZE)
  private val outPixels = IntArray(SIZE * SIZE)
  private val resized = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
  private val result = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  init {
    val options = CompiledModel.Options(accelerator)
    model = CompiledModel.create(modelFile.absolutePath, options, null)
    inBufs = model.createInputBuffers()
    outBufs = model.createOutputBuffers()
    Log.i(TAG, "$accelerator compiled OK — ${inBufs.size} in / ${outBufs.size} out")
  }

  /**
   * Matte the portrait and composite the foreground over [bgColor]. Returns the composited 512x512
   * bitmap + inference time (ms).
   */
  fun matte(bitmap: Bitmap, bgColor: Int): Pair<Bitmap, Long> {
    val t = System.nanoTime()
    val canvas = Canvas(resized)
    matrix.setScale(SIZE.toFloat() / bitmap.width, SIZE.toFloat() / bitmap.height)
    canvas.drawBitmap(bitmap, matrix, paint)
    resized.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
    val plane = SIZE * SIZE
    for (i in 0 until plane) {
      val p = pixels[i]
      inputFloats[i] = (((p shr 16) and 0xFF) / 255f - 0.5f) / 0.5f
      inputFloats[plane + i] = (((p shr 8) and 0xFF) / 255f - 0.5f) / 0.5f
      inputFloats[2 * plane + i] = ((p and 0xFF) / 255f - 0.5f) / 0.5f
    }
    inBufs[0].writeFloat(inputFloats)

    model.run(inBufs, outBufs)
    val matte = outBufs[0].readFloat() // [512*512] alpha 0..1

    compositePixels(matte, pixels, bgColor, outPixels, plane)
    result.setPixels(outPixels, 0, SIZE, 0, 0, SIZE, SIZE)
    return result to ((System.nanoTime() - t) / 1_000_000)
  }

  override fun close() {
    model.close()
    if (!resized.isRecycled) resized.recycle()
    if (!result.isRecycled) result.recycle()
  }
}
