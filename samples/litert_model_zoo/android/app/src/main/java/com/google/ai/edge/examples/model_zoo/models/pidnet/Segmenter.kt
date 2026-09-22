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

package com.google.ai.edge.examples.model_zoo.models.pidnet

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
 * PIDNet-S real-time semantic segmentation on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 1024, 1024] NCHW, RGB, ImageNet-normalized. Output: [1, 19, 128, 128] Cityscapes
 * class logits at 1/8 resolution.
 *
 * We argmax over the 19 classes per pixel and paint a 128x128 Cityscapes-colored label map, which
 * the overlay view scales up over the camera preview. PIDNet-S is a pure CNN — the whole graph runs
 * on the GPU delegate (no CPU fallback).
 */
class Segmenter(
  context: Context,
  modelDir: File,
  accelerator: Accelerator = Accelerator.GPU,
  modelFileName: String = "pidnet_s.tflite",
) : AutoCloseable {

  companion object {
    private const val TAG = "PIDNet"
    const val INPUT = 1024
    const val OUT = 128
    const val N_CLASS = 19
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    /** Per-pixel argmax/color block from segment(), exposed for JVM fixtures. */
    internal fun colorPixel(logits: FloatArray, i: Int, hw: Int): Int {
      var best = 0
      var bestV = logits[i]
      for (c in 1 until N_CLASS) {
        val v = logits[c * hw + i]
        if (v > bestV) {
          bestV = v
          best = c
        }
      }
      return CityscapesPalette.COLORS[best]
    }
  }

  private val model: CompiledModel
  private val inBufs: List<TensorBuffer>
  private val outBufs: List<TensorBuffer>

  private val inputFloats = FloatArray(3 * INPUT * INPUT)
  private val pixels = IntArray(INPUT * INPUT)
  private val resized = Bitmap.createBitmap(INPUT, INPUT, Bitmap.Config.ARGB_8888)
  private val labelBitmap = Bitmap.createBitmap(OUT, OUT, Bitmap.Config.ARGB_8888)
  private val labelPixels = IntArray(OUT * OUT)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  init {
    val options = CompiledModel.Options(accelerator)
    model =
      try {
        CompiledModel.create(File(modelDir, modelFileName).absolutePath, options, null)
      } catch (failure: Exception) {
        resized.recycle()
        labelBitmap.recycle()
        throw failure
      }
    inBufs =
      try {
        model.createInputBuffers()
      } catch (failure: Exception) {
        model.close()
        resized.recycle()
        labelBitmap.recycle()
        throw failure
      }
    outBufs =
      try {
        model.createOutputBuffers()
      } catch (failure: Exception) {
        inBufs.forEach { it.close() }
        model.close()
        resized.recycle()
        labelBitmap.recycle()
        throw failure
      }
    Log.i(TAG, "$accelerator compiled OK — ${inBufs.size} in / ${outBufs.size} out")
  }

  /** Segment one frame. Returns a 128x128 Cityscapes-colored label bitmap + time (ms). */
  fun segment(bitmap: Bitmap): Pair<Bitmap, Long> {
    val t = System.nanoTime()
    // Preprocess -> NCHW, RGB, ImageNet-normalized
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
    val logits = outBufs[0].readFloat() // [19, 128, 128] (NCHW, batch dropped)

    // argmax over 19 classes per pixel -> colored label map
    val hw = OUT * OUT
    for (i in 0 until hw) {
      labelPixels[i] = colorPixel(logits, i, hw)
    }
    labelBitmap.setPixels(labelPixels, 0, OUT, 0, 0, OUT, OUT)
    val ms = (System.nanoTime() - t) / 1_000_000
    return labelBitmap to ms
  }

  override fun close() {
    inBufs.forEach { it.close() }
    outBufs.forEach { it.close() }
    model.close()
    if (!resized.isRecycled) resized.recycle()
    if (!labelBitmap.isRecycled) labelBitmap.recycle()
  }
}
