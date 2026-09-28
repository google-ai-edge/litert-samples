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

package com.google.ai.edge.examples.litert_model_zoo.models.crowdcount

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.examples.litert_model_zoo.common.CompiledModelRunner
import com.google.ai.edge.litert.Accelerator
import java.io.File

/**
 * DM-Count crowd counting on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 512, 512] NCHW, RGB, ImageNet-normalized. Output: [1, 1, 64, 64] density map — sum
 * of the map = estimated person count.
 *
 * VGG19 backbone + conv regression head (NeurIPS 2020) — pure CNN, fully GPU. The mid-graph 2x
 * bilinear upsample (align_corners=True, banned on the GPU delegate) is re-authored exactly as two
 * constant-matrix multiplies at conversion time.
 */
class CrowdCounter(context: Context, modelFile: File, accelerator: Accelerator = Accelerator.GPU) :
  AutoCloseable {

  companion object {
    internal fun sumDensity(density: FloatArray): Float {
      var sum = 0f
      for (v in density) {
        sum += v
      }
      return sum
    }

    private const val TAG = "DMCount"
    const val SIZE = 512
    const val OUT = 64
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
  }

  /** One inference result: [density] is an OUT×OUT map whose sum is [count]. */
  data class Result(val density: FloatArray, val count: Float, val inferenceMs: Long)

  private val runner: CompiledModelRunner
  private val inputFloats = FloatArray(3 * SIZE * SIZE)
  private val pixels = IntArray(SIZE * SIZE)
  private val resized = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  init {
    try {
      runner = CompiledModelRunner.fromFile(modelFile.absolutePath, accelerator)
    } catch (failure: Throwable) {
      if (!resized.isRecycled) {
        resized.recycle()
      }
      throw failure
    }
    Log.i(
      TAG,
      "$accelerator compiled OK — ${runner.inputBuffers.size} in / " +
        "${runner.outputBuffers.size} out",
    )
  }

  /** Runs DM-Count on [bitmap] and returns the density map + person count + time. */
  fun count(bitmap: Bitmap): Result {
    val t = System.nanoTime()
    Canvas(resized)
      .drawBitmap(
        bitmap,
        matrix.apply { setScale(SIZE.toFloat() / bitmap.width, SIZE.toFloat() / bitmap.height) },
        paint,
      )
    resized.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
    val plane = SIZE * SIZE
    for (i in 0 until plane) {
      val p = pixels[i]
      inputFloats[i] = (((p shr 16) and 0xFF) / 255f - MEAN[0]) / STD[0]
      inputFloats[plane + i] = (((p shr 8) and 0xFF) / 255f - MEAN[1]) / STD[1]
      inputFloats[2 * plane + i] = ((p and 0xFF) / 255f - MEAN[2]) / STD[2]
    }
    runner.inputBuffers[0].writeFloat(inputFloats)
    runner.run()
    val density = runner.outputBuffers[0].readFloat() // [64*64] non-negative density
    val sum = sumDensity(density)
    return Result(density, sum, (System.nanoTime() - t) / 1_000_000)
  }

  override fun close() {
    runner.close()
    if (!resized.isRecycled) {
      resized.recycle()
    }
  }
}
