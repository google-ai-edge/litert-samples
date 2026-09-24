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

package com.google.ai.edge.examples.model_zoo.models.liveness

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
 * Silent-Face (MiniFASNetV2) face liveness / anti-spoofing on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 80, 80] NCHW, BGR, x/255 (a face crop). Output: [1, 3] softmax — class 1 =
 * live/real, 0 & 2 = spoof (print / replay).
 *
 * MiniFASNetV2 — pure CNN, fully GPU (PReLU lowers to relu ops, zero patches). ~5 ms, 1.85 MB.
 */
class LivenessDetector(
  context: Context,
  modelFile: File,
  accelerator: Accelerator = Accelerator.GPU,
) : AutoCloseable {

  companion object {
    private const val TAG = "Liveness"
    const val SIZE = 80

    internal fun decode(o: FloatArray): Pair<Boolean, Float> {
      val live = o[1]
      val isLive = o[1] >= o[0] && o[1] >= o[2]
      return isLive to live
    }
  }

  private val model: CompiledModel
  private val inBufs: List<TensorBuffer>
  private val outBufs: List<TensorBuffer>
  private val inputFloats = FloatArray(3 * SIZE * SIZE)
  private val pixels = IntArray(SIZE * SIZE)
  private val resized = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  init {
    model = CompiledModel.create(modelFile.absolutePath, CompiledModel.Options(accelerator), null)
    inBufs = model.createInputBuffers()
    outBufs = model.createOutputBuffers()
    Log.i(TAG, "$accelerator compiled OK — ${inBufs.size} in / ${outBufs.size} out")
  }

  /** Returns (isLive, liveScore, ms) from a face crop bitmap. */
  fun detect(faceCrop: Bitmap): Triple<Boolean, Float, Long> {
    val t = System.nanoTime()
    Canvas(resized)
      .drawBitmap(
        faceCrop,
        matrix.apply {
          setScale(SIZE.toFloat() / faceCrop.width, SIZE.toFloat() / faceCrop.height)
        },
        paint,
      )
    resized.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
    val plane = SIZE * SIZE
    for (i in 0 until plane) {
      val p = pixels[i]
      inputFloats[i] = (p and 0xFF) / 255f // B
      inputFloats[plane + i] = ((p shr 8) and 0xFF) / 255f // G
      inputFloats[2 * plane + i] = ((p shr 16) and 0xFF) / 255f // R
    }
    inBufs[0].writeFloat(inputFloats)
    model.run(inBufs, outBufs)
    val o = outBufs[0].readFloat() // [3] softmax
    val (isLive, live) = decode(o)
    return Triple(isLive, live, (System.nanoTime() - t) / 1_000_000)
  }

  override fun close() {
    model.close()
    if (!resized.isRecycled) resized.recycle()
  }
}
