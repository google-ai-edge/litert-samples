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

package com.google.ai.edge.examples.pphumanseg

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
 * PP-HumanSeg human/portrait segmentation on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 192, 192, 3] NHWC, BGR, (x/255 - 0.5) / 0.5.
 * Output: [1, 192, 192, 2] NHWC softmax; argmax over the last dim -> human mask (1 = person).
 *
 * Pure CNN (converted with onnx2tf, no patches). ~36 ms/frame, 6 MB.
 */
class PortraitSegmenter(context: Context, modelFileName: String = "pphumanseg.tflite") : AutoCloseable {

  companion object {
    private const val TAG = "PP-HumanSeg"
    const val S = 192
  }

  private val model = CompiledModel.create(context.assets, modelFileName, CompiledModel.Options(Accelerator.GPU), null)
  private val inBufs: List<TensorBuffer> = model.createInputBuffers()
  private val outBufs: List<TensorBuffer> = model.createOutputBuffers()

  private val inputFloats = FloatArray(S * S * 3)
  private val pixels = IntArray(S * S)
  private val resized = Bitmap.createBitmap(S, S, Bitmap.Config.ARGB_8888)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
  private val mask = ByteArray(S * S)

  init { Log.i(TAG, "GPU compiled OK — ${inBufs.size} in / ${outBufs.size} out") }

  /** Returns an S*S byte mask (1 = person) + time (ms). */
  fun segment(bitmap: Bitmap): Pair<ByteArray, Long> {
    val t = System.nanoTime()
    Canvas(resized).drawBitmap(
      bitmap, matrix.apply { setScale(S.toFloat() / bitmap.width, S.toFloat() / bitmap.height) }, paint)
    resized.getPixels(pixels, 0, S, 0, 0, S, S)
    for (i in 0 until S * S) {
      val p = pixels[i]
      inputFloats[i * 3] = ((p and 0xFF) / 255f - 0.5f) / 0.5f              // B
      inputFloats[i * 3 + 1] = (((p shr 8) and 0xFF) / 255f - 0.5f) / 0.5f  // G
      inputFloats[i * 3 + 2] = (((p shr 16) and 0xFF) / 255f - 0.5f) / 0.5f // R
    }
    inBufs[0].writeFloat(inputFloats)
    model.run(inBufs, outBufs)
    val out = outBufs[0].readFloat()   // [S*S*2] NHWC: [bg, human] per pixel
    for (i in 0 until S * S) mask[i] = if (out[i * 2 + 1] > out[i * 2]) 1 else 0
    return mask to ((System.nanoTime() - t) / 1_000_000)
  }

  override fun close() {
    model.close()
    if (!resized.isRecycled) resized.recycle()
  }
}
