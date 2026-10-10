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

package com.google.ai.edge.examples.twinlitenet

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

/**
 * TwinLiteNet drivable-area + lane-line segmentation on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 360, 640] NCHW, RGB, x/255.
 * Output: two [1, 2, 360, 640] logit maps — drivable_area and lane_line (argmax over 2 classes).
 *
 * ESPNet-based, pure CNN, fully on the GPU delegate (ConvTranspose2d → ZeroStuffConvT2d). 3.1 MB.
 */
class TwinLiteSegmenter(modelPath: String) : AutoCloseable {

  companion object {
    private const val TAG = "TwinLiteNet"
    const val W = 640
    const val H = 360
  }

  private val model = CompiledModel.create(modelPath, CompiledModel.Options(Accelerator.GPU), null)
  private val inBufs: List<TensorBuffer> = model.createInputBuffers()
  private val outBufs: List<TensorBuffer> = model.createOutputBuffers()

  private val inputFloats = FloatArray(3 * H * W)
  private val pixels = IntArray(W * H)
  private val resized = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_8888)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
  private val daMask = ByteArray(W * H)
  private val llMask = ByteArray(W * H)

  init { Log.i(TAG, "GPU compiled OK — ${inBufs.size} in / ${outBufs.size} out") }

  /** Returns (drivableMask, laneMask) each W*H bytes (1 = foreground) + time (ms). */
  fun segment(bitmap: Bitmap): Triple<ByteArray, ByteArray, Long> {
    val t = System.nanoTime()
    Canvas(resized).drawBitmap(
      bitmap,
      matrix.apply { setScale(W.toFloat() / bitmap.width, H.toFloat() / bitmap.height) },
      paint)
    resized.getPixels(pixels, 0, W, 0, 0, W, H)
    val plane = W * H
    for (i in 0 until plane) {
      val p = pixels[i]
      inputFloats[i] = ((p shr 16) and 0xFF) / 255f
      inputFloats[plane + i] = ((p shr 8) and 0xFF) / 255f
      inputFloats[2 * plane + i] = (p and 0xFF) / 255f
    }
    inBufs[0].writeFloat(inputFloats)
    model.run(inBufs, outBufs)
    val da = outBufs[0].readFloat()
    val ll = outBufs[1].readFloat()
    for (i in 0 until plane) {
      daMask[i] = if (da[plane + i] > da[i]) 1 else 0
      llMask[i] = if (ll[plane + i] > ll[i]) 1 else 0
    }
    return Triple(daMask, llMask, (System.nanoTime() - t) / 1_000_000)
  }

  override fun close() {
    model.close()
    if (!resized.isRecycled) {
      resized.recycle()
    }
  }
}
