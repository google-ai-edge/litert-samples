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

package com.google.ai.edge.examples.dewarpnet

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

/**
 * DewarpNet document unwarping on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 256, 256] NCHW, BGR, x/255.
 * Output: [1, 2, 128, 128] backward-mapping grid (~[-1,1]).
 *
 * WCNet (UNet) → BMNet (DenseNet), pure CNN, fully on the GPU delegate. The grid_sample
 * unwarp (bilinear) runs host-side.
 */
class DocumentDewarper(modelPath: String) : AutoCloseable {

  companion object {
    private const val TAG = "DewarpNet"
    const val SIZE = 256; const val BM = 128; const val OUT = 384
  }

  private val model = CompiledModel.create(modelPath, CompiledModel.Options(Accelerator.GPU), null)
  private val inBufs: List<TensorBuffer> = model.createInputBuffers()
  private val outBufs: List<TensorBuffer> = model.createOutputBuffers()

  private val inputFloats = FloatArray(3 * SIZE * SIZE)
  private val inPixels = IntArray(SIZE * SIZE)
  private val resized = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
  private var srcPixels = IntArray(0); private var srcW = 0; private var srcH = 0
  private val outPixels = IntArray(OUT * OUT)

  init { Log.i(TAG, "GPU compiled OK — ${inBufs.size} in / ${outBufs.size} out") }

  fun dewarp(bitmap: Bitmap): Pair<Bitmap, Long> {
    val t = System.nanoTime()
    Canvas(resized).drawBitmap(
      bitmap, matrix.apply { setScale(SIZE.toFloat() / bitmap.width, SIZE.toFloat() / bitmap.height) }, paint)
    resized.getPixels(inPixels, 0, SIZE, 0, 0, SIZE, SIZE)
    val plane = SIZE * SIZE
    for (i in 0 until plane) {
      val p = inPixels[i]
      inputFloats[i] = (p and 0xFF) / 255f
      inputFloats[plane + i] = ((p shr 8) and 0xFF) / 255f
      inputFloats[2 * plane + i] = ((p shr 16) and 0xFF) / 255f
    }
    inBufs[0].writeFloat(inputFloats)
    model.run(inBufs, outBufs)
    val bm = outBufs[0].readFloat()

    if (srcW != bitmap.width || srcH != bitmap.height) {
      srcW = bitmap.width; srcH = bitmap.height; srcPixels = IntArray(srcW * srcH)
    }
    bitmap.getPixels(srcPixels, 0, srcW, 0, 0, srcW, srcH)

    val bmPlane = BM * BM
    for (oy in 0 until OUT) {
      val by = oy * (BM - 1f) / (OUT - 1)
      for (ox in 0 until OUT) {
        val bx = ox * (BM - 1f) / (OUT - 1)
        val gx = sampleBM(bm, 0, bmPlane, bx, by); val gy = sampleBM(bm, 1, bmPlane, bx, by)
        outPixels[oy * OUT + ox] = sampleSrc((gx + 1f) * 0.5f * (srcW - 1), (gy + 1f) * 0.5f * (srcH - 1))
      }
    }
    return Bitmap.createBitmap(outPixels, OUT, OUT, Bitmap.Config.ARGB_8888) to ((System.nanoTime() - t) / 1_000_000)
  }

  private fun sampleBM(bm: FloatArray, plane: Int, bmPlane: Int, x: Float, y: Float): Float {
    val x0 = x.toInt().coerceIn(0, BM - 1); val y0 = y.toInt().coerceIn(0, BM - 1)
    val x1 = (x0 + 1).coerceAtMost(BM - 1); val y1 = (y0 + 1).coerceAtMost(BM - 1)
    val fx = x - x0; val fy = y - y0; val base = plane * bmPlane
    val v00 = bm[base + y0 * BM + x0]; val v01 = bm[base + y0 * BM + x1]
    val v10 = bm[base + y1 * BM + x0]; val v11 = bm[base + y1 * BM + x1]
    return (v00 * (1 - fx) + v01 * fx) * (1 - fy) + (v10 * (1 - fx) + v11 * fx) * fy
  }

  private fun sampleSrc(x: Float, y: Float): Int {
    if (x < 0 || y < 0 || x > srcW - 1 || y > srcH - 1) return 0xFF000000.toInt()
    val x0 = x.toInt(); val y0 = y.toInt()
    val x1 = (x0 + 1).coerceAtMost(srcW - 1); val y1 = (y0 + 1).coerceAtMost(srcH - 1)
    val fx = x - x0; val fy = y - y0
    val p00 = srcPixels[y0 * srcW + x0]; val p01 = srcPixels[y0 * srcW + x1]
    val p10 = srcPixels[y1 * srcW + x0]; val p11 = srcPixels[y1 * srcW + x1]
    val r = l2((p00 shr 16) and 0xFF, (p01 shr 16) and 0xFF, (p10 shr 16) and 0xFF, (p11 shr 16) and 0xFF, fx, fy)
    val g = l2((p00 shr 8) and 0xFF, (p01 shr 8) and 0xFF, (p10 shr 8) and 0xFF, (p11 shr 8) and 0xFF, fx, fy)
    val b = l2(p00 and 0xFF, p01 and 0xFF, p10 and 0xFF, p11 and 0xFF, fx, fy)
    return (0xFF shl 24) or (r shl 16) or (g shl 8) or b
  }

  private fun l2(a: Int, b: Int, c: Int, d: Int, fx: Float, fy: Float): Int =
    ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy).toInt().coerceIn(0, 255)

  override fun close() {
    model.close()
    if (!resized.isRecycled) resized.recycle()
  }
}
