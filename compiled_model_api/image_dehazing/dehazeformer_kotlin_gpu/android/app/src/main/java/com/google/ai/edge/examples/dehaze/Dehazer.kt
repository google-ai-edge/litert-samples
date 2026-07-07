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

package com.google.ai.edge.examples.dehaze

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

/**
 * DehazeFormer-MCT image dehazing on LiteRT CompiledModel (GPU).
 *
 * The basenet (Swin-style windowed attention) runs on the GPU at a fixed 256x256 and
 * predicts 72 per-pixel curve parameters (3 out-channels x 3 in-channels x 8 levels).
 * The curves are applied to the FULL-resolution image host-side (trilinear lookup, the
 * official grid_sample mapping), so output resolution is independent of the network.
 *
 * Input : [1, 3, 256, 256] NCHW, RGB in [-1, 1].
 * Output: [1, 72, 256, 256] curve parameters.
 */
class Dehazer(modelPath: String) : AutoCloseable {

  companion object {
    private const val TAG = "Dehaze"
    const val SIZE = 256
    private const val LEVELS = 8
    private const val PLANE = SIZE * SIZE
  }

  private val model = CompiledModel.create(modelPath, CompiledModel.Options(Accelerator.GPU), null)
  private val inBufs: List<TensorBuffer> = model.createInputBuffers()
  private val outBufs: List<TensorBuffer> = model.createOutputBuffers()
  private val inputFloats = FloatArray(3 * PLANE)
  private val pixels256 = IntArray(PLANE)
  private val resized = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  init {
    Log.i(TAG, "GPU compiled OK — ${inBufs.size} in / ${outBufs.size} out")
  }

  /** Runs DehazeFormer on [bitmap] and returns the dehazed full-res bitmap + time (ms). */
  fun dehaze(bitmap: Bitmap): Pair<Bitmap, Long> {
    val start = System.nanoTime()
    Canvas(resized).drawBitmap(
      bitmap,
      matrix.apply { setScale(SIZE.toFloat() / bitmap.width, SIZE.toFloat() / bitmap.height) },
      paint,
    )
    resized.getPixels(pixels256, 0, SIZE, 0, 0, SIZE, SIZE)
    for (i in 0 until PLANE) {
      val p = pixels256[i]
      inputFloats[i] = ((p shr 16) and 0xFF) / 127.5f - 1f
      inputFloats[PLANE + i] = ((p shr 8) and 0xFF) / 127.5f - 1f
      inputFloats[2 * PLANE + i] = (p and 0xFF) / 127.5f - 1f
    }
    inBufs[0].writeFloat(inputFloats)
    model.run(inBufs, outBufs)
    val param = outBufs[0].readFloat()
    val out = applyCurves(bitmap, param)
    return out to ((System.nanoTime() - start) / 1_000_000)
  }

  /**
   * Applies the predicted curves to the full-res image: for each output channel c,
   * out[c] = sum_i trilinear(curve[c][i], depth = in_i, y, x) — the exact
   * grid_sample(bilinear, border, align_corners=true) of the official MCT mapping.
   */
  private fun applyCurves(bitmap: Bitmap, param: FloatArray): Bitmap {
    val w = bitmap.width
    val h = bitmap.height
    val srcPixels = IntArray(w * h)
    val dstPixels = IntArray(w * h)
    bitmap.getPixels(srcPixels, 0, w, 0, 0, w, h)
    // align_corners=true bilinear taps from image space to the 256x256 param grid
    val u0 = IntArray(w)
    val u1 = IntArray(w)
    val fu = FloatArray(w)
    val v0 = IntArray(h)
    val v1 = IntArray(h)
    val fv = FloatArray(h)
    for (x in 0 until w) {
      val u = x.toFloat() / (w - 1) * (SIZE - 1)
      u0[x] = u.toInt()
      u1[x] = (u0[x] + 1).coerceAtMost(SIZE - 1)
      fu[x] = u - u0[x]
    }
    for (y in 0 until h) {
      val v = y.toFloat() / (h - 1) * (SIZE - 1)
      v0[y] = v.toInt()
      v1[y] = (v0[y] + 1).coerceAtMost(SIZE - 1)
      fv[y] = v - v0[y]
    }
    val depthScale = LEVELS - 1f
    for (y in 0 until h) {
      val row = y * w
      val rv0 = v0[y] * SIZE
      val rv1 = v1[y] * SIZE
      val wy1 = fv[y]
      val wy0 = 1f - wy1
      for (x in 0 until w) {
        val p = srcPixels[row + x]
        val wx1 = fu[x]
        val wx0 = 1f - wx1
        val w00 = wy0 * wx0
        val w01 = wy0 * wx1
        val w10 = wy1 * wx0
        val w11 = wy1 * wx1
        val i00 = rv0 + u0[x]
        val i01 = rv0 + u1[x]
        val i10 = rv1 + u0[x]
        val i11 = rv1 + u1[x]
        var r = 0f
        var g = 0f
        var b = 0f
        for (ci in 0 until 3) {
          val px = when (ci) {
            0 -> (p shr 16) and 0xFF
            1 -> (p shr 8) and 0xFF
            else -> p and 0xFF
          }
          val d = px * depthScale / 255f
          val d0 = d.toInt().coerceAtMost(LEVELS - 1)
          val d1 = (d0 + 1).coerceAtMost(LEVELS - 1)
          val fd = d - d0
          for (c in 0 until 3) {
            val base = (c * 3 + ci) * LEVELS
            val p0 = (base + d0) * PLANE
            val p1 = (base + d1) * PLANE
            val s0 = param[p0 + i00] * w00 + param[p0 + i01] * w01 +
              param[p0 + i10] * w10 + param[p0 + i11] * w11
            val s1 = param[p1 + i00] * w00 + param[p1 + i01] * w01 +
              param[p1 + i10] * w10 + param[p1 + i11] * w11
            val s = s0 * (1f - fd) + s1 * fd
            when (c) {
              0 -> r += s
              1 -> g += s
              else -> b += s
            }
          }
        }
        dstPixels[row + x] = (0xFF shl 24) or
          (toByteRange(r) shl 16) or (toByteRange(g) shl 8) or toByteRange(b)
      }
    }
    return Bitmap.createBitmap(dstPixels, w, h, Bitmap.Config.ARGB_8888)
  }

  /** [-1,1] model output -> 0..255 display value. */
  private fun toByteRange(value: Float): Int {
    val clamped = if (value < -1f) -1f else if (value > 1f) 1f else value
    return ((clamped * 0.5f + 0.5f) * 255f + 0.5f).toInt().coerceIn(0, 255)
  }

  override fun close() {
    model.close()
    if (!resized.isRecycled) {
      resized.recycle()
    }
  }
}
