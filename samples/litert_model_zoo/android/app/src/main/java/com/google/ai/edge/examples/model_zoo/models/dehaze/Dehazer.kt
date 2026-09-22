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

package com.google.ai.edge.examples.model_zoo.models.dehaze

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
 * DehazeFormer-MCT image dehazing on LiteRT CompiledModel (GPU).
 *
 * The basenet (DehazeFormer, Swin-style windowed attention) runs on the GPU at a fixed 256x256 and
 * predicts 72 per-pixel curve parameters (3 out-channels x 3 in-channels x 8 curve levels). The
 * curves are then applied to the FULL-resolution frame host-side (trilinear lookup — the official
 * grid_sample step), so output resolution is independent of the network resolution.
 *
 * Input : [1, 3, 256, 256] NCHW, RGB in [-1, 1]. Output: [1, 72, 256, 256] curve parameters.
 */
class Dehazer(context: Context, modelFile: File, accelerator: Accelerator = Accelerator.GPU) :
  AutoCloseable {

  companion object {
    private const val TAG = "Dehaze"
    const val SIZE = 256
    private const val LEVELS = 8
    private const val PLANE = SIZE * SIZE
  }

  private val model: CompiledModel
  private val inBufs: List<TensorBuffer>
  private val outBufs: List<TensorBuffer>
  private val inputFloats = FloatArray(3 * PLANE)
  private val pixels256 = IntArray(PLANE)
  private val resized = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  // Full-res mapping caches (allocated per frame size, camera frames are stable)
  private var mapW = 0
  private var mapH = 0
  private lateinit var srcPixels: IntArray
  private lateinit var dstPixels: IntArray
  private lateinit var u0: IntArray
  private lateinit var u1: IntArray
  private lateinit var fu: FloatArray
  private lateinit var v0: IntArray
  private lateinit var v1: IntArray
  private lateinit var fv: FloatArray

  init {
    model = CompiledModel.create(modelFile.absolutePath, CompiledModel.Options(accelerator), null)
    inBufs = model.createInputBuffers()
    outBufs = model.createOutputBuffers()
    Log.i(TAG, "$accelerator compiled OK — ${inBufs.size} in / ${outBufs.size} out")
  }

  /** Runs DehazeFormer on [bitmap] and returns the dehazed full-res bitmap + time (ms). */
  fun dehaze(bitmap: Bitmap): Pair<Bitmap, Long> {
    val t = System.nanoTime()
    Canvas(resized)
      .drawBitmap(
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
    val param = outBufs[0].readFloat() // [72 * 256 * 256]
    val out = applyCurves(bitmap, param)
    return out to ((System.nanoTime() - t) / 1_000_000)
  }

  /**
   * Applies the predicted curves to the full-res frame: for each output channel c, out[c] = sum_i
   * trilinear(curve[c][i], depth = in_i, y, x) — the exact grid_sample(bilinear, border,
   * align_corners=true) of the official MCT mapping.
   */
  private fun applyCurves(bitmap: Bitmap, param: FloatArray): Bitmap {
    ensureMaps(bitmap.width, bitmap.height)
    val w = bitmap.width
    val h = bitmap.height
    bitmap.getPixels(srcPixels, 0, w, 0, 0, w, h)
    val depthScale = LEVELS - 1f // d = pixel/255 * 7, align_corners over 8 levels
    for (y in 0 until h) {
      val row = y * w
      val rv0 = v0[y] * SIZE
      val rv1 = v1[y] * SIZE
      val wy1 = fv[y]
      val wy0 = 1f - wy1
      for (x in 0 until w) {
        val p = srcPixels[row + x]
        val cu0 = u0[x]
        val cu1 = u1[x]
        val wx1 = fu[x]
        val wx0 = 1f - wx1
        val w00 = wy0 * wx0
        val w01 = wy0 * wx1
        val w10 = wy1 * wx0
        val w11 = wy1 * wx1
        val i00 = rv0 + cu0
        val i01 = rv0 + cu1
        val i10 = rv1 + cu0
        val i11 = rv1 + cu1
        var r = 0f
        var g = 0f
        var b = 0f
        for (ci in 0 until 3) {
          val px =
            when (ci) {
              0 -> (p shr 16) and 0xFF
              1 -> (p shr 8) and 0xFF
              else -> p and 0xFF
            }
          val d = px * depthScale / 255f
          val d0 = d.toInt().coerceAtMost(LEVELS - 1)
          val d1 = (d0 + 1).coerceAtMost(LEVELS - 1)
          val fd = d - d0
          for (c in 0 until 3) {
            val base = ((c * 3 + ci) * LEVELS)
            val p0 = (base + d0) * PLANE
            val p1 = (base + d1) * PLANE
            val s0 =
              param[p0 + i00] * w00 +
                param[p0 + i01] * w01 +
                param[p0 + i10] * w10 +
                param[p0 + i11] * w11
            val s1 =
              param[p1 + i00] * w00 +
                param[p1 + i01] * w01 +
                param[p1 + i10] * w10 +
                param[p1 + i11] * w11
            val s = s0 * (1f - fd) + s1 * fd
            when (c) {
              0 -> r += s
              1 -> g += s
              else -> b += s
            }
          }
        }
        dstPixels[row + x] =
          (0xFF shl 24) or (toByteRange(r) shl 16) or (toByteRange(g) shl 8) or toByteRange(b)
      }
    }
    return Bitmap.createBitmap(dstPixels, w, h, Bitmap.Config.ARGB_8888)
  }

  /** [-1,1] model output -> 0..255 display value. */
  internal fun toByteRange(value: Float): Int {
    val clamped = if (value < -1f) -1f else if (value > 1f) 1f else value
    return ((clamped * 0.5f + 0.5f) * 255f + 0.5f).toInt().coerceIn(0, 255)
  }

  /** Precomputes the align_corners=true bilinear taps from frame space to the 256 param grid. */
  private fun ensureMaps(w: Int, h: Int) {
    if (w == mapW && h == mapH) return
    mapW = w
    mapH = h
    srcPixels = IntArray(w * h)
    dstPixels = IntArray(w * h)
    u0 = IntArray(w)
    u1 = IntArray(w)
    fu = FloatArray(w)
    v0 = IntArray(h)
    v1 = IntArray(h)
    fv = FloatArray(h)
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
  }

  override fun close() {
    model.close()
    if (!resized.isRecycled) resized.recycle()
  }
}
