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

// Port of WeVisOne-VL's _masks_from_logits (Tencent, Apache-2.0; see NOTICE).
package com.google.ai.edge.examples.image_cutout

import kotlin.math.exp

/**
 * The mask decoder's logits to the photo's alpha matte, as `WeVisOneVL._masks_from_logits` does it:
 * sigmoid at 1024 x 1024, then bilinear (half-pixel centres, align_corners False, like torch's
 * upsample_bilinear2d) to the photo's height x width, in float32.
 */
object AlphaMath {
  fun sigmoid(logits: FloatArray): FloatArray =
    FloatArray(logits.size) { 1f / (1f + exp(-logits[it])) }

  private class Axis(val i0: IntArray, val i1: IntArray, val l0: FloatArray, val l1: FloatArray)

  private fun axis(outSize: Int, inSize: Int): Axis {
    val scale = inSize.toFloat() / outSize.toFloat()
    val i0 = IntArray(outSize)
    val i1 = IntArray(outSize)
    val l0 = FloatArray(outSize)
    val l1 = FloatArray(outSize)
    for (d in 0 until outSize) {
      val src = maxOf(scale * (d + 0.5f) - 0.5f, 0f)
      val lo = minOf(src.toInt(), inSize - 1)
      val lambda = (src - lo).coerceIn(0f, 1f)
      i0[d] = lo
      i1[d] = if (lo < inSize - 1) lo + 1 else lo
      l1[d] = lambda
      l0[d] = 1f - lambda
    }
    return Axis(i0, i1, l0, l1)
  }

  /** [src] = inHeight x inWidth row-major -> outHeight x outWidth. */
  fun resizeBilinear(
    src: FloatArray,
    inHeight: Int,
    inWidth: Int,
    outHeight: Int,
    outWidth: Int,
  ): FloatArray {
    require(src.size == inHeight * inWidth)
    val ya = axis(outHeight, inHeight)
    val xa = axis(outWidth, inWidth)
    val out = FloatArray(outHeight * outWidth)
    for (y in 0 until outHeight) {
      val r0 = ya.i0[y] * inWidth
      val r1 = ya.i1[y] * inWidth
      val h0 = ya.l0[y]
      val h1 = ya.l1[y]
      val o = y * outWidth
      for (x in 0 until outWidth) {
        val c0 = xa.i0[x]
        val c1 = xa.i1[x]
        val w0 = xa.l0[x]
        val w1 = xa.l1[x]
        out[o + x] =
          h0 * (w0 * src[r0 + c0] + w1 * src[r0 + c1]) +
            h1 * (w0 * src[r1 + c0] + w1 * src[r1 + c1])
      }
    }
    return out
  }

  /** The photo's alpha matte from the [size] x [size] logits. */
  fun alpha(logits: FloatArray, size: Int, height: Int, width: Int): FloatArray =
    resizeBilinear(sigmoid(logits), size, size, height, width)
}
