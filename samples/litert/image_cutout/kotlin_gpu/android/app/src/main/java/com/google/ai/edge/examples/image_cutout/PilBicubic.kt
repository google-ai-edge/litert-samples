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

// Port of the bicubic resampling of Pillow 12.2.0 (MIT-CMU License; see NOTICE).
package com.google.ai.edge.examples.image_cutout

import kotlin.math.ceil

/**
 * Pillow 12.2.0's `Image.resize(size, Image.BICUBIC)` for 8-bit RGB, pixel for pixel: the reference
 * ran the photo through it to 512 x 512 for the vision encoder and to 1024 x 1024 for the mask
 * decoder, and the vision encoder gives different tokens for other resamplers (in the conversion's
 * checks, a runtime resize with another bicubic kernel changed 28 of 29 answers).
 *
 * The algorithm of Pillow's src/libImaging/Resample.c (12.2.0): a bicubic kernel (a = -0.5) widened
 * by the scale when shrinking, coefficients normalized to sum 1 and rounded to 22-bit fixed point,
 * a horizontal pass over the rows the vertical pass reads, rounded and clipped to 8 bits, then the
 * vertical pass. (Pillow 13 runs the vertical pass first when the height shrinks much more than the
 * width, which changes some pixels; the reference used 12.2.0.)
 *
 * Pillow: Copyright (c) 1997-2011 by Secret Labs AB, Copyright (c) 1995-2011 by Fredrik Lundh and
 * contributors, Copyright (c) 2010 by Jeffrey 'Alex' Clark and contributors; MIT-CMU License
 * (licenses/Pillow-MIT-CMU.txt).
 */
object PilBicubic {
  private const val PRECISION_BITS = 32 - 8 - 2

  private class Coeffs(val ksize: Int, val bounds: IntArray, val kk: IntArray)

  private fun bicubic(x0: Double): Double {
    val a = -0.5
    val x = if (x0 < 0.0) -x0 else x0
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a
    return 0.0
  }

  /** precompute_coeffs + normalize_coeffs_8bpc for box [0, inSize) -> outSize. */
  private fun coeffs(inSize: Int, outSize: Int): Coeffs {
    val scale = inSize.toFloat().toDouble() / outSize
    val filterscale = if (scale < 1.0) 1.0 else scale
    val support = 2.0 * filterscale
    val ksize = ceil(support).toInt() * 2 + 1
    val pre = DoubleArray(outSize * ksize)
    val bounds = IntArray(outSize * 2)
    val ss = 1.0 / filterscale
    for (xx in 0 until outSize) {
      val center = (xx + 0.5) * scale
      var ww = 0.0
      var xmin = (center - support + 0.5).toInt()
      if (xmin < 0) {
        xmin = 0
      }
      var xmax = (center + support + 0.5).toInt()
      if (xmax > inSize) {
        xmax = inSize
      }
      xmax -= xmin
      val base = xx * ksize
      for (x in 0 until xmax) {
        val w = bicubic((x + xmin - center + 0.5) * ss)
        pre[base + x] = w
        ww += w
      }
      if (ww != 0.0) {
        for (x in 0 until xmax) {
          pre[base + x] /= ww
        }
      }
      bounds[xx * 2] = xmin
      bounds[xx * 2 + 1] = xmax
    }
    val one = (1 shl PRECISION_BITS).toDouble()
    val kk =
      IntArray(pre.size) { i ->
        val v = pre[i]
        if (v < 0) (-0.5 + v * one).toInt() else (0.5 + v * one).toInt()
      }
    return Coeffs(ksize, bounds, kk)
  }

  private fun clip8(acc: Int): Int {
    val v = acc shr PRECISION_BITS
    return if (v < 0) 0 else if (v > 255) 255 else v
  }

  /**
   * [rgb] = width x height x 3 bytes, row-major; returns outWidth x outHeight x 3 bytes. A
   * dimension that does not change skips its pass, as in Pillow.
   */
  fun resize(rgb: ByteArray, width: Int, height: Int, outWidth: Int, outHeight: Int): ByteArray {
    require(rgb.size == width * height * 3) {
      "rgb holds ${rgb.size} bytes, not ${width}x${height}x3"
    }
    val needHorizontal = outWidth != width
    val needVertical = outHeight != height
    val horiz = coeffs(width, outWidth)
    val vert = coeffs(height, outHeight)
    val yFirst = vert.bounds[0]
    val yLast = vert.bounds[outHeight * 2 - 2] + vert.bounds[outHeight * 2 - 1]

    var src = rgb
    var srcWidth = width
    if (needHorizontal) {
      // Shift the vertical bounds to the rows the horizontal pass produces.
      for (i in 0 until outHeight) {
        vert.bounds[i * 2] -= yFirst
      }
      val tempHeight = yLast - yFirst
      val temp = ByteArray(outWidth * tempHeight * 3)
      val half = 1 shl (PRECISION_BITS - 1)
      for (yy in 0 until tempHeight) {
        val inRow = (yy + yFirst) * width * 3
        val outRow = yy * outWidth * 3
        for (xx in 0 until outWidth) {
          val xmin = horiz.bounds[xx * 2]
          val xmax = horiz.bounds[xx * 2 + 1]
          val k = xx * horiz.ksize
          var s0 = half
          var s1 = half
          var s2 = half
          var p = inRow + xmin * 3
          for (x in 0 until xmax) {
            val c = horiz.kk[k + x]
            s0 += (rgb[p].toInt() and 0xFF) * c
            s1 += (rgb[p + 1].toInt() and 0xFF) * c
            s2 += (rgb[p + 2].toInt() and 0xFF) * c
            p += 3
          }
          val o = outRow + xx * 3
          temp[o] = clip8(s0).toByte()
          temp[o + 1] = clip8(s1).toByte()
          temp[o + 2] = clip8(s2).toByte()
        }
      }
      src = temp
      srcWidth = outWidth
    }
    if (!needVertical) return if (src === rgb) rgb.copyOf() else src

    val out = ByteArray(srcWidth * outHeight * 3)
    val half = 1 shl (PRECISION_BITS - 1)
    for (yy in 0 until outHeight) {
      val ymin = vert.bounds[yy * 2]
      val ymax = vert.bounds[yy * 2 + 1]
      val k = yy * vert.ksize
      val outRow = yy * srcWidth * 3
      for (xx in 0 until srcWidth) {
        var s0 = half
        var s1 = half
        var s2 = half
        var p = (ymin * srcWidth + xx) * 3
        val stride = srcWidth * 3
        for (y in 0 until ymax) {
          val c = vert.kk[k + y]
          s0 += (src[p].toInt() and 0xFF) * c
          s1 += (src[p + 1].toInt() and 0xFF) * c
          s2 += (src[p + 2].toInt() and 0xFF) * c
          p += stride
        }
        val o = outRow + xx * 3
        out[o] = clip8(s0).toByte()
        out[o + 1] = clip8(s1).toByte()
        out[o + 2] = clip8(s2).toByte()
      }
    }
    return out
  }
}
