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

// Port of the bicubic resampling of Pillow 13.0.0 (MIT-CMU License; see NOTICE). Adapted from the
// Pillow 12.2.0 port of the image cutout sample proposed in google-ai-edge/litert-samples#490
// (samples/litert/image_cutout/kotlin_gpu), with Pillow 13's choice of which pass runs first.
package com.google.ai.edge.examples.math_to_latex

import kotlin.math.ceil

/**
 * Pillow 13.0.0's `Image.resize(size, Image.BICUBIC)` for 8-bit RGB, pixel for pixel. The model
 * card's page script (and the checks behind the card's numbers) prepare every picture with Pillow,
 * so the app prepares the same pixels.
 *
 * The algorithm of Pillow's src/libImaging/Resample.c (13.0.0): a bicubic kernel (a = -0.5) widened
 * by the scale when shrinking, coefficients normalized to sum 1 and rounded to 22-bit fixed point,
 * and two passes, each rounded and clipped to 8 bits. The horizontal pass runs first (over only the
 * rows the vertical pass reads), unless the height shrinks by more than twice as many pixels as the
 * width does; then the vertical pass runs first, over every column.
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
   * Horizontal pass: [rows] rows of [src] ([srcWidth] wide), starting at row [firstRow], to
   * [outWidth] columns.
   */
  private fun horizontal(
    src: ByteArray,
    srcWidth: Int,
    firstRow: Int,
    rows: Int,
    outWidth: Int,
    c: Coeffs,
  ): ByteArray {
    val out = ByteArray(outWidth * rows * 3)
    val half = 1 shl (PRECISION_BITS - 1)
    for (yy in 0 until rows) {
      val inRow = (yy + firstRow) * srcWidth * 3
      val outRow = yy * outWidth * 3
      for (xx in 0 until outWidth) {
        val xmin = c.bounds[xx * 2]
        val xmax = c.bounds[xx * 2 + 1]
        val k = xx * c.ksize
        var s0 = half
        var s1 = half
        var s2 = half
        var p = inRow + xmin * 3
        for (x in 0 until xmax) {
          val w = c.kk[k + x]
          s0 += (src[p].toInt() and 0xFF) * w
          s1 += (src[p + 1].toInt() and 0xFF) * w
          s2 += (src[p + 2].toInt() and 0xFF) * w
          p += 3
        }
        val o = outRow + xx * 3
        out[o] = clip8(s0).toByte()
        out[o + 1] = clip8(s1).toByte()
        out[o + 2] = clip8(s2).toByte()
      }
    }
    return out
  }

  /** Vertical pass: every column of [src] ([width] wide) to [outHeight] rows. */
  private fun vertical(src: ByteArray, width: Int, outHeight: Int, c: Coeffs): ByteArray {
    val out = ByteArray(width * outHeight * 3)
    val half = 1 shl (PRECISION_BITS - 1)
    val stride = width * 3
    for (yy in 0 until outHeight) {
      val ymin = c.bounds[yy * 2]
      val ymax = c.bounds[yy * 2 + 1]
      val k = yy * c.ksize
      val outRow = yy * stride
      for (xx in 0 until width) {
        var s0 = half
        var s1 = half
        var s2 = half
        var p = ymin * stride + xx * 3
        for (y in 0 until ymax) {
          val w = c.kk[k + y]
          s0 += (src[p].toInt() and 0xFF) * w
          s1 += (src[p + 1].toInt() and 0xFF) * w
          s2 += (src[p + 2].toInt() and 0xFF) * w
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

  /**
   * [rgb] = width x height x 3 bytes, row-major; returns outWidth x outHeight x 3 bytes. A
   * dimension that does not change skips its pass, as in Pillow.
   */
  fun resize(rgb: ByteArray, width: Int, height: Int, outWidth: Int, outHeight: Int): ByteArray {
    require(rgb.size == width * height * 3) {
      "rgb holds ${rgb.size} bytes, not ${width}x${height}x3"
    }
    require(outWidth > 0 && outHeight > 0) { "empty output size ${outWidth}x$outHeight" }
    val needHorizontal = outWidth != width
    val needVertical = outHeight != height
    if (!needHorizontal && !needVertical) return rgb.copyOf()
    val horizontalFirst =
      !((height - outHeight) > 0 && (height - outHeight) > (width - outWidth) * 2)
    val vert = coeffs(height, outHeight)
    if (horizontalFirst) {
      var src = rgb
      var srcWidth = width
      if (needHorizontal) {
        val horiz = coeffs(width, outWidth)
        val yFirst = vert.bounds[0]
        val yLast = vert.bounds[outHeight * 2 - 2] + vert.bounds[outHeight * 2 - 1]
        // Shift the vertical bounds to the rows the horizontal pass produces.
        for (i in 0 until outHeight) {
          vert.bounds[i * 2] -= yFirst
        }
        src = horizontal(rgb, width, yFirst, yLast - yFirst, outWidth, horiz)
        srcWidth = outWidth
      }
      return if (needVertical) vertical(src, srcWidth, outHeight, vert) else src
    }
    // The height shrinks the most: the vertical pass first, over every column.
    val tall = vertical(rgb, width, outHeight, vert)
    return if (needHorizontal) {
      horizontal(tall, width, 0, outHeight, outWidth, coeffs(width, outWidth))
    } else {
      tall
    }
  }
}
