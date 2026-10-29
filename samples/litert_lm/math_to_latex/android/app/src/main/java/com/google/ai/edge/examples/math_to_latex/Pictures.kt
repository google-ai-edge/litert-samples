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

package com.google.ai.edge.examples.math_to_latex

import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/** An 8-bit RGB picture: [width] x [height] pixels, row-major, 3 bytes a pixel. */
class Rgb(val width: Int, val height: Int, val bytes: ByteArray) {
  init {
    require(width > 0 && height > 0) { "empty picture ${width}x$height" }
    require(bytes.size == width * height * 3) {
      "an RGB ${width}x$height picture holds ${width * height * 3} bytes, not ${bytes.size}"
    }
  }

  /** The pixels of the box [x0, x1) x [y0, y1). */
  fun crop(x0: Int, y0: Int, x1: Int, y1: Int): Rgb {
    require(0 <= x0 && x0 < x1 && x1 <= width && 0 <= y0 && y0 < y1 && y1 <= height) {
      "box ($x0, $y0, $x1, $y1) is not inside ${width}x$height"
    }
    val w = x1 - x0
    val out = ByteArray(w * (y1 - y0) * 3)
    for (y in y0 until y1) {
      System.arraycopy(bytes, (y * width + x0) * 3, out, (y - y0) * w * 3, w * 3)
    }
    return Rgb(w, y1 - y0, out)
  }

  fun resize(outWidth: Int, outHeight: Int): Rgb =
    Rgb(outWidth, outHeight, PilBicubic.resize(bytes, width, height, outWidth, outHeight))

  /**
   * Turns the picture upright by its EXIF orientation (1-8), as Pillow's `ImageOps.exif_transpose`
   * does. Any other value leaves it as it is.
   */
  fun orient(orientation: Int): Rgb {
    val w = width
    val h = height
    // (output width, output height, source x and y of the output pixel (x, y))
    val (ow, oh, from) =
      when (orientation) {
        2 -> Triple(w, h) { x: Int, y: Int -> (w - 1 - x) to y } // flip left-right
        3 -> Triple(w, h) { x: Int, y: Int -> (w - 1 - x) to (h - 1 - y) } // rotate 180
        4 -> Triple(w, h) { x: Int, y: Int -> x to (h - 1 - y) } // flip top-bottom
        5 -> Triple(h, w) { x: Int, y: Int -> y to x } // transpose
        6 -> Triple(h, w) { x: Int, y: Int -> y to (h - 1 - x) } // rotate 90 clockwise
        7 -> Triple(h, w) { x: Int, y: Int -> (w - 1 - y) to (h - 1 - x) } // transverse
        8 -> Triple(h, w) { x: Int, y: Int -> (w - 1 - y) to x } // rotate 90 counter-clockwise
        else -> return this
      }
    val out = ByteArray(ow * oh * 3)
    for (y in 0 until oh) {
      for (x in 0 until ow) {
        val (sx, sy) = from(x, y)
        System.arraycopy(bytes, (sy * w + sx) * 3, out, (y * ow + x) * 3, 3)
      }
    }
    return Rgb(ow, oh, out)
  }

  fun sha256(): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
}

/** The pictures of one request and the layout the chat template needs for them. */
class Prepared(val pictures: List<Rgb>, val sliceCols: Int?) {
  /** The extra context of the message: `slice_cols` only when the pictures are a tiled page. */
  val extraContext: Map<String, Any>
    get() = if (sliceCols == null) emptyMap() else mapOf("slice_cols" to sliceCols)
}

/**
 * The model card's page script (`ocr_page.py`, linked from the card), in Kotlin: the bundle resizes
 * every picture to 448 x 448 without keeping the aspect ratio, so the app prepares each picture at
 * 448 x 448 itself.
 * - A region (an area up to twice 448 x 448): centred on a white square, then resized to 448 x 448.
 * - A larger picture: the model processor's own grid (an overview plus at most 9 slices), the whole
 *   picture padded to a white square as the overview, then every grid cell resized to 448 x 448,
 *   sent as the pictures of one message with `slice_cols` (the grid's columns) as extra context,
 *   which the bundle's chat template reads to mark the 2nd and later pictures as slices, row by
 *   row.
 */
object Pictures {
  /** The bundle's picture size: 448 x 448 = 256 image tokens. */
  const val SIZE = 448

  /** MiniCPMOcrImageProcessor.max_slice_nums. */
  const val MAX_SLICES = 9

  /** A picture larger than this many times 448 x 448 is tiled. */
  const val TILE_ABOVE = 2.0

  /** The most pictures one request sends: the overview and [MAX_SLICES] slices. */
  const val MAX_PICTURES = MAX_SLICES + 1

  private const val WHITE = 0xFF.toByte()

  /**
   * [rows, cols] of the processor's slicing, or null when it does not slice (transformers
   * MiniCPMOcrImageProcessor.get_sliced_grid, Apache-2.0, the same arithmetic).
   */
  fun slicedGrid(
    width: Int,
    height: Int,
    maxSlices: Int = MAX_SLICES,
    scale: Int = SIZE,
  ): IntArray? {
    val logRatio = ln(width.toDouble() / height)
    val ratio = width.toDouble() * height / (scale.toDouble() * scale)
    val multiple = min(ceil(ratio).toInt(), maxSlices)
    if (multiple <= 1) return null
    var best = intArrayOf(1, 1)
    var minError = Double.POSITIVE_INFINITY
    for (numSlices in intArrayOf(multiple - 1, multiple, multiple + 1)) {
      if (numSlices == 1 || numSlices > maxSlices) continue
      for (rows in 1..numSlices) {
        if (numSlices % rows == 0) {
          val cols = numSlices / rows
          val error = abs(logRatio - ln(cols.toDouble() / rows))
          if (error < minError || (error == minError && rows > best[0])) {
            best = intArrayOf(rows, cols)
            minError = error
          }
        }
      }
    }
    return best
  }

  /** The picture centred on a white square (Pillow's paste offsets), resized to [size] x [size]. */
  fun padSquare(img: Rgb, size: Int = SIZE): Rgb {
    val side = max(img.width, img.height)
    val square = ByteArray(side * side * 3) { WHITE }
    val left = (side - img.width) / 2
    val top = (side - img.height) / 2
    for (y in 0 until img.height) {
      System.arraycopy(
        img.bytes,
        y * img.width * 3,
        square,
        ((top + y) * side + left) * 3,
        img.width * 3,
      )
    }
    return Rgb(side, side, square).resize(size, size)
  }

  /** Python's round(): halves to even. */
  private fun pyRound(v: Double): Int = Math.rint(v).toInt()

  /** Every cell of [grid] (rows, cols) resized to [size] x [size], row by row. */
  fun cells(img: Rgb, grid: IntArray, size: Int = SIZE): List<Rgb> {
    val (rows, cols) = grid[0] to grid[1]
    val w = img.width.toDouble()
    val h = img.height.toDouble()
    val out = ArrayList<Rgb>(rows * cols)
    for (r in 0 until rows) {
      for (c in 0 until cols) {
        val crop =
          img.crop(
            pyRound(w * c / cols),
            pyRound(h * r / rows),
            pyRound(w * (c + 1) / cols),
            pyRound(h * (r + 1) / rows),
          )
        out += crop.resize(size, size)
      }
    }
    return out
  }

  /**
   * The pictures for [img]: one padded square, or (when [img] is larger than [TILE_ABOVE] times 448
   * x 448 and the processor would slice it) the overview and the grid's cells with their layout.
   */
  fun prepare(img: Rgb, tile: Boolean = true): Prepared {
    val grid = slicedGrid(img.width, img.height)
    val big = img.width.toDouble() * img.height > TILE_ABOVE * SIZE * SIZE
    if (grid == null || !tile || !big) return Prepared(listOf(padSquare(img)), null)
    return Prepared(listOf(padSquare(img)) + cells(img, grid), grid[1])
  }
}
