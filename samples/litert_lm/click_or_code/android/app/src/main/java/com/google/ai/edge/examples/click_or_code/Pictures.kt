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

package com.google.ai.edge.examples.click_or_code

import java.security.MessageDigest

/** An 8-bit RGB picture: [width] x [height] pixels, row-major, 3 bytes a pixel. */
class Rgb(val width: Int, val height: Int, val bytes: ByteArray) {
  init {
    require(width > 0 && height > 0) { "empty picture ${width}x$height" }
    require(bytes.size == width * height * 3) {
      "an RGB ${width}x$height picture holds ${width * height * 3} bytes, not ${bytes.size}"
    }
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

/**
 * The picture the model file reads: one 1920 x 1088 frame (2,040 image tokens), the grid the model's
 * own processor gives a 1920 x 1080 desktop screenshot. LiteRT-LM resizes every picture to that size
 * without keeping its aspect ratio, so the app makes the frame itself, with Pillow's bicubic filter,
 * as the model card's script does (`cowork_step.py`); portrait phone screens are stretched too. The
 * model's point is a fraction of the whole frame, so the stretch does not move it relative to what
 * is on the screen.
 */
object Pictures {
  const val WIDTH = 1920
  const val HEIGHT = 1088

  fun stretch(img: Rgb): Rgb = if (img.width == WIDTH && img.height == HEIGHT) img else img.resize(WIDTH, HEIGHT)
}
