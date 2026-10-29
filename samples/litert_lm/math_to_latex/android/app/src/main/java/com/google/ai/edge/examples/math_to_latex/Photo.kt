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

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.exifinterface.media.ExifInterface
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

/** Decoding the user's picture and encoding the 448 x 448 pictures the runtime reads. */
object Photo {
  /** Pictures above this many pixels on their long side are decoded at a power-of-two scale. */
  const val MAX_SIDE = 4096

  /**
   * Decodes [bytes] (any format BitmapFactory reads), turns it upright by its EXIF orientation and
   * returns its pixels as they are stored: no color-space conversion, as Pillow reads them. A
   * picture longer than [MAX_SIDE] is decoded at the smallest power-of-two scale that fits.
   */
  fun decode(bytes: ByteArray): Rgb {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
    if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
      throw IllegalArgumentException("not a picture this phone can decode")
    }
    var sample = 1
    while (maxOf(bounds.outWidth, bounds.outHeight) / sample > MAX_SIDE) {
      sample *= 2
    }
    val options =
      BitmapFactory.Options().apply {
        inPreferredConfig = Bitmap.Config.ARGB_8888
        inSampleSize = sample
        inScaled = false
      }
    val bitmap =
      BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options)
        ?: throw IllegalArgumentException("not a picture this phone can decode")
    val rgb =
      try {
        toRgb(bitmap)
      } finally {
        bitmap.recycle()
      }
    val orientation =
      runCatching {
          ExifInterface(ByteArrayInputStream(bytes))
            .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        }
        .getOrDefault(ExifInterface.ORIENTATION_NORMAL)
    return rgb.orient(orientation)
  }

  /**
   * A copy of the picture for the screen, upright, at most about [maxSide] pixels on its long side.
   * Boxes drawn on it are kept as fractions of its size, so they map onto [decode]'s pixels.
   */
  fun decodeForScreen(bytes: ByteArray, maxSide: Int = 2048): Bitmap {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
    var sample = 1
    while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxSide) {
      sample *= 2
    }
    val bitmap =
      BitmapFactory.decodeByteArray(
        bytes,
        0,
        bytes.size,
        BitmapFactory.Options().apply { inSampleSize = sample },
      ) ?: throw IllegalArgumentException("not a picture this phone can decode")
    val orientation =
      runCatching {
          ExifInterface(ByteArrayInputStream(bytes))
            .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        }
        .getOrDefault(ExifInterface.ORIENTATION_NORMAL)
    val matrix = android.graphics.Matrix()
    when (orientation) {
      ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.setScale(-1f, 1f)
      ExifInterface.ORIENTATION_ROTATE_180 -> matrix.setRotate(180f)
      ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.setScale(1f, -1f)
      ExifInterface.ORIENTATION_TRANSPOSE -> {
        matrix.setRotate(90f)
        matrix.postScale(-1f, 1f)
      }
      ExifInterface.ORIENTATION_ROTATE_90 -> matrix.setRotate(90f)
      ExifInterface.ORIENTATION_TRANSVERSE -> {
        matrix.setRotate(-90f)
        matrix.postScale(-1f, 1f)
      }
      ExifInterface.ORIENTATION_ROTATE_270 -> matrix.setRotate(-90f)
      else -> return bitmap
    }
    val upright = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    if (upright !== bitmap) bitmap.recycle()
    return upright
  }

  /** The bitmap's stored bytes (RGBA in memory for ARGB_8888), alpha dropped. */
  private fun toRgb(bitmap: Bitmap): Rgb {
    val w = bitmap.width
    val h = bitmap.height
    val rgba = ByteBuffer.allocate(bitmap.rowBytes * h)
    bitmap.copyPixelsToBuffer(rgba)
    val src = rgba.array()
    val out = ByteArray(w * h * 3)
    for (y in 0 until h) {
      var s = y * bitmap.rowBytes
      var o = y * w * 3
      for (x in 0 until w) {
        out[o] = src[s]
        out[o + 1] = src[s + 1]
        out[o + 2] = src[s + 2]
        s += 4
        o += 3
      }
    }
    return Rgb(w, h, out)
  }

  /** A bitmap of [img] for the screen. */
  fun toBitmap(img: Rgb): Bitmap {
    val pixels = IntArray(img.width * img.height)
    val b = img.bytes
    for (i in pixels.indices) {
      pixels[i] =
        (0xFF shl 24) or
          ((b[i * 3].toInt() and 0xFF) shl 16) or
          ((b[i * 3 + 1].toInt() and 0xFF) shl 8) or
          (b[i * 3 + 2].toInt() and 0xFF)
    }
    return Bitmap.createBitmap(pixels, img.width, img.height, Bitmap.Config.ARGB_8888)
  }

  /** [img] as PNG bytes (lossless: the runtime decodes the same pixels). */
  fun png(img: Rgb): ByteArray {
    val bitmap = toBitmap(img)
    try {
      val out = ByteArrayOutputStream()
      check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)) { "PNG encoding failed" }
      return out.toByteArray()
    } finally {
      bitmap.recycle()
    }
  }
}
