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

package com.google.ai.edge.examples.ocr_verification

import android.content.ContentResolver
import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.graphics.Rect
import android.net.Uri
import java.io.File
import java.nio.ByteBuffer
import kotlin.math.roundToInt

/**
 * A photo as the app reads it: the decoded 8-bit RGB values at full size (no colour conversion:
 * what Pillow's convert("RGB") gives for a picture file), upright (a camera photo's EXIF orientation
 * is applied by the decoder, as the user sees it), and a smaller copy for the screen. The model's
 * region pictures are crops of these values.
 */
class Photo(
  val name: String,
  val rgb: HostMath.Rgb,
  /** The photo for the text recognizer and the screen (ARGB_8888, the same pixels). */
  val bitmap: Bitmap,
) {
  val width: Int
    get() = rgb.width

  val height: Int
    get() = rgb.height

  /**
   * The region of a text line: its box grown by a margin of a fifth of its height (at least 4
   * pixels), inside the photo.
   */
  fun crop(box: Rect): Pair<HostMath.Rgb, Rect> {
    val margin = maxOf(4, (box.height() * 0.2f).roundToInt())
    val r =
      Rect(
        (box.left - margin).coerceAtLeast(0),
        (box.top - margin).coerceAtLeast(0),
        (box.right + margin).coerceAtMost(width),
        (box.bottom + margin).coerceAtMost(height),
      )
    require(r.width() > 0 && r.height() > 0) { "empty region" }
    return WebRenderer.crop(rgb, r.left, r.top, r.width(), r.height()) to r
  }

  companion object {
    /** Photos above this many pixels are decoded at half size per side (and again). */
    const val MAX_PIXELS = 24_000_000L

    fun fromUri(resolver: ContentResolver, uri: Uri, name: String): Photo =
      fromBitmap(name, decode(ImageDecoder.createSource(resolver, uri)))

    fun fromFile(file: File): Photo = fromBitmap(file.name, decode(ImageDecoder.createSource(file)))

    fun fromAsset(context: Context, path: String): Photo =
      fromBitmap(path.substringAfterLast('/'), decode(ImageDecoder.createSource(context.assets, path)))

    /** A picture file decoded as the model's host reads it (for the check mode). */
    fun rgbOfFile(file: File): HostMath.Rgb {
      val bmp = decode(ImageDecoder.createSource(file))
      return toRgb(bmp).also { bmp.recycle() }
    }

    private fun decode(source: ImageDecoder.Source): Bitmap =
      ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
        decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
        decoder.isUnpremultipliedRequired = true
        // keep the file's own values: no conversion to the screen's colour space (a 16-bit PNG
        // comes back as RGBA_F16 and is refused in toRgb)
        decoder.setTargetColorSpace(
          info.colorSpace ?: android.graphics.ColorSpace.get(android.graphics.ColorSpace.Named.SRGB)
        )
        var s = 1
        val pixels = info.size.width.toLong() * info.size.height
        while (pixels / (s.toLong() * s) > MAX_PIXELS) s *= 2
        if (s > 1) decoder.setTargetSampleSize(s)
      }

    private fun toRgb(bitmap: Bitmap): HostMath.Rgb {
      if (bitmap.config == Bitmap.Config.RGBA_F16) {
        throw Refused("not_8_bit", "a 16-bit picture: the model's host reads 8-bit pictures only")
      }
      val argb = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap else bitmap.copy(Bitmap.Config.ARGB_8888, false)
      val w = argb.width
      val h = argb.height
      // copyPixelsToBuffer keeps the stored values (getPixels would convert to sRGB); R, G, B, A
      val rgba = ByteBuffer.allocate(w * h * 4)
      argb.copyPixelsToBuffer(rgba)
      val src = rgba.array()
      val rgb = ByteArray(w * h * 3)
      for (i in 0 until w * h) {
        rgb[i * 3] = src[i * 4]
        rgb[i * 3 + 1] = src[i * 4 + 1]
        rgb[i * 3 + 2] = src[i * 4 + 2]
      }
      if (argb !== bitmap) argb.recycle()
      return HostMath.Rgb(rgb, w, h)
    }

    private fun fromBitmap(name: String, bitmap: Bitmap): Photo {
      val rgb = toRgb(bitmap)
      val px = IntArray(rgb.width * rgb.height)
      for (i in px.indices) {
        px[i] =
          (0xFF shl 24) or
            ((rgb.bytes[i * 3].toInt() and 0xFF) shl 16) or
            ((rgb.bytes[i * 3 + 1].toInt() and 0xFF) shl 8) or
            (rgb.bytes[i * 3 + 2].toInt() and 0xFF)
      }
      bitmap.recycle()
      return Photo(name, rgb, Bitmap.createBitmap(px, rgb.width, rgb.height, Bitmap.Config.ARGB_8888))
    }
  }
}
