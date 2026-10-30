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

package com.google.ai.edge.examples.photo_voice_chat

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import android.os.Build
import java.io.InputStream

/**
 * A photo as the model sees it: the center square of the picture (upright by its EXIF orientation), scaled to
 * [VisionEncoder.SIZE] x [VisionEncoder.SIZE]; [rgb] = its pixels (row-major RGB bytes), [square] = the same pixels for
 * the screen. A picture that is already that size and square is used as it is (the bundled examples).
 */
class PhotoInput(val name: String, val rgb: ByteArray, val square: Bitmap, val width: Int, val height: Int)

/** Why a file cannot be a photo, as the sentence the screen shows. */
class UnusableImage(message: String) : Exception(message)

object Photos {
  private const val SIZE = VisionEncoder.SIZE

  fun fromUri(context: Context, uri: Uri, name: String): PhotoInput {
    if (Build.VERSION.SDK_INT >= 28) return fromUriDecoder(context, uri, name)
    fun open(): InputStream = context.contentResolver.openInputStream(uri) ?: throw UnusableImage("Could not open $name.")
    val orientation = try {
      open().use { ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL) }
    } catch (e: UnusableImage) {
      throw e
    } catch (e: Exception) {
      ExifInterface.ORIENTATION_NORMAL
    }
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    try {
      open().use { BitmapFactory.decodeStream(it, null, bounds) }
    } catch (e: UnusableImage) {
      throw e
    } catch (e: Exception) {
      throw UnusableImage("Could not open $name: ${e.message}")
    }
    if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
      throw UnusableImage("Could not read $name as a photo (JPEG, PNG, WebP or HEIC): the file has no picture this phone can decode.")
    }
    // decode at a power-of-two scale whose short side still covers SIZE (a 12 MP photo needs no 48 MB bitmap)
    var sample = 1
    while (minOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= SIZE) sample *= 2
    val opts = BitmapFactory.Options().apply { inSampleSize = sample; inPreferredConfig = Bitmap.Config.ARGB_8888 }
    val decoded = try {
      open().use { BitmapFactory.decodeStream(it, null, opts) }
    } catch (e: Exception) {
      null
    } ?: throw UnusableImage("Could not read $name as a photo (JPEG, PNG, WebP or HEIC): the picture data is damaged or cut short.")
    return fromBitmap(name, upright(decoded, orientation), bounds.outWidth, bounds.outHeight)
  }

  /**
   * Android 9 and later: ImageDecoder (EXIF orientation applied by the decoder). A picture that is cut short or damaged
   * is refused: BitmapFactory returns such a file with its missing part gray, and the model would answer about it.
   */
  private fun fromUriDecoder(context: Context, uri: Uri, name: String): PhotoInput {
    var width = 0
    var height = 0
    val bitmap = try {
      ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.contentResolver, uri)) { decoder, info, _ ->
        width = info.size.width
        height = info.size.height
        decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
        // decode at a power-of-two scale whose short side still covers SIZE (a 12 MP photo needs no 48 MB bitmap)
        var sample = 1
        while (minOf(width, height) / (sample * 2) >= SIZE) sample *= 2
        decoder.setTargetSampleSize(sample)
        decoder.setOnPartialImageListener { false }
      }
    } catch (e: ImageDecoder.DecodeException) {
      throw UnusableImage("Could not read $name as a photo (JPEG, PNG, WebP or HEIC): the picture data is damaged or cut short.")
    } catch (e: Exception) {
      throw UnusableImage("Could not read $name as a photo (JPEG, PNG, WebP or HEIC): ${e.message ?: e.javaClass.simpleName}")
    }
    return fromBitmap(name, bitmap, width, height)
  }

  fun fromAsset(context: Context, path: String, name: String): PhotoInput {
    val opts = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
    val b = context.assets.open(path).use { BitmapFactory.decodeStream(it, null, opts) } ?: throw UnusableImage("Could not read $path.")
    return fromBitmap(name, b, b.width, b.height)
  }

  /** The center square, scaled to SIZE x SIZE (bilinear); as it is when it already is SIZE x SIZE. */
  fun fromBitmap(name: String, src: Bitmap, width: Int, height: Int): PhotoInput {
    val side = minOf(src.width, src.height)
    val cropped = if (src.width == src.height) src else Bitmap.createBitmap(src, (src.width - side) / 2, (src.height - side) / 2, side, side)
    val square = if (cropped.width == SIZE) cropped else Bitmap.createScaledBitmap(cropped, SIZE, SIZE, true)
    val px = IntArray(SIZE * SIZE)
    square.getPixels(px, 0, SIZE, 0, 0, SIZE, SIZE)
    val rgb = ByteArray(SIZE * SIZE * 3)
    for (i in px.indices) {
      val c = px[i]
      rgb[3 * i] = (c shr 16 and 0xff).toByte()
      rgb[3 * i + 1] = (c shr 8 and 0xff).toByte()
      rgb[3 * i + 2] = (c and 0xff).toByte()
    }
    return PhotoInput(name, rgb, square, width, height)
  }

  private fun upright(b: Bitmap, orientation: Int): Bitmap {
    val m = Matrix()
    when (orientation) {
      ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
      ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
      ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
      ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
      ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
      ExifInterface.ORIENTATION_TRANSPOSE -> { m.postRotate(90f); m.postScale(-1f, 1f) }
      ExifInterface.ORIENTATION_TRANSVERSE -> { m.postRotate(270f); m.postScale(-1f, 1f) }
      else -> return b
    }
    return Bitmap.createBitmap(b, 0, 0, b.width, b.height, m, true)
  }
}
