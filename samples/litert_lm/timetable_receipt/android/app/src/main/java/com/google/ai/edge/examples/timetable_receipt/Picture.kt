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

package com.google.ai.edge.examples.timetable_receipt

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.InputStream
import java.nio.ByteBuffer
import kotlin.math.max

/** A picture the user chose, ready for the screen and for the model. */
class Picture(
  val name: String,
  /** Upright, at most [PREVIEW_SIDE] pixels on its long side. */
  val preview: Bitmap,
  /** The model's input: the upright picture stretched to [MODEL_SIDE] x [MODEL_SIDE], as PNG. */
  val modelPng: ByteArray,
  /** The upright size of the original picture. */
  val width: Int,
  val height: Int,
)

class PictureException(message: String) : Exception(message)

/**
 * Reads pictures for the model. The bundle's vision encoder takes a fixed 512 x 512 picture, and
 * LiteRT-LM 0.18.1 was measured to stretch whatever it is given to that size without looking at
 * EXIF orientation, so the app turns the photo upright first (ImageDecoder applies the EXIF
 * orientation) and does the stretch itself; the model then gets exactly the pixels the app made.
 */
object Pictures {
  const val MODEL_SIDE = 512
  private const val PREVIEW_SIDE = 1280
  private const val MAX_FILE_BYTES = 40L * 1024 * 1024

  fun fromUri(context: Context, uri: Uri, name: String): Picture {
    val bytes =
      try {
        context.contentResolver.openInputStream(uri)?.use { readCapped(it) }
          ?: throw PictureException("Could not open $name.")
      } catch (e: IOException) {
        throw PictureException("Could not read $name: ${e.message}")
      } catch (e: SecurityException) {
        throw PictureException("This app may not read $name.")
      }
    return fromBytes(bytes, name)
  }

  fun fromAsset(context: Context, path: String, name: String): Picture =
    fromBytes(context.assets.open(path).use { readCapped(it) }, name)

  fun fromBytes(bytes: ByteArray, name: String): Picture {
    if (bytes.isEmpty()) throw PictureException("$name is empty.")
    var width = 0
    var height = 0
    val preview =
      decode(bytes, name) { w, h ->
        width = w
        height = h
        val scale = PREVIEW_SIDE.toFloat() / max(w, h)
        if (scale < 1f) Pair((w * scale).toInt().coerceAtLeast(1), (h * scale).toInt().coerceAtLeast(1))
        else Pair(w, h)
      }
    val model = decode(bytes, name) { _, _ -> Pair(MODEL_SIDE, MODEL_SIDE) }
    val png = ByteArrayOutputStream().use { out ->
      model.compress(Bitmap.CompressFormat.PNG, 100, out)
      out.toByteArray()
    }
    model.recycle()
    return Picture(name, preview, png, width, height)
  }

  /** Decodes [bytes] upright, scaled during the decode to the size [target] picks. */
  private fun decode(bytes: ByteArray, name: String, target: (Int, Int) -> Pair<Int, Int>): Bitmap {
    val source = ImageDecoder.createSource(ByteBuffer.wrap(bytes))
    try {
      return ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
        decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
        // A file that ends early decodes with a grey band; refuse it instead.
        decoder.setOnPartialImageListener { false }
        val (w, h) = target(info.size.width, info.size.height)
        decoder.setTargetSize(w, h)
      }
    } catch (e: ImageDecoder.DecodeException) {
      val why = when (e.error) {
        ImageDecoder.DecodeException.SOURCE_INCOMPLETE -> "the file ends early (it is incomplete)"
        ImageDecoder.DecodeException.SOURCE_MALFORMED_DATA -> "its picture data is damaged"
        else -> "it is not a picture this phone can decode"
      }
      throw PictureException("Could not read $name as a picture: $why.")
    } catch (e: IOException) {
      throw PictureException("Could not read $name as a picture (JPEG, PNG, WebP or HEIF).")
    }
  }

  private fun readCapped(input: InputStream): ByteArray {
    val out = ByteArrayOutputStream()
    val buf = ByteArray(1 shl 16)
    var total = 0L
    while (true) {
      val n = input.read(buf)
      if (n < 0) break
      total += n
      if (total > MAX_FILE_BYTES) throw PictureException("The picture is larger than 40 MB.")
      out.write(buf, 0, n)
    }
    return out.toByteArray()
  }
}
