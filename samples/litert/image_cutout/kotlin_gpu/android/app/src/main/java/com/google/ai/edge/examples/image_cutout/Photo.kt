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

package com.google.ai.edge.examples.image_cutout

import android.content.ContentResolver
import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer

/**
 * A photo as the model reads it: the decoded RGB bytes (unpremultiplied, the file's own colour
 * values, no colour conversion: what PIL's convert("RGB") gives) at full size, and a smaller copy
 * for the screen.
 */
class Photo(
  val name: String,
  val rgb: ByteArray,
  val width: Int,
  val height: Int,
  val display: Bitmap,
  /** The display copy takes every [displayStep]-th pixel of every [displayStep]-th row. */
  val displayStep: Int,
) {
  /** SHA-256 of the RGB bytes, the fixtures' `pixel_sha256`. */
  val pixelSha256: String by lazy { ModelFiles.sha256(rgb) }

  companion object {
    /**
     * Photos above this many pixels are decoded at half size per side (and again) to bound the
     * memory.
     */
    const val MAX_PIXELS = 40_000_000L
    const val DISPLAY_SIDE = 2048

    fun fromUri(resolver: ContentResolver, uri: Uri, name: String): Photo =
      fromBitmap(name, decode(resolver, uri))

    fun fromFile(file: File): Photo = fromBitmap(file.name, decodeFile(file))

    fun fromAsset(context: Context, path: String): Photo =
      context.assets.open(path).use { input ->
        val bytes = input.readBytes()
        fromBitmap(path.substringAfterLast('/'), decodeBytes(bytes))
      }

    private fun decode(resolver: ContentResolver, uri: Uri): Bitmap =
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
        ImageDecoder.decodeBitmap(ImageDecoder.createSource(resolver, uri)) { decoder, info, _ ->
          configure(decoder, info.size.width.toLong() * info.size.height)
        }
      } else {
        resolver.openInputStream(uri).use { decodeBytes(requireNotNull(it).readBytes()) }
      }

    private fun decodeFile(file: File): Bitmap =
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
        ImageDecoder.decodeBitmap(ImageDecoder.createSource(file)) { decoder, info, _ ->
          configure(decoder, info.size.width.toLong() * info.size.height)
        }
      } else {
        decodeBytes(file.readBytes())
      }

    private fun decodeBytes(bytes: ByteArray): Bitmap =
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
        ImageDecoder.decodeBitmap(ImageDecoder.createSource(ByteBuffer.wrap(bytes))) {
          decoder,
          info,
          _ ->
          configure(decoder, info.size.width.toLong() * info.size.height)
        }
      } else {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
        val options =
          BitmapFactory.Options().apply {
            inPremultiplied = false
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inSampleSize = sampleSize(bounds.outWidth.toLong() * bounds.outHeight)
          }
        requireNotNull(BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options)) {
          "not an image"
        }
      }

    private fun configure(decoder: ImageDecoder, pixels: Long) {
      decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
      decoder.isUnpremultipliedRequired = true
      val s = sampleSize(pixels)
      if (s > 1) {
        decoder.setTargetSampleSize(s)
      }
    }

    private fun sampleSize(pixels: Long): Int {
      var s = 1
      while (pixels / (s.toLong() * s) > MAX_PIXELS) {
        s *= 2
      }
      return s
    }

    private fun fromBitmap(name: String, bitmap: Bitmap): Photo {
      val argb =
        if (bitmap.config == Bitmap.Config.ARGB_8888) {
          bitmap
        } else {
          bitmap.copy(Bitmap.Config.ARGB_8888, false)
        }
      val w = argb.width
      val h = argb.height
      // copyPixelsToBuffer keeps the stored values (getPixels would convert to sRGB); ARGB_8888 is
      // R, G, B, A
      val rgba = ByteBuffer.allocateDirect(w * h * 4)
      argb.copyPixelsToBuffer(rgba)
      rgba.rewind()
      val rgb = ByteArray(w * h * 3)
      val row = ByteArray(w * 4)
      for (y in 0 until h) {
        rgba.get(row)
        var o = y * w * 3
        for (x in 0 until w) {
          rgb[o] = row[x * 4]
          rgb[o + 1] = row[x * 4 + 1]
          rgb[o + 2] = row[x * 4 + 2]
          o += 3
        }
      }
      val step = maxOf(1, (maxOf(w, h) + DISPLAY_SIDE - 1) / DISPLAY_SIDE)
      val display = displayCopy(rgb, w, h, step)
      if (argb !== bitmap) {
        argb.recycle()
      }
      bitmap.recycle()
      return Photo(name, rgb, w, h, display, step)
    }

    /** An opaque screen copy of at most [DISPLAY_SIDE] pixels a side (nearest sampling). */
    private fun displayCopy(rgb: ByteArray, w: Int, h: Int, step: Int): Bitmap {
      val dw = maxOf(1, w / step)
      val dh = maxOf(1, h / step)
      val px = IntArray(dw * dh)
      for (y in 0 until dh) {
        for (x in 0 until dw) {
          val i = ((y * step) * w + x * step) * 3
          px[y * dw + x] =
            (0xFF shl 24) or
              ((rgb[i].toInt() and 0xFF) shl 16) or
              ((rgb[i + 1].toInt() and 0xFF) shl 8) or
              (rgb[i + 2].toInt() and 0xFF)
        }
      }
      return Bitmap.createBitmap(px, dw, dh, Bitmap.Config.ARGB_8888)
    }
  }
}

/**
 * The union of the instances' alpha mattes (the largest value at each pixel), at the photo's size.
 */
fun combinedAlpha(instances: List<Instance>, size: Int): FloatArray {
  val out = FloatArray(size)
  for (inst in instances) {
    for (i in 0 until size) {
      if (inst.alpha[i] > out[i]) {
        out[i] = inst.alpha[i]
      }
    }
  }
  return out
}

/** The photo with [alpha] as its transparency, at full size (the PNG the app saves). */
fun cutoutBitmap(photo: Photo, alpha: FloatArray): Bitmap {
  val px = IntArray(photo.width * photo.height)
  for (i in px.indices) {
    val a = (alpha[i] * 255f + 0.5f).toInt().coerceIn(0, 255)
    px[i] =
      (a shl 24) or
        ((photo.rgb[i * 3].toInt() and 0xFF) shl 16) or
        ((photo.rgb[i * 3 + 1].toInt() and 0xFF) shl 8) or
        (photo.rgb[i * 3 + 2].toInt() and 0xFF)
  }
  return Bitmap.createBitmap(px, photo.width, photo.height, Bitmap.Config.ARGB_8888)
}

/**
 * Screen-size cut-out and tint layer, sampled from the full-size [alpha] like the photo's display
 * copy.
 */
fun displayLayers(photo: Photo, alpha: FloatArray, color: Int): Pair<Bitmap, Bitmap> {
  val dw = photo.display.width
  val dh = photo.display.height
  val step = photo.displayStep
  val cut = IntArray(dw * dh)
  val tint = IntArray(dw * dh)
  val rgb = color and 0xFFFFFF
  for (y in 0 until dh) {
    for (x in 0 until dw) {
      val i = (y * step) * photo.width + x * step
      val a = (alpha[i] * 255f + 0.5f).toInt().coerceIn(0, 255)
      val j = y * dw + x
      cut[j] =
        (a shl 24) or
          ((photo.rgb[i * 3].toInt() and 0xFF) shl 16) or
          ((photo.rgb[i * 3 + 1].toInt() and 0xFF) shl 8) or
          (photo.rgb[i * 3 + 2].toInt() and 0xFF)
      tint[j] = ((a * 0.55f).toInt() shl 24) or rgb
    }
  }
  return Bitmap.createBitmap(cut, dw, dh, Bitmap.Config.ARGB_8888) to
    Bitmap.createBitmap(tint, dw, dh, Bitmap.Config.ARGB_8888)
}

/** Saves [bitmap] as a PNG in Pictures/WeVisOne; returns where it went. */
fun savePng(context: Context, bitmap: Bitmap, name: String): String {
  if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
    val values =
      ContentValues().apply {
        put(MediaStore.Images.Media.DISPLAY_NAME, name)
        put(MediaStore.Images.Media.MIME_TYPE, "image/png")
        put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/WeVisOne")
      }
    val resolver = context.contentResolver
    val uri =
      requireNotNull(resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)) {
        "MediaStore insert failed"
      }
    resolver.openOutputStream(uri).use { out ->
      bitmap.compress(Bitmap.CompressFormat.PNG, 100, requireNotNull(out))
    }
    return "Pictures/WeVisOne/$name"
  }
  val dir =
    File(context.getExternalFilesDir(Environment.DIRECTORY_PICTURES), "WeVisOne").apply { mkdirs() }
  val file = File(dir, name)
  FileOutputStream(file).use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
  return file.absolutePath
}
