/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.ai.edge.examples.yolact

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri

/** Decodes a bitmap bundled in the app's assets. */
fun Context.decodeAssetBitmap(assetName: String): Bitmap =
  assets.open(assetName).use { BitmapFactory.decodeStream(it) }
    ?: error("Cannot decode asset image: $assetName")

/** Decodes a gallery image and rotates it upright based on its EXIF orientation. */
fun Context.loadOrientedBitmap(uri: Uri): Bitmap {
  val bitmap =
    contentResolver.openInputStream(uri).use { BitmapFactory.decodeStream(it) }
      ?: error("Cannot decode image: $uri")
  val degrees =
    contentResolver.openInputStream(uri).use { stream ->
      when (ExifInterface(stream!!).getAttributeInt(ExifInterface.TAG_ORIENTATION, 1)) {
        ExifInterface.ORIENTATION_ROTATE_90 -> 90f
        ExifInterface.ORIENTATION_ROTATE_180 -> 180f
        ExifInterface.ORIENTATION_ROTATE_270 -> 270f
        else -> 0f
      }
    }
  if (degrees == 0f) return bitmap
  val rotation = Matrix().apply { postRotate(degrees) }
  return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, rotation, true)
}

/** Squashes a bitmap to [size] x [size], matching the model's fixed square input. */
fun Bitmap.squareResize(size: Int): Bitmap = Bitmap.createScaledBitmap(this, size, size, true)

/** Flattens the bitmap to a row-major RGB float array in the [0, 255] range. */
fun Bitmap.toRgbFloatArray(): FloatArray {
  val pixelCount = width * height
  val pixels = IntArray(pixelCount)
  getPixels(pixels, 0, width, 0, 0, width, height)
  val rgb = FloatArray(pixelCount * 3)
  for (i in 0 until pixelCount) {
    val pixel = pixels[i]
    rgb[i * 3] = ((pixel shr 16) and 0xFF).toFloat()
    rgb[i * 3 + 1] = ((pixel shr 8) and 0xFF).toFloat()
    rgb[i * 3 + 2] = (pixel and 0xFF).toFloat()
  }
  return rgb
}
