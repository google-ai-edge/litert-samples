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

package com.google.ai.edge.examples.face_restoration

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri

/** Longest-edge cap for a decoded gallery image, to keep face detection fast and memory bounded. */
private const val MAX_INPUT_SIZE = 1024

/**
 * Decodes a gallery image, downsampling it so its longest edge is at most [MAX_INPUT_SIZE] px, and
 * rotates it upright based on its EXIF orientation. Returns null if the image cannot be decoded.
 */
fun Context.loadBitmap(uri: Uri): Bitmap? {
  val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
  contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
  var sampleSize = 1
  while (
    opts.outWidth / sampleSize > MAX_INPUT_SIZE || opts.outHeight / sampleSize > MAX_INPUT_SIZE
  ) {
    sampleSize *= 2
  }
  val decodeOpts = BitmapFactory.Options().apply { inSampleSize = sampleSize }
  val bitmap =
    contentResolver.openInputStream(uri)?.use {
      BitmapFactory.decodeStream(it, null, decodeOpts)
    } ?: return null
  val rotation =
    contentResolver.openInputStream(uri)?.use {
      when (
        ExifInterface(it)
          .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
      ) {
        ExifInterface.ORIENTATION_ROTATE_90 -> 90f
        ExifInterface.ORIENTATION_ROTATE_180 -> 180f
        ExifInterface.ORIENTATION_ROTATE_270 -> 270f
        else -> 0f
      }
    } ?: 0f
  if (rotation == 0f) return bitmap
  val m = Matrix().apply { postRotate(rotation) }
  val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
  bitmap.recycle()
  return rotated
}
