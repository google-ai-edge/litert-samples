package com.example.mobilenetlitert

import android.graphics.Bitmap
import androidx.core.graphics.scale

object ImagePreprocessor {
  const val INPUT_SIZE = 224
  const val CHANNELS = 3
  const val INPUT_FLOAT_COUNT = INPUT_SIZE * INPUT_SIZE * CHANNELS

  fun preprocessBitmap(bitmap: Bitmap): FloatArray {
    val scaled = bitmap.scale(INPUT_SIZE, INPUT_SIZE, true)
    val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
    scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
    return preprocessPixels(pixels)
  }

  fun preprocessPixels(pixels: IntArray): FloatArray {
    require(pixels.size == INPUT_SIZE * INPUT_SIZE) {
      "Expected ${INPUT_SIZE * INPUT_SIZE} pixels, got ${pixels.size}."
    }

    val output = FloatArray(INPUT_FLOAT_COUNT)
    pixels.forEachIndexed { index, pixel ->
      val base = index * CHANNELS
      output[base] = ((pixel shr 16) and 0xFF) / 255.0f
      output[base + 1] = ((pixel shr 8) and 0xFF) / 255.0f
      output[base + 2] = (pixel and 0xFF) / 255.0f
    }
    return output
  }
}
