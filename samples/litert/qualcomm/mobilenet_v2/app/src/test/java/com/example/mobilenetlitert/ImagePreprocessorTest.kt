package com.example.mobilenetlitert

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ImagePreprocessorTest {
  @Test
  fun preprocessPixels_returnsRgbFloatTensorInRange() {
    val red = 0x00FF0000
    val green = 0x0000FF00
    val blue = 0x000000FF
    val pixels =
      IntArray(ImagePreprocessor.INPUT_SIZE * ImagePreprocessor.INPUT_SIZE) { index ->
        when (index % 3) {
          0 -> red
          1 -> green
          else -> blue
        }
      }

    val tensor = ImagePreprocessor.preprocessPixels(pixels)

    assertEquals(ImagePreprocessor.INPUT_FLOAT_COUNT, tensor.size)
    assertEquals(1.0f, tensor[0], 0.0001f)
    assertEquals(0.0f, tensor[1], 0.0001f)
    assertEquals(0.0f, tensor[2], 0.0001f)
    assertEquals(0.0f, tensor[3], 0.0001f)
    assertEquals(1.0f, tensor[4], 0.0001f)
    assertEquals(0.0f, tensor[5], 0.0001f)
    assertTrue(tensor.all { it in 0.0f..1.0f })
  }
}
