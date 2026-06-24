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

package com.google.ai.edge.examples.super_resolution

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.SystemClock
import android.util.Log
import androidx.core.graphics.scale
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext

/**
 * Runs VDSR single-image super-resolution with the LiteRT CompiledModel API.
 *
 * Model: vdsr_256_fp16.tflite
 *  - input : 1 x 256 x 256 x 1 float32, the luminance (Y) of a bicubic-upscaled image, range [0,1]
 *  - output: 1 x 256 x 256 x 1 float32, the refined (sharper) Y, range [0,1]
 *
 * VDSR refines the luminance (Y) of the input; the chroma (Cb/Cr) is kept unchanged and recombined.
 * The result is a vertical "Input vs VDSR" comparison.
 */
class SuperResolutionHelper(private val context: Context) {

  val enhanced: SharedFlow<EnhanceResult>
    get() = _enhanced

  private val _enhanced =
    MutableSharedFlow<EnhanceResult>(extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST)

  val error: SharedFlow<Throwable?>
    get() = _error

  private val _error = MutableSharedFlow<Throwable?>()

  private var enhancer: Enhancer? = null
  private val singleThreadDispatcher = Dispatchers.IO.limitedParallelism(1, "ModelDispatcher")

  suspend fun initEnhancer(acceleratorEnum: AcceleratorEnum = AcceleratorEnum.GPU) {
    cleanup()
    try {
      withContext(singleThreadDispatcher) {
        val model =
          CompiledModel.create(
            context.assets,
            MODEL_FILE,
            CompiledModel.Options(toAccelerator(acceleratorEnum)),
            null,
          )
        enhancer = Enhancer(model)
        Log.d(TAG, "Created a super-resolver ($acceleratorEnum)")
      }
    } catch (e: Exception) {
      Log.i(TAG, "Create LiteRT from $MODEL_FILE failed: ${e.message}")
      _error.emit(e)
    }
  }

  suspend fun cleanup() {
    try {
      withContext(singleThreadDispatcher) {
        enhancer?.cleanup()
        enhancer = null
      }
    } catch (e: Exception) {
      Log.e(TAG, "Error during cleanup: ${e.message}")
    }
  }

  suspend fun enhance(bitmap: Bitmap) {
    try {
      withContext(singleThreadDispatcher) {
        enhancer?.enhance(bitmap)?.let { if (isActive) _enhanced.emit(it) }
      }
    } catch (e: Exception) {
      Log.i(TAG, "Super-resolution error: ${e.message}")
      _error.emit(e)
    }
  }

  private class Enhancer(private val model: CompiledModel) {
    private val inputBuffers = model.createInputBuffers()
    private val outputBuffers = model.createOutputBuffers()

    fun cleanup() {
      inputBuffers.forEach { it.close() }
      outputBuffers.forEach { it.close() }
      model.close()
    }

    fun enhance(bitmap: Bitmap): EnhanceResult {
      val input = bitmap.scale(SIZE, SIZE, true)

      val numPixels = SIZE * SIZE
      val pixels = IntArray(numPixels)
      input.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
      val y = FloatArray(numPixels)
      val cb = FloatArray(numPixels)
      val cr = FloatArray(numPixels)
      for (i in 0 until numPixels) {
        val p = pixels[i]
        val r = Color.red(p).toFloat()
        val g = Color.green(p).toFloat()
        val b = Color.blue(p).toFloat()
        y[i] = (0.299f * r + 0.587f * g + 0.114f * b) / 255f
        cb[i] = -0.168736f * r - 0.331264f * g + 0.5f * b + 128f
        cr[i] = 0.5f * r - 0.418688f * g - 0.081312f * b + 128f
      }

      val start = SystemClock.uptimeMillis()
      inputBuffers[0].writeFloat(y)
      model.run(inputBuffers, outputBuffers)
      val srY = outputBuffers[0].readFloat() // SIZE * SIZE refined Y in [0,1]
      val inferenceTime = SystemClock.uptimeMillis() - start

      val sr = IntArray(numPixels)
      for (i in 0 until numPixels) {
        sr[i] = ycbcrToArgb(srY[i] * 255f, cb[i], cr[i])
      }
      val vdsr = Bitmap.createBitmap(sr, SIZE, SIZE, Bitmap.Config.ARGB_8888)

      return EnhanceResult(compare(input, vdsr), inferenceTime)
    }

    /** Stack the bicubic image over the VDSR image, with captions, into one comparison bitmap. */
    private fun compare(top: Bitmap, bottom: Bitmap): Bitmap {
      val gap = 8
      val out = Bitmap.createBitmap(SIZE, SIZE * 2 + gap, Bitmap.Config.ARGB_8888)
      val canvas = Canvas(out)
      canvas.drawColor(Color.BLACK)
      canvas.drawBitmap(top, 0f, 0f, null)
      canvas.drawBitmap(bottom, 0f, (SIZE + gap).toFloat(), null)
      caption(canvas, "Input", 0f)
      caption(canvas, "VDSR (LiteRT)", (SIZE + gap).toFloat())
      return out
    }

    private fun caption(canvas: Canvas, label: String, top: Float) {
      val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = 22f }
      val bgPaint = Paint().apply { color = Color.argb(150, 0, 0, 0) }
      val pad = 8f
      val tw = textPaint.measureText(label)
      canvas.drawRect(6f, top + 6f, 6f + tw + 2 * pad, top + 6f + 34f, bgPaint)
      canvas.drawText(label, 6f + pad, top + 6f + 25f, textPaint)
    }

    private fun ycbcrToArgb(yv: Float, cbv: Float, crv: Float): Int {
      val cbd = cbv - 128f
      val crd = crv - 128f
      val r = (yv + 1.402f * crd).toInt().coerceIn(0, 255)
      val g = (yv - 0.344136f * cbd - 0.714136f * crd).toInt().coerceIn(0, 255)
      val b = (yv + 1.772f * cbd).toInt().coerceIn(0, 255)
      return (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }
  }

  data class EnhanceResult(val overlay: Bitmap, val inferenceTime: Long)

  enum class AcceleratorEnum {
    CPU,
    GPU,
  }

  private companion object {
    const val TAG = "SuperResolution"
    const val MODEL_FILE = "vdsr_256_fp16.tflite"
    const val SIZE = 256

    fun toAccelerator(acceleratorEnum: AcceleratorEnum): Accelerator =
      when (acceleratorEnum) {
        AcceleratorEnum.CPU -> Accelerator.CPU
        AcceleratorEnum.GPU -> Accelerator.GPU
      }
  }
}
