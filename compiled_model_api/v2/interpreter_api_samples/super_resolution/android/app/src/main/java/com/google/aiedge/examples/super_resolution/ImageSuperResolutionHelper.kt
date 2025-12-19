/*
 * Copyright 2024 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.aiedge.examples.super_resolution

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.nio.FloatBuffer

class ImageSuperResolutionHelper(private val context: Context) {
    companion object {
        private const val TAG = "ImageSuperResolution"
    }

    private var interpreter: Interpreter? = null

    val superResolutionFlow: SharedFlow<Result>
        get() = _superResolution
    private val _superResolution = MutableSharedFlow<Result>(
        extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    init {
        initClassifier()
    }

    /** Init a Interpreter from ESRGAN model.*/
    fun initClassifier(delegate: Delegate = Delegate.CPU) {
        interpreter = try {
            val litertBuffer = FileUtil.loadMappedFile(context, "ESRGAN.tflite")
            Log.i(TAG, "Done creating TFLite buffer from ESRGAN model")
            val options = Interpreter.Options().apply {
                numThreads = 4
            }
            Interpreter(litertBuffer, options)
        } catch (e: Exception) {
            Log.e(TAG, "Initializing TensorFlow Lite has failed with error: ${e.message}")
            return
        }
    }

    /*
     *Performs super-resolution processing on a provided bitmap.
     */
    suspend fun makeSuperResolution(bitmap: Bitmap) {
        if (interpreter == null) return

        withContext(Dispatchers.IO) {
            val startTime = SystemClock.uptimeMillis()
            val (_, h, w, _) = interpreter?.getInputTensor(0)?.shape() ?: return@withContext
            val imageProcessor =
                ImageProcessor
                    .Builder()
                    .add(ResizeOp(h, w, ResizeOp.ResizeMethod.BILINEAR))
                    .add(NormalizeOp(0f, 1f))
                    .build()

            // Preprocess the image and convert it into a TensorImage for segmentation.
            val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))
            val outputBitmap = process(tensorImage)
            val inferenceTime = SystemClock.uptimeMillis() - startTime

            _superResolution.emit(Result(bitmap = outputBitmap, inferenceTime = inferenceTime))
        }
    }

    /*
     * Convert the output from the interpreter float array to a Bitmap
     */
    private fun floatArrayToBitmap(floatArray: FloatArray, width: Int, height: Int): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height)

        val minVal = floatArray.minOrNull() ?: 0f
        val maxVal = floatArray.maxOrNull() ?: 1f
        val normalizedArray = floatArray.map {
            (it - minVal) / (maxVal - minVal).coerceAtLeast(
                1.0E-5F
            )
        }

        for (i in pixels.indices) {
            val r = (normalizedArray[i * 3] * 255).toInt()
            val g = (normalizedArray[i * 3 + 1] * 255).toInt()
            val b = (normalizedArray[i * 3 + 2] * 255).toInt()
            pixels[i] = Color.argb(255, r, g, b)
        }

        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }

    private fun process(tensorImage: TensorImage): Bitmap {
        val (_, w, h, c) = interpreter!!.getOutputTensor(0).shape()
        val outputBuffer = FloatBuffer.allocate(h * w * c)
        val inputBuffer = TensorImage.createFrom(tensorImage, DataType.FLOAT32).buffer
        interpreter?.run(inputBuffer, outputBuffer)

        return floatArrayToBitmap(outputBuffer.array(), w, h)
    }

    data class Result(
        val bitmap: Bitmap? = null,
        val inferenceTime: Long = 0L
    )

    enum class Delegate {
        CPU
    }
}