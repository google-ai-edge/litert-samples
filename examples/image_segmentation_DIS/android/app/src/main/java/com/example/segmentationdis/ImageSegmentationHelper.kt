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

package com.example.segmentationdis

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.nio.FloatBuffer
import kotlin.math.max

class ImageSegmentationHelper(private val context: Context) {
    companion object {
        private const val TAG = "SegmentationDis"
    }

    /** As the result of sound classification, this value emits map of probabilities */
    val segmentation: SharedFlow<SegmentationResult>
        get() = _segmentation
    private val _segmentation = MutableSharedFlow<SegmentationResult>(
        extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    val error: SharedFlow<Throwable?>
        get() = _error
    private val _error = MutableSharedFlow<Throwable?>()

    private var interpreter: Interpreter? = null

    private var currentDelegate = Delegate.CPU

    /** Init a Interpreter from Delegate and Model.*/
    suspend fun initClassifier(
        delegate: Delegate? = null,
    ) {
        delegate?.let { currentDelegate = it }
        interpreter = try {
            val litertBuffer = FileUtil.loadMappedFile(context, "isnet_tfl_drq.tflite")
            val options = Interpreter.Options().apply {
                numThreads = 4
                useNNAPI = currentDelegate == Delegate.NNAPI
            }
            Interpreter(litertBuffer, options)
        } catch (e: Exception) {
            _error.emit(e)
            null
        }
    }

    /**
     * Start read model and get bitmap result
     */
    suspend fun segment(bitmap: Bitmap) {
        try {
            val newBitmap = addBlackBorderToSquare(bitmap)
            withContext(Dispatchers.IO) {
                if (interpreter == null) return@withContext
                val startTime = SystemClock.uptimeMillis()

                val (_, h, w, _) = interpreter?.getInputTensor(0)?.shape() ?: return@withContext
                val imageProcessor =
                    ImageProcessor.Builder()
                        .add(ResizeOp(h, w, ResizeOp.ResizeMethod.BILINEAR))
                        .add(NormalizeOp(0f, 1f))
                        .build()

                val tensorImageFromBitmap = TensorImage.fromBitmap(newBitmap)
                // Preprocess the image and convert it into a TensorImage for segmentation.
                val tensorImageWithImageProcessor = imageProcessor.process(tensorImageFromBitmap)
                val bitmapOutput = getBitmap(tensorImageWithImageProcessor)
                val bitmapCrop = cropBlackBorderBitmap(bitmapOutput, bitmap)
                val bitmapResult = scaleBitmap(bitmapCrop, bitmap)
                val inferenceTime = SystemClock.uptimeMillis() - startTime
                if (isActive) {
                    _segmentation.emit(SegmentationResult(bitmapResult, inferenceTime))
                }
            }
        } catch (e: Exception) {
            Log.i(TAG, "Image segment error occurred: ${e.message}")
            _error.emit(e)
        }
    }

    /**
     * Convert bitmap to Square
     */
    private fun addBlackBorderToSquare(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height

        val squareSize = max(width, height)

        val result = Bitmap.createBitmap(squareSize, squareSize, Bitmap.Config.ARGB_8888)
        result.eraseColor(Color.BLACK)
        val canvas = Canvas(result)

        val x = (squareSize - width) / 2
        val y = (squareSize - height) / 2

        canvas.drawBitmap(bitmap, x.toFloat(), y.toFloat(), null)

        return result
    }

    /**
     * Create output and run model
     */
    private fun getBitmap(tensorImage: TensorImage): Bitmap {
        val (_, h, w, c) = interpreter!!.getOutputTensor(0).shape()
        val outputBuffer = FloatBuffer.allocate(h * w * c)
        val inputBuffer = TensorImage.createFrom(tensorImage, DataType.FLOAT32).buffer

        outputBuffer.rewind()
        interpreter?.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()

        val array32 = FloatArray(h * w * c)
        outputBuffer.get(array32)

        return floatArrayToBitmap(array32, w, h)
    }

    /**
     * Convert float array to bitmap
     */
    private fun floatArrayToBitmap(floatArray: FloatArray, width: Int, height: Int): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        for (h in 0 until height) {
            for (w in 0 until width) {
                // Get the index in the float array
                val index = h * width + w

                // Get the float value, ensuring it's between 0.0 and 1.0
                val value = floatArray[index].coerceIn(0.0f, 1.0f)

                // Set gray value
                val grayValue = (value * 255).toInt()
                val alpha = if (value > 0.95f) 0 else 255

                val color =
                    Color.argb(alpha, grayValue, grayValue, grayValue)
                bitmap.setPixel(w, h, color)
            }
        }

        return bitmap
    }

    /**
     * Compare old bitmap to crop
     */
    private fun cropBlackBorderBitmap(bitmap: Bitmap, oldBitmap: Bitmap): Bitmap {
        val isImageHorizontal = oldBitmap.width.toFloat() < oldBitmap.height.toFloat()
        val percent =
            if (isImageHorizontal) oldBitmap.width.toFloat() / oldBitmap.height.toFloat()
            else oldBitmap.height.toFloat() / oldBitmap.width.toFloat()

        return Bitmap.createBitmap(
            bitmap,
            if (isImageHorizontal) ((1 - percent) / 2 * bitmap.width).toInt() else 0,
            if (isImageHorizontal) 0 else ((1 - percent) / 2 * bitmap.height).toInt(),
            if (isImageHorizontal) (bitmap.width * percent).toInt() else bitmap.width,
            if (isImageHorizontal) bitmap.height else (bitmap.height * percent).toInt()
        )
    }

    /**
     * Scale bitmap before emit result
     */
    private fun scaleBitmap(bitmap: Bitmap, oldBitmap: Bitmap): Bitmap {
        val percentScale = oldBitmap.width.toFloat() / bitmap.width.toFloat()

        return Bitmap.createScaledBitmap(
            bitmap,
            (bitmap.width * percentScale).toInt(),
            (bitmap.height * percentScale).toInt(),
            false
        )
    }

    enum class Delegate {
        CPU, NNAPI
    }

    data class SegmentationResult(
        val bitmap: Bitmap,
        val inferenceTime: Long
    )
}
