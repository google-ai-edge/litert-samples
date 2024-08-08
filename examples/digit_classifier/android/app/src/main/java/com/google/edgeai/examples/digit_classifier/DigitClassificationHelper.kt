/*
 * Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

package com.google.edgeai.examples.digit_classifier

import android.content.Context
import android.graphics.Bitmap
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
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.FloatBuffer

class DigitClassificationHelper(private val context: Context) {
    /** The TFLite interpreter instance.  */
    private var interpreter: Interpreter? = null

    companion object {
        private const val TAG = "DigitClassificationHelper"
    }

    /** As the result of digit classification, this value emits digit and score */
    val classification: SharedFlow<Pair<String, Float>>
        get() = _classification
    private val _classification = MutableSharedFlow<Pair<String, Float>>(
        extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    init {
        initHelper()
    }

    /** Init a Interpreter from asset*/
    private fun initHelper() {
        interpreter = try {
            val tfliteBuffer = FileUtil.loadMappedFile(context, "digit_classifier.tflite")
            Log.i(TAG, "Done creating TFLite buffer from asset")
            Interpreter(tfliteBuffer, Interpreter.Options())
        } catch (e: Exception) {
            Log.e(TAG, "Initializing TensorFlow Lite has failed with error: ${e.message}")
            return
        }
    }

    suspend fun classify(bitmap: Bitmap) {
        withContext(Dispatchers.IO) {
            if (interpreter == null) return@withContext

            // Get the input tensor shape from the interpreter.
            val (_, h, w, _) = interpreter?.getInputTensor(0)?.shape() ?: return@withContext

            // Build an image processor for pre-processing the input image.
            val imageProcessor =
                ImageProcessor.Builder().add(ResizeOp(h, w, ResizeOp.ResizeMethod.BILINEAR))
                    .add(NormalizeOp(0f, 1f)).build()

            // Preprocess the image and convert it into a TensorImage for classification.
            val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))
            val output = classifyWithTFLite(tensorImage)
            val (digit, score) = findResult(output)
            _classification.emit(Pair(digit.toString(), score))
        }
    }

    private fun classifyWithTFLite(tensorImage: TensorImage): FloatArray {
        val outputShape = interpreter!!.getOutputTensor(0).shape()
        val outputBuffer = FloatBuffer.allocate(outputShape[1])
        val inputBuffer = TensorBuffer.createFrom(tensorImage.tensorBuffer, DataType.FLOAT32).buffer

        inputBuffer.rewind()
        outputBuffer.rewind()
        interpreter?.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()
        val output = FloatArray(outputBuffer.capacity())
        outputBuffer.get(output)
        return output
    }

    /**
     * Finds the index and value of the maximum element in a non-empty float array.
     *
     * @param array The input float array.
     * @return A pair containing the index and value of the maximum element.
     * @throws AssertionError If the input array is empty.
     */
    private fun findResult(array: FloatArray): Pair<Int, Float> {
        assert(array.isNotEmpty()) {
            "Input array must not be empty"
        }

        var maxIndex = 0
        var maxValue = array[0]

        for (i in array.indices) {
            if (array[i] > maxValue) {
                maxValue = array[i]
                maxIndex = i
            }
        }

        return Pair(maxIndex, maxValue)
    }
}