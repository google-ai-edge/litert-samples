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

package com.google.aiedge.examples.digit_classifier

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.withContext

class DigitClassificationHelper(private val context: Context) {
    private var model: CompiledModel? = null
    
    // Use a single thread to avoid race conditions with model execution
    private val dispatcher = Dispatchers.IO.limitedParallelism(1)

    companion object {
        private const val TAG = "DigitClassificationHelper"
        
        fun toAccelerator(acceleratorEnum: AcceleratorEnum): Accelerator {
            return when (acceleratorEnum) {
                AcceleratorEnum.CPU -> Accelerator.CPU
                AcceleratorEnum.GPU -> Accelerator.GPU
            }
        }
    }

    /** As the result of digit classification, this value emits digit and score */
    val classification: SharedFlow<Pair<String, Float>>
        get() = _classification
    private val _classification = MutableSharedFlow<Pair<String, Float>>(
        extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    enum class AcceleratorEnum {
        CPU,
        GPU,
    }

    suspend fun initClassifier(acceleratorEnum: AcceleratorEnum = AcceleratorEnum.CPU) {
        cleanup()
        try {
            withContext(dispatcher) {
                model = CompiledModel.create(
                    context.assets,
                    "digit_classifier.tflite",
                    CompiledModel.Options(toAccelerator(acceleratorEnum)),
                    null
                )
                Log.i(TAG, "Created a CompiledModel with $acceleratorEnum")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Initializing CompiledModel has failed with error: ${e.message}")
        }
    }

    fun cleanup() {
        model?.close()
        model = null
    }

    suspend fun classify(bitmap: Bitmap) {
        withContext(dispatcher) {
            val localModel = model ?: return@withContext

            try {
                // 1. Preprocessing
                // Resize to 28x28
                val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 28, 28, true)
                // Convert to grayscale and normalize to 0..1
                val inputFloatArray = convertBitmapToFloatArray(scaledBitmap)

                // 2. Execution
                val inputBuffers = localModel.createInputBuffers()
                val outputBuffers = localModel.createOutputBuffers()

                inputBuffers[0].writeFloat(inputFloatArray)
                localModel.run(inputBuffers, outputBuffers)
                
                val outputFloatArray = outputBuffers[0].readFloat()

                // Cleanup buffers
                inputBuffers.forEach { it.close() }
                outputBuffers.forEach { it.close() }

                // 3. Postprocessing
                val (digit, score) = findResult(outputFloatArray)
                _classification.emit(Pair(digit.toString(), score))

            } catch (e: Exception) {
                Log.e(TAG, "Error during classification: ${e.message}")
            }
        }
    }

    private fun convertBitmapToFloatArray(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // The original sample used ImageProcessor without grayscale conversion, 
        // implying it sent 3 channels (RGB) to the model.
        // It also used NormalizeOp(0f, 1f), which effectively keeps the 0-255 range 
        // when converting from the default Bitmap uint8 values to Float32.
        
        // Target: [1, 28, 28, 3]
        val output = FloatArray(width * height * 3) 
        
        for (i in pixels.indices) {
            val pixel = pixels[i]
            
            // Extract RGB (ignore alpha)
            val r = Color.red(pixel).toFloat()
            val g = Color.green(pixel).toFloat()
            val b = Color.blue(pixel).toFloat()
            
            // Don't divide by 255.0f to match original 0..255 range behavior
            val baseIndex = i * 3
            output[baseIndex] = r
            output[baseIndex + 1] = g
            output[baseIndex + 2] = b
        }
        return output
    }

    /**
     * Finds the index and value of the maximum element in a non-empty float array.
     */
    private fun findResult(array: FloatArray): Pair<Int, Float> {
        if (array.isEmpty()) return Pair(-1, 0f)

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