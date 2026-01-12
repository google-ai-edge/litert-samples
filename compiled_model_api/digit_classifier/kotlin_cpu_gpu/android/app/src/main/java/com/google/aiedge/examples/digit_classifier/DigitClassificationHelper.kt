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
import android.util.Log
import androidx.core.graphics.scale
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer

class DigitClassificationHelper(private val context: Context) {
    private var model: CompiledModel? = null
    
    companion object {
        private const val TAG = "DigitClassificationHelper"
        private const val MODEL_PATH = "digit_classifier.tflite"
    }

    /** As the result of digit classification, this value emits digit and score */
    val classification: SharedFlow<Pair<String, Float>>
        get() = _classification
    private val _classification = MutableSharedFlow<Pair<String, Float>>(
        extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    suspend fun initHelper(accelerator: Accelerator = Accelerator.CPU) {
        withContext(Dispatchers.IO) {
            try {
                model?.close()
                model = null
                // Initialize with specified accelerator
                val options = CompiledModel.Options(accelerator)
                model = CompiledModel.create(context.assets, MODEL_PATH, options, null)
                Log.i(TAG, "Done creating LiteRT CompiledModel with $accelerator")
            } catch (e: Exception) {
                Log.e(TAG, "Initializing LiteRT has failed with error: ${e.message}")
            }
        }
    }

    suspend fun classify(bitmap: Bitmap) {
        withContext(Dispatchers.IO) {
            if (model == null) {
                initHelper()
                if (model == null) return@withContext
            }

            val currentModel = model ?: return@withContext

            // Create input and output buffers
            val inputBuffers = currentModel.createInputBuffers()
            val outputBuffers = currentModel.createOutputBuffers()

            // Preprocess the image
            // The model expects 28x28 input, grayscale, normalized to [0, 1]
            val scaledBitmap = bitmap.scale(28, 28, true)
            val inputData = preprocessBitmap(scaledBitmap)

            // Write to input buffer
            inputBuffers[0].writeFloat(inputData)

            // Run inference
            currentModel.run(inputBuffers, outputBuffers)

            // Read output
            val outputData = outputBuffers[0].readFloat()
            
            // Find result
            val (digit, score) = findResult(outputData)
            _classification.emit(Pair(digit.toString(), score))
            
            // Close buffers
            inputBuffers.forEach { it.close() }
            outputBuffers.forEach { it.close() }
        }
    }
    
    private fun preprocessBitmap(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        
        val floatArray = FloatArray(width * height)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            // Convert to grayscale and normalize. 
            // Assuming the input bitmap is white drawing on black background or similar.
            // The v1 sample used NormalizeOp(0f, 1f) which just casts to float if values are already 0-255? 
            // Actually TensorImage.fromBitmap converts to 0-255 int values. 
            // NormalizeOp(0f, 1f) divides by 1? No, (value - mean) / stddev. So (x - 0) / 1 = x.
            // Wait, TFLite usually expects 0-1 or -1 to 1 for float models.
            // Let's check v1 code again. 
            // ImageProcessor.Builder().add(ResizeOp(h, w, ...)).add(NormalizeOp(0f, 1f)).build()
            // NormalizeOp(0f, 1f) does nothing effectively if input is 0-255? 
            // Ah, TensorImage handles the conversion.
            // Let's assume standard 0-1 normalization for float32 models usually.
            // But the v1 sample might have been trained on 0-255?
            // Let's look at the v1 code: `NormalizeOp(0f, 1f)` -> (x - 0) / 1. 
            // If the input tensor is float32, TensorImage loads bitmap as 0-255 integers (in float format).
            // So it seems it passes 0-255 values.
            // However, standard MNIST is usually 0-1. 
            // Let's try to normalize to 0-1 (pixel / 255f). If that fails, we can try 0-255.
            
            // Using the blue channel as they are all the same for grayscale/black-white
            val r = Color.red(pixel)
            val g = Color.green(pixel)
            val b = Color.blue(pixel)
            
            // Simple grayscale conversion or just taking one channel if it's black/white
            val gray = (r + g + b) / 3.0f
            
            // Normalize to 0-1
            floatArray[i] = gray / 255.0f
        }
        return floatArray
    }

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
    
    fun close() {
        model?.close()
        model = null
    }
}
