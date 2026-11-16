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
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext

class DigitClassificationHelper(private val context: Context) {
    private var model: CompiledModel? = null
    private var environment: Environment? = null

    // Use a single thread to avoid race conditions with model initialization and execution
    private val dispatcher = Dispatchers.IO.limitedParallelism(1)

    companion object {
        private const val TAG = "DigitClassificationHelper"
        private const val MODEL_ASSET = "digit_classifier.tflite"

        fun toAccelerator(acceleratorEnum: AcceleratorEnum): Accelerator {
            return when (acceleratorEnum) {
                AcceleratorEnum.CPU -> Accelerator.CPU
                AcceleratorEnum.GPU -> Accelerator.GPU
                AcceleratorEnum.NPU -> Accelerator.NPU
            }
        }

        /**
         * Finds the index and value of the maximum element in a non-empty float array.
         */
        fun findResult(array: FloatArray): Pair<Int, Float> {
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

    /** As the result of digit classification, this value emits digit and score */
    val classification: SharedFlow<Pair<String, Float>>
        get() = _classification
    private val _classification = MutableSharedFlow<Pair<String, Float>>(
        extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    private val _status = MutableStateFlow("Initializing...")
    val status: StateFlow<String> = _status.asStateFlow()

    enum class AcceleratorEnum {
        CPU,
        GPU,
        NPU,
    }

    suspend fun initClassifier(acceleratorEnum: AcceleratorEnum = AcceleratorEnum.CPU): Boolean {
        return withContext(dispatcher) {
            releaseResources()
            try {
                val appContext = context.applicationContext
                val accelerator = toAccelerator(acceleratorEnum)
                val createdEnv: Environment? = if (acceleratorEnum == AcceleratorEnum.NPU) {
                    runCatching {
                        System.loadLibrary("QnnSystem")
                        System.loadLibrary("QnnHtp")
                    }
                    val provider = BuiltinNpuAcceleratorProvider(appContext)
                    val npuEnv = Environment.create(appContext, provider)
                    val available = npuEnv.getAvailableAccelerators()
                    if (Accelerator.NPU !in available) {
                        npuEnv.close()
                        throw IllegalStateException(
                            "NPU accelerator unavailable on this device (available: ${available.joinToString()}). " +
                                "Bundle QNN libraries or run on a device with supported NPU hardware."
                        )
                    }
                    npuEnv
                } else {
                    Environment.create(appContext)
                }

                val options = CompiledModel.Options(accelerator).apply {
                    if (acceleratorEnum == AcceleratorEnum.NPU) {
                        qualcommOptions = CompiledModel.QualcommOptions(
                            htpPerformanceMode =
                                CompiledModel.QualcommOptions.HtpPerformanceMode.HIGH_PERFORMANCE
                        )
                    }
                }

                environment = createdEnv
                model = CompiledModel.create(
                    context.assets,
                    MODEL_ASSET,
                    options,
                    createdEnv
                )
                val label = if (acceleratorEnum == AcceleratorEnum.NPU) "NPU (JIT)" else acceleratorEnum.name
                _status.value = "Ready on $label"
                Log.i(TAG, "Created a CompiledModel with $acceleratorEnum")
                true
            } catch (e: Exception) {
                releaseResources()
                val reason = e.message ?: e.javaClass.simpleName
                _status.value = "${acceleratorEnum.name} unavailable: $reason"
                Log.e(TAG, "Initializing CompiledModel on $acceleratorEnum failed: $reason", e)
                false
            }
        }
    }

    fun cleanup() {
        releaseResources()
    }

    private fun releaseResources() {
        runCatching { model?.close() }
        model = null
        runCatching { environment?.close() }
        environment = null
    }

    suspend fun classify(bitmap: Bitmap) {
        withContext(dispatcher) {
            val localModel = model ?: return@withContext

            try {
                // 1. Preprocessing: Resize to 28x28
                val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 28, 28, true)
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
                Log.e(TAG, "Error during classification: ${e.message}", e)
            }
        }
    }

    private fun convertBitmapToFloatArray(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // Target: [1, 28, 28, 3] in [0, 255] range
        val output = FloatArray(width * height * 3)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = Color.red(pixel).toFloat()
            val g = Color.green(pixel).toFloat()
            val b = Color.blue(pixel).toFloat()

            val baseIndex = i * 3
            output[baseIndex] = r
            output[baseIndex + 1] = g
            output[baseIndex + 2] = b
        }
        return output
    }
}