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

package com.google.aiedge.examples.phototalk

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.support.metadata.MetadataExtractor
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer

data class ClassificationResult(
    val label: String,
    val confidence: Float
)

class ImageClassifierHelper(private val context: Context) {

    companion object {
        private const val TAG = "PhotoTalk_Classifier"
        private const val MODEL_NAME = "efficientnet_lite0.tflite"
    }

    private var model: CompiledModel? = null
    private var labels: List<String> = emptyList()

    suspend fun initClassifier() = withContext(Dispatchers.IO) {
        try {
            val assetManager = context.assets
            val modelStream = assetManager.open(MODEL_NAME)
            val modelBytes = modelStream.readBytes()
            val modelBuffer = ByteBuffer.wrap(modelBytes)
            modelStream.close()

            // Extract labels from model metadata
            val metadataExtractor = MetadataExtractor(modelBuffer)
            if (metadataExtractor.hasMetadata()) {
                val labelStream = metadataExtractor.getAssociatedFile("labels_without_background.txt")
                    ?: metadataExtractor.getAssociatedFile("labels.txt")
                if (labelStream != null) {
                    labels = BufferedReader(InputStreamReader(labelStream)).readLines()
                }
            }

            // Create LiteRT CompiledModel with GPU/CPU hardware acceleration
            model = try {
                CompiledModel.create(assetManager, MODEL_NAME, CompiledModel.Options(Accelerator.GPU), null)
            } catch (e: Exception) {
                Log.w(TAG, "GPU acceleration failed, falling back to CPU", e)
                CompiledModel.create(assetManager, MODEL_NAME, CompiledModel.Options(Accelerator.CPU), null)
            }
            Log.i(TAG, "LiteRT CompiledModel classifier initialized successfully.")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize LiteRT image classifier", e)
            throw e
        }
    }

    suspend fun classify(bitmap: Bitmap): ClassificationResult? = withContext(Dispatchers.IO) {
        val currentModel = model ?: run {
            initClassifier()
            model ?: return@withContext null
        }

        try {
            val width = 224
            val height = 224
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, width, height, true)
            val inputFloatArray = normalize(scaledBitmap, 127.5f, 127.5f)

            val inputBuffers = currentModel.createInputBuffers()
            val outputBuffers = currentModel.createOutputBuffers()

            inputBuffers[0].writeFloat(inputFloatArray)
            currentModel.run(inputBuffers, outputBuffers)

            val probabilities = outputBuffers[0].readFloat()

            inputBuffers.forEach { it.close() }
            outputBuffers.forEach { it.close() }

            // Find top-1 classification index
            var maxIndex = 0
            var maxProb = 0.0f
            for (i in probabilities.indices) {
                if (probabilities[i] > maxProb) {
                    maxProb = probabilities[i]
                    maxIndex = i
                }
            }

            val labelName = labels.getOrElse(maxIndex) { "Object #$maxIndex" }
            ClassificationResult(label = labelName, confidence = maxProb)
        } catch (e: Exception) {
            Log.e(TAG, "Classification error", e)
            null
        }
    }

    private fun normalize(image: Bitmap, mean: Float, stddev: Float): FloatArray {
        val width = image.width
        val height = image.height
        val numPixels = width * height
        val pixelsIntArray = IntArray(numPixels)
        val outputFloatArray = FloatArray(numPixels * 3)
        image.getPixels(pixelsIntArray, 0, width, 0, 0, width, height)
        for (i in 0 until numPixels) {
            val pixel = pixelsIntArray[i]
            val r = android.graphics.Color.red(pixel).toFloat()
            val g = android.graphics.Color.green(pixel).toFloat()
            val b = android.graphics.Color.blue(pixel).toFloat()
            val idx = i * 3
            outputFloatArray[idx] = (r - mean) / stddev
            outputFloatArray[idx + 1] = (g - mean) / stddev
            outputFloatArray[idx + 2] = (b - mean) / stddev
        }
        return outputFloatArray
    }

    fun close() {
        model?.close()
        model = null
    }
}
