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

package com.google.aiedge.examples.imageclassification

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.metadata.MetadataExtractor
import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.FloatBuffer

class ImageClassificationHelper(
    private val context: Context,
    private var options: Options = Options(),
) {
    class Options(
        /** The enum contains the model file name, relative to the assets/ directory */
        var model: Model = DEFAULT_MODEL,
        /** The delegate for running computationally intensive operations*/
        var delegate: Delegate = DEFAULT_DELEGATE,
        /** Number of output classes of the TFLite model.  */
        var resultCount: Int = DEFAULT_RESULT_COUNT,
        /** Probability value above which a class is labeled as active (i.e., detected) the display.  */
        var probabilityThreshold: Float = DEFAULT_THRESHOLD,
        /** Number of threads to be used for ops that support multi-threading.
         * threadCount>= -1. Setting numThreads to 0 has the effect of disabling multithreading,
         * which is equivalent to setting numThreads to 1. If unspecified, or set to the value -1,
         * the number of threads used will be implementation-defined and platform-dependent.
         * */
        var threadCount: Int = DEFAULT_THREAD_COUNT
    )

    companion object {
        private const val TAG = "ImageClassification"

        val DEFAULT_MODEL = Model.EfficientNetLite0
        val DEFAULT_DELEGATE = Delegate.CPU
        const val DEFAULT_RESULT_COUNT = 3
        const val DEFAULT_THRESHOLD = 0.3f
        const val DEFAULT_THREAD_COUNT = 2

    }

    /** As the result of sound classification, this value emits map of probabilities */
    val classification: SharedFlow<ClassificationResult>
        get() = _classification
    private val _classification = MutableSharedFlow<ClassificationResult>(
        extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    val error: SharedFlow<Throwable?>
        get() = _error
    private val _error = MutableSharedFlow<Throwable?>()

    private var interpreter: Interpreter? = null
    private lateinit var labels: List<String>

    /** Init a Interpreter from [Model] with [Delegate]*/
    suspend fun initClassifier() {
        interpreter = try {
            val litertBuffer = FileUtil.loadMappedFile(context, options.model.fileName)
            Log.i(TAG, "Done creating TFLite buffer from ${options.model.fileName}")
            val options = Interpreter.Options().apply {
                numThreads = options.threadCount
                useNNAPI = options.delegate == Delegate.NNAPI
            }
            labels = getModelMetadata(litertBuffer)
            Interpreter(litertBuffer, options)
        } catch (e: Exception) {
            Log.i(TAG, "Create TFLite from ${options.model.fileName} is failed ${e.message}")
            _error.emit(e)
            null
        }
    }

    fun setOptions(options: Options) {
        this.options = options
    }

    suspend fun classify(bitmap: Bitmap, rotationDegrees: Int) {
        try {
            withContext(Dispatchers.IO) {
                if (interpreter == null) return@withContext
                val startTime = SystemClock.uptimeMillis()

                val rotation = -rotationDegrees / 90
                val (_, h, w, _) = interpreter?.getInputTensor(0)?.shape() ?: return@withContext
                val imageProcessor =
                    ImageProcessor.Builder().add(ResizeOp(h, w, ResizeOp.ResizeMethod.BILINEAR))
                        .add(Rot90Op(rotation)).add(NormalizeOp(127.5f, 127.5f)).build()

                // Preprocess the image and convert it into a TensorImage for classification.
                val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))
                val output = classifyWithTFLite(tensorImage)

                val outputList = output.map {
                    /** Scores in range 0..1.0 for each of the output classes. */
                    if (it < options.probabilityThreshold) 0f else it
                }

                val categories = labels.zip(outputList).map {
                    Category(label = it.first, score = it.second)
                }.sortedByDescending { it.score }.take(options.resultCount)

                val inferenceTime = SystemClock.uptimeMillis() - startTime
                if (isActive) {
                    _classification.emit(ClassificationResult(categories, inferenceTime))
                }
            }
        } catch (e: Exception) {
            Log.i(TAG, "Image classification error occurred: ${e.message}")
            _error.emit(e)
        }
    }

    private fun classifyWithTFLite(tensorImage: TensorImage): FloatArray {
        val outputShape = interpreter!!.getOutputTensor(0).shape()
        val outputBuffer = FloatBuffer.allocate(outputShape[1])

        outputBuffer.rewind()
        interpreter?.run(tensorImage.tensorBuffer.buffer, outputBuffer)
        outputBuffer.rewind()
        val output = FloatArray(outputBuffer.capacity())
        outputBuffer.get(output)
        return output
    }


    /** Load metadata from model*/
    private fun getModelMetadata(litertBuffer: ByteBuffer): List<String> {
        val metadataExtractor = MetadataExtractor(litertBuffer)
        val labels = mutableListOf<String>()
        if (metadataExtractor.hasMetadata()) {
            val inputStream = metadataExtractor.getAssociatedFile("labels_without_background.txt")
            labels.addAll(readFileInputStream(inputStream))
            Log.i(
                TAG, "Successfully loaded model metadata ${metadataExtractor.associatedFileNames}"
            )
        }
        return labels
    }

    /** Retrieve Map<String, Int> from metadata file */
    private fun readFileInputStream(inputStream: InputStream): List<String> {
        val reader = BufferedReader(InputStreamReader(inputStream))

        val list = mutableListOf<String>()
        var index = 0
        var line = ""
        while (reader.readLine().also { if (it != null) line = it } != null) {
            list.add(line)
            index++
        }

        reader.close()
        return list
    }


    enum class Delegate {
        CPU, NNAPI
    }

    enum class Model(val fileName: String) {
        EfficientNetLite0("efficientnet_lite0.tflite"), EfficientNetLite2("efficientnet_lite2.tflite")
    }

    data class ClassificationResult(
        val categories: List<Category>, val inferenceTime: Long
    )

    data class Category(val label: String, val score: Float)
}