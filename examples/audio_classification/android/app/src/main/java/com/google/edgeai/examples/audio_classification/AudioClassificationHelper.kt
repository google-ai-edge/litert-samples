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

package com.google.edgeai.examples.audio_classification

import android.annotation.SuppressLint
import android.content.Context
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.metadata.MetadataExtractor
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.FloatBuffer

/**
 * Performs classification on sound.
 *
 * <p>The API supports models which accept sound input via {@code AudioRecord} and one classification output tensor.
 * The output of the recognition is emitted as LiveData of Map.
 *
 */
class AudioClassificationHelper(private val context: Context, val options: Options = Options()) {
    class Options(
        /** Overlap factor of recognition period */
        var overlapFactor: Float = DEFAULT_OVERLAP,
        /** Probability value above which a class is labeled as active (i.e., detected) the display.  */
        var probabilityThreshold: Float = DEFAULT_PROBABILITY_THRESHOLD,
        /** The enum contains the model file name, relative to the assets/ directory */
        var currentModel: TFLiteModel = DEFAULT_MODEL,
        /** The delegate for running computationally intensive operations*/
        var delegate: Delegate = DEFAULT_DELEGATE,
        /** Number of output classes of the TFLite model.  */
        var resultCount: Int = DEFAULT_RESULT_COUNT,
        /** Number of threads to be used for ops that support multi-threading.
         * threadCount>= -1. Setting numThreads to 0 has the effect of disabling multithreading,
         * which is equivalent to setting numThreads to 1. If unspecified, or set to the value -1,
         * the number of threads used will be implementation-defined and platform-dependent.
         * */
        var threadCount: Int = DEFAULT_THREAD_COUNT
    )

    /** As the result of sound classification, this value emits map of probabilities */
    val probabilities: SharedFlow<Pair<List<Category>, Long>>
        get() = _probabilities
    private val _probabilities = MutableSharedFlow<Pair<List<Category>, Long>>(
        extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )


    /** The TFLite interpreter instance.  */
    private var interpreter: Interpreter? = null

    private var job: Job? = null

    private lateinit var labels: List<String>

    private var previousAudioArray: FloatArray? = null

    private var audioManager: AudioManager? = null

    /** Stop, cancel or reset all necessary variable*/
    fun stop() {
        previousAudioArray = null
        job?.cancel()
        audioManager?.stopRecord()
        interpreter?.resetVariableTensors()
        interpreter?.close()
        interpreter = null
    }

    suspend fun setupInterpreter() {
        interpreter = try {
            val litertBuffer = FileUtil.loadMappedFile(context, options.currentModel.fileName)
            Log.i(TAG, "Done creating TFLite buffer from ${options.currentModel}")
            labels = getModelMetadata(litertBuffer)
            Interpreter(litertBuffer, Interpreter.Options().apply {
                numThreads = options.threadCount
                useNNAPI = options.delegate == Delegate.NNAPI
            })
        } catch (e: IOException) {
            throw IOException("Failed to load TFLite model - ${e.message}")
        } catch (e: Exception) {
            throw Exception("Failed to create Interpreter - ${e.message}")
        }
    }

    /*
    * Starts sound classification, which triggers running on IO Thread
    */
    @SuppressLint("MissingPermission")
    suspend fun startRecord() {
        withContext(Dispatchers.IO) {
            // Inspect input and output specs.
            val inputShape = interpreter?.getInputTensor(0)?.shape() ?: return@withContext

            /**
             * YAMNET input: float32[15600]
             * Speech Command input: float32[1,44032]
             */
            val modelInputLength =
                inputShape[if (options.currentModel == TFLiteModel.YAMNET) 0 else 1]
            audioManager = AudioManager(
                options.currentModel.sampleRate,
                modelInputLength,
                options.overlapFactor
            )

            previousAudioArray = FloatArray(0)
            audioManager!!.record().collect {
                val array = convertShortToFloat(it)
                startRecognition(array)
            }
        }
    }

    private suspend fun startRecognition(audioArray: FloatArray) {
        // Inspect input and output specs.
        val inputShape = interpreter?.getInputTensor(0)?.shape() ?: return

        /**
         * YAMNET input: float32[15600]
         * Speech Command input: float32[1,44032]
         */
        val modelInputLength = inputShape[if (options.currentModel == TFLiteModel.YAMNET) 0 else 1]

        val outputShape = interpreter?.getOutputTensor(0)?.shape() ?: return
        val modelNumClasses = outputShape[1]
        // Fill the array with NaNs initially.
        val predictionProbs = FloatArray(modelNumClasses) { Float.NaN }

        val inputBuffer = FloatBuffer.allocate(modelInputLength)

        coroutineScope {
            if (modelInputLength <= 0) {
                Log.e(TAG, "Switches: Cannot start recognition because model is unavailable.")
                return@coroutineScope
            }
            val outputBuffer = FloatBuffer.allocate(modelNumClasses)
            // Put audio data to buffer
            if (previousAudioArray != null) {
                // Put previous audio array first
                inputBuffer.put(previousAudioArray)
                // Case: overlap > 50%
                if (audioArray.size < modelInputLength - previousAudioArray!!.size) {
                    previousAudioArray = previousAudioArray?.plus(audioArray)
                    return@coroutineScope
                }
                // Case: overlap < 50%
                else if (audioArray.size > modelInputLength - previousAudioArray!!.size) {
                    val range = IntRange(0, (modelInputLength * options.overlapFactor).toInt() - 1)
                    previousAudioArray = audioArray.sliceArray(range)
                    inputBuffer.put(previousAudioArray)
                }
                // Put the remaining missing data
                else {
                    inputBuffer.put(audioArray)
                    previousAudioArray = FloatArray(0)
                }
            }

            val startTime = SystemClock.uptimeMillis()
            inputBuffer.rewind()
            outputBuffer.rewind()
            interpreter?.run(inputBuffer, outputBuffer)

            outputBuffer.rewind()
            outputBuffer.get(predictionProbs) // Copy data to predictionProbs.

            val probList = predictionProbs.map {
                /** Scores in range 0..1.0 for each of the output classes. */
                if (it < options.probabilityThreshold) 0f else it
            }

            val categories = labels.zip(probList).map {
                Category(label = it.first, score = it.second)
            }.sortedByDescending { it.score }.take(options.resultCount)
            val inferenceTime = SystemClock.uptimeMillis() - startTime
            _probabilities.emit(
                Pair(categories, inferenceTime)
            )
        }
    }

    /** Load metadata from model*/
    private suspend fun getModelMetadata(litertBuffer: ByteBuffer): List<String> {
        val metadataExtractor = MetadataExtractor(litertBuffer)
        val labels = mutableListOf<String>()
        if (metadataExtractor.hasMetadata()) {
            val inputStream = metadataExtractor.getAssociatedFile(options.currentModel.labelFile)
            labels.addAll(readFileInputStream(inputStream))
            Log.i(
                TAG, "Successfully loaded model metadata ${metadataExtractor.associatedFileNames}"
            )
        }
        return labels
    }

    /** Retrieve Map<String, Int> from metadata file */
    private suspend fun readFileInputStream(inputStream: InputStream): List<String> {
        return withContext(Dispatchers.IO) {
            val reader = BufferedReader(InputStreamReader(inputStream))

            val list = mutableListOf<String>()
            var index = 0
            var line = ""
            while (reader.readLine().also { if (it != null) line = it } != null) {
                list.add(line)
                index++
            }

            reader.close()
            list
        }
    }

    private fun convertShortToFloat(shortAudio: ShortArray): FloatArray {
        val audioLength = shortAudio.size
        val floatAudio = FloatArray(audioLength)

        // Loop and convert each short value to float
        for (i in 0 until audioLength) {
            floatAudio[i] = shortAudio[i].toFloat() / Short.MAX_VALUE
        }
        return floatAudio
    }

    companion object {
        private const val TAG = "SoundClassifier"

        val DEFAULT_MODEL = TFLiteModel.YAMNET
        val DEFAULT_DELEGATE = Delegate.CPU
        const val DEFAULT_THREAD_COUNT = 2
        const val DEFAULT_RESULT_COUNT = 3
        const val DEFAULT_OVERLAP = 0f
        const val DEFAULT_PROBABILITY_THRESHOLD = 0.3f
    }

    enum class Delegate {
        CPU, NNAPI
    }

    enum class TFLiteModel(val fileName: String, val labelFile: String, val sampleRate: Int) {
        YAMNET(
            "yamnet.tflite",
            "yamnet_label_list.txt",
            16000
        ),

        SpeechCommand(
            "speech.tflite",
            "probability_labels.txt",
            44100
        )
    }
}

data class Category(val label: String, val score: Float)
