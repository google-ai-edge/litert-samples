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

package com.google.edgeai.examples.text_classification

import android.content.Context
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
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
import java.nio.IntBuffer

class TextClassificationHelper(private val context: Context) {
    /** The TFLite interpreter instance.  */
    private var interpreter: Interpreter? = null
    private val vocabularyMap = mutableMapOf<String, Int>()

    init {
        initClassifier()
    }

    /** As the result of sound classification, this value emits map of probabilities */
    val percentages: SharedFlow<Pair<FloatArray, Long>>
        get() = _percentages
    private val _percentages = MutableSharedFlow<Pair<FloatArray, Long>>(
        extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    val error: SharedFlow<Throwable?>
        get() = _error
    private val _error = MutableSharedFlow<Throwable?>()

    var completableDeferred: CompletableDeferred<Unit>? = null

    /** Init a Interpreter from [model]. View Model enum here [TFLiteModel]*/
    fun initClassifier(model: TFLiteModel = TFLiteModel.MobileBERT) {
        interpreter = try {
            val tfliteBuffer = FileUtil.loadMappedFile(context, model.fileName)
            Log.i(TAG, "Done creating TFLite buffer from ${model.fileName}")
            loadModelMetadata(tfliteBuffer)
            Interpreter(tfliteBuffer, Interpreter.Options())
        } catch (e: Exception) {
            Log.i(TAG, "Create TFLite from ${model.fileName} is failed ${e.message}")
            return
        }
    }

    /** Stop current Interpreter*/
    fun stopClassify() {
        if (interpreter != null) {
            interpreter!!.close()
            interpreter = null
        }
    }


    /** Run classify [inputText] using TFLite Interpreter*/
    suspend fun classify(inputText: String) {
        withContext(Dispatchers.IO) {
            if (interpreter == null) return@withContext
            val inputShape = interpreter!!.getInputTensor(0)?.shape() ?: return@withContext
            val outputShape = interpreter!!.getOutputTensor(0)?.shape() ?: return@withContext

            val inputBuffer = IntBuffer.allocate(inputShape[1])
            val outputBuffer = FloatBuffer.allocate(outputShape[1])

            val tokenizerText = tokenizeText(inputText)
            if (tokenizerText.size > inputShape[1]) {
                Log.e(TAG, "The number of word exceeds the limit")
                _error.emit(Throwable("The number of word exceeds the limit, please input the number of word <= ${inputShape[1]}"))
                return@withContext
            }
            completableDeferred?.await()
            inputBuffer.put(tokenizerText.toIntArray())
            completableDeferred = CompletableDeferred()

            inputBuffer.rewind()
            outputBuffer.rewind()

            val startTime = SystemClock.uptimeMillis()
            interpreter!!.run(inputBuffer, outputBuffer)
            val inferenceTime = SystemClock.uptimeMillis() - startTime

            val output = FloatArray(outputBuffer.capacity())
            outputBuffer.rewind()
            outputBuffer.get(output)
            /*
             * MobileBERT labels: negative & positive
             * AverageWordVec: 0 & 1
             */
            _percentages.tryEmit(Pair(output, inferenceTime))
        }
    }

    /** Load metadata from model*/
    private fun loadModelMetadata(tfliteBuffer: ByteBuffer) {
        val metadataExtractor = MetadataExtractor(tfliteBuffer)
        if (metadataExtractor.hasMetadata()) {
            val vocalBuffer = metadataExtractor.getAssociatedFile("vocab.txt")
            vocabularyMap.putAll(getVocabulary(vocalBuffer))
            Log.i(TAG, "Successfully loaded model metadata")
        }
    }

    /** Tokenize the text from String to int[], based on the index of words in `voca.txt` */
    private fun tokenizeText(inputText: String): List<Int> {
        return try {
            val nonePunctuationText = removePunctuation(inputText)
            val result = nonePunctuationText.split(" ")
            val ids = mutableListOf<Int>()
            result.forEach { text ->
                if (vocabularyMap.containsKey(text)) {
                    ids.add(vocabularyMap[text]!!)
                } else {
                    ids.add(0)
                }
            }
            Log.i(TAG, "tokenizeText: $ids")
            return ids
        } catch (e: IOException) {
            Log.e(TAG, "Failed to read vocabulary.txt: ${e.message}")
            emptyList()
        }
    }

    /** Remove punctuation to reduce unnecessary inputs*/
    private fun removePunctuation(text: String): String {
        return text.replace("[^a-zA-Z0-9 ]".toRegex(), "")
    }

    /** Retrieve vocabularies from "vocab.txt" file metadata*/
    private fun getVocabulary(inputStream: InputStream): Map<String, Int> {
        val reader = BufferedReader(InputStreamReader(inputStream))

        val map = mutableMapOf<String, Int>()
        var index = 0
        var line = ""
        while (reader.readLine().also { if (it != null) line = it } != null) {
            map[line] = index
            index++
        }

        reader.close()
        Log.d(TAG, "loadVocabulary: ${map.size}")
        return map
    }

    companion object {
        private const val TAG = "TextClassifier"
    }

    enum class TFLiteModel(val fileName: String) {
        MobileBERT("mobile_bert.tflite"),
        AverageWordVec("word_vec.tflite")
    }
}