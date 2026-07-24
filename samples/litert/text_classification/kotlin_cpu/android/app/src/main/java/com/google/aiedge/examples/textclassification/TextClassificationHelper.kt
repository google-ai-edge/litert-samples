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

package com.google.aiedge.examples.textclassification

import android.content.Context
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.withContext
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.metadata.MetadataExtractor
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Accelerator
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer

class TextClassificationHelper(private val context: Context) {
    /** The LiteRT CompiledModel instance.  */
    private var model: CompiledModel? = null
    private val vocabularyMap = mutableMapOf<String, Int>()
    
    // Store current model to infer input sizes dynamically
    private var currentModel: Model = Model.MobileBERT

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

    /** Init a CompiledModel from [model]. View Model enum here [Model]*/
    fun initClassifier(modelEnum: Model = Model.MobileBERT) {
        try {
            currentModel = modelEnum
            // Load vocabulary using MetadataExtractor
            val litertBuffer = FileUtil.loadMappedFile(context, modelEnum.fileName)
            loadModelMetadata(litertBuffer)
            
            // Create CompiledModel
            model = CompiledModel.create(
                context.assets,
                modelEnum.fileName,
                CompiledModel.Options(Accelerator.CPU),
                null
            )
            Log.i(TAG, "Done creating CompiledModel from ${modelEnum.fileName} on CPU")
        } catch (e: Exception) {
            Log.e(TAG, "Create CompiledModel from ${modelEnum.fileName} has failed: ${e.message}")
        }
    }

    /** Stop current model*/
    fun stopClassify() {
        model?.close()
        model = null
    }


    /** Run classify [inputText] using LiteRT CompiledModel API*/
    suspend fun classify(inputText: String) {
        withContext(Dispatchers.IO) {
            val localModel = model ?: return@withContext
            
            try {
                // 1. Prepare buffers
                val inputBuffers = localModel.createInputBuffers()
                val outputBuffers = localModel.createOutputBuffers()
                
                // MobileBERT expects 128 tokens (512 bytes buffer), 
                // AverageWordVec expects 256 tokens (1024 bytes buffer).
                val expectedInputSize = if (currentModel == Model.MobileBERT) 128 else 256

                val tokenizerText = tokenizeText(inputText)
                if (tokenizerText.size > expectedInputSize) {
                    Log.e(TAG, "The number of word exceeds the limit: $expectedInputSize")
                    _error.emit(Throwable("The number of word exceeds the limit, please input the number of word <= $expectedInputSize"))
                    
                    for (buffer in inputBuffers) { buffer.close() }
                    for (buffer in outputBuffers) { buffer.close() }
                    return@withContext
                }
                
                // Write input_ids
                val paddedIntArray = IntArray(expectedInputSize) { index ->
                    if (index < tokenizerText.size) tokenizerText[index] else 0
                }
                inputBuffers[0].writeInt(paddedIntArray)

                // If the model expects multiple inputs (like MobileBERT), we must initialize them 
                // to prevent raw native uninitialized memory garbage from crashing the ML operations.
                if (inputBuffers.size > 1) {
                    // input_mask -> 1s for valid tokens, 0 for padded space
                    val inputMaskArray = IntArray(expectedInputSize) { index ->
                        if (index < tokenizerText.size) 1 else 0
                    }
                    inputBuffers[1].writeInt(inputMaskArray)
                }

                if (inputBuffers.size > 2) {
                    // segment_ids -> 0s for all tokens
                    val segmentIdsArray = IntArray(expectedInputSize) { 0 }
                    inputBuffers[2].writeInt(segmentIdsArray)
                }

                val startTime = SystemClock.uptimeMillis()
                localModel.run(inputBuffers, outputBuffers)
                val inferenceTime = SystemClock.uptimeMillis() - startTime

                val outputFloatArray = outputBuffers[0].readFloat()
                
                // Close buffers to free memory
                for (buffer in inputBuffers) { buffer.close() }
                for (buffer in outputBuffers) { buffer.close() }

                /*
                 * MobileBERT labels: negative & positive
                 * AverageWordVec: 0 & 1
                 */
                _percentages.tryEmit(Pair(outputFloatArray, inferenceTime))
            } catch (e: Exception) {
                Log.e(TAG, "Error during classification: ${e.message}")
                _error.emit(Throwable("Error running inference: ${e.message}"))
            }
        }
    }

    /** Load metadata from model*/
    private fun loadModelMetadata(litertBuffer: ByteBuffer) {
        val metadataExtractor = MetadataExtractor(litertBuffer)
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

    enum class Model(val fileName: String) {
        MobileBERT("mobile_bert.tflite"),
        AverageWordVec("word_vec.tflite")
    }
}