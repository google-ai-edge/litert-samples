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

package com.google.edgeai.examples.reinforcement_learning

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.FloatBuffer

class ReinforcementLearningHelper(private val context: Context) {
    /** The TFLite interpreter instance.  */
    private var interpreter: Interpreter? = null

    init {
        initHelper()
    }


    /** Init a Interpreter from asset*/
    private fun initHelper() {
        interpreter = try {
            val tfliteBuffer = FileUtil.loadMappedFile(context, "planestrike.tflite")
            Log.i(TAG, "Done creating TFLite buffer from asset")
            Interpreter(tfliteBuffer, Interpreter.Options())
        } catch (e: Exception) {
            Log.i(TAG, "Initializing TensorFlow Lite has failed with error: ${e.message}")
            return
        }
    }

    suspend fun predict(board: List<List<Int>>): Int {
        return withContext(Dispatchers.IO) {
            assert(interpreter != null) {
                "Interpreter is null sos please init Interpreter first"
            }
            val outputShape = interpreter!!.getOutputTensor(0).shape()

            val outputBuffer = FloatBuffer.allocate(outputShape[1])

            val inputArray = arrayOfIntArraysToFloatArrays(board)

            outputBuffer.rewind()

            interpreter!!.run(inputArray, outputBuffer)
            outputBuffer.rewind()

            val output = FloatArray(outputBuffer.capacity())

            outputBuffer.get(output)
            val max = output.max()
            val maxId = output.indexOfFirst {
                it == max
            }
            maxId
        }
    }


    private fun arrayOfIntArraysToFloatArrays(board: List<List<Int>>): Array<FloatArray> {
        return board.map {
            it.map { value -> value.toFloat() }.toFloatArray()
        }.toTypedArray()
    }


    companion object {
        private const val TAG = "ReinforcementLearningHelper"
    }
}
