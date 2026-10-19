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

package android.template.ui.classify

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer

private val environment by lazy { Environment.create() }

/** Runs model.tflite from the app's assets on the CPU. */
class Classifier(context: Context) : AutoCloseable {
    private val model = CompiledModel.create(
        context.assets,
        "model.tflite",
        CompiledModel.Options(Accelerator.CPU),
        environment,
    )
    private val inputs = mutableListOf<TensorBuffer>()
    private val outputs = mutableListOf<TensorBuffer>()

    init {
        try {
            inputs += model.createInputBuffers()
            outputs += model.createOutputBuffers()
        } catch (e: Exception) {
            close()
            throw e
        }
    }

    /** One score per class. */
    fun classify(input: FloatArray): FloatArray {
        inputs[0].writeFloat(input)
        model.run(inputs, outputs)
        return outputs[0].readFloat()
    }

    override fun close() {
        (inputs + outputs).forEach { it.close() }
        model.close()
    }
}
