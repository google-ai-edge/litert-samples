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

package com.google.ai.edge.examples.flux2_klein

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.io.File

/** Runs one int8 graph of the chunked FLUX.2-klein model on the LiteRT CompiledModel GPU. */
object ChunkRunner {

  /**
   * Loads one graph on the GPU, feeds [inputs] in convert-argument order, runs it, and returns
   * every output, freeing the native buffers and the graph before returning.
   *
   * The twelve graphs total 6.2 GB, past both the 2 GB flatbuffer load limit and the GPU memory
   * budget, so exactly one is resident at a time. FP32 compute is forced because the modulated
   * (adaLN) blocks overflow fp16 to NaN on the GPU delegate. A single shared [env] is reused across
   * every call: a per-call (null) environment leaks the ML Drift OpenCL context and aborts the
   * process after roughly twenty compiles. Every [com.google.ai.edge.litert.TensorBuffer] is closed
   * because leaked native buffers accumulate until the process is killed part-way through a run.
   *
   * @param env shared LiteRT environment reused across all graphs.
   * @param name graph file name inside [dir].
   * @param dir directory holding the staged `.tflite` graphs.
   * @param inputs input tensors in the graph's convert-argument order.
   * @return the graph's outputs as flat float arrays, in signature order.
   */
  fun gpu(
    env: Environment,
    name: String,
    dir: File,
    inputs: List<FloatArray>,
  ): List<FloatArray> {
    val options = CompiledModel.Options(Accelerator.GPU)
    options.gpuOptions =
      CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
    val model = CompiledModel.create(File(dir, name).absolutePath, options, env)
    val inputBuffers = model.createInputBuffers()
    val outputBuffers = model.createOutputBuffers()
    inputs.forEachIndexed { index, values -> inputBuffers[index].writeFloat(values) }
    model.run(inputBuffers, outputBuffers)
    val outputs = outputBuffers.map { it.readFloat() }
    inputBuffers.forEach { it.close() }
    outputBuffers.forEach { it.close() }
    model.close()
    return outputs
  }
}
