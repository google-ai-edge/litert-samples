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

package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File

/**
 * One compiled decision graph with its input and output buffers addressed by signature name; the
 * graphs' storage order differs from their signature order, so index-based buffers would mix them
 * up. On the GPU every graph computes in FP32: each model card measured changed answers at the
 * default GPU precision.
 */
internal class DecisionGraph
private constructor(
  private val model: CompiledModel,
  private val inputs: Map<String, TensorBuffer>,
  private val outputs: Map<String, TensorBuffer>,
) : AutoCloseable {
  /** Input names in the order the graph was created with. */
  val inputNames: List<String>
    get() = inputs.keys.toList()

  fun write(name: String, values: FloatArray) {
    inputs.getValue(name).writeFloat(values)
  }

  /** Runs the graph; the readback in [read] is the call that waits for the GPU. */
  fun run() {
    model.run(inputs, outputs, SIGNATURE)
  }

  fun read(name: String): FloatArray = outputs.getValue(name).readFloat()

  @Volatile private var closed = false

  /** Releases the buffers, then the model. Safe to call more than once. */
  override fun close() {
    if (closed) return
    closed = true
    try {
      (inputs.values + outputs.values).forEach { runCatching { it.close() } }
    } finally {
      model.close()
    }
  }

  companion object {
    private const val SIGNATURE = "serving_default"

    /** Threads for the CPU path, as on the model cards. */
    private const val CPU_THREADS = 4

    /**
     * Compiles [file] for [accelerator] and allocates the named buffers. [inputNames] may be given
     * as a function of the input shapes, for graphs whose converter named the inputs `args_N`.
     */
    fun create(
      file: File,
      accelerator: Accelerator,
      inputNames: (dimensions: (String) -> List<Int>) -> List<String>,
      outputNames: List<String>,
    ): DecisionGraph {
      check(file.isFile) { "Model not downloaded: ${file.name}" }
      val options =
        CompiledModel.Options(accelerator).apply {
          if (accelerator == Accelerator.GPU) {
            gpuOptions =
              CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
          } else {
            cpuOptions = CompiledModel.CpuOptions(numThreads = CPU_THREADS)
          }
        }
      val model = CompiledModel.create(file.absolutePath, options, null)
      val inputs = linkedMapOf<String, TensorBuffer>()
      val outputs = linkedMapOf<String, TensorBuffer>()
      try {
        val names =
          inputNames { name ->
            model.getInputTensorType(name, SIGNATURE).layout?.dimensions.orEmpty()
          }
        names.forEach { inputs[it] = model.createInputBuffer(it, SIGNATURE) }
        outputNames.forEach { outputs[it] = model.createOutputBuffer(it, SIGNATURE) }
        return DecisionGraph(model, inputs, outputs)
      } catch (failure: Throwable) {
        (inputs.values + outputs.values).forEach { runCatching { it.close() } }
        model.close()
        throw failure
      }
    }

    /** [create] for a graph whose input names are fixed. */
    fun create(
      file: File,
      accelerator: Accelerator,
      inputNames: List<String>,
      outputNames: List<String>,
    ): DecisionGraph = create(file, accelerator, { inputNames }, outputNames)

    fun milliseconds(startNanos: Long): Double = (System.nanoTime() - startNanos) / 1_000_000.0
  }
}
