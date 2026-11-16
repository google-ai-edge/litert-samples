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

package __MODULE_PACKAGE__
// Vendored from common/kotlin/CompiledModelRunner.kt — edit the canonical and run tools/sync_common.py --apply.

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer

/**
 * Thin lifecycle wrapper around LiteRT [CompiledModel].
 *
 * Owns the compiled model plus one pre-allocated set of input/output buffers and releases all of
 * them in [close]. [TensorBuffer] is `AutoCloseable` too — forgetting to close the buffers leaks
 * native memory even when the model itself is closed. Construction is all-or-nothing: if a buffer
 * cannot be allocated, everything created before it is closed and the exception propagates, so a
 * caller that falls back from GPU to CPU never leaves a half-built GPU model behind.
 *
 * API notes that are easy to miss:
 * - `CompiledModel` with [Accelerator.GPU] requires every op in the graph to be GPU-compatible.
 *   There is no CPU fallback — an unsupported op fails compilation.
 * - [run] enqueues work on the GPU and may return before the computation has finished; the
 *   readback ([readOutput]) is the call that waits. Always benchmark run + readback together,
 *   never run() alone.
 * - For stateful/recurrent models, create a second buffer set and feed step N's output buffers as
 *   step N+1's inputs via [run] (buffer ping-pong) instead of copying state through the host each
 *   step.
 */
class CompiledModelRunner private constructor(private val model: CompiledModel) : AutoCloseable {

  /** Pre-allocated input buffers, index-aligned with the model's input tensors. */
  val inputBuffers: List<TensorBuffer>

  /** Pre-allocated output buffers, index-aligned with the model's output tensors. */
  val outputBuffers: List<TensorBuffer>

  init {
    var inputs: List<TensorBuffer>? = null
    try {
      inputs = model.createInputBuffers()
      inputBuffers = inputs
      outputBuffers = model.createOutputBuffers()
    } catch (failure: Throwable) {
      inputs?.forEach { runCatching { it.close() } }
      runCatching { model.close() }
      throw failure
    }
  }

  companion object {
    /**
     * Compiles a model bundled in `assets/`. The module's Gradle config must set
     * `androidResources { noCompress += "tflite" }` so the asset stays mmappable.
     */
    fun fromAssets(
      context: Context,
      fileName: String,
      vararg accelerators: Accelerator = arrayOf(Accelerator.GPU),
    ): CompiledModelRunner =
      CompiledModelRunner(
        CompiledModel.create(context.assets, fileName, CompiledModel.Options(*accelerators), null)
      )

    /** Compiles a model from an absolute file path — the pattern for downloaded models. */
    fun fromFile(
      path: String,
      vararg accelerators: Accelerator = arrayOf(Accelerator.GPU),
    ): CompiledModelRunner = fromFile(path, CompiledModel.Options(*accelerators), null)

    /**
     * Compiles a model from an absolute file path with explicit [options] (GPU precision, NPU
     * settings) and an optional shared [environment], which the caller keeps owning.
     */
    fun fromFile(
      path: String,
      options: CompiledModel.Options,
      environment: Environment?,
    ): CompiledModelRunner = CompiledModelRunner(CompiledModel.create(path, options, environment))
  }

  /** Writes [data] into input tensor [index]. */
  fun writeInput(index: Int, data: FloatArray) {
    inputBuffers[index].writeFloat(data)
  }

  /** Runs the model on the pre-allocated buffer sets (asynchronous — see class KDoc). */
  fun run() {
    model.run(inputBuffers, outputBuffers)
  }

  /** Runs the model on caller-provided buffers (buffer ping-pong, multi-graph pipelines). */
  fun run(inputs: List<TensorBuffer>, outputs: List<TensorBuffer>) {
    model.run(inputs, outputs)
  }

  /** Reads output tensor [index] back to the host. This is the call that waits for the GPU. */
  fun readOutput(index: Int): FloatArray = outputBuffers[index].readFloat()

  @Volatile private var closed = false

  /** Releases the buffers, then the model. Safe to call more than once. */
  override fun close() {
    if (closed) return
    closed = true
    outputBuffers.forEach { runCatching { it.close() } }
    inputBuffers.forEach { runCatching { it.close() } }
    model.close()
  }
}
