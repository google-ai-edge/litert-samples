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

package com.google.ai.edge.examples.ocr_verification

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.Closeable
import java.io.File

/**
 * The graph files of one kind (`tower_N<N>[_qb<Q>]_fp16fc.tflite` or
 * `decide_L<L>[_qb<Q>]_f16sfc.tflite`) in the model folder, one file per size: a request takes the
 * smallest size that holds it, and a file is built on the GPU the first time a request needs it. At
 * most [keep] files stay built; the least recently used one is closed first.
 */
class BucketGraphs(
  private val env: Environment,
  dir: File,
  private val kind: Kind,
  private val keep: Int,
) : Closeable {
  enum class Kind(
    val prefix: String,
    val form: String,
    val signature: String,
    val inputs: List<String>,
    val output: String,
  ) {
    /** The picture tower at float32: its rows are wrong at the GPU's default precision. */
    TOWER("tower_N", "fp16fc", "tower", listOf("patches", "valid", "pos", "cos", "sin"), "rows"),
    /** The decision graph at the GPU's default precision (its final norm fits float16). */
    DECIDE("decide_L", "f16sfc", "decide", listOf("embeds", "valid", "cos", "sin"), "hidden"),
  }

  private class Built(
    val size: Int,
    val model: CompiledModel,
    val inputs: Map<String, TensorBuffer>,
    val output: TensorBuffer,
  ) : Closeable {
    override fun close() {
      inputs.values.forEach { runCatching { it.close() } }
      runCatching { output.close() }
      runCatching { model.close() }
    }
  }

  /** size -> file, smallest first. */
  val files: Map<Int, File> = discover(dir, kind)
  private val built = LinkedHashMap<Int, Built>()

  /** Build time of the last file built, in ms (0 when the request found it built). */
  var lastBuildMs = 0.0
    private set

  val maxSize: Int
    get() = files.keys.maxOrNull() ?: 0

  fun sizeFor(n: Int): Int? = files.keys.firstOrNull { n <= it }

  fun isBuilt(size: Int): Boolean = built.containsKey(size)

  /** Builds the file of [size] if needed and runs it: inputs by name, the output's values back. */
  fun run(size: Int, feed: Map<String, FloatArray>): FloatArray {
    val g = graph(size)
    for (name in kind.inputs) {
      g.inputs.getValue(name).writeFloat(feed.getValue(name))
    }
    g.model.run(g.inputs, mapOf(kind.output to g.output), kind.signature)
    return g.output.readFloat()
  }

  private fun graph(size: Int): Built {
    built.remove(size)?.let {
      built[size] = it
      lastBuildMs = 0.0
      return it
    }
    while (built.size >= keep) {
      val oldest = built.keys.first()
      built.remove(oldest)?.close()
    }
    val file = files[size] ?: error("no ${kind.prefix}$size file in the model folder")
    val t = System.nanoTime()
    val options =
      CompiledModel.Options(Accelerator.GPU).apply {
        gpuOptions =
          CompiledModel.GpuOptions(
            constantTensorSharing = true,
            precision =
              if (kind == Kind.TOWER) {
                CompiledModel.GpuOptions.Precision.FP32
              } else {
                CompiledModel.GpuOptions.Precision.DEFAULT
              },
          )
      }
    val model = CompiledModel.create(file.absolutePath, options, env)
    val b =
      try {
        // buffers by name: createInputBuffers() lists the subgraph's own tensor order
        Built(
          size,
          model,
          kind.inputs.associateWith { model.createInputBuffer(it, kind.signature) },
          model.createOutputBuffer(kind.output, kind.signature),
        )
      } catch (e: Throwable) {
        model.close()
        throw e
      }
    lastBuildMs = (System.nanoTime() - t) / 1e6
    built[size] = b
    return b
  }

  /** Closes every built file (the files stay in the folder). */
  fun release() {
    built.values.forEach { it.close() }
    built.clear()
  }

  override fun close() = release()

  companion object {
    fun discover(dir: File, kind: Kind): Map<Int, File> {
      val pattern = Regex("^${kind.prefix}(\\d+)(?:_qb\\d+)?_${kind.form}\\.tflite$")
      val out = sortedMapOf<Int, File>()
      dir.listFiles()?.forEach { f ->
        pattern.matchEntire(f.name)?.let { m -> out[m.groupValues[1].toInt()] = f }
      }
      return out
    }
  }
}
