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

package com.google.ai.edge.examples.zero_shot_classification

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.Closeable
import java.io.File
import java.util.concurrent.Callable
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors

/** Single-row multilingual Laya execution; GPU explicitly requests FP32 with no CPU fallback. */
class LayaEngine(context: Context, val storage: Storage = Storage.WFP16) : Closeable {
  /** Storage changes graph weights only; GPU arithmetic remains explicitly FP32 in both cases. */
  enum class Storage(val argument: String) {
    WFP16("wfp16"),
    FP32("fp32");

    companion object {
      /** Parses a supported intent selector, rejecting unknown values. */
      fun fromArgument(value: String): Storage =
        when (value.lowercase()) {
          "wfp16" -> WFP16
          "fp32" -> FP32
          else -> error("Unknown graph storage: $value; expected wfp16 or fp32")
        }
    }
  }

  /** Selects the accelerator explicitly; GPU creation never silently falls back. */
  enum class Backend(val accelerator: Accelerator) {
    GPU(Accelerator.GPU),
    CPU(Accelerator.CPU);

    companion object {
      /** Parses a supported intent selector, rejecting unknown values. */
      fun fromArgument(value: String): Backend =
        when (value.lowercase()) {
          "gpu" -> GPU
          "cpu" -> CPU
          else -> error("Unknown accelerator: $value; expected gpu or cpu")
        }
    }
  }

  /** Includes output readback because GPU run() can enqueue work asynchronously. */
  data class GraphTiming(val writeMs: Double, val enqueueMs: Double, val readbackMs: Double) {
    /** Total write, enqueue, and readback duration in milliseconds. */
    val totalMs: Double
      get() = writeMs + enqueueMs + readbackMs

    /** Emits named timing components for machine-readable gate reports. */
    fun toMap(): Map<String, Double> =
      linkedMapOf(
        "write_ms" to writeMs,
        "run_enqueue_ms" to enqueueMs,
        "readback_ms" to readbackMs,
        "write_run_read_ms" to totalMs,
      )
  }

  /** Raw model tensors and host/graph timings for one question row. */
  data class RawResult(
    val tokenLogits: FloatArray,
    val pooledCls: FloatArray,
    val markerLogits: FloatArray,
    val actFeatures: FloatArray,
    val actLogits: FloatArray,
    val mainTiming: GraphTiming,
    val actTiming: GraphTiming?,
    val embeddingLookupMs: Double,
    val featureMs: Double,
    val callMs: Double,
  ) {
    /** Rejects any NaN or infinity before displaying a decoded answer. */
    val finite: Boolean
      get() =
        tokenLogits.all { it.isFinite() } &&
          pooledCls.all { it.isFinite() } &&
          markerLogits.all { it.isFinite() } &&
          actFeatures.all { it.isFinite() } &&
          actLogits.all { it.isFinite() }

    /** Main plus act graph durations, excluding the separate embedding lookup. */
    val graphMs: Double
      get() = mainTiming.totalMs + (actTiming?.totalMs ?: 0.0)
  }

  /** Calibrated answer with its built sequence and complete per-question timing. */
  data class AnswerResult(
    val sequence: LayaSequence,
    val answer: Map<String, Any?>,
    val raw: RawResult,
    val prepareMs: Double,
    val decodeMs: Double,
    /** Total write, enqueue, and readback duration in milliseconds. */
    val totalMs: Double,
  )

  private data class Key(val window: Int, val backend: Backend, val storage: Storage)

  private class Graph(
    val model: CompiledModel,
    val inputs: Map<String, TensorBuffer>,
    val outputs: Map<String, TensorBuffer>,
  ) : Closeable {
    override fun close() {
      try {
        (inputs.values + outputs.values).forEach { it.close() }
      } finally {
        model.close()
      }
    }
  }

  private val filesDir = context.applicationContext.filesDir
  private val mainGraphs = linkedMapOf<Key, Graph>()
  private val actGraphs = linkedMapOf<Backend, Graph>()
  private val embeddingInputs = linkedMapOf<Int, FloatArray>()
  private val embeddings: LayaEmbeddings
  private val tokenizer: LayaTokenizer
  /** Installed temperatures used by the interactive calibrated decoder. */
  val calibration: LayaCalibration
  /** Tokenizer initialization duration in milliseconds. */
  val tokenizerLoadMs: Double
  /** Metadata parsing and table mapping duration in milliseconds. */
  val embeddingLoadMs: Double
  /** Metadata hash recorded in each device gate report. */
  val embeddingTableSha256: String
    get() = embeddings.sha256

  private var closed = false

  init {
    REQUIRED_FILES.forEach { requireFile(it) }
    val started = System.nanoTime()
    tokenizer = LayaTokenizer(requireFile("tokenizer.json"))
    tokenizerLoadMs = milliseconds(System.nanoTime() - started)
    calibration = LayaCalibration.load(requireFile("laya_ml_calibration.json"))
    val embeddingStarted = System.nanoTime()
    embeddings =
      LayaEmbeddings(requireFile("token_embeddings_fp16.bin"), requireFile("token_embeddings.json"))
    embeddingLoadMs = milliseconds(System.nanoTime() - embeddingStarted)
  }

  /**
   * Compile separately from measured calls so the first graph call can remain a cold observation.
   */
  fun initialize(backend: Backend, window: Int = 256) =
    LayaProcessRuntime.call {
      checkOpen()
      requireWindow(window)
      mainGraph(window, backend)
      actGraph(backend)
      Unit
    }

  /** The gate and interactive app use exactly the same tokenizer and prompt builder. */
  fun prepare(
    state: Any?,
    question: Map<String, Any?>,
    window: Int = 256,
    questionId: String = "",
  ): LayaSequence {
    checkOpen()
    requireWindow(window)
    return LayaPromptBuilder(tokenizer, maxLen = window, headMaxLen = 256)
      .build(state, question, questionId)
  }

  /**
   * One main invocation and one action-head invocation, including both readbacks. Named signature
   * buffers avoid the different signature versus FlatBuffer tensor orders in these artifacts.
   * Nonfinite main output is preserved in the result, and is never sent through the action head.
   */
  fun runRaw(sequence: LayaSequence, backend: Backend, window: Int = 256): RawResult =
    LayaProcessRuntime.call {
      checkOpen()
      requireWindow(window)
      require(sequence.ids.size <= window) { "Sequence exceeds window $window" }
      require(sequence.markers.size == sequence.question.optionCount) { "Missing option marker" }
      val main = mainGraph(window, backend)
      val act = actGraph(backend)
      val attention = FloatArray(window) { if (it < sequence.ids.size) 1f else 0f }
      val qtype = FloatArray(3).also { it[sequence.question.qtype] = 1f }

      val lookupStarted = System.nanoTime()
      val embeds = embeddingInputs.getOrPut(window) { FloatArray(window * LayaEmbeddings.WIDTH) }
      embeddings.gather(sequence.ids, window, embeds)
      val started = System.nanoTime()
      val embeddingLookupMs = milliseconds(started - lookupStarted)
      main.inputs.getValue("inputs_embeds").writeFloat(embeds)
      main.inputs.getValue("attention_mask").writeFloat(attention)
      main.inputs.getValue("qtype_onehot").writeFloat(qtype)
      val written = System.nanoTime()
      main.model.run(main.inputs, main.outputs, SIGNATURE)
      val enqueued = System.nanoTime()
      val tokens = main.outputs.getValue("token_logits").readFloat()
      val pooled = main.outputs.getValue("pooled_cls").readFloat()
      val read = System.nanoTime()
      require(tokens.size == window && pooled.size == 768) { "Unexpected main graph output shape" }
      val mainTiming = timing(started, written, enqueued, read)
      val markers = LayaDecoder.gather(tokens, sequence.markers)
      if (!tokens.all { it.isFinite() } || !pooled.all { it.isFinite() }) {
        return@call RawResult(
          tokens,
          pooled,
          markers,
          FloatArray(4) { Float.NaN },
          FloatArray(2) { Float.NaN },
          mainTiming,
          null,
          embeddingLookupMs,
          milliseconds(System.nanoTime() - read),
          milliseconds(System.nanoTime() - lookupStarted),
        )
      }
      val features = LayaDecoder.actFeatures(markers)
      val featured = System.nanoTime()
      act.inputs.getValue("pooled_cls").writeFloat(pooled)
      act.inputs.getValue("feats").writeFloat(features)
      val actWritten = System.nanoTime()
      act.model.run(act.inputs, act.outputs, SIGNATURE)
      val actEnqueued = System.nanoTime()
      val action = act.outputs.getValue("act_logits").readFloat()
      val finished = System.nanoTime()
      require(action.size == 2) { "Unexpected action graph output shape" }
      RawResult(
        tokens,
        pooled,
        markers,
        features,
        action,
        mainTiming,
        timing(featured, actWritten, actEnqueued, finished),
        embeddingLookupMs,
        milliseconds(featured - read),
        milliseconds(finished - lookupStarted),
      )
    }

  /** Runs builder, both graphs, and calibrated decoding with separate phase timings. */
  fun answer(
    state: Any?,
    question: Map<String, Any?>,
    backend: Backend,
    questionId: String = "",
    window: Int = 256,
  ): AnswerResult {
    val started = System.nanoTime()
    val sequence = prepare(state, question, window, questionId)
    val prepared = System.nanoTime()
    val raw = runRaw(sequence, backend, window)
    check(raw.finite) {
      "Nonfinite model output on ${backend.name}; inspect the device gate report"
    }
    val decodeStarted = System.nanoTime()
    val decoded =
      LayaDecoder.decode(raw.markerLogits, raw.actLogits, sequence.question, calibration)
    val finished = System.nanoTime()
    return AnswerResult(
      sequence,
      decoded,
      raw,
      milliseconds(prepared - started),
      milliseconds(finished - decodeStarted),
      milliseconds(finished - started),
    )
  }

  private fun mainGraph(window: Int, backend: Backend): Graph =
    mainGraphs.getOrPut(Key(window, backend, storage)) {
      createGraph(
        mainFilename(window),
        backend,
        listOf("attention_mask", "inputs_embeds", "qtype_onehot"),
        listOf("pooled_cls", "token_logits"),
      )
    }

  private fun actGraph(backend: Backend): Graph =
    actGraphs.getOrPut(backend) {
      createGraph(
        "laya_ml_act_head_fp32.tflite",
        backend,
        listOf("feats", "pooled_cls"),
        listOf("act_logits"),
      )
    }

  private fun createGraph(
    filename: String,
    backend: Backend,
    inputNames: List<String>,
    outputNames: List<String>,
  ): Graph {
    val options =
      CompiledModel.Options(backend.accelerator).apply {
        if (backend == Backend.GPU) {
          gpuOptions = CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
        } else {
          cpuOptions = CompiledModel.CpuOptions(numThreads = 4)
        }
      }
    val model =
      CompiledModel.create(
        requireFile(filename).absolutePath,
        options,
        LayaProcessRuntime.environment(),
      )
    val inputs = linkedMapOf<String, TensorBuffer>()
    val outputs = linkedMapOf<String, TensorBuffer>()
    try {
      inputNames.forEach { inputs[it] = model.createInputBuffer(it, SIGNATURE) }
      outputNames.forEach { outputs[it] = model.createOutputBuffer(it, SIGNATURE) }
      return Graph(model, inputs, outputs)
    } catch (failure: Throwable) {
      (inputs.values + outputs.values).forEach { it.close() }
      model.close()
      throw failure
    }
  }

  private fun requireFile(name: String): File =
    File(filesDir, name).also {
      check(it.isFile) { "Missing $name. Download the model files on the start screen." }
    }

  private fun checkOpen() = check(!closed) { "LayaEngine is closed" }

  /** Resolves the selected fixed-window storage variant without changing GPU precision. */
  fun mainFilename(window: Int): String = "laya_ml_s${window}_embeds_${storage.argument}.tflite"

  /** Closes model buffers and releases table references on the native runtime thread. */
  override fun close() =
    LayaProcessRuntime.call {
      if (!closed) {
        closed = true
        try {
          (mainGraphs.values + actGraphs.values).forEach { it.close() }
        } finally {
          mainGraphs.clear()
          actGraphs.clear()
          embeddingInputs.clear()
          embeddings.close()
        }
      }
    }

  companion object {
    /** Runtime pin shared with the version catalog and gate metadata. */
    const val LITERT_VERSION = "2.2.0"
    private const val SIGNATURE = "serving_default"
    /** Shared files required by either graph storage choice. */
    val REQUIRED_FILES =
      listOf(
        "laya_ml_act_head_fp32.tflite",
        "laya_ml_calibration.json",
        "tokenizer.json",
        "token_embeddings_fp16.bin",
        "token_embeddings.json",
      )

    private fun requireWindow(window: Int) =
      require(window == 256 || window == 512) {
        "Only multilingual windows 256 and 512 are supported"
      }

    private fun milliseconds(nanos: Long) = nanos / 1_000_000.0

    private fun timing(start: Long, written: Long, enqueued: Long, read: Long) =
      GraphTiming(
        milliseconds(written - start),
        milliseconds(enqueued - written),
        milliseconds(read - enqueued),
      )
  }
}

/** One native thread and one Environment for the lifetime of this application process. */
private object LayaProcessRuntime {
  private val executor =
    Executors.newSingleThreadExecutor { runnable ->
      Thread(runnable, "Laya-LiteRT").apply { isDaemon = true }
    }
  private var sharedEnvironment: Environment? = null

  fun environment(): Environment =
    sharedEnvironment ?: Environment.create().also { sharedEnvironment = it }

  fun <T> call(block: () -> T): T =
    try {
      executor.submit(Callable { block() }).get()
    } catch (failure: ExecutionException) {
      throw failure.cause ?: failure
    }
}
