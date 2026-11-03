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

package com.google.ai.edge.examples.voicemail_summary

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future

/**
 * Where a graph runs: `cpu`, `gpu` (with its precision: fp32, fp16acc32, fp16, default) or `npu`. On the CPU,
 * [weightCache] is XNNPACK's weight cache FILE: the packed weights live there once for every signature of the graph
 * (the language model has seven; without the file XNNPACK packs the int4 weights once per signature, and the phone
 * killed the app at 2.8 GB RSS + 1.1 GB swap while the second signature packed).
 */
data class Accel(
  val kind: String,
  val precision: String = "fp32",
  val threads: Int = 4,
  val weightCache: String? = null,
  /** The HTP's performance mode on the NPU: burst (default here: one answer is a burst of short calls), high,
   *  sustained, balanced or default (the dispatch library's own, which ran the depth pass at 9 ms against 2.2 ms). */
  val npuPerf: String = "burst",
  /** GPU, a graph of many signatures (the language model): the GPU settings LiteRT-LM gives its decoder on Android =
   *  one copy of the weights for every signature (constant tensor sharing), quantized weights read as they are,
   *  weights in textures; with [gpuCacheDir], the compiled programs are kept in that dir under [gpuCacheKey]. */
  val sharedWeights: Boolean = false,
  val gpuCacheDir: String? = null,
  val gpuCacheKey: String? = null,
) {
  fun options(): CompiledModel.Options {
    val cpu = CompiledModel.CpuOptions(numThreads = threads, xnnPackWeightCachePath = weightCache)
    return when (kind) {
      "cpu" -> CompiledModel.Options(Accelerator.CPU).apply { cpuOptions = cpu }
      "gpu" -> CompiledModel.Options(Accelerator.GPU).apply {
        val p = when (precision) {
          "fp32" -> CompiledModel.GpuOptions.Precision.FP32
          "fp16acc32" -> CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM
          "fp16" -> CompiledModel.GpuOptions.Precision.FP16
          else -> CompiledModel.GpuOptions.Precision.DEFAULT
        }
        gpuOptions = if (!sharedWeights) CompiledModel.GpuOptions(precision = p) else CompiledModel.GpuOptions(
          constantTensorSharing = true,
          allowSrcQuantizedFcConvOps = true,
          precision = p,
          preferTextureWeights = true,
          serializationDir = gpuCacheDir,
          modelCacheKey = gpuCacheKey,
          serializeProgramCache = if (gpuCacheDir != null) true else null,
        )
      }
      "npu" -> CompiledModel.Options(Accelerator.NPU).apply {
        cpuOptions = cpu
        val mode = when (npuPerf) {
          "burst" -> CompiledModel.QualcommOptions.HtpPerformanceMode.BURST
          "high" -> CompiledModel.QualcommOptions.HtpPerformanceMode.HIGH_PERFORMANCE
          "sustained" -> CompiledModel.QualcommOptions.HtpPerformanceMode.SUSTAINED_HIGH_PERFORMANCE
          "balanced" -> CompiledModel.QualcommOptions.HtpPerformanceMode.BALANCED
          else -> null
        }
        if (mode != null) qualcommOptions = CompiledModel.QualcommOptions(htpPerformanceMode = mode)
      }
      else -> error("accelerator $kind")
    }
  }

  override fun toString(): String = when (kind) {
    "gpu" -> "gpu($precision${if (sharedWeights) ",shared" else ""})"
    "cpu" -> "cpu($threads)"
    "npu" -> "npu($npuPerf)"
    else -> kind
  }

  /** The same accelerator with a weight cache file in [dir] for [model] (the name follows the file's bytes): XNNPACK's
   *  on the CPU; on the GPU, the compiled programs of a [sharedWeights] graph (in [dir]/gpu). */
  fun cached(dir: File, model: File, tag: String): Accel = when {
    kind == "cpu" -> copy(weightCache = File(dir, "$tag-${model.length()}-${model.lastModified()}.xnnpack_cache").absolutePath)
    kind == "gpu" && sharedWeights ->
      copy(gpuCacheDir = File(dir, "gpu").apply { mkdirs() }.absolutePath, gpuCacheKey = "$tag-${model.length()}-${model.lastModified()}")
    else -> this
  }
}

/**
 * One compiled graph with its own buffers, called by name; the wall time of every call (write + run + read, [ns]) and
 * its parts: the host -> buffer copies ([writeNs], through [write] / [writeInts]), the run ([runNs]) and the
 * buffer -> host copies ([readNs], through [read]). On the GPU the run only queues the work: the read waits for it.
 */
class Graph(file: File, val signature: String, val accel: Accel, env: Environment) : AutoCloseable {
  val model: CompiledModel = CompiledModel.create(file.absolutePath, accel.options(), env)
  private val inputs = HashMap<String, TensorBuffer>()
  private val outputs = HashMap<String, TensorBuffer>()
  var calls = 0L
    private set
  var ns = 0L
    private set
  var writeNs = 0L
    private set
  var runNs = 0L
    private set
  var readNs = 0L
    private set

  fun input(name: String): TensorBuffer = inputs.getOrPut(name) { model.createInputBuffer(name, signature) }
  fun output(name: String): TensorBuffer = outputs.getOrPut(name) { model.createOutputBuffer(name, signature) }

  fun write(name: String, v: FloatArray) {
    val t0 = System.nanoTime()
    input(name).writeFloat(v)
    writeNs += System.nanoTime() - t0
  }

  fun writeInts(name: String, v: IntArray) {
    val t0 = System.nanoTime()
    input(name).writeInt(v)
    writeNs += System.nanoTime() - t0
  }

  fun read(name: String): FloatArray {
    val t0 = System.nanoTime()
    val v = output(name).readFloat()
    readNs += System.nanoTime() - t0
    return v
  }

  /** Runs with the inputs written before (by [input]) into every output named in [outs]. */
  fun run(outs: List<String>) {
    for (o in outs) output(o)
    val t0 = System.nanoTime()
    model.run(inputs, outputs, signature)
    runNs += System.nanoTime() - t0
    calls++
  }

  inline fun <T> timed(block: () -> T): T {
    val t0 = System.nanoTime()
    try {
      return block()
    } finally {
      addNs(System.nanoTime() - t0)
    }
  }

  fun addNs(n: Long) {
    ns += n
  }

  override fun close() = model.close()
}

/**
 * The depth decoder split in two graphs: `depth_proj` (the LM hidden -> the 12 codebook inputs, once per frame) and
 * `depth_pass` (one maskgit pass: inputs + the codes revealed so far + which codebooks are still hidden -> 12 x 2017
 * logits). A frame is one projection and three passes (the schedule [4, 4, 4]). The two graphs are created by the
 * engine (the pass may be an NPU form with a CPU fallback).
 */
class DepthDecoder(val projGraph: Graph, val passGraph: Graph) : AutoCloseable {

  fun project(hidden: FloatArray) {
    projGraph.timed {
      projGraph.write("hidden", hidden)
      projGraph.run(listOf("depth_in"))
      passGraph.write("depth_in", projGraph.read("depth_in"))
    }
  }

  /** One pass on the inputs of the last [project]; returns logits [12 * 2017]. */
  fun pass(codes: IntArray, masked: FloatArray): FloatArray = passGraph.timed {
    passGraph.writeInts("codes", codes)
    passGraph.write("masked", masked)
    passGraph.run(listOf("logits"))
    passGraph.read("logits")
  }

  override fun close() {
    projGraph.close()
    passGraph.close()
  }
}

/** A speech frame's 12 codes -> the LM input of the next slot (the sum of the 12 codebook embeddings). */
class FrameEmbed(file: File, accel: Accel, env: Environment) : AutoCloseable {
  val graph = Graph(file, "frame_embed", accel, env)

  fun embed(codes: IntArray): FloatArray = graph.timed {
    graph.writeInts("codes", codes)
    graph.run(listOf("embeds"))
    graph.read("embeds")
  }

  override fun close() = graph.close()
}

/**
 * The codec decoder: [framesPerCall] codec frames (12 codes each) + its streaming state -> windowed time frames
 * (6 per codec frame, 1280 samples each), overlap-added at hop 320 by [CodecStream]. The state (the last 29 keys and
 * values of the three attention layers, the last 6 inputs of the five conv layers) goes back in on the next call;
 * a short last chunk is padded with code 0 and its extra steps dropped (the decoder is causal).
 * The samples of each chunk go to [sink], in order. With [overlap] the decoder has a thread of its own (the graph is
 * created and called only there): a full chunk is decoded while the language model generates the next frames, [push]
 * returns at once (it waits only for the chunk before, done long before: 12 frames take ~0.4 s, a chunk ~20 ms), and
 * [finish] returns when the last samples are out. The calls and their inputs are the same either way.
 */
class CodecDecoder(file: File, accel: Accel, env: Environment, window: FloatArray, val framesPerCall: Int = 12,
                   val overlap: Boolean = false) : AutoCloseable {
  private val thread: ExecutorService? =
    if (overlap) Executors.newSingleThreadExecutor { r -> Thread(r, "codec").apply { isDaemon = true } } else null
  val graph: Graph = try {
    onCodecThread { Graph(file, "decode", accel, env) }
  } catch (e: Throwable) {
    thread?.shutdown()
    throw e
  }
  private val stream = CodecStream(window)
  private val stateNames = listOf("k0", "v0", "k1", "v1", "k2", "v2", "c0", "c1", "c2", "c3", "c4")
  private val state = HashMap<String, FloatArray>()
  private val pending = ArrayList<IntArray>()
  private var done = 0
  private var last: Future<*>? = null
  /** Receives each chunk's final samples (on the codec thread with [overlap]). */
  var sink: (FloatArray) -> Unit = {}
  /** The overlap-add's own time (host). */
  val olaNs: Long get() = stream.ns

  init {
    reset()
  }

  private fun <T> onCodecThread(block: () -> T): T {
    val t = thread ?: return block()
    return try {
      t.submit(java.util.concurrent.Callable { block() }).get()
    } catch (e: java.util.concurrent.ExecutionException) {
      throw e.cause ?: e
    }
  }

  fun reset() {
    last?.let { runCatching { it.get() } }
    last = null
    pending.clear()
    onCodecThread {
      for (n in stateNames) state[n] = FloatArray(LmRunner.elements(graph.model.getInputTensorType(n, "decode")))
      done = 0
      stream.reset()
    }
  }

  /** Adds one frame's codes; a full chunk is decoded and its samples go to [sink]. */
  fun push(codes: IntArray) {
    pending.add(codes.copyOf())
    if (pending.size < framesPerCall) return
    submit(pending.toList(), finalize = false)
    pending.clear()
  }

  /** The rest of the audio (the pending frames, padded, or the held tail) to [sink]; returns when all of it is out. */
  fun finish() {
    if (pending.isNotEmpty()) submit(pending.toList(), finalize = true) else submit(null, finalize = true)
    pending.clear()
    last?.get()
    last = null
  }

  /** Decodes [chunk] (null = only the held tail) here or on the codec thread; a failure of the chunk before surfaces. */
  private fun submit(chunk: List<IntArray>?, finalize: Boolean) {
    val job = { sink(if (chunk == null) stream.finish() else call(chunk, finalize)) }
    val t = thread
    if (t == null) {
      job()
      return
    }
    last?.get()
    last = t.submit(job)
  }

  private fun call(chunk: List<IntArray>, finalize: Boolean): FloatArray {
    val valid = chunk.size
    val f = framesPerCall
    val frames = graph.timed {
      // codes [1, 12, F]: codebook k, frame j at k * F + j; frames past `valid` are code 0
      val codes = IntArray(NUM_CODEBOOKS * f)
      for (j in 0 until valid) for (k in 0 until NUM_CODEBOOKS) codes[k * f + j] = chunk[j][k]
      graph.writeInts("codes", codes)
      graph.writeInts("start", intArrayOf(done * FRAME_STACK))
      for (n in stateNames) graph.write(n, state.getValue(n))
      graph.run(listOf("frames") + stateNames.map { it + "_out" })
      for (n in stateNames) state[n] = graph.read(n + "_out")
      graph.read("frames")
    }
    done += valid
    return stream.push(frames, valid * FRAME_STACK, finalize)
  }

  override fun close() {
    runCatching { last?.get() }
    onCodecThread { graph.close() }
    thread?.shutdown()
  }

  companion object {
    const val NUM_CODEBOOKS = 12
    const val FRAME_STACK = 6
  }
}

/**
 * Overlap-add of the codec's windowed frames (hop 320, n_fft 1280), divided by the overlap-added window power
 * (envelope); the first 480 samples of the stream are dropped, the last 960 held until the next call, and on the last
 * call the final 480 dropped instead (LiquidCodecModel.decode, streaming branch).
 */
class CodecStream(window: FloatArray) {
  private val nFft = window.size
  private val hop = 320
  private val trim = (nFft - hop) / 2
  private val tailLen = 2 * trim
  private val w2 = FloatArray(nFft) { window[it] * window[it] }
  private var tailAudio: FloatArray? = null
  private var tailEnv: FloatArray? = null
  private var started = false
  /** Time spent in [push] (the host's overlap-add and envelope division). */
  var ns = 0L
    private set

  fun reset() {
    tailAudio = null
    tailEnv = null
    started = false
  }

  /** [frames] holds `count` frames of [nFft] samples. */
  fun push(frames: FloatArray, count: Int, finalize: Boolean): FloatArray {
    val t0 = System.nanoTime()
    try {
      return pushUntimed(frames, count, finalize)
    } finally {
      ns += System.nanoTime() - t0
    }
  }

  private fun pushUntimed(frames: FloatArray, count: Int, finalize: Boolean): FloatArray {
    val size = (count - 1) * hop + nFft
    val audio = FloatArray(size)
    val env = FloatArray(size)
    for (i in 0 until count) {
      val o = i * hop
      val src = i * nFft
      for (s in 0 until nFft) {
        audio[o + s] += frames[src + s]
        env[o + s] += w2[s]
      }
    }
    tailAudio?.let { t -> for (s in t.indices) audio[s] += t[s] }
    tailEnv?.let { t -> for (s in t.indices) env[s] += t[s] }
    var from = 0
    if (!started) {
      from = trim
      started = true
    }
    val to: Int
    if (finalize) {
      to = size - trim
      tailAudio = null
      tailEnv = null
    } else {
      to = size - tailLen
      tailAudio = audio.copyOfRange(size - tailLen, size)
      tailEnv = env.copyOfRange(size - tailLen, size)
    }
    return FloatArray(maxOf(0, to - from)) { audio[from + it] / env[from + it] }
  }

  fun finish(): FloatArray {
    val a = tailAudio ?: return FloatArray(0)
    val e = tailEnv!!
    tailAudio = null
    tailEnv = null
    val n = a.size - trim
    return FloatArray(n) { a[it] / e[it] }
  }
}

/**
 * The bundle's audio section: 30 s of 16 kHz PCM as [1, 3000, 160] -> `features` [1, 375, 2048] (+ a uint8 `mask` the
 * app does not read: the valid token count is computed on the host with the same formula as the graph and the
 * feature extractor: the last non-zero sample, frames = 1 + ceil((n - 400) / 160), three halvings).
 */
class AudioEncoder(file: File, accel: Accel, env: Environment) : AutoCloseable {
  val graph = Graph(file, "encode", accel, env)

  /** Returns the valid tokens' feature rows ([n][2048]) of [pcm16k] (at most 30 s). */
  fun encode(pcm16k: FloatArray): List<FloatArray> {
    val n = minOf(pcm16k.size, WINDOW)
    val audio = FloatArray(WINDOW)
    System.arraycopy(pcm16k, 0, audio, 0, n)
    val tokens = validTokens(audio)
    val feats = graph.timed {
      graph.write("audio", audio)
      graph.run(listOf("features", "mask"))
      graph.read("features")
    }
    return List(tokens) { t -> feats.copyOfRange(t * HIDDEN, (t + 1) * HIDDEN) }
  }

  override fun close() = graph.close()

  companion object {
    const val WINDOW = 480_000
    const val HIDDEN = 2048

    fun validTokens(audio: FloatArray): Int {
      var last = -1
      for (i in audio.indices.reversed()) if (audio[i] != 0f) {
        last = i
        break
      }
      val nValid = (last + 1).toDouble()
      val frames = maxOf(1.0, 1.0 + Math.ceil((nValid - 400.0) / 160.0))
      val l2 = Math.floor((frames + 1.0) / 2.0)
      val l4 = Math.floor((l2 + 1.0) / 2.0)
      return Math.floor((l4 + 1.0) / 2.0).toInt()
    }
  }
}
