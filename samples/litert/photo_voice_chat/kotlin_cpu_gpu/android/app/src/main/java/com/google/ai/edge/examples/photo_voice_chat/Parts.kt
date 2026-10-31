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

package com.google.ai.edge.examples.photo_voice_chat

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future

/**
 * Where a graph runs: `cpu`, `gpu` (with its precision: fp32, fp16acc32, fp16, default) or `npu` (a Qualcomm NPU,
 * compiled on the phone when the graph is created; the ops it cannot take stay on the CPU). On the CPU, [weightCache]
 * is XNNPACK's weight cache FILE: the packed weights live there once for every signature of the graph (the language
 * model has seven; without the file XNNPACK packs the weights once per signature).
 */
data class Accel(
  val kind: String,
  val precision: String = "fp32",
  val threads: Int = 4,
  val weightCache: String? = null,
  /** GPU, a graph of many signatures (the language model): the GPU settings LiteRT-LM gives its decoder on Android =
   *  one copy of the weights for every signature (constant tensor sharing), quantized weights read as they are,
   *  weights in textures; with [gpuCacheDir], the compiled programs are kept in that dir under [gpuCacheKey] (also for
   *  a graph of one signature, the vision encoder and adapter: [programCached]). */
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
        gpuOptions = if (!sharedWeights && gpuCacheDir == null) CompiledModel.GpuOptions(precision = p)
        else if (!sharedWeights) CompiledModel.GpuOptions(
          precision = p,
          serializationDir = gpuCacheDir,
          modelCacheKey = gpuCacheKey,
          serializeProgramCache = true,
        ) else CompiledModel.GpuOptions(
          constantTensorSharing = true,
          allowSrcQuantizedFcConvOps = true,
          precision = p,
          preferTextureWeights = true,
          serializationDir = gpuCacheDir,
          modelCacheKey = gpuCacheKey,
          serializeProgramCache = if (gpuCacheDir != null) true else null,
        )
      }
      "npu" -> CompiledModel.Options(Accelerator.NPU, Accelerator.CPU).apply {
        cpuOptions = cpu
        qualcommOptions = CompiledModel.QualcommOptions(
          htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.HIGH_PERFORMANCE)
      }
      else -> error("accelerator $kind")
    }
  }

  override fun toString(): String = when (kind) {
    "gpu" -> "gpu($precision${if (sharedWeights) ",shared" else ""})"
    "cpu" -> "cpu($threads)"
    else -> kind
  }

  companion object {
    /** The accelerator names a depth decoder variant can carry ("<form>@<name>"; Lfm3OmniEngine.Config). */
    val NAMES = listOf("cpu", "gpu32", "gpu16a", "gpu16", "npu")

    /** cpu, gpu32 (float32), gpu16a (float16 with float32 accumulation), gpu16 (float16) or npu. */
    fun named(name: String, threads: Int = 4): Accel = when (name) {
      "cpu" -> Accel("cpu", threads = threads)
      "gpu32" -> Accel("gpu", "fp32")
      "gpu16a" -> Accel("gpu", "fp16acc32")
      "gpu16" -> Accel("gpu", "fp16")
      "npu" -> Accel("npu", threads = threads)
      else -> throw IllegalArgumentException("The accelerator $name is not one of ${NAMES.joinToString()}.")
    }
  }

  /** The same accelerator with a weight cache file in [dir] for [model] (the name follows the file's bytes): XNNPACK's
   *  on the CPU; on the GPU, the compiled programs of a [sharedWeights] graph (in [dir]/gpu). */
  fun cached(dir: File, model: File, tag: String): Accel = when {
    kind == "cpu" -> copy(weightCache = File(dir, "$tag-${model.length()}-${model.lastModified()}.xnnpack_cache").absolutePath)
    kind == "gpu" && sharedWeights ->
      copy(gpuCacheDir = File(dir, "gpu").apply { mkdirs() }.absolutePath, gpuCacheKey = "$tag-${model.length()}-${model.lastModified()}")
    else -> this
  }

  /** On the GPU, the same accelerator with the graph's compiled programs kept in [dir]/gpu under a key of [tag], the
   *  precision and [model]'s size and time (a program compiled at one precision is never read at another, and each graph
   *  has its own key); other accelerators unchanged. Without it, opening the vision encoder on a Pixel 11 Pro's GPU
   *  compiled its programs every time (13.0-24.9 s); with it, an open after the first reads them (1.0-1.7 s; README.md,
   *  Performance). */
  fun programCached(dir: File, model: File, tag: String): Accel = if (kind != "gpu") this else
    copy(gpuCacheDir = File(dir, "gpu").apply { mkdirs() }.absolutePath, gpuCacheKey = "$tag-$precision-${model.length()}-${model.lastModified()}")
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

  fun readInts(name: String): IntArray {
    val t0 = System.nanoTime()
    val v = output(name).readInt()
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
 * The depth decoder: the 12 codes of one speech frame from the LM hidden state, in three maskgit passes (the schedule
 * [4, 4, 4]). Four forms of the same computation, one per set of graph files (README.md, "Performance"):
 *   step   `depth_step`, one call per pass: the hidden state + the codes revealed so far + which codebooks are still
 *          hidden -> 12 x 2017 logits; the host samples the codes (SpeechGenerator), three calls a frame. [project]
 *          writes the frame's hidden state once, the passes read it from the same input buffer.
 *   split  `depth_project` once a frame (the hidden state -> the passes' input, 12 x 1024) and `depth_passes` per pass
 *          (that input + codes + hidden flags -> logits): the projection runs once a frame instead of three times.
 *   frame  `depth_frame`, one call a frame with the sampler inside the graph (temperature [FRAME_TEMPERATURE] and
 *          repetition penalty [FRAME_PENALTY] are its constants): the hidden state + the three passes' reveal rows +
 *          one uniform draw per codebook + which codes are in each codebook's penalty window -> the 12 codes.
 *   frame2 the frame form with the 12 codes as its only output (no logits to copy out) and each draw scaled by its
 *          codebook's own cdf total (the code = the number of cdf values at or below the draw times that total).
 * The counters add up every call of the form's graphs (write + run + read, and those parts). [variant] is the name the
 * engine opened it under ("<form>" or "<form>@<accelerator>"), [accel] where its graphs run.
 */
class DepthDecoder private constructor(
  val form: String,
  private val stepGraph: Graph?,
  private val projectGraph: Graph?,
  private val passesGraph: Graph?,
  private val frameGraph: Graph?,
) : AutoCloseable {
  val graphs: List<Graph> = listOfNotNull(stepGraph, projectGraph, passesGraph, frameGraph)
  /** True for step and split (the host samples each pass's codes), false for frame and frame2 (the graph samples). */
  val samplesOnHost: Boolean = frameGraph == null
  var variant: String = form
  val accel: String get() = graphs.first().accel.toString()

  val calls: Long get() = graphs.sumOf { it.calls }
  val ns: Long get() = graphs.sumOf { it.ns }
  val writeNs: Long get() = graphs.sumOf { it.writeNs }
  val runNs: Long get() = graphs.sumOf { it.runNs }
  val readNs: Long get() = graphs.sumOf { it.readNs }

  /** step / split: the frame's hidden state, once a frame (split: through `depth_project` into the passes' input). */
  fun project(hidden: FloatArray) {
    val s = stepGraph
    if (s != null) {
      s.timed { s.write("hidden", hidden) }
      return
    }
    val p = projectGraph ?: error("the depth form $form samples in its graph")
    val q = passesGraph!!
    val depthIn = p.timed {
      p.write("hidden", hidden)
      p.run(PROJECT_OUTPUTS)
      p.read("depth_in")
    }
    q.timed { q.write("depth_in", depthIn) }
  }

  /** step / split: one pass on the hidden state of the last [project]; returns logits [12 * 2017]. */
  fun pass(codes: IntArray, masked: FloatArray): FloatArray {
    val g = stepGraph ?: passesGraph ?: error("the depth form $form samples in its graph")
    return g.timed {
      g.writeInts("codes", codes)
      g.write("masked", masked)
      g.run(PASS_OUTPUTS)
      g.read("logits")
    }
  }

  /**
   * frame / frame2: the whole frame in one call. [reveal] = the three passes' rows (1 on the codebooks revealed at that
   * pass), [u] = one uniform draw per codebook, [presence] = 12 x 2017 (1 where that code is in the codebook's penalty
   * window). Returns the 12 codes (the frame graph's logits output is not read).
   */
  fun frame(hidden: FloatArray, reveal: Array<FloatArray>, u: FloatArray, presence: FloatArray): IntArray {
    val g = frameGraph ?: error("the depth form $form samples on the host")
    return g.timed {
      g.write("hidden", hidden)
      g.write("reveal0", reveal[0])
      g.write("reveal1", reveal[1])
      g.write("reveal2", reveal[2])
      g.write("u", u)
      g.write("presence", presence)
      g.run(if (form == FRAME2) FRAME2_OUTPUTS else FRAME_OUTPUTS)
      g.readInts("codes")
    }
  }

  override fun close() {
    for (g in graphs) runCatching { g.close() }
  }

  companion object {
    const val STEP = "step"
    const val SPLIT = "split"
    const val FRAME = "frame"
    const val FRAME2 = "frame2"
    val FORMS = listOf(STEP, SPLIT, FRAME, FRAME2)
    /** The frame graph's sampler constants (the model card's speech sampling). */
    const val FRAME_TEMPERATURE = 0.4f
    const val FRAME_PENALTY = 1.2f
    /**
     * The largest uniform draw given to the frame graph, 1 - 2^-11. The graph draws a code as the number of cdf values
     * at or below the draw, and its float32 cdf can total a little under 1; a draw above that total would count every
     * code, which the graph turns into the last code = the end of audio, and the speech would stop mid-sentence. With
     * the cap the top 2^-11 of the range goes to the code at that point of the cdf.
     */
    const val FRAME_U_MAX = 1f - 1f / 2048
    private val PROJECT_OUTPUTS = listOf("depth_in")
    private val PASS_OUTPUTS = listOf("logits")
    private val FRAME_OUTPUTS = listOf("codes", "logits")
    private val FRAME2_OUTPUTS = listOf("codes")

    fun step(graph: Graph) = DepthDecoder(STEP, graph, null, null, null)

    fun split(project: Graph, passes: Graph) = DepthDecoder(SPLIT, null, project, passes, null)

    fun frame(graph: Graph) = DepthDecoder(FRAME, null, null, null, graph)

    fun frame2(graph: Graph) = DepthDecoder(FRAME2, null, null, null, graph)
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
 * The bundle's audio section: 30 s of 16 kHz PCM as [1, 3000, 160] -> `features` [1, 375, 2048] + a uint8 `mask` (1 =
 * a row of the clip). The rows the language model sees are the mask's rows (LiteRT-LM's rule: up to the last 1); the
 * host formula of the graph and the feature extractor (the last non-zero sample, frames = 1 + ceil((n - 400) / 160),
 * three halvings) is kept as a check ([lastValid] = what the mask said, [lastHost] = the formula).
 */
class AudioEncoder(file: File, accel: Accel, env: Environment) : AutoCloseable {
  val graph = Graph(file, "encode", accel, env)

  var lastValid = -1
    private set
  var lastHost = -1
    private set

  /** Returns the valid tokens' feature rows ([n][2048]) of [pcm16k] (at most 30 s). */
  fun encode(pcm16k: FloatArray): List<FloatArray> {
    val n = minOf(pcm16k.size, WINDOW)
    val audio = FloatArray(WINDOW)
    System.arraycopy(pcm16k, 0, audio, 0, n)
    lastHost = validTokens(audio)
    var mask: ByteArray? = null
    val feats = graph.timed {
      graph.write("audio", audio)
      graph.run(listOf("features", "mask"))
      // the uint8 mask through the byte reader; when that reader refuses the type, the host formula (the same rule)
      mask = runCatching { graph.output("mask").readInt8() }.getOrNull()
      graph.read("features")
    }
    val m = mask
    var tokens = lastHost
    if (m != null) {
      tokens = 0
      for (i in m.indices.reversed()) if (m[i].toInt() != 0) {
        tokens = i + 1
        break
      }
    }
    lastValid = if (m != null) tokens else -1
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

/**
 * The bundle's two vision sections: a photo as [1, 512, 512, 3] floats (RGB, 0..1; the normalization is inside the
 * graph) -> the encoder's 1,024 patch rows [1, 1024, 1024] -> the adapter's 256 rows [1, 256, 2048] (each 2 x 2 block of
 * patches = one row, in the order of the `<|image|>` placeholders).
 */
class VisionEncoder(encoderFile: File, adapterFile: File, accel: Accel, adapterAccel: Accel, env: Environment) : AutoCloseable {
  val encoder = Graph(encoderFile, SIGNATURE, accel, env)
  val adapter: Graph = try {
    Graph(adapterFile, SIGNATURE, adapterAccel, env)
  } catch (t: Throwable) {
    encoder.close()
    throw t
  }

  /** [rgb] = SIZE x SIZE x 3 bytes (row-major RGB); returns the 256 rows the language model reads at the placeholders. */
  fun encode(rgb: ByteArray): List<FloatArray> {
    require(rgb.size == SIZE * SIZE * 3) { "the picture must be $SIZE x $SIZE RGB" }
    val x = FloatArray(rgb.size) { (rgb[it].toInt() and 0xff) / 255f }
    val patches = encoder.timed {
      encoder.write(INPUT, x)
      encoder.run(listOf(OUTPUT))
      encoder.read(OUTPUT)
    }
    val rows = adapter.timed {
      adapter.write(INPUT, patches)
      adapter.run(listOf(OUTPUT))
      adapter.read(OUTPUT)
    }
    val n = rows.size / AudioEncoder.HIDDEN
    return List(n) { t -> rows.copyOfRange(t * AudioEncoder.HIDDEN, (t + 1) * AudioEncoder.HIDDEN) }
  }

  override fun close() {
    encoder.close()
    adapter.close()
  }

  companion object {
    const val SIZE = 512
    const val ROWS = 256
    const val SIGNATURE = "serving_default"
    const val INPUT = "args_0"
    const val OUTPUT = "output_0"
  }
}
