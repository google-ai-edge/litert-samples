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

package com.google.ai.edge.examples.image_cutout

import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.ShortBuffer
import java.nio.channels.FileChannel

/**
 * How the language model graph (7 signatures, 56 KV states) runs. [needMb]: the MemAvailable the
 * language model takes when it is built with its buffers, rounded up (Galaxy S26, LiteRT 2.3.0:
 * 4.76 GB for the fp16 form, 3.25 GB for the int8 form; the CPU and fp32 forms are estimates).
 */
enum class DecoderForm(
  val key: String,
  val label: String,
  val needMb: Long,
  val file: ModelFiles.Spec,
) {
  /** fp16 weights, GPU with fp16 storage and fp32 accumulation. */
  GPU_FP16_ACC32("gpu-fp16acc32", "GPU (fp16, fp32 accumulation)", 4_900, ModelFiles.DECODER_FP16),

  /** int8 weights (dynamic-range, per channel), GPU fp32. */
  GPU_INT8_FP32("gpu-int8-fp32", "GPU (int8 weights, fp32)", 3_400, ModelFiles.DECODER_INT8),

  /** fp16 weights, GPU fp32 (on a 12 GB phone the weights alone fill the memory). */
  GPU_FP32("gpu-fp32", "GPU (fp32)", 8_200, ModelFiles.DECODER_FP16),

  /** fp16 weights, CPU (XNNPACK unpacks them to fp32). */
  CPU_FP32("cpu", "CPU (fp32)", 8_100, ModelFiles.DECODER_FP16),
}

/** The language model form whose [DecoderForm.key] is [key], or null. */
fun decoderFormOf(key: String?): DecoderForm? = DecoderForm.entries.firstOrNull { it.key == key }

/** Engine settings: a button run uses the defaults; the debug autorun may set them per plan. */
data class RunOptions(
  /** Free memory (MemAvailable) every stage leaves, in MB, on top of its own need. */
  val reserveMb: Long = WeVisOneEngine.DEFAULT_RESERVE_MB,
  /**
   * False only for a guarded measurement of a form the check refuses (the stage is built anyway).
   */
  val memoryCheck: Boolean = true,
  /** Keep the GPU's compiled programs in the cache directory (quicker builds after the first). */
  val gpuProgramCache: Boolean = true,
  val cpuThreads: Int = 6,
)

/** A stage that would not fit: the app reports it and builds nothing. */
class NotEnoughMemoryException(val stage: String, val availableKb: Long, val neededKb: Long) :
  RuntimeException(
    "Not enough memory for the $stage: ${gb(availableKb)} GB available, ${gb(neededKb)} GB needed."
  ) {
  companion object {
    fun gb(kb: Long): String = "%.1f".format(kb / 1_048_576.0)
  }
}

class MissingFilesException(val problems: List<ModelFiles.Problem>) :
  RuntimeException(problems.joinToString(" ") { it.message })

class CancelledException : RuntimeException("Stopped.")

/**
 * One cut-out: the box (bins), the <OBJ> hidden state, the mask decoder's logits and the photo-size
 * alpha.
 */
class Instance(
  val bins: IntArray,
  val tSeg: FloatArray,
  val logits: FloatArray,
  val alpha: FloatArray,
)

class Prediction(
  val promptIds: IntArray,
  val newIds: IntArray,
  val response: String,
  val answer: Answer,
  val instances: List<Instance>,
  val width: Int,
  val height: Int,
  val timing: LinkedHashMap<String, Any?>,
  val memory: List<Map<String, Any?>>,
  val stopped: Boolean,
)

/**
 * The pixel path of WeVisOne-VL-2B on LiteRT CompiledModel: a photo and an instruction in, the
 * answer, its boxes and points, and an alpha matte for every `<OBJ>` out. The steps:
 *
 * 1. Prompt: the fixed system turn, the image first in the user turn ([Prompts]), tokenized pieces
 *    around 256 `<|image_pad|>` ids
 * 2. Vision: the photo to 512 x 512 with Pillow's bicubic ([PilBicubic]), u8 * float32(1/255) ->
 *    encoder [1,1024,4096] (merger inputs of layers 23 | 5 | 11 | 17) -> adapter [1,256,8196]
 * 3. Embeddings: text ids from the host fp16 table (2048 wide, zero to 8196), the image rows from
 *    the adapter
 * 4. Decoder: positions 0..n-1; prompt[:-1] through the prefill signatures that fit exactly
 *    (largest first), the last prompt token through decode, then greedy decode to `<|im_end|>`; the
 *    56 KV states stay in two sets of buffers that swap roles every call
 * 5. t_seg: the decode step whose input token is `<OBJ>` returns its hidden state
 * 6. Mask: taps reshaped to [1,1024,32,32], t_seg, the box, the photo to 1024 x 1024 / 255, the
 *    box's Fourier features ([BoxFeatures]) -> logits [1,1024,1024] -> sigmoid -> bilinear to the
 *    photo size
 *
 * Graphs live only as long as memory allows: before each stage the engine reads MemAvailable and
 * closes the graphs the stage does not use when the stage would not fit beside them; when it would
 * not fit alone it throws [NotEnoughMemoryException] and builds nothing.
 */
class WeVisOneEngine(
  private val modelDir: File,
  private val cacheDir: File,
  private val tokenizer: QwenBpeTokenizer,
  private val boxFeatures: BoxFeatures,
  val form: DecoderForm,
  val options: RunOptions = RunOptions(),
) : Closeable {
  private val gpuProgramCache = options.gpuProgramCache
  private val cpuThreads = options.cpuThreads
  private val reserveMb = options.reserveMb
  private val memoryCheck = options.memoryCheck
  private val env: Environment = Environment.create()
  private var vision: Pair<CompiledModel, CompiledModel>? = null
  private var decoder: Decoder? = null
  private var mask: CompiledModel? = null
  private val table: ShortBuffer
  private val tableChannel: FileChannel

  /** The current run's Stop request, read between stages and between tokens (set by [predict]). */
  private var stopRequested: () -> Boolean = { false }

  private fun checkStop() {
    if (stopRequested()) throw CancelledException()
  }

  init {
    val file = File(modelDir, ModelFiles.EMBEDDINGS.name)
    tableChannel = RandomAccessFile(file, "r").channel
    table =
      tableChannel
        .map(FileChannel.MapMode.READ_ONLY, 0, tableChannel.size())
        .order(ByteOrder.LITTLE_ENDIAN)
        .asShortBuffer()
    require(table.capacity() == VOCAB * HIDDEN) {
      "embedding table holds ${table.capacity()} values"
    }
  }

  fun interface Progress {
    fun stage(text: String)
  }

  /**
   * [stop] is the caller's Stop request for this run: it is read, never reset, so a Stop pressed
   * before this call (while the files were checked or the graphs built) still ends the run.
   */
  fun predict(
    rgb: ByteArray,
    width: Int,
    height: Int,
    instruction: String,
    stop: () -> Boolean,
    progress: Progress,
  ): Prediction {
    stopRequested = stop
    checkStop()
    val timing = LinkedHashMap<String, Any?>()
    val memory = ArrayList<Map<String, Any?>>()
    val t0 = System.nanoTime()
    val promptIds =
      tokenizer.encode(Prompts.BEFORE_IMAGE) +
        IntArray(N_IMAGE) { IMAGE_PAD } +
        tokenizer.encode(Prompts.afterImage(instruction))
    require(promptIds.size + 1 < CACHE) {
      "the prompt is ${promptIds.size} tokens; the cache holds $CACHE"
    }

    // Nothing loaded yet: the stages run one after another at the least, so the largest one must
    // fit before the
    // first graph is built (on an 8 GB Pixel 8a the vision graphs alone took MemAvailable from 3.1
    // to 0.3 GB).
    if (memoryCheck && vision == null && decoder == null && mask == null) {
      val largest =
        listOf(
            "vision encoder" to VISION_NEED_MB,
            "language model" to form.needMb,
            "mask decoder" to MASK_NEED_MB,
          )
          .maxBy { it.second }
      val neededKb = (largest.second + reserveMb) * 1024
      val available = DeviceInfo.memAvailableKb()
      memory.add(reading("room for the largest stage (${largest.first})", "needed_kb" to neededKb))
      if (available in 0 until neededKb) {
        throw NotEnoughMemoryException(largest.first, available, neededKb)
      }
    }

    // 2. vision
    progress.stage("Reading the photo")
    var t = System.nanoTime()
    val u8 = PilBicubic.resize(rgb, width, height, VIS, VIS)
    val x512 = FloatArray(u8.size) { (u8[it].toInt() and 0xFF) * INV_255 }
    timing["resize_512_ms"] = ms(t)
    val (enc, ada) = visionGraphs(memory)
    checkStop()
    progress.stage("Vision encoder")
    t = System.nanoTime()
    val encIn = enc.createInputBuffer("args_0", "serving_default")
    val encOut = enc.createOutputBuffer("output_0", "serving_default")
    val encoded: FloatArray
    try {
      encIn.writeFloat(x512)
      enc.run(mapOf("args_0" to encIn), mapOf("output_0" to encOut), "serving_default")
      encoded = encOut.readFloat()
    } finally {
      encIn.close()
      encOut.close()
    }
    timing["vision_encoder_ms"] = ms(t)
    check(encoded.size == N_PATCH * 4 * TAP) { "encoder output ${encoded.size}" }
    t = System.nanoTime()
    val adaIn = ada.createInputBuffer("args_0", "serving_default")
    val adaOut = ada.createOutputBuffer("output_0", "serving_default")
    val imageRows: FloatArray
    try {
      adaIn.writeFloat(encoded)
      ada.run(mapOf("args_0" to adaIn), mapOf("output_0" to adaOut), "serving_default")
      imageRows = adaOut.readFloat()
    } finally {
      adaIn.close()
      adaOut.close()
    }
    timing["vision_adapter_ms"] = ms(t)
    check(imageRows.size == N_IMAGE * WIDTH) { "adapter output ${imageRows.size}" }
    checkStop()

    // 3-5. decoder
    val dec = decoderGraph(memory)
    checkStop()
    progress.stage("Answering")
    val gen =
      dec.generate(promptIds, imageRows, MAX_NEW, timing) { n ->
        progress.stage("Answering: $n tokens")
      }
    val response = tokenizer.decode(gen.newIds)
    val answer = Answer(response)
    timing["answer_ms"] = ms(t0)

    // 6. masks
    val instances = ArrayList<Instance>()
    val count = minOf(answer.segBoxes.size, gen.objHidden.size)
    if (count > 0 && !gen.stopped) {
      progress.stage("Cutting out")
      t = System.nanoTime()
      val c1024 = PilBicubic.resize(rgb, width, height, CANVAS, CANVAS)
      val canvas = FloatArray(3 * CANVAS * CANVAS)
      val plane = CANVAS * CANVAS
      for (i in 0 until plane) {
        canvas[i] = (c1024[i * 3].toInt() and 0xFF) / 255f
        canvas[plane + i] = (c1024[i * 3 + 1].toInt() and 0xFF) / 255f
        canvas[2 * plane + i] = (c1024[i * 3 + 2].toInt() and 0xFF) / 255f
      }
      val taps = taps(encoded)
      timing["canvas_and_taps_ms"] = ms(t)
      val m = maskGraph(memory)
      val maskMs = ArrayList<Double>()
      val alphaMs = ArrayList<Double>()
      val names = (0..7).map { "args_$it" }
      val inputs = names.associateWith { m.createInputBuffer(it, "serving_default") }
      val output = m.createOutputBuffer("output_0", "serving_default")
      try {
        for (k in 0 until 4) {
          inputs.getValue(names[k]).writeFloat(taps[k])
        }
        inputs.getValue("args_6").writeFloat(canvas)
        for (i in 0 until count) {
          checkStop()
          val bins = answer.segBoxes[i]
          val box = BoxFeatures.normalized(bins)
          val tSeg = gen.objHidden[i]
          t = System.nanoTime()
          inputs.getValue("args_4").writeFloat(tSeg)
          inputs.getValue("args_5").writeFloat(box)
          inputs.getValue("args_7").writeFloat(boxFeatures.features(box))
          m.run(inputs, mapOf("output_0" to output), "serving_default")
          val logits = output.readFloat()
          maskMs.add(ms(t))
          t = System.nanoTime()
          val alpha = AlphaMath.alpha(logits, CANVAS, height, width)
          alphaMs.add(ms(t))
          instances.add(Instance(bins, tSeg, logits, alpha))
        }
      } finally {
        inputs.values.forEach { it.close() }
        output.close()
      }
      timing["mask_decoder_ms"] = maskMs
      timing["alpha_host_ms"] = alphaMs
    }
    timing["total_ms"] = ms(t0)
    timing["backends"] = LinkedHashMap(backends)
    memory.add(reading("done"))
    return Prediction(
      promptIds,
      gen.newIds,
      response,
      answer,
      instances,
      width,
      height,
      timing,
      memory,
      gen.stopped,
    )
  }

  /**
   * The encoder output's slices as the mask decoder's tap inputs: layers 5, 11, 17, 23, each
   * [1,1024,32,32].
   */
  private fun taps(encoded: FloatArray): List<FloatArray> = MASK_TAP_SLICE.map { j ->
    val out = FloatArray(TAP * N_PATCH)
    for (p in 0 until N_PATCH) {
      val row = p * 4 * TAP + j * TAP
      for (c in 0 until TAP) {
        out[c * N_PATCH + p] = encoded[row + c]
      }
    }
    out
  }

  // --------------------------------------------------------------------------- graph lifetimes
  private fun visionGraphs(
    memory: MutableList<Map<String, Any?>>
  ): Pair<CompiledModel, CompiledModel> {
    vision?.let {
      return it
    }
    ensureRoom("vision encoder", VISION_NEED_MB, closeFirst = listOf("mask", "decoder"), memory)
    val t = System.nanoTime()
    val fp32 = CompiledModel.GpuOptions.Precision.FP32
    val enc = gpuOrCpu(ModelFiles.VISION_ENCODER, fp32)
    val ada =
      try {
        gpuOrCpu(ModelFiles.VISION_ADAPTER, fp32)
      } catch (e: Throwable) {
        enc.close()
        throw e
      }
    memory.add(reading("vision built", "build_ms" to ms(t), "backends" to LinkedHashMap(backends)))
    return (enc to ada).also { vision = it }
  }

  /**
   * Where each graph runs: "gpu", or "cpu (the GPU refused: ...)" when the phone's GPU delegate
   * refused it.
   */
  val backends = LinkedHashMap<String, String>()

  /**
   * A one-signature graph on the GPU at [precision]; when the phone's GPU delegate refuses it (an
   * op it does not take, e.g. TRANSPOSE_CONV on some Mali drivers), on the CPU (XNNPACK, fp32)
   * instead, recorded in [backends]. The language model never falls back: its form is the user's
   * choice.
   */
  private fun gpuOrCpu(
    spec: ModelFiles.Spec,
    precision: CompiledModel.GpuOptions.Precision,
  ): CompiledModel =
    try {
      CompiledModel.create(file(spec), gpuOptions(precision, spec), env).also {
        backends[spec.what] = "gpu"
      }
    } catch (e: Exception) {
      Log.w(TAG, "GPU refused ${spec.name}, building it on the CPU", e)
      val cpu =
        CompiledModel.Options(Accelerator.CPU).apply {
          cpuOptions = CompiledModel.CpuOptions(numThreads = cpuThreads)
        }
      CompiledModel.create(file(spec), cpu, env).also {
        backends[spec.what] =
          "cpu (the GPU refused: " +
            "${e.message?.lineSequence()?.lastOrNull { l -> l.isNotBlank() }?.take(160)})"
      }
    }

  private fun decoderGraph(memory: MutableList<Map<String, Any?>>): Decoder {
    decoder?.let {
      return it
    }
    // the vision graphs are not used again in this run: they go first
    ensureRoom("language model", form.needMb, closeFirst = listOf("vision", "mask"), memory)
    val t = System.nanoTime()
    val options =
      when (form) {
        DecoderForm.CPU_FP32 ->
          CompiledModel.Options(Accelerator.CPU).apply {
            cpuOptions =
              CompiledModel.CpuOptions(
                numThreads = cpuThreads,
                xnnPackWeightCachePath = File(cacheDir, "xnnpack_decoder.cache").absolutePath,
              )
          }
        DecoderForm.GPU_FP16_ACC32 ->
          gpuOptions(
            CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM,
            form.file,
            share = true,
          )
        DecoderForm.GPU_INT8_FP32,
        DecoderForm.GPU_FP32 ->
          gpuOptions(CompiledModel.GpuOptions.Precision.FP32, form.file, share = true)
      }
    val model = CompiledModel.create(file(form.file), options, env)
    val d =
      try {
        Decoder(model)
      } catch (e: Throwable) {
        model.close()
        throw e
      }
    backends["language model"] = form.key
    memory.add(reading("language model built", "build_ms" to ms(t)))
    return d.also { decoder = it }
  }

  private fun maskGraph(memory: MutableList<Map<String, Any?>>): CompiledModel {
    mask?.let {
      return it
    }
    ensureRoom("mask decoder", MASK_NEED_MB, closeFirst = listOf("vision", "decoder"), memory)
    val t = System.nanoTime()
    val m = gpuOrCpu(ModelFiles.MASK_DECODER, CompiledModel.GpuOptions.Precision.FP16)
    memory.add(
      reading("mask decoder built", "build_ms" to ms(t), "backends" to LinkedHashMap(backends))
    )
    return m.also { mask = it }
  }

  /**
   * Makes room for a stage that needs [needMb] more than MemAvailable must keep ([reserveMb]):
   * closes the resident graphs in [closeFirst] order until the stage fits, and throws
   * [NotEnoughMemoryException] (building nothing) when it does not fit with all of them closed.
   */
  private fun ensureRoom(
    stage: String,
    needMb: Long,
    closeFirst: List<String>,
    memory: MutableList<Map<String, Any?>>,
  ) {
    val neededKb = (needMb + reserveMb) * 1024
    fun fits() = DeviceInfo.memAvailableKb().let { it < 0 || it >= neededKb }
    for (name in closeFirst) {
      if (fits()) break
      when (name) {
        "vision" -> closeVision(memory)
        "mask" -> closeMask(memory)
        "decoder" -> closeDecoder(memory)
      }
    }
    val available = DeviceInfo.memAvailableKb()
    memory.add(reading("room for $stage", "needed_kb" to neededKb))
    if (memoryCheck && available in 0 until neededKb) {
      throw NotEnoughMemoryException(stage, available, neededKb)
    }
  }

  private fun closeVision(memory: MutableList<Map<String, Any?>>) {
    vision?.let {
      it.first.close()
      it.second.close()
      vision = null
      memory.add(reading("vision closed"))
    }
  }

  private fun closeMask(memory: MutableList<Map<String, Any?>>) {
    mask?.let {
      it.close()
      mask = null
      memory.add(reading("mask decoder closed"))
    }
  }

  private fun closeDecoder(memory: MutableList<Map<String, Any?>>) {
    decoder?.let {
      it.close()
      decoder = null
      memory.add(reading("language model closed"))
    }
  }

  /**
   * GPU only (no CPU fallback: a graph the GPU does not take fails to build). With
   * [gpuProgramCache] the compiled programs are kept in the cache directory under a key of the
   * file's name and size and the precision.
   */
  private fun gpuOptions(
    precision: CompiledModel.GpuOptions.Precision,
    spec: ModelFiles.Spec,
    share: Boolean = false,
  ) =
    CompiledModel.Options(Accelerator.GPU).apply {
      gpuOptions =
        CompiledModel.GpuOptions(
          constantTensorSharing = if (share) true else null,
          precision = precision,
          serializationDir =
            if (gpuProgramCache) File(cacheDir, "gpu").apply { mkdirs() }.absolutePath else null,
          modelCacheKey =
            if (gpuProgramCache) {
              "${spec.name.substringBefore('.')}_${spec.bytes}_${precision.name.lowercase()}"
            } else {
              null
            },
          serializeProgramCache = if (gpuProgramCache) true else null,
        )
    }

  private fun file(spec: ModelFiles.Spec): String = File(modelDir, spec.name).absolutePath

  private fun reading(event: String, vararg extra: Pair<String, Any?>): Map<String, Any?> {
    val status = DeviceInfo.selfStatusKb()
    return linkedMapOf<String, Any?>(
        "event" to event,
        "t_ms" to System.currentTimeMillis(),
        "mem_available_kb" to DeviceInfo.memAvailableKb(),
        "vm_rss_kb" to status["VmRSS"],
        "vm_hwm_kb" to status["VmHWM"],
        "vm_swap_kb" to status["VmSwap"],
        "resident" to
          listOfNotNull(
            "vision".takeIf { vision != null },
            "decoder".takeIf { decoder != null },
            "mask".takeIf { mask != null },
          ),
      )
      .apply { extra.forEach { put(it.first, it.second) } }
  }

  /**
   * One text token's 8196-wide input: the 2048 embedding values from the fp16 table, zeros after.
   */
  private fun embed(token: Int, out: FloatArray, at: Int) {
    val base = token * HIDDEN
    for (i in 0 until HIDDEN) {
      out[at + i] = Half.toFloat(table.get(base + i).toInt())
    }
    java.util.Arrays.fill(out, at + HIDDEN, at + WIDTH, 0f)
  }

  override fun close() {
    val ignored = ArrayList<Map<String, Any?>>()
    closeVision(ignored)
    closeMask(ignored)
    closeDecoder(ignored)
    runCatching { tableChannel.close() }
    env.close()
  }

  class Generation(val newIds: IntArray, val objHidden: List<FloatArray>, val stopped: Boolean)

  /**
   * The language model with persistent buffers: per-signature inputs and outputs, two KV sets that
   * swap roles.
   */
  private inner class Decoder(private val model: CompiledModel) : Closeable {
    private val kvNames =
      (0 until LAYERS).map { "kv_cache_k_$it" } + (0 until LAYERS).map { "kv_cache_v_$it" }
    private val signatures = PREFILL_LENGTHS.map { "prefill_$it" } + "decode"
    private val buffers = ArrayList<TensorBuffer>()
    private val inputs = signatures.associateWith { sig ->
      listOf("embeddings", "input_pos", "mask").associateWith {
        model.createInputBuffer(it, sig).also(buffers::add)
      }
    }
    private val outputs = signatures.associateWith { sig ->
      (listOf("hidden") + if (sig == "decode") listOf("logits") else emptyList()).associateWith {
        model.createOutputBuffer(it, sig).also(buffers::add)
      }
    }
    private val kvA = kvNames.associateWith {
      model.createInputBuffer(it, "decode").also(buffers::add)
    }
    private val kvB = kvNames.associateWith {
      model.createOutputBuffer(it, "decode").also(buffers::add)
    }
    private var cur = kvA
    private var nxt = kvB
    private val zeros = FloatArray(KV_FLOATS)

    private fun reset() {
      for (n in kvNames) {
        kvA.getValue(n).writeFloat(zeros)
        kvB.getValue(n).writeFloat(zeros)
      }
      cur = kvA
      nxt = kvB
    }

    /**
     * One call: rows of [emb] at [start] until start + len, positions [pos0] onward; returns
     * (hidden, logits).
     */
    private fun step(
      sig: String,
      emb: FloatArray,
      start: Int,
      len: Int,
      pos0: Int,
    ): Pair<FloatArray, FloatArray?> {
      val io = inputs.getValue(sig)
      val e =
        if (start == 0 && emb.size == len * WIDTH) {
          emb
        } else {
          emb.copyOfRange(start * WIDTH, (start + len) * WIDTH)
        }
      io.getValue("embeddings").writeFloat(e)
      io.getValue("input_pos").writeInt(IntArray(len) { pos0 + it })
      val mask = FloatArray(len * CACHE)
      for (r in 0 until len) {
        val p = pos0 + r
        java.util.Arrays.fill(mask, r * CACHE + p + 1, (r + 1) * CACHE, NEG)
      }
      io.getValue("mask").writeFloat(mask)
      val out = outputs.getValue(sig)
      model.run(io + cur, out + nxt, sig)
      val swap = cur
      cur = nxt
      nxt = swap
      // reading the outputs also waits for the GPU work before the next call writes its inputs
      val hidden = out.getValue("hidden").readFloat()
      val logits = out["logits"]?.readFloat()
      return hidden to logits
    }

    fun generate(
      prompt: IntArray,
      imageRows: FloatArray,
      maxNew: Int,
      timing: MutableMap<String, Any?>,
      onToken: (Int) -> Unit,
    ): Generation {
      val n = prompt.size
      val emb = FloatArray(n * WIDTH)
      var k = 0
      for ((i, tok) in prompt.withIndex()) {
        if (tok == IMAGE_PAD) {
          System.arraycopy(imageRows, k * WIDTH, emb, i * WIDTH, WIDTH)
          k++
        } else {
          embed(tok, emb, i * WIDTH)
        }
      }
      check(k == N_IMAGE) { "the prompt holds $k image positions" }
      var t = System.nanoTime()
      reset()
      timing["kv_reset_ms"] = ms(t)
      t = System.nanoTime()
      var p = 0
      val plan = ArrayList<String>()
      for (len in PREFILL_LENGTHS) {
        while (n - 1 - p >= len) {
          step("prefill_$len", emb, p, len, p)
          plan.add("$len")
          p += len
        }
      }
      timing["prefill_ms"] = ms(t)
      timing["prefill_tokens"] = n - 1
      timing["prefill_plan"] = plan.joinToString("+")
      val newIds = ArrayList<Int>()
      val objHidden = ArrayList<FloatArray>()
      val stepMs = ArrayList<Double>()
      val row = FloatArray(WIDTH)
      System.arraycopy(emb, (n - 1) * WIDTH, row, 0, WIDTH)
      var pos = n - 1
      var fed = prompt[n - 1]
      var stopped = false
      while (true) {
        t = System.nanoTime()
        val (hidden, logits) = step("decode", row, 0, 1, pos)
        stepMs.add(ms(t))
        if (fed == OBJ && newIds.isNotEmpty()) {
          objHidden.add(hidden)
        }
        if (newIds.size >= maxNew) break
        if (stopRequested()) {
          stopped = true
          break
        }
        val next = argmax(logits!!)
        newIds.add(next)
        onToken(newIds.size)
        if (next == IM_END || next == END_OF_TEXT) break
        if (pos + 1 >= CACHE - 1) break
        embed(next, row, 0)
        pos += 1
        fed = next
      }
      if (
        !stopped &&
          newIds.isNotEmpty() &&
          newIds.last() == OBJ &&
          objHidden.size < newIds.count { it == OBJ }
      ) {
        // cut at the cap right after an <OBJ>: one more step reads its hidden state
        embed(OBJ, row, 0)
        objHidden.add(step("decode", row, 0, 1, pos + 1).first)
      }
      timing["decode_steps"] = stepMs.size
      timing["decode_step_ms_median"] = median(stepMs)
      timing["decode_ms"] = stepMs.sum()
      return Generation(newIds.toIntArray(), objHidden, stopped)
    }

    override fun close() {
      buffers.forEach { runCatching { it.close() } }
      model.close()
    }
  }

  companion object {
    private const val TAG = "WeVisOne"
    const val VIS = 512
    const val CANVAS = 1024
    const val N_IMAGE = 256
    const val N_PATCH = 1024
    const val TAP = 1024
    const val HIDDEN = 2048
    const val WIDTH = 8196
    const val VOCAB = 152_680
    const val CACHE = 2048
    const val LAYERS = 28
    const val KV_FLOATS = 8 * 2048 * 128
    const val IMAGE_PAD = 151655
    const val OBJ = 152678
    const val IM_END = 151645
    const val END_OF_TEXT = 151643
    const val MAX_NEW = 1024
    /**
     * The prefill signatures' lengths: a prompt goes through the largest that fit, in this order.
     */
    val PREFILL_LENGTHS = intArrayOf(1024, 256, 64, 16, 4, 1)
    private val NEG = (-0.7 * Float.MAX_VALUE).toFloat()
    private val INV_255 = (1.0 / 255.0).toFloat()

    /**
     * The encoder output's 1024-wide slices are layers 23 | 5 | 11 | 17; the mask decoder takes 5,
     * 11, 17, 23.
     */
    private val MASK_TAP_SLICE = intArrayOf(1, 2, 3, 0)

    /**
     * MemAvailable a stage takes, in MB, rounded up from the Galaxy S26 readings (vision 1.52-1.56
     * GB, mask decoder 0.46-0.51 GB with LiteRT 2.3.0).
     */
    const val VISION_NEED_MB = 1_700L
    const val MASK_NEED_MB = 600L
    const val DEFAULT_RESERVE_MB = 800L

    fun ms(since: Long): Double = (System.nanoTime() - since) / 1e6

    fun median(xs: List<Double>): Double? =
      if (xs.isEmpty()) {
        null
      } else {
        xs.sorted().let { s ->
          if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2
        }
      }

    /** The first index of the largest value (numpy's argmax). */
    fun argmax(xs: FloatArray): Int {
      var best = 0
      var v = xs[0]
      for (i in 1 until xs.size) {
        if (xs[i] > v) {
          v = xs[i]
          best = i
        }
      }
      return best
    }

    init {
      Log.d(TAG, "LiteRT ${BuildConfig.LITERT_VERSION}")
    }
  }
}
