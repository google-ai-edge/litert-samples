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

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.NpuCompatibilityChecker
import java.io.File
import java.io.InputStreamReader
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.zip.Inflater
import java.util.zip.InflaterInputStream
import org.json.JSONObject

/**
 * LFM3-Omni-2B on LiteRT: the files and the graphs.
 *
 * Files (in the app's files dir, or its external files dir for `adb push`; README.md), as the model repository names
 * them:
 *   LFM3-Omni-2B_int8.litertlm                 the bundle: the language model, its embedder, the audio encoder, the
 *                                              vision encoder and adapter, and the tokenizer (the same file LiteRT-LM
 *                                              runs for text answers)
 *   the depth decoder, one of its forms (DepthDecoder; the app opens [DEFAULT_DEPTH_FORM], placed by
 *   [defaultDepthVariant]):
 *     LFM3-Omni-2B_depth_frame2_f32.tflite     frame2: the whole frame and its sampling in one call, the codes as
 *                                              the only output
 *     LFM3-Omni-2B_depth_frame_f32.tflite      frame: the same with the logits as an output too
 *     LFM3-Omni-2B_depth_step_f32.tflite       step: one maskgit pass a call (three per speech frame)
 *     LFM3-Omni-2B_depth_project_f32.tflite    split: the projection, once a frame
 *     LFM3-Omni-2B_depth_passes_f32.tflite       and one maskgit pass a call (three per speech frame)
 *   LFM3-Omni-2B_frame_embed_f32.tflite        speech frame -> the next LM input
 *   LFM3-Omni-2B_codec_decoder_f12_f32.tflite  codec decoder, 12 frames (0.96 s) per call
 */
class Lfm3OmniEngine private constructor(
  val tokenizer: Lfm3Tokenizer,
  val lm: LmRunner,
  /** The depth decoder variants that were opened, by name: [Config.depthForm] and [Config.moreDepthForms]. */
  val depthForms: Map<String, DepthDecoder>,
  val frame: FrameEmbed,
  val codec: CodecDecoder,
  val audioEncoder: AudioEncoder,
  val vision: VisionEncoder,
  val config: Config,
  val load: JSONObject,
) : AutoCloseable {

  /** Where each part runs. The model card's speech path runs on the CPU (the codec decoder must stay there). */
  data class Config(
    val lm: Accel = Accel("cpu"),
    val depth: Accel = Accel("cpu"),
    val frame: Accel = Accel("cpu"),
    val codec: Accel = Accel("cpu"),
    val audio: Accel = Accel("cpu"),
    val vision: Accel = defaultVisionAccel(),
    /** The depth decoder of every answer: a form (DepthDecoder.FORMS) on [depth], or "<form>@<accelerator>" with one
     *  of Accel.NAMES (e.g. "frame2@gpu16"). */
    val depthForm: String = defaultDepthVariant(),
    /** More variants to open next to it, for the measuring tool's answers that name one. */
    val moreDepthForms: List<String> = emptyList(),
  )

  /** The default variant's depth decoder. */
  val depth: DepthDecoder get() = depthForms.getValue(config.depthForm)

  /** The depth decoder of [variant] (null = the default one); the variant must have been opened. */
  fun depth(variant: String?): DepthDecoder = if (variant == null) depth else depthForms[variant]
    ?: throw IllegalStateException("the depth decoder $variant is not open (open: ${depthForms.keys.joinToString()})")

  val generator = SpeechGenerator(lm, depth, frame)

  /** Where each part runs after the fallbacks, for the status line: e.g. "language model CPU · depth decoder CPU · ...". */
  fun placement(): String {
    fun kind(key: String) = load.optString(key).substringBefore('(').let { k ->
      if (load.optString(key).contains("fallback")) "CPU" else k.uppercase()
    }
    return "language model ${kind("lm_accel")} · speech decoder ${kind("depth_accel")} · audio encoder ${kind("audio_accel")} · " +
      "vision encoder ${kind("vision_accel")}"
  }

  override fun close() {
    lm.close()
    for (d in depthForms.values) d.close()
    frame.close()
    codec.close()
    audioEncoder.close()
    vision.close()
  }

  /** Not enough free memory for the next stage: nothing more was loaded, and what was loaded is closed. */
  class NotEnoughMemory(message: String) : Exception(message)

  companion object {
    const val TAG = "PhotoVoiceChat"
    const val DEPTH_STEP = "LFM3-Omni-2B_depth_step_f32.tflite"
    const val DEPTH_PROJECT = "LFM3-Omni-2B_depth_project_f32.tflite"
    const val DEPTH_PASSES = "LFM3-Omni-2B_depth_passes_f32.tflite"
    const val DEPTH_FRAME = "LFM3-Omni-2B_depth_frame_f32.tflite"
    const val DEPTH_FRAME2 = "LFM3-Omni-2B_depth_frame2_f32.tflite"
    /** The depth decoder form the app opens (DepthDecoder; README.md, "How it runs" and "Performance"). */
    const val DEFAULT_DEPTH_FORM = DepthDecoder.FRAME2
    /**
     * Where the depth decoder runs on a Qualcomm phone (Accel.NAMES): on a Galaxy S26 the depth decoder on the GPU at
     * fp16 made a second of speech in 0.75 s, against 0.96 s on the CPU, and its speech passed the same checks; on a
     * Pixel 11 Pro the GPU was slower than the CPU (README.md, Performance).
     */
    const val QUALCOMM_DEPTH_ACCEL = "gpu16"

    /**
     * The depth decoder the app opens: [DEFAULT_DEPTH_FORM] on [QUALCOMM_DEPTH_ACCEL] on a Qualcomm phone, on the CPU on
     * other phones and on Android versions before 12 (which do not name the maker of the chip). A GPU that cannot open
     * the graph leaves it on the CPU (the load record names the reason).
     */
    fun defaultDepthVariant(): String =
      if (Build.VERSION.SDK_INT >= 31 && Build.SOC_MANUFACTURER == "QTI") "$DEFAULT_DEPTH_FORM@$QUALCOMM_DEPTH_ACCEL"
      else DEFAULT_DEPTH_FORM

    /**
     * Where the vision encoder and adapter run on a Qualcomm phone and on a Google Tensor G6 phone (Accel.NAMES). On a
     * Galaxy S26 the GPU at float32 read a 12 s video in 11.0 s against 16.2 s with them on the CPU (the 12 pictures in
     * 2.60 s against 6.38 s); float16 read it in 10.5 s, but with it the spoken answers about Video 2 named a picture
     * shown after the line. On a Pixel 11 Pro float16 read it in 23.6 s against 34.5 s (3.36 s against 14.2 s), and its
     * answers passed the same checks (README.md, Performance).
     */
    const val QUALCOMM_VISION_ACCEL = "gpu32"
    const val TENSOR_G6_VISION_ACCEL = "gpu16"

    /**
     * The vision encoder's and adapter's accelerator: [QUALCOMM_VISION_ACCEL] on a Qualcomm phone, [TENSOR_G6_VISION_ACCEL]
     * on a Google Tensor G6 phone, the CPU on other phones and on Android versions before 12 (as [defaultDepthVariant]).
     * A GPU that cannot open the graphs leaves them on the CPU (the load record names the reason).
     */
    fun defaultVisionAccel(): Accel = when {
      Build.VERSION.SDK_INT >= 31 && Build.SOC_MANUFACTURER == "QTI" -> Accel.named(QUALCOMM_VISION_ACCEL)
      Build.VERSION.SDK_INT >= 31 && Build.SOC_MANUFACTURER == "Google" && Build.SOC_MODEL == "Tensor G6" ->
        Accel.named(TENSOR_G6_VISION_ACCEL)
      else -> Accel("cpu")
    }
    const val FRAME = "LFM3-Omni-2B_frame_embed_f32.tflite"
    const val CODEC = "LFM3-Omni-2B_codec_decoder_f12_f32.tflite"
    const val SAMPLE_RATE = 24000

    /**
     * Free memory each loading stage needs (bytes), checked before the stage with [RESERVE] on top; the largest stage
     * is checked first, before anything is loaded (a stage that cannot fit must not start: on a phone low on memory
     * the system kills other apps first). Stages: the language model (decoder + embedder), the speech decoder (depth
     * decoder, frame embedding, codec decoder), the audio encoder, the vision encoder and adapter. The weights are
     * packed into XNNPACK's cache files and mapped from them, so a stage takes little of the free memory: on a Galaxy
     * S26's first start (the caches written) the four stages took 38, 117, 12 and 84 MB of it (the app's RSS 1.1 GB
     * after loading); the needs below leave room for the answers' own memory and for other phones.
     */
    val STAGE_NEED = linkedMapOf(
      "language model" to 1_200_000_000L,
      "speech decoder" to 500_000_000L,
      "audio encoder" to 500_000_000L,
      "vision encoder" to 500_000_000L,
    )
    const val RESERVE = 500_000_000L

    /** The dirs searched for the model files: the app's files dir, then its external files dir. */
    fun modelDirs(context: Context): List<File> = listOfNotNull(context.filesDir, context.getExternalFilesDir(null))

    fun find(context: Context, name: String): File? = modelDirs(context).map { File(it, name) }.firstOrNull { it.isFile }

    /** The graphs the app copies out of the bundle: its name for each, and the model type its section has. */
    private val GRAPHS = linkedMapOf(
      "decoder" to "tf_lite_prefill_decode",
      "embedder" to "tf_lite_embedder",
      "audio" to "tf_lite_audio_encoder_hw",
      "vision_encoder" to "tf_lite_vision_encoder",
      "vision_adapter" to "tf_lite_vision_adapter",
    )

    private fun LitertlmHeader.graph(type: String) =
      sections.firstOrNull { it.type == LitertlmHeader.TFLITE_MODEL && it.modelType == type }

    private fun LitertlmHeader.weights(type: String) =
      sections.firstOrNull { it.type == LitertlmHeader.TFLITE_WEIGHTS && it.modelType == type }

    private fun LitertlmHeader.tokenizer() = sections.firstOrNull { it.type == LitertlmHeader.HF_TOKENIZER_ZLIB }

    /** The bundle on this phone and its header: a whole .litertlm file whose header lists every section the app reads,
     *  with its weights inside each graph or in TFLiteWeights sections. */
    fun findBundle(context: Context): Pair<File, LitertlmHeader>? = modelDirs(context).flatMap { d ->
      d.listFiles()?.filter { it.isFile && it.name.endsWith(".litertlm") } ?: emptyList()
    }.firstNotNullOfOrNull { f ->
      LitertlmHeader.read(f)
        ?.takeIf { h -> h.fileBytes == f.length() && h.tokenizer() != null && GRAPHS.values.all { h.graph(it) != null } }
        ?.let { f to it }
    }

    /** Each section's published SHA-256s, in both forms of the bundle, and the decoder's signatures. */
    private fun bundleFacts(context: Context) = JSONObject(context.assets.open("bundle_sections.json").bufferedReader().readText())

    /** The name a graph gives its weights (its external buffer group): LiteRT opens a file of that name in the graph's folder. */
    private const val WEIGHTS_FILE = "tflite_weights"

    /** The graph files of a depth decoder form. */
    fun depthFiles(form: String): List<String> = when (form) {
      DepthDecoder.STEP -> listOf(DEPTH_STEP)
      DepthDecoder.SPLIT -> listOf(DEPTH_PROJECT, DEPTH_PASSES)
      DepthDecoder.FRAME -> listOf(DEPTH_FRAME)
      DepthDecoder.FRAME2 -> listOf(DEPTH_FRAME2)
      else -> throw IllegalArgumentException("The depth form $form is not one of ${DepthDecoder.FORMS.joinToString()}.")
    }

    /** "<form>" or "<form>@<accelerator>" -> the form and its accelerator ([default] when the variant names none). */
    fun depthVariant(variant: String, default: Accel): Pair<String, Accel> {
      val form = variant.substringBefore('@')
      depthFiles(form)
      return form to (if ('@' in variant) Accel.named(variant.substringAfter('@'), default.threads) else default)
    }

    /**
     * The NPU graphs' environment: LiteRT's Qualcomm runtime, loaded from the app's native library dir. An APK has it
     * only when built with the Qualcomm NPU libraries (README.md); without them, or on another phone, this says why
     * and the graph opens on the CPU instead (openWithFallback).
     */
    private fun npuEnvironment(context: Context): Environment {
      val provider = BuiltinNpuAcceleratorProvider(context, NpuCompatibilityChecker.Qualcomm)
      check(provider.isDeviceSupported()) { "this phone has no Qualcomm NPU that LiteRT supports" }
      val dir = File(provider.getLibraryDir())
      val skel = dir.list().orEmpty().any { it.startsWith("libQnnHtpV") && it.endsWith("Skel.so") }
      val missing = NPU_LIBRARIES.filter { !File(dir, it).isFile } + (if (skel) emptyList() else listOf("libQnnHtpV<NN>Skel.so"))
      check(missing.isEmpty()) { "this APK has no Qualcomm NPU runtime (missing ${missing.joinToString()})" }
      val env = Environment.create(provider)
      check(Accelerator.NPU in env.getAvailableAccelerators()) { "LiteRT could not load the Qualcomm NPU runtime" }
      return env
    }

    /** The Qualcomm runtime's files the on-phone compile loads, besides the SoC's libQnnHtpV<NN>Stub / Skel. */
    private val NPU_LIBRARIES = listOf("libLiteRtDispatch_Qualcomm.so", "libLiteRtCompilerPlugin_Qualcomm.so",
      "libQnnHtp.so", "libQnnSystem.so", "libQnnHtpPrepare.so")

    /** What is missing before [open] can run (with the default depth form), as one sentence; null when every required
     *  file is there. */
    fun missing(context: Context): String? {
      val names = (depthFiles(DEFAULT_DEPTH_FORM) + listOf(FRAME, CODEC)).filter { find(context, it) == null }
      val bundle = findBundle(context)
      if (names.isEmpty() && bundle != null) return null
      val list = (if (bundle == null) listOf("the .litertlm bundle") else emptyList()) + names
      val dir = context.getExternalFilesDir(null)?.absolutePath ?: context.filesDir.absolutePath
      return "The model files are not on this phone yet. Missing: ${list.joinToString(", ")}. " +
        "Push them into $dir with install_to_device.sh (README.md), then reopen the app."
    }

    fun availableMemory(context: Context): Long {
      val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
      return ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }.availMem
    }

    /** VmRSS of this process (bytes), for the load record. */
    fun rss(): Long = runCatching {
      File("/proc/self/status").readLines().first { it.startsWith("VmRSS:") }.filter { it.isDigit() }.toLong() * 1024
    }.getOrDefault(-1L)

    private fun gb(b: Long) = "%.1f GB".format(b / 1e9)

    /**
     * Opens everything: the bundle's sections copied out once (to files/lfm3_sections/<file size>-<file time>/, checked by
     * size and the TFLite magic, and by SHA-256 the first time), the tokenizer inflated from the bundle, the graphs
     * compiled, each stage after a memory check. [log] receives one line per step.
     */
    fun open(context: Context, config: Config, log: (String) -> Unit = {}): Lfm3OmniEngine {
      val load = JSONObject()
      val stages = JSONObject()
      load.put("stages", stages)
      val t0 = System.nanoTime()
      val (bundle, header) = findBundle(context) ?: throw IllegalStateException(missing(context) ?: "no bundle")
      missing(context)?.let { throw IllegalStateException(it) }
      fun gate(stage: String, first: Boolean = false) {
        val need = (if (first) STAGE_NEED.values.max() else STAGE_NEED.getValue(stage)) + RESERVE
        val avail = availableMemory(context)
        stages.put("before_$stage", JSONObject().put("available", avail).put("rss", rss()).put("need", need))
        if (avail < need) {
          throw NotEnoughMemory("Not enough free memory to load the $stage: ${gb(avail)} available, ${gb(need)} needed. " +
            (if (first) "Nothing was loaded. " else "What was loaded is closed again. ") + "Close other apps and reopen this one.")
        }
      }
      fun after(stage: String) = stages.put("after_$stage", JSONObject().put("available", availableMemory(context)).put("rss", rss()))
      gate("language model", first = true)
      load.put("bundle", bundle.absolutePath)
      load.put("weights_in_sections", header.weightsInSections)
      val facts = bundleFacts(context)
      val published = facts.getJSONObject("section_sha256")
      val sectionsDir = File(context.filesDir, "lfm3_sections")
      val outDir = File(sectionsDir, "${bundle.length()}-${bundle.lastModified()}").apply { mkdirs() }
      // The copies of another file would only take up room.
      sectionsDir.listFiles()?.filter { it != outDir }?.forEach { it.deleteRecursively() }
      val extracted = HashMap<String, File>()
      fun copyOut(name: String, s: LitertlmHeader.Section, f: File, graph: Boolean) {
        val ok = File(f.parentFile, "${f.nameWithoutExtension}.verified")
        val size = s.end - s.begin
        if (!(f.length() == size && ok.isFile)) {
          log("Preparing the model, once: copying the $name section out of ${bundle.name} (${size / 1_000_000} MB)")
          val tc = System.nanoTime()
          val sha = copyRange(bundle, s.begin, size, f, graph)
          load.put("${name.replace(' ', '_')}_copy_ms", (System.nanoTime() - tc) / 1e6)
          val known = published.optJSONArray(name)
          check(known != null && (0 until known.length()).any { known.getString(it) == sha }) {
            "$name section: sha256 $sha is not one of the published files'"
          }
          ok.writeText(sha)
        }
      }
      for ((name, type) in GRAPHS) {
        val s = checkNotNull(header.graph(type))
        // Every weights file has the same name, so a graph with one gets a folder of its own.
        val weights = header.weights(type)
        val dir = if (weights == null) outDir else File(outDir, name).apply { mkdirs() }
        val f = File(dir, "$name.tflite")
        copyOut(name, s, f, graph = true)
        if (weights != null) copyOut("$name weights", weights, File(dir, WEIGHTS_FILE), graph = false)
        extracted[name] = f
      }
      log("Reading the tokenizer")
      val tt = System.nanoTime()
      val tok = checkNotNull(header.tokenizer())
      val tokenizer = RandomAccessFile(bundle, "r").use { raf ->
        val bytes = ByteArray((tok.end - tok.begin).toInt())
        raf.seek(tok.begin)
        raf.readFully(bytes)
        val reader = InputStreamReader(InflaterInputStream(bytes.inputStream(8, bytes.size - 8), Inflater()), Charsets.UTF_8)
        Lfm3Tokenizer.load(reader.buffered(1 shl 16))
      }
      load.put("tokenizer_ms", (System.nanoTime() - tt) / 1e6)
      fun need(name: String) = find(context, name) ?: throw IllegalStateException(missing(context) ?: name)
      val env = Environment.create()
      // the NPU graphs' environment, created with the first NPU graph
      var npuEnv: Environment? = null
      fun envFor(a: Accel): Environment =
        if (a.kind != "npu") env else npuEnv ?: npuEnvironment(context).also { npuEnv = it }
      val cacheDir = File(context.filesDir, "xnnpack_cache").apply { mkdirs() }
      val opened = ArrayList<AutoCloseable>()
      try {
        log("Loading the language model")
        val tl = System.nanoTime()
        val sig = facts.optJSONObject("decoder_signatures")
        val decoderFile = extracted.getValue("decoder")
        val embedderFile = extracted.getValue("embedder")
        val lm = LmRunner(decoderFile, embedderFile, config.lm.cached(cacheDir, decoderFile, "decoder").options(),
          Accel("cpu").cached(cacheDir, embedderFile, "embedder").options(), env,
          sig?.getJSONArray("prefill")?.let { a -> IntArray(a.length()) { a.getInt(it) } },
          sig?.getJSONArray("state")?.let { a -> List(a.length()) { a.getString(it) } }, inPlace = config.lm.kind == "cpu")
        opened.add(lm)
        load.put("lm_load_ms", (System.nanoTime() - tl) / 1e6)
        load.put("lm_accel", config.lm.toString())
        after("language model")

        gate("speech decoder")
        log("Loading the speech decoder")
        val td = System.nanoTime()
        val depthForms = LinkedHashMap<String, DepthDecoder>()
        for (variant in (listOf(config.depthForm) + config.moreDepthForms).distinct()) {
          val (form, accel) = depthVariant(variant, config.depth)
          val files = depthFiles(form).map { name ->
            find(context, name) ?: throw IllegalStateException("The depth form $form needs $name, which is not on this " +
              "phone. Push it with install_to_device.sh (README.md), then reopen the app.")
          }
          depthForms[variant] = openWithFallback(load, "depth_$variant", accel, files[0], cacheDir) { a ->
            val e = envFor(a)
            when (form) {
              DepthDecoder.STEP -> DepthDecoder.step(Graph(files[0], "depth_step", a, e))
              DepthDecoder.FRAME -> DepthDecoder.frame(Graph(files[0], "depth_frame", a, e))
              DepthDecoder.FRAME2 -> DepthDecoder.frame2(Graph(files[0], "depth_frame", a, e))
              else -> {
                val project = Graph(files[0], "depth_project", a, e)
                try {
                  DepthDecoder.split(project, Graph(files[1], "depth_passes", a.cached(cacheDir, files[1], "depth_passes"), e))
                } catch (t: Throwable) {
                  project.close()
                  throw t
                }
              }
            }
          }.also { it.variant = variant; opened.add(it) }
        }
        load.put("depth_form", config.depthForm)
        load.put("depth_forms", org.json.JSONArray(depthForms.keys.toList()))
        load.put("depth_accel", load.optString("depth_${config.depthForm}_accel"))
        load.put("depth_load_ms", (System.nanoTime() - td) / 1e6)
        val frameFile = need(FRAME)
        val frame = openWithFallback(load, "frame", config.frame, frameFile, cacheDir) { FrameEmbed(frameFile, it, env) }.also { opened.add(it) }
        val window = context.assets.open("codec_window_f32.bin").use { s ->
          val b = ByteBuffer.wrap(s.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
          FloatArray(b.remaining() / 4) { b.getFloat() }
        }
        val codecFile = need(CODEC)
        val codec = openWithFallback(load, "codec", config.codec, codecFile, cacheDir) { CodecDecoder(codecFile, it, env, window, overlap = true) }
          .also { opened.add(it) }
        after("speech decoder")

        gate("audio encoder")
        log("Loading the audio encoder")
        val audioFile = extracted.getValue("audio")
        val audio = openWithFallback(load, "audio", config.audio, audioFile, cacheDir) { AudioEncoder(audioFile, it, env) }.also { opened.add(it) }
        after("audio encoder")

        gate("vision encoder")
        log("Loading the vision encoder")
        val encFile = extracted.getValue("vision_encoder")
        val adpFile = extracted.getValue("vision_adapter")
        val vision = openWithFallback(load, "vision", config.vision, encFile, cacheDir) {
          VisionEncoder(encFile, adpFile, it.programCached(cacheDir, encFile, "vision_encoder"),
            if (it.kind == "cpu") Accel("cpu", threads = it.threads).cached(cacheDir, adpFile, "vision_adapter")
            else it.programCached(cacheDir, adpFile, "vision_adapter"), env)
        }.also { opened.add(it) }
        after("vision encoder")
        load.put("open_ms", (System.nanoTime() - t0) / 1e6)
        load.put("available_accelerators", runCatching { env.getAvailableAccelerators().joinToString(",") }.getOrDefault("?"))
        log("Ready")
        return Lfm3OmniEngine(tokenizer, lm, depthForms, frame, codec, audio, vision, config, load)
      } catch (t: Throwable) {
        for (o in opened.reversed()) runCatching { o.close() }
        throw t
      }
    }

    /** A graph on its accelerator; when a GPU or NPU form cannot be created, the CPU, with the reason in [load]. */
    private fun <T> openWithFallback(load: JSONObject, name: String, accel: Accel, model: File, cacheDir: File, open: (Accel) -> T): T {
      val t0 = System.nanoTime()
      return try {
        open(accel.cached(cacheDir, model, name)).also { load.put("${name}_accel", accel.toString()) }
      } catch (e: Exception) {
        if (accel.kind == "cpu") throw e
        Log.w(TAG, "$name on $accel failed, CPU instead: $e")
        load.put("${name}_${accel.kind}_error", "${e.javaClass.simpleName}: ${e.message}".take(400))
        open(Accel("cpu", threads = accel.threads).cached(cacheDir, model, name)).also { load.put("${name}_accel", "cpu(fallback)") }
      } finally {
        load.put("${name}_load_ms", (System.nanoTime() - t0) / 1e6)
      }
    }

    /** Copies [size] bytes from [src] at [begin] into [dst]; returns their SHA-256 (hex). A [graph] must be a TFLite flatbuffer. */
    private fun copyRange(src: File, begin: Long, size: Long, dst: File, graph: Boolean): String {
      val md = MessageDigest.getInstance("SHA-256")
      val tmp = File(dst.path + ".part")
      RandomAccessFile(src, "r").use { raf ->
        raf.seek(begin)
        tmp.outputStream().buffered(1 shl 20).use { out ->
          val buf = ByteArray(1 shl 20)
          var left = size
          while (left > 0) {
            val n = raf.read(buf, 0, minOf(buf.size.toLong(), left).toInt())
            check(n > 0) { "short read of ${src.name}" }
            out.write(buf, 0, n)
            md.update(buf, 0, n)
            left -= n
          }
        }
      }
      if (graph) {
        val magic = ByteArray(4)
        RandomAccessFile(tmp, "r").use { it.seek(4); it.readFully(magic) }
        check(String(magic, Charsets.US_ASCII) == "TFL3") { "${dst.name}: not a TFLite flatbuffer" }
      }
      check(tmp.renameTo(dst)) { "rename ${tmp.name}" }
      return md.digest().joinToString("") { "%02x".format(it) }
    }
  }
}
