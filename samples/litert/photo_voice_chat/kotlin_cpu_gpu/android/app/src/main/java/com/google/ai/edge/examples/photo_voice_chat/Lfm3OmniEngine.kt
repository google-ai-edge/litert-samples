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
import android.util.Log
import com.google.ai.edge.litert.Environment
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
 *   LFM3-Omni-2B_depth_step_f32.tflite         depth decoder, one maskgit pass (three per speech frame)
 *   LFM3-Omni-2B_frame_embed_f32.tflite        speech frame -> the next LM input
 *   LFM3-Omni-2B_codec_decoder_f12_f32.tflite  codec decoder, 12 frames (0.96 s) per call
 */
class Lfm3OmniEngine private constructor(
  val tokenizer: Lfm3Tokenizer,
  val lm: LmRunner,
  val depth: DepthDecoder,
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
    val vision: Accel = Accel("cpu"),
  )

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
    depth.close()
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
    const val FRAME = "LFM3-Omni-2B_frame_embed_f32.tflite"
    const val CODEC = "LFM3-Omni-2B_codec_decoder_f12_f32.tflite"
    const val SAMPLE_RATE = 24000
    private val SECTIONS = listOf("decoder", "embedder", "audio", "vision_encoder", "vision_adapter")

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

    fun findBundle(context: Context, sections: JSONObject): File? = modelDirs(context).flatMap { d ->
      d.listFiles()?.filter { it.isFile && it.name.endsWith(".litertlm") } ?: emptyList()
    }.firstOrNull { it.length() == sections.getLong("bundle_bytes") }

    private fun sections(context: Context) = JSONObject(context.assets.open("bundle_sections.json").bufferedReader().readText())

    /** What is missing before [open] can run, as one sentence; null when every required file is there. */
    fun missing(context: Context): String? {
      val sections = sections(context)
      val names = listOf(DEPTH_STEP, FRAME, CODEC).filter { find(context, it) == null }
      val bundle = findBundle(context, sections)
      if (names.isEmpty() && bundle != null) return null
      val list = (if (bundle == null) listOf("the .litertlm bundle (${sections.getLong("bundle_bytes")} bytes)") else emptyList()) + names
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
     * Opens everything: the bundle's sections copied out once (to files/lfm3_sections/<sha256 prefix>/, checked by
     * size and the TFLite magic, and by SHA-256 the first time), the tokenizer inflated from the bundle, the graphs
     * compiled, each stage after a memory check. [log] receives one line per step.
     */
    fun open(context: Context, config: Config, log: (String) -> Unit = {}): Lfm3OmniEngine {
      val load = JSONObject()
      val stages = JSONObject()
      load.put("stages", stages)
      val t0 = System.nanoTime()
      val sections = sections(context)
      val bundle = findBundle(context, sections) ?: throw IllegalStateException(missing(context) ?: "no bundle")
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
      val secs = sections.getJSONObject("sections")
      val outDir = File(context.filesDir, "lfm3_sections/" + sections.getString("bundle_sha256").take(16)).apply { mkdirs() }
      val extracted = HashMap<String, File>()
      for (name in SECTIONS) {
        val s = secs.getJSONObject(name)
        val f = File(outDir, "$name.tflite")
        val ok = File(outDir, "$name.verified")
        val size = s.getLong("end") - s.getLong("begin")
        if (!(f.length() == size && ok.isFile)) {
          log("Preparing the model, once: copying the $name section out of ${bundle.name} (${size / 1_000_000} MB)")
          val tc = System.nanoTime()
          val sha = copyRange(bundle, s.getLong("begin"), size, f)
          load.put("${name}_copy_ms", (System.nanoTime() - tc) / 1e6)
          check(sha == s.getString("sha256")) { "$name section: sha256 $sha != ${s.getString("sha256")}" }
          ok.writeText(sha)
        }
        extracted[name] = f
      }
      log("Reading the tokenizer")
      val tt = System.nanoTime()
      val tok = secs.getJSONObject("tokenizer")
      val tokenizer = RandomAccessFile(bundle, "r").use { raf ->
        val begin = tok.getLong("begin")
        val bytes = ByteArray((tok.getLong("end") - begin).toInt())
        raf.seek(begin)
        raf.readFully(bytes)
        val reader = InputStreamReader(InflaterInputStream(bytes.inputStream(8, bytes.size - 8), Inflater()), Charsets.UTF_8)
        Lfm3Tokenizer.load(reader.buffered(1 shl 16))
      }
      load.put("tokenizer_ms", (System.nanoTime() - tt) / 1e6)
      fun need(name: String) = find(context, name) ?: throw IllegalStateException(missing(context) ?: name)
      val env = Environment.create()
      val cacheDir = File(context.filesDir, "xnnpack_cache").apply { mkdirs() }
      val opened = ArrayList<AutoCloseable>()
      try {
        log("Loading the language model")
        val tl = System.nanoTime()
        val sig = sections.optJSONObject("decoder_signatures")
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
        val depthFile = need(DEPTH_STEP)
        val depth = openWithFallback(load, "depth", config.depth, depthFile, cacheDir) { DepthDecoder(Graph(depthFile, "depth_step", it, env)) }
          .also { opened.add(it) }
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
          VisionEncoder(encFile, adpFile, it, if (it.kind == "cpu") Accel("cpu", threads = it.threads).cached(cacheDir, adpFile, "vision_adapter") else it, env)
        }.also { opened.add(it) }
        after("vision encoder")
        load.put("open_ms", (System.nanoTime() - t0) / 1e6)
        load.put("available_accelerators", runCatching { env.getAvailableAccelerators().joinToString(",") }.getOrDefault("?"))
        log("Ready")
        return Lfm3OmniEngine(tokenizer, lm, depth, frame, codec, audio, vision, config, load)
      } catch (t: Throwable) {
        for (o in opened.reversed()) runCatching { o.close() }
        throw t
      }
    }

    /** A graph on its accelerator; when a GPU form cannot be created, the CPU, with the reason in [load]. */
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

    /** Copies [size] bytes from [src] at [begin] into [dst]; returns their SHA-256 (hex). */
    private fun copyRange(src: File, begin: Long, size: Long, dst: File): String {
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
      val magic = ByteArray(4)
      RandomAccessFile(tmp, "r").use { it.seek(4); it.readFully(magic) }
      check(String(magic, Charsets.US_ASCII) == "TFL3") { "${dst.name}: not a TFLite flatbuffer" }
      check(tmp.renameTo(dst)) { "rename ${tmp.name}" }
      return md.digest().joinToString("") { "%02x".format(it) }
    }
  }
}
