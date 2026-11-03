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

import android.app.ActivityManager
import android.content.Context
import android.os.Build
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
 * LFM3-Audio-2B on LiteRT: the files and the graphs.
 *
 * Files (in the app's files dir, or its external files dir for `adb push`; README.md):
 *   lfm3-audio-2b_int4_b128_c2048_h30.litertlm   the bundle: the language model, its embedder, the audio encoder and
 *                                               the tokenizer (the same file LiteRT-LM runs for text answers)
 *   lfm3_audio_depth_proj_r4b.tflite            depth decoder, projection (once per speech frame)
 *   lfm3_audio_depth_pass_r4b_int8.tflite       depth decoder, one maskgit pass (three per frame), CPU: int8
 *                                               weights, float32 activations
 *   lfm3_audio_frame_embed_f32.tflite           speech frame -> the next LM input
 *   lfm3_audio_codec_decoder_f12_r4b2.tflite    codec decoder, 12 frames (0.96 s) per call
 *   lfm3_audio_depth_pass_r5b_aot.tflite        optional: the depth pass compiled for the Hexagon NPU of an SM8850
 *                                               phone (Galaxy S26), used by an APK built with the NPU libraries
 */
class Lfm3AudioEngine private constructor(
  val tokenizer: Lfm3Tokenizer,
  val lm: LmRunner,
  val depth: DepthDecoder,
  val frame: FrameEmbed,
  val codec: CodecDecoder,
  val audioEncoder: AudioEncoder,
  val config: Config,
  val load: JSONObject,
) : AutoCloseable {

  data class Config(
    val lm: Accel = Accel("cpu"),
    val depthProj: Accel = Accel("cpu"),
    val depthPass: Accel = Accel("cpu"),
    val frame: Accel = Accel("cpu"),
    val codec: Accel = Accel("gpu", "fp32"),
    val audio: Accel = Accel("gpu", "fp32"),
    val depthPassFile: String = DEPTH_PASS,
  )

  val generator = SpeechGenerator(lm, depth, frame)

  /** Where each part runs after the fallbacks, for the status line: e.g. "language model CPU, depth decoder NPU, ...". */
  fun placement(): String {
    fun kind(key: String) = load.optString(key).substringBefore('(').let { k ->
      if (load.optString(key).contains("fallback")) "CPU" else k.uppercase()
    }
    return "language model ${kind("lm_accel")} · depth decoder ${kind("depth_pass_accel")} · codec ${kind("codec_accel")} · " +
      "audio encoder ${kind("audio_accel")}"
  }

  override fun close() {
    lm.close()
    depth.close()
    frame.close()
    codec.close()
    audioEncoder.close()
  }

  /** Not enough free memory for the next stage: nothing more was loaded, and what was loaded is closed. */
  class NotEnoughMemory(message: String) : Exception(message)

  companion object {
    const val TAG = "VoicemailSummary"
    const val DEPTH_PROJ = "lfm3_audio_depth_proj_r4b.tflite"
    const val DEPTH_PASS = "lfm3_audio_depth_pass_r4b_int8.tflite"
    const val DEPTH_PASS_NPU_SM8850 = "lfm3_audio_depth_pass_r5b_aot.tflite"
    const val FRAME = "lfm3_audio_frame_embed_f32.tflite"
    const val CODEC = "lfm3_audio_codec_decoder_f12_r4b2.tflite"
    const val SAMPLE_RATE = 24000

    /**
     * Free memory each loading stage needs (bytes), checked before the stage with [RESERVE] on top; the largest stage
     * is checked first, before anything is loaded (a stage that cannot fit must not start: on a phone low on memory
     * the system kills other apps first). Stages: the language model (decoder + embedder, the CPU packing its int4
     * weights), the speech parts (depth decoder, frame embedding, codec decoder on the GPU), the audio encoder (GPU).
     */
    val STAGE_NEED = linkedMapOf(
      "language model" to 2_600_000_000L,
      "speech decoder" to 800_000_000L,
      "audio encoder" to 800_000_000L,
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
      val names = listOf(DEPTH_PROJ, DEPTH_PASS, FRAME, CODEC).filter { find(context, it) == null }
      val bundle = findBundle(context, sections)
      if (names.isEmpty() && bundle != null) return null
      val list = (if (bundle == null) listOf("the .litertlm bundle (${sections.getLong("bundle_bytes")} bytes)") else emptyList()) + names
      val dir = context.getExternalFilesDir(null)?.absolutePath ?: context.filesDir.absolutePath
      return "The model files are not on this phone yet. Missing: ${list.joinToString(", ")}. " +
        "Push them into $dir with install_to_device.sh (README.md), then reopen the app."
    }

    /**
     * This phone's configuration: the depth decoder's pass on the Hexagon NPU when the APK carries the NPU libraries,
     * the phone is an SM8850 and its compiled file is there; everything else as the speech sample measured fastest
     * (language model, depth projection and frame embedding on the CPU, codec decoder and audio encoder on the GPU).
     */
    fun defaultConfig(context: Context): Config {
      val soc = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else ""
      return if (BuildConfig.NPU_LIBS && soc == "SM8850" && find(context, DEPTH_PASS_NPU_SM8850) != null) {
        Config(depthPass = Accel("npu"), depthPassFile = DEPTH_PASS_NPU_SM8850)
      } else Config()
    }

    /** The NPU's libraries are wired into the Environment only for a configuration that runs a graph on the NPU. */
    fun environment(context: Context, npu: Boolean): Environment = if (npu && BuildConfig.NPU_LIBS) {
      val dir = context.applicationInfo.nativeLibraryDir
      Environment.create(context, mapOf(Environment.Option.DispatchLibraryDir to dir, Environment.Option.CompilerPluginLibraryDir to dir))
    } else Environment.create()

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
    fun open(context: Context, config: Config, log: (String) -> Unit = {}): Lfm3AudioEngine {
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
      for (name in listOf("decoder", "embedder", "audio")) {
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
      val env = environment(context, npu = config.depthPass.kind == "npu")
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
        val projFile = need(DEPTH_PROJ)
        val proj = Graph(projFile, "depth_proj", config.depthProj.cached(cacheDir, projFile, "depth_proj"), env).also { opened.add(it) }
        // the NPU form is a compiled file of its own: when it cannot be created, the int8 pass on the CPU
        val pass = try {
          val f = need(config.depthPassFile)
          Graph(f, "depth_pass", config.depthPass.cached(cacheDir, f, "depth_pass"), env).also {
            load.put("depth_pass_accel", config.depthPass.toString()); load.put("depth_pass_file", f.name)
          }
        } catch (e: Exception) {
          if (config.depthPass.kind == "cpu") throw e
          Log.w(TAG, "depth pass on ${config.depthPass} failed, the CPU instead: $e")
          load.put("depth_pass_${config.depthPass.kind}_error", "${e.javaClass.simpleName}: ${e.message}".take(400))
          val f = need(DEPTH_PASS)
          Graph(f, "depth_pass", Accel("cpu").cached(cacheDir, f, "depth_pass"), env).also {
            load.put("depth_pass_accel", "cpu(fallback)"); load.put("depth_pass_file", f.name)
          }
        }
        opened.add(pass)
        val depth = DepthDecoder(proj, pass)
        load.put("depth_load_ms", (System.nanoTime() - td) / 1e6)
        val frameFile = need(FRAME)
        val frame = FrameEmbed(frameFile, config.frame.cached(cacheDir, frameFile, "frame"), env).also { opened.add(it) }
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
        load.put("open_ms", (System.nanoTime() - t0) / 1e6)
        load.put("available_accelerators", runCatching { env.getAvailableAccelerators().joinToString(",") }.getOrDefault("?"))
        log("Ready")
        return Lfm3AudioEngine(tokenizer, lm, depth, frame, codec, audio, config, load)
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
