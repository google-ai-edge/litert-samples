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

package com.google.ai.edge.examples.voice_assistant.check

import android.app.KeyguardManager
import android.content.Context
import android.os.Build
import android.os.PowerManager
import android.provider.Settings
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.examples.voice_assistant.Runtime
import com.google.ai.edge.examples.voice_assistant.VoiceAssistantException
import com.google.ai.edge.examples.voice_assistant.data.DownloadStatus
import com.google.ai.edge.examples.voice_assistant.data.ModelCatalog
import com.google.ai.edge.examples.voice_assistant.data.ModelStore
import com.google.ai.edge.examples.voice_assistant.tts.KittenSynthesizer
import com.google.ai.edge.examples.voice_assistant.tts.LiteRtSpeaker
import com.google.ai.edge.examples.voice_assistant.tts.NpzVoices
import com.google.ai.edge.examples.voice_assistant.tts.Speaker
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.StandardOpenOption
import java.util.Locale
import kotlin.math.abs
import kotlin.math.sqrt
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter

/**
 * Device check for the KittenTTS vocoder on CompiledModel. For ten fixed replies it runs the G2P,
 * the predictor and the prosody graph once and gives the same inputs (asr, f0, n, har, style) to
 * three vocoders: the sample's (CompiledModel, CPU, 4 threads, through LiteRtDynamicShape), the
 * same graph on CompiledModel with LiteRT's default CPU options (one thread, as the
 * text_to_speech_streaming sample opens it), and the same file on the classic Interpreter API (4
 * threads, XNNPACK), the way this sample ran its vocoder before, opened only here as the reference.
 * It compares the waveforms (length, max |Δ|, correlation), times the vocoders on a second, warm
 * pass over the replies, and times the sample's whole synthesis as the voice loop calls it (G2P to
 * trimmed waveform, with the default voice and speed, the ten replies once and then warm).
 *
 * Before it runs: the KittenTTS files in the app's store (one load in the app, or the files
 * side-loaded into its external files dir, see the README).
 *
 * ```
 * ./gradlew :app:connectedDebugAndroidTest \
 *   -Pandroid.injected.androidTest.leaveApksInstalledAfterRun=true \
 *   -Pandroid.testInstrumentationRunnerArguments.class=\
 * com.google.ai.edge.examples.voice_assistant.check.VocoderParityCheck
 * adb logcat -d -s voice-assistant-check | grep RESULT
 * ```
 *
 * Keep the APKs installed: uninstalling the app deletes its model store. The last line is `RESULT
 * step=all ok=true ...` when every step passed: for every reply the sample's vocoder gave as many
 * samples as the reference, and the warm pass gave the same samples as the first one. The check
 * sets no threshold on the difference between the waveforms: its lines report it.
 */
@RunWith(AndroidJUnit4::class)
class VocoderParityCheck {
  private val ctx = InstrumentationRegistry.getInstrumentation().targetContext

  @Test
  fun vocoderParity(): Unit = runBlocking {
    val failures = ArrayList<String>()
    fun step(name: String, ok: Boolean, values: String) {
      if (!ok) {
        failures += name
      }
      Log.i(TAG, "RESULT step=$name ok=$ok $values")
    }
    Log.i(TAG, "device=${Build.MODEL} build=${Build.DISPLAY} package=${ctx.packageName} ${state()}")
    var speaker: Speaker? = null
    val graphs = ArrayList<AutoCloseable>()
    try {
      // 1. the KittenTTS files, from the app's store (the side-loaded copies are imported first)
      val catalog =
        ModelCatalog.parse(ctx.assets.open("models.json").bufferedReader().use { it.readText() })
      val store = ModelStore(File(ctx.filesDir, "models"))
      val kitten = catalog.entry(ModelCatalog.KITTEN)
      if (store.inspect(kitten).status != DownloadStatus.READY) {
        store.sideLoad(kitten, ctx.getExternalFilesDir(null))
      }
      check(store.inspect(kitten).status == DownloadStatus.READY) {
        "${kitten.id} is not in the store: load the app once or side-load its files (README)"
      }
      val file = { name: String -> store.file(kitten, name) }

      // 2. the sample's speaker as the voice loop calls it: the ten replies once, then warm
      val symbols = { ctx.assets.open("symbols.json").bufferedReader().use { it.readText() } }
      val s = LiteRtSpeaker.open(file, symbols).also { speaker = it }
      val ids = REPLIES.map { s.phonemeIds(it) }
      val first = REPLIES.map { s.synthesize(it) }
      val warm = REPLIES.map { s.synthesize(it) }
      val same = warm.indices.count { warm[it].samples.contentEquals(first[it].samples) }
      val total = warm.map { it.timing.totalMs }
      val rtf = warm.map { it.timing.totalMs / 1000 / (it.samples.size.toDouble() / it.sampleRate) }
      step(
        "synthesis",
        same == REPLIES.size,
        "replies=${REPLIES.size} chars=${REPLIES.minOf(::chars)}-${REPLIES.maxOf(::chars)} " +
          "ms_total_median=${f1(median(total))} ms_total_min=${f1(total.min())} " +
          "ms_total_max=${f1(total.max())} " +
          "ms_synth_median=${f1(median(warm.map { it.timing.synthMs }))} " +
          "ms_g2p_median=${f1(median(warm.map { it.timing.g2pMs }))} " +
          "rtf_median=${f3(median(rtf))} first_call_ms_total=${f1(first[0].timing.totalMs)} " +
          "same_pcm_as_first=$same/${REPLIES.size}",
      )
      s.closeAndJoin()
      speaker = null

      // 3. the vocoders on the same inputs: the sample's, the same graph on one thread, and the
      // reference on the Interpreter
      val voice = LiteRtSpeaker.VOICES[0]
      val table = checkNotNull(NpzVoices.read(file("voices.npz"))[voice]) { "no voice $voice" }
      val speed = LiteRtSpeaker.graphSpeed(1f, LiteRtSpeaker.SPEED_PRIORS[voice] ?: 1.0)
      Runtime.call {
        val loadMs = LinkedHashMap<String, Long>()
        val synth =
          KittenSynthesizer.open(
              file("kitten_predictor.tflite"),
              file("kitten_prosody.tflite"),
              file("kitten_vocoder.tflite"),
              LiteRtSpeaker.CPU_THREADS,
              LiteRtSpeaker.TAIL_TRIM,
              LiteRtSpeaker.MIN_SAMPLES,
              loadMs,
              xnnpackOff = KittenSynthesizer.XNNPACK_OFF,
            )
            .also { graphs += it }
        var t0 = System.nanoTime()
        val oneThread =
          CompiledModel.create(
              file("kitten_vocoder.tflite").absolutePath,
              CompiledModel.Options(Accelerator.CPU),
              null,
            )
            .also { graphs += it }
        loadMs["vocoder_1_thread"] = (System.nanoTime() - t0) / 1_000_000
        t0 = System.nanoTime()
        val options =
          Interpreter.Options().setNumThreads(LiteRtSpeaker.CPU_THREADS).setUseXNNPACK(true)
        val reference =
          Interpreter(withInputShapes(file("kitten_vocoder.tflite"), PLACEHOLDERS), options).also {
            graphs += it
          }
        loadMs["vocoder_interpreter"] = (System.nanoTime() - t0) / 1_000_000
        step(
          "load",
          true,
          "load_ms " + loadMs.entries.joinToString(" ") { "${it.key}=${it.value}" },
        )

        // First pass: the inputs once per reply, the three waveforms, their comparison.
        val inputs = ArrayList<KittenSynthesizer.VocoderInputs>()
        val outputs = ArrayList<List<FloatArray>>()
        for ((i, text) in REPLIES.withIndex()) {
          val x = synth.vocoderInputs(ids[i], table.row(minOf(chars(text), table.rows - 1)), speed)
          val timed = timedVocoders(synth, oneThread, reference, x)
          val (compiled, one, interpreter) = timed.map { it.first }
          val same = compiled.size == interpreter.size
          step(
            "parity",
            same && compiled.all { it.isFinite() },
            "i=$i chars=${chars(text)} frames=${x.frames} samples_compiled=${compiled.size} " +
              "samples_interpreter=${interpreter.size} " +
              "max_abs=${e3(maxAbs(compiled, interpreter))} " +
              "corr=${f6(corr(compiled, interpreter))} " +
              "peak_interpreter=${f3(interpreter.maxOf { abs(it) }.toDouble())} " +
              "max_abs_1_thread=${e3(maxAbs(one, compiled))} " +
              "first_ms_compiled=${f1(timed[0].second)} first_ms_1_thread=${f1(timed[1].second)} " +
              "first_ms_interpreter=${f1(timed[2].second)}",
          )
          inputs += x
          outputs += listOf(compiled, one, interpreter)
        }

        // Second pass, warm: the same inputs again, timed.
        val ms = List(3) { ArrayList<Double>() }
        var repeated = 0
        for ((i, x) in inputs.withIndex()) {
          val timed = timedVocoders(synth, oneThread, reference, x)
          val again = (0 until 3).map { timed[it].first.contentEquals(outputs[i][it]) }
          if (again[0]) {
            repeated++
          }
          for (k in 0 until 3) {
            ms[k] += timed[k].second
          }
          Log.i(
            TAG,
            "RESULT info warm i=$i chars=${chars(REPLIES[i])} frames=${x.frames} " +
              "ms_compiled=${f1(timed[0].second)} ms_1_thread=${f1(timed[1].second)} " +
              "ms_interpreter=${f1(timed[2].second)} same_as_first=$again",
          )
        }
        step(
          "vocoder_timing",
          repeated == inputs.size,
          "warm_calls=${inputs.size} " +
            "compiled_4_threads_ms_median=${f1(median(ms[0]))} min=${f1(ms[0].min())} " +
            "max=${f1(ms[0].max())} compiled_1_thread_ms_median=${f1(median(ms[1]))} " +
            "min=${f1(ms[1].min())} max=${f1(ms[1].max())} " +
            "interpreter_4_threads_ms_median=${f1(median(ms[2]))} min=${f1(ms[2].min())} " +
            "max=${f1(ms[2].max())} compiled_same_as_first=$repeated/${inputs.size}",
        )
        graphs.forEach { it.close() }
        graphs.clear()
      }
    } catch (e: VoiceAssistantException) {
      step("exception", false, "error=${e.code} reason=${q(e.message.orEmpty())}")
    } catch (e: Exception) {
      Log.e(TAG, "failed", e)
      step("exception", false, "error=${e.javaClass.simpleName} message=${q(e.message.orEmpty())}")
    } finally {
      runCatching { speaker?.closeAndJoin() }
      runCatching { Runtime.call { graphs.forEach { g -> runCatching { g.close() } } } }
      Log.i(
        TAG,
        "RESULT step=all ok=${failures.isEmpty()} failed=$failures device=${Build.MODEL} " +
          "build=${Build.DISPLAY} ${state()}",
      )
    }
    assertTrue("failed steps: $failures (adb logcat -d -s $TAG)", failures.isEmpty())
  }

  /**
   * The three vocoders on [x], in this order: the sample's, the one-thread CompiledModel, the
   * Interpreter. Each waveform comes with its wall time in ms.
   */
  private fun timedVocoders(
    synth: KittenSynthesizer,
    oneThread: CompiledModel,
    reference: Interpreter,
    x: KittenSynthesizer.VocoderInputs,
  ): List<Pair<FloatArray, Double>> {
    val out = ArrayList<Pair<FloatArray, Double>>(3)
    var t0 = System.nanoTime()
    val compiled = synth.vocode(x)
    out += compiled to (System.nanoTime() - t0) / 1e6
    t0 = System.nanoTime()
    val one = KittenSynthesizer.vocode(oneThread, x)
    out += one to (System.nanoTime() - t0) / 1e6
    t0 = System.nanoTime()
    val interpreter = interpreterVocode(reference, x)
    out += interpreter to (System.nanoTime() - t0) / 1e6
    return out
  }

  /**
   * The vocoder on the Interpreter, as this sample ran it before: each input resized to the call's
   * shape (re-allocating when any changed), the variable tensors reset, then the run, and the
   * waveform copied out.
   */
  private fun interpreterVocode(it: Interpreter, x: KittenSynthesizer.VocoderInputs): FloatArray {
    val feeds =
      mapOf(
        "asr" to (intArrayOf(1, x.frames, ASR_DIM) to x.asr),
        "f0" to (intArrayOf(1, x.f0.size) to x.f0),
        "n" to (intArrayOf(1, x.noise.size) to x.noise),
        "har" to (intArrayOf(1, x.har.size / HAR_DIM, HAR_DIM) to x.har),
        "style" to (intArrayOf(1, KittenSynthesizer.STYLE_DIM) to x.style),
      )
    var resized = false
    val inputs = arrayOfNulls<Any>(it.inputTensorCount)
    for (i in 0 until it.inputTensorCount) {
      val tensor = it.getInputTensor(i)
      val name = KittenSynthesizer.canonical(tensor.name())
      val (shape, data) = feeds[name] ?: throw IllegalStateException("no feed for input '$name'")
      if (!tensor.shape().contentEquals(shape)) {
        it.resizeInput(i, shape)
        resized = true
      }
      inputs[i] =
        ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.nativeOrder()).also { b ->
          b.asFloatBuffer().put(data)
        }
    }
    if (resized) {
      it.allocateTensors()
    }
    it.resetVariableTensors()
    it.runForMultipleInputsOutputs(inputs, emptyMap())
    val b = it.getOutputTensor(0).asReadOnlyBuffer().order(ByteOrder.nativeOrder()).asFloatBuffer()
    return FloatArray(b.remaining()).also { b.get(it) }
  }

  /** Airplane mode, the screen and the keyguard, the thermal status. */
  private fun state(): String {
    val power = ctx.getSystemService(Context.POWER_SERVICE) as PowerManager
    val keyguard = ctx.getSystemService(Context.KEYGUARD_SERVICE) as KeyguardManager
    val airplane =
      Settings.Global.getInt(ctx.contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0) == 1
    return "airplane_mode=$airplane screen_interactive=${power.isInteractive} " +
      "keyguard_locked=${keyguard.isKeyguardLocked} thermal_status=${power.currentThermalStatus}"
  }

  private fun chars(text: String) = text.codePointCount(0, text.length)

  /** Over the shorter length when the lengths differ. */
  private fun maxAbs(a: FloatArray, b: FloatArray): Double {
    var m = 0.0
    for (i in 0 until minOf(a.size, b.size)) {
      m = maxOf(m, abs(a[i].toDouble() - b[i]))
    }
    return m
  }

  /** Pearson's correlation, over the shorter length when the lengths differ. */
  private fun corr(a: FloatArray, b: FloatArray): Double {
    val n = minOf(a.size, b.size)
    var sa = 0.0
    var sb = 0.0
    for (i in 0 until n) {
      sa += a[i]
      sb += b[i]
    }
    val ma = sa / n
    val mb = sb / n
    var ab = 0.0
    var aa = 0.0
    var bb = 0.0
    for (i in 0 until n) {
      val da = a[i] - ma
      val db = b[i] - mb
      ab += da * db
      aa += da * da
      bb += db * db
    }
    return ab / sqrt(aa * bb)
  }

  private fun median(v: List<Double>): Double {
    val s = v.sorted()
    return if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2
  }

  private fun f1(v: Double) = String.format(Locale.US, "%.1f", v)

  private fun f3(v: Double) = String.format(Locale.US, "%.3f", v)

  private fun f6(v: Double) = String.format(Locale.US, "%.6f", v)

  private fun e3(v: Double) = String.format(Locale.US, "%.3e", v)

  private fun q(s: String) = "\"" + s.take(300).replace("\n", " ").replace("\"", "'") + "\""

  private companion object {
    const val TAG = "voice-assistant-check"
    const val ASR_DIM = 128
    const val HAR_DIM = 22

    /** The ten replies of the README's synthesis measurements, 34 to 77 characters. */
    val REPLIES =
      listOf(
        "Alarm set for seven thirty tomorrow morning.",
        "Your timer for ten minutes is running.",
        "It is three fifteen in the afternoon.",
        "You have two events tomorrow: a team standup at nine and the dentist at five.",
        "Done. I added the meeting to your calendar.",
        "Sorry, I could not find a calendar on this phone.",
        "The alarm is set for half past nine tonight.",
        "Timer started: forty five minutes.",
        "Good morning! Everything is running on this phone, offline.",
        "I set an alarm for eight and a timer for twenty minutes.",
      )

    /**
     * The vocoder's inputs carry placeholder shapes that disagree with each other (asr [1,1,128]
     * but f0 and n [1,1] and har [1,1,22], where f0 and n are 2T long and har 120T+1). The Java
     * Interpreter allocates at the placeholders when it is created, and XNNPACK refuses them
     * ("XNNPack delegate failed to reshape runtime", while Python's interpreter resizes first and
     * never meets them). So the reference opens with the placeholders of T = 1, and every call
     * resizes to its own T anyway. CompiledModel opens the file as it is.
     */
    val PLACEHOLDERS =
      mapOf("f0" to intArrayOf(1, 2), "n" to intArrayOf(1, 2), "har" to intArrayOf(1, 121, HAR_DIM))

    /**
     * [file] mapped copy-on-write with the default shapes of the named inputs of subgraph 0
     * rewritten to [shapes]: the `shape` vectors of the TFLite flatbuffer (Model.subgraphs 2 ->
     * SubGraph.tensors 0 / inputs 1 -> Tensor.shape 0 / name 3), same rank only. A private mapping
     * needs a channel open for writing, but its changes never reach the file: the verified file on
     * disk stays as it is.
     */
    fun withInputShapes(file: File, shapes: Map<String, IntArray>): ByteBuffer {
      val mapped =
        FileChannel.open(file.toPath(), StandardOpenOption.READ, StandardOpenOption.WRITE).use {
          it.map(FileChannel.MapMode.PRIVATE, 0, it.size())
        }
      val b = mapped.duplicate().order(ByteOrder.LITTLE_ENDIAN)
      fun field(table: Int, index: Int): Int {
        val vt = table - b.getInt(table)
        val slot = 4 + 2 * index
        if (slot + 2 > (b.getShort(vt).toInt() and 0xffff)) {
          return -1
        }
        val off = b.getShort(vt + slot).toInt() and 0xffff
        return if (off == 0) -1 else table + off
      }
      fun deref(p: Int): Int {
        check(p >= 0) { "${file.name}: a flatbuffer field is missing" }
        return p + b.getInt(p)
      }
      val subgraph = deref(deref(field(b.getInt(0), 2)) + 4)
      val tensors = deref(field(subgraph, 0))
      val inputs = deref(field(subgraph, 1))
      val done = HashSet<String>()
      for (i in 0 until b.getInt(inputs)) {
        val tensor = deref(tensors + 4 + 4 * b.getInt(inputs + 4 + 4 * i))
        val s = deref(field(tensor, 3))
        val bytes = ByteArray(b.getInt(s))
        b.position(s + 4)
        b.get(bytes)
        val name = KittenSynthesizer.canonical(String(bytes, Charsets.UTF_8))
        val dims = shapes[name] ?: continue
        val v = deref(field(tensor, 0))
        check(b.getInt(v) == dims.size) {
          "${file.name}: input '$name' has rank ${b.getInt(v)}, expected ${dims.size}"
        }
        for (k in dims.indices) {
          b.putInt(v + 4 + 4 * k, dims[k])
        }
        done += name
      }
      check(done == shapes.keys) { "${file.name}: inputs ${shapes.keys - done} not found" }
      return mapped
    }
  }
}
