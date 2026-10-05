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

// Adapted from this repository's samples/litert/text_to_speech_streaming (KittenSynthesizer.kt),
// by way of john-rocky/hfmodels-android (commit 3086d647):
// litert/src/main/kotlin/io/github/johnrocky/hfmodels/litert/KittenSynthesizer.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.tts

import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.StandardOpenOption
import org.tensorflow.lite.Interpreter

/**
 * KittenTTS nano (StyleTTS2 + ISTFTNet, 24 kHz): three graphs with a dynamic sequence length.
 *
 * ```
 * ids [1,N] + style [1,256] + speed [1] -> predictor -> t_en [1,N,128], durations [N], d [1,N,256]
 *   host: repeat each row by its duration -> en [1,T,256], asr [1,T,128]
 *   en + style -> prosody -> f0 [1,2T], n [1,2T], har [1,120T+1,22]
 *   asr, f0, n, har, style -> vocoder -> wav [1,600T]
 * ```
 *
 * All three run on the classic Interpreter API: the predictor and prosody graphs keep their fused
 * LSTM state in variable tensors, which CompiledModel does not load (b/365299994), and the vocoder
 * runs there too (this repository's text_to_speech_streaming sample drives it through
 * CompiledModel with its own JNI resize). Before every run each graph is resized to the call's
 * shapes and its variable tensors are reset: a same-length second call would otherwise start from
 * the previous call's LSTM state. The end of the waveform is trimmed as the pip package does.
 * Created, run and closed on the LiteRT thread.
 */
internal class KittenSynthesizer
private constructor(
  private val predictor: Interpreter,
  private val prosody: Interpreter,
  private val vocoder: Interpreter,
  private val tailTrim: Int,
  private val minSamples: Int,
) : Closeable {
  class Result(val samples: FloatArray, val frames: Int, val durations: IntArray)

  /** [ids]: symbol ids with the 0 at each end; [style]: one row of the voice's table. */
  fun synthesize(ids: IntArray, style: FloatArray, speed: Float): Result {
    val n = ids.size
    val p =
      run(
        predictor,
        mapOf(
          "input_ids" to Feed(intArrayOf(1, n), ints(ids)),
          "style" to Feed(intArrayOf(1, STYLE_DIM), floats(style)),
          "speed" to Feed(intArrayOf(1), floats(floatArrayOf(speed))),
        ),
      )
    val tEn = p.floats("StatefulPartitionedCall:0", n * ASR_DIM)
    val durations = p.ints("StatefulPartitionedCall:1", n)
    val d = p.floats("StatefulPartitionedCall:2", n * D_DIM)
    var frames = 0
    for (x in durations) {
      check(x >= 0) { "the predictor gave a negative duration ($x)" }
      frames += x
    }
    check(frames > 0) { "the predictor gave 0 frames for $n symbols" }

    val q =
      run(
        prosody,
        mapOf(
          "en" to Feed(intArrayOf(1, frames, D_DIM), floats(repeatRows(d, durations, D_DIM))),
          "style" to Feed(intArrayOf(1, STYLE_DIM), floats(style)),
        ),
      )
    val f0 = q.floats("StatefulPartitionedCall:0", null)
    val noise = q.floats("StatefulPartitionedCall:1", f0.size)
    val har = q.floats("StatefulPartitionedCall:2", null)
    check(har.size % HAR_DIM == 0) {
      "prosody har has ${har.size} floats, not a multiple of $HAR_DIM"
    }

    val asr = repeatRows(tEn, durations, ASR_DIM)
    val v =
      run(
        vocoder,
        mapOf(
          "asr" to Feed(intArrayOf(1, frames, ASR_DIM), floats(asr)),
          "f0" to Feed(intArrayOf(1, f0.size), floats(f0)),
          "n" to Feed(intArrayOf(1, noise.size), floats(noise)),
          "har" to Feed(intArrayOf(1, har.size / HAR_DIM, HAR_DIM), floats(har)),
          "style" to Feed(intArrayOf(1, STYLE_DIM), floats(style)),
        ),
      )
    val wav = v.floats(null, SAMPLES_PER_FRAME * frames)
    return Result(trim(wav, tailTrim, minSamples), frames, durations)
  }

  /**
   * Closes all three graphs even when one throws; the first failure is rethrown after the others
   * closed.
   */
  override fun close() {
    var failure: Throwable? = null
    for (graph in listOf(predictor, prosody, vocoder)) {
      runCatching { graph.close() }
        .onFailure { t ->
          val first = failure
          if (first != null) {
            first.addSuppressed(t)
          } else {
            failure = t
          }
        }
    }
    failure?.let { throw it }
  }

  private class Feed(val shape: IntArray, val data: ByteBuffer)

  /**
   * The outputs of the run that just finished, copied out (the tensor memory is reused by the next
   * allocation).
   */
  private class Outputs(private val it: Interpreter) {
    private fun buffer(name: String?): ByteBuffer {
      val i =
        if (name == null) {
          0
        } else {
          (0 until it.outputTensorCount).firstOrNull { i -> it.getOutputTensor(i).name() == name }
            ?: throw IllegalStateException("no output '$name'")
        }
      return it.getOutputTensor(i).asReadOnlyBuffer().order(ByteOrder.nativeOrder())
    }

    /** [name] null = the only output. [expected] null = any size. */
    fun floats(name: String?, expected: Int?): FloatArray {
      val b = buffer(name).asFloatBuffer()
      check(expected == null || b.remaining() == expected) {
        "output ${name ?: "0"} has ${b.remaining()} floats, expected $expected"
      }
      return FloatArray(b.remaining()).also { b.get(it) }
    }

    fun ints(name: String, expected: Int): IntArray {
      val b = buffer(name).asIntBuffer()
      check(b.remaining() == expected) {
        "output $name has ${b.remaining()} ints, expected $expected"
      }
      return IntArray(b.remaining()).also { b.get(it) }
    }
  }

  companion object {
    const val SAMPLES_PER_FRAME = 600
    const val STYLE_DIM = 256
    private const val D_DIM = 256
    private const val ASR_DIM = 128
    private const val HAR_DIM = 22

    private val PREDICTOR_INPUTS = setOf("input_ids", "style", "speed")
    private val PROSODY_INPUTS = setOf("en", "style")
    private val VOCODER_INPUTS = setOf("asr", "f0", "n", "har", "style")
    private val SPC_OUTPUTS =
      setOf("StatefulPartitionedCall:0", "StatefulPartitionedCall:1", "StatefulPartitionedCall:2")

    /** The three graphs by the names `loadMs`, `afterEach` and `xnnpackOff` use. */
    val GRAPHS = listOf("predictor", "prosody", "vocoder")

    /**
     * The graphs that run without the XNNPACK delegate: the predictor (its peak memory). On a
     * Galaxy S26 (2026-10-03, fp32) the predictor without XNNPACK took the peak RSS after the load
     * from 624 MB to 334 MB and the median synthesis from 288 ms to 301 ms, with the same frames.
     */
    val XNNPACK_OFF: Set<String> = setOf("predictor")

    /**
     * Opens the three graphs (Interpreter, [threads] threads, XNNPACK except on the graphs in
     * [xnnpackOff]) and checks their input and output names; [loadMs] gets each graph's
     * construction time, [afterEach] is called with its key. On the LiteRT thread. Throws the
     * runtime's exception unchanged; the caller maps it.
     */
    fun open(
      predictorFile: File,
      prosodyFile: File,
      vocoderFile: File,
      threads: Int,
      tailTrim: Int,
      minSamples: Int,
      loadMs: MutableMap<String, Long>,
      xnnpackOff: Set<String> = emptySet(),
      afterEach: (String) -> Unit = {},
    ): KittenSynthesizer {
      val opened = ArrayList<Interpreter>(3)
      fun graph(
        key: String,
        file: File,
        inputs: Set<String>,
        outputs: Set<String>?,
        placeholders: Map<String, IntArray> = emptyMap(),
      ): Interpreter {
        val t0 = System.nanoTime()
        val options =
          Interpreter.Options().setNumThreads(threads).setUseXNNPACK(key !in xnnpackOff)
        val it =
          if (placeholders.isEmpty()) {
            Interpreter(file, options)
          } else {
            Interpreter(withInputShapes(file, placeholders), options)
          }
        opened += it
        loadMs[key] = (System.nanoTime() - t0) / 1_000_000
        afterEach(key)
        val ins =
          (0 until it.inputTensorCount).map { i -> canonical(it.getInputTensor(i).name()) }.toSet()
        val outs = (0 until it.outputTensorCount).map { i -> it.getOutputTensor(i).name() }.toSet()
        val outputsMatch = if (outputs == null) outs.size == 1 else outs == outputs
        check(ins == inputs && outputsMatch) {
          "$key graph ${file.name}: inputs $ins, outputs $outs; expected inputs $inputs, " +
            "outputs ${outputs ?: "one"}"
        }
        return it
      }
      try {
        return KittenSynthesizer(
          graph("predictor", predictorFile, PREDICTOR_INPUTS, SPC_OUTPUTS),
          graph("prosody", prosodyFile, PROSODY_INPUTS, SPC_OUTPUTS),
          graph("vocoder", vocoderFile, VOCODER_INPUTS, null, VOCODER_PLACEHOLDERS),
          tailTrim,
          minSamples,
        )
      } catch (t: Throwable) {
        opened.forEach { runCatching { it.close() } }
        throw t
      }
    }

    /**
     * The vocoder's inputs carry placeholder shapes that disagree with each other (asr [1,1,128]
     * but f0 and n [1,1] and har [1,1,22], where f0 and n are 2T long and har 120T+1). The Java
     * Interpreter allocates at the placeholders when it is created, and XNNPACK refuses them
     * ("XNNPack delegate failed to reshape runtime"; Python's interpreter resizes first and never
     * meets them). So the vocoder is opened with the placeholders of T = 1; every call resizes to
     * its own T anyway.
     */
    private val VOCODER_PLACEHOLDERS =
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
        val name = canonical(String(bytes, Charsets.UTF_8))
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

    /**
     * "serving_default_x:0" and "x" -> "x", so signature and signature-less graphs feed the same
     * way.
     */
    fun canonical(name: String) = name.removePrefix("serving_default_").substringBefore(':')

    /**
     * Resizes each input to its feed's shape (re-allocating when any changed), resets the variable
     * tensors and runs. Inputs are matched by [canonical] name.
     */
    private fun run(it: Interpreter, feeds: Map<String, Feed>): Outputs {
      var resized = false
      val inputs = arrayOfNulls<Any>(it.inputTensorCount)
      for (i in 0 until it.inputTensorCount) {
        val tensor = it.getInputTensor(i)
        val name = canonical(tensor.name())
        val feed = feeds[name] ?: throw IllegalStateException("no feed for input '$name'")
        if (!tensor.shape().contentEquals(feed.shape)) {
          it.resizeInput(i, feed.shape)
          resized = true
        }
        inputs[i] = feed.data.rewind()
      }
      if (resized) {
        it.allocateTensors()
      }
      it.resetVariableTensors()
      it.runForMultipleInputsOutputs(inputs, emptyMap())
      return Outputs(it)
    }

    /** `np.repeat(x[0], durations, axis=0)` for a row-major [1, N, dim] tensor. */
    fun repeatRows(x: FloatArray, durations: IntArray, dim: Int): FloatArray {
      val out = FloatArray(durations.sum() * dim)
      var write = 0
      for (row in durations.indices) {
        repeat(durations[row]) {
          System.arraycopy(x, row * dim, out, write, dim)
          write += dim
        }
      }
      return out
    }

    /**
     * The pip package drops the last [tailTrim] samples of every chunk; keeps at least
     * min([minSamples], all). Clamped to [-1, 1].
     */
    fun trim(wav: FloatArray, tailTrim: Int, minSamples: Int): FloatArray {
      val n = maxOf(wav.size - tailTrim, minOf(minSamples, wav.size))
      return FloatArray(n) { wav[it].coerceIn(-1f, 1f) }
    }

    private fun floats(data: FloatArray): ByteBuffer =
      ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.nativeOrder()).also {
        it.asFloatBuffer().put(data)
      }

    private fun ints(data: IntArray): ByteBuffer =
      ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.nativeOrder()).also {
        it.asIntBuffer().put(data)
      }
  }
}
