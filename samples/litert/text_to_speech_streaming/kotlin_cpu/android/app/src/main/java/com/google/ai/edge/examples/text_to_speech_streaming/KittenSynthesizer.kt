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

package com.google.ai.edge.examples.text_to_speech_streaming

import android.content.Context
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.tensorflow.lite.Interpreter

/**
 * KittenTTS nano 0.8 (StyleTTS2 + ISTFTNet, 15M params, 24 kHz) on the LiteRT CPU (XNNPACK)
 * interpreter. All three graphs have a **dynamic sequence length** — any sentence runs on the same
 * graphs with no padding buckets, so compute scales with the text instead of a fixed bucket.
 *
 * ```
 * token ids [1,N] + style [1,256] + speed [1]
 *   -> predictor -> d [1,N,256], t_en [1,N,128], durations [N]
 *   -> (host: repeat rows by durations) -> en [1,T,256], asr [1,T,128]
 *   -> prosody  -> f0 [1,2T], noise [1,2T], harmonics [1,120T+1,22]
 *   -> vocoder  -> waveform [1,600T] @ 24 kHz
 * ```
 *
 * The host glue between graphs is the row-repeat alignment (verified bit-exact against the
 * reference ONNX `Loop`). Because the graphs are dynamic, every call resizes the inputs to the
 * actual sentence length and lets the interpreter re-propagate shapes — this is the piece the
 * fixed-shape CompiledModel path cannot express, and why this sample uses the Interpreter API.
 */
class KittenSynthesizer(context: Context) : Closeable {

  /** One synthesized sentence: float PCM at 24 kHz plus per-stage timing for the demo UI. */
  data class Result(val audio: FloatArray, val tokens: Int, val frames: Int, val synthMs: Long)

  private val predictor = loadModel(context, PREDICTOR_MODEL)
  private val prosody = loadModel(context, PROSODY_MODEL)
  private val vocoder = loadModel(context, VOCODER_MODEL)

  /** [VOICES.size, STYLE_ROWS, STYLE_DIM] — style vectors indexed by voice and text length. */
  private val voiceTable: FloatArray

  /** Total size of the three graphs on disk, for the "small model" readout in the UI. */
  val modelBytes: Long =
    listOf(PREDICTOR_MODEL, PROSODY_MODEL, VOCODER_MODEL).sumOf { File(context.filesDir, it).length() }

  init {
    val voicesFile = File(context.filesDir, VOICES_FILE)
    check(voicesFile.exists()) {
      "Voices not found: $VOICES_FILE. Push it first:\n  ./install_to_device.sh"
    }
    val bytes = voicesFile.readBytes()
    check(bytes.size == VOICES.size * STYLE_ROWS * STYLE_DIM * 4) {
      "Unexpected $VOICES_FILE size ${bytes.size}"
    }
    val floats = FloatArray(bytes.size / 4)
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(floats)
    voiceTable = floats
  }

  private fun loadModel(context: Context, name: String): Interpreter {
    val file = File(context.filesDir, name)
    check(file.exists()) { "Model not found: $name. Push it first:\n  ./install_to_device.sh" }
    return Interpreter(file, Interpreter.Options().setNumThreads(NUM_THREADS))
  }

  /**
   * The style vector conditioning every graph. KittenTTS ships one style row per *text length*
   * (the model expects a length-conditioned style), so the lookup uses the sentence's character
   * count — same as the upstream pip package.
   */
  private fun styleFor(voice: Int, textLength: Int): FloatArray {
    val row = minOf(textLength, STYLE_ROWS - 1)
    val offset = (voice * STYLE_ROWS + row) * STYLE_DIM
    return voiceTable.copyOfRange(offset, offset + STYLE_DIM)
  }

  /**
   * Synthesizes one sentence.
   *
   * @param symbolIds symbol ids from [KittenG2P.phonemize] (no BOS/EOS — added here).
   * @param voice index into [VOICES].
   * @param textLength character count of the source text (selects the style row).
   * @param speed speech speed; 1.0 = normal.
   */
  fun synthesize(symbolIds: IntArray, voice: Int, textLength: Int, speed: Float = 1f): Result {
    val startNanos = System.nanoTime()
    val style = styleFor(voice, textLength)

    // Predictor: [0] + ids + [0] -> per-token features + integer durations.
    val tokenCount = symbolIds.size + 2
    val ids = IntArray(tokenCount)
    symbolIds.copyInto(ids, 1)
    val predictorOut =
      run(
        predictor,
        mapOf(
          "input_ids" to Feed(intArrayOf(1, tokenCount), intBuffer(ids)),
          "style" to Feed(intArrayOf(1, STYLE_DIM), floatBuffer(style)),
          "speed" to Feed(intArrayOf(1), floatBuffer(floatArrayOf(speed))),
        ),
      )
    val tEn = predictorOut.float("StatefulPartitionedCall:0") // [1,N,128]
    val durations = predictorOut.int("StatefulPartitionedCall:1") // [N]
    val d = predictorOut.float("StatefulPartitionedCall:2") // [1,N,256]

    // Length-regulate: repeat each token's feature row by its predicted duration.
    val frames = durations.sum()
    val en = repeatRows(d, durations, D_DIM)
    val asr = repeatRows(tEn, durations, ASR_DIM)

    // Prosody: harmonic F0 / noise / harmonics envelope for the whole utterance.
    val prosodyOut =
      run(
        prosody,
        mapOf(
          "en" to Feed(intArrayOf(1, frames, D_DIM), floatBuffer(en)),
          "style" to Feed(intArrayOf(1, STYLE_DIM), floatBuffer(style)),
        ),
      )
    val f0 = prosodyOut.float("StatefulPartitionedCall:0") // [1,2T]
    val noise = prosodyOut.float("StatefulPartitionedCall:1") // [1,2T]
    val harmonics = prosodyOut.float("StatefulPartitionedCall:2") // [1,120T+1,22]

    // Vocoder (ISTFTNet): aligned text features + prosody -> waveform.
    val vocoderOut =
      run(
        vocoder,
        mapOf(
          "asr" to Feed(intArrayOf(1, frames, ASR_DIM), floatBuffer(asr)),
          "f0" to Feed(intArrayOf(1, f0.size), floatBuffer(f0)),
          "n" to Feed(intArrayOf(1, noise.size), floatBuffer(noise)),
          "har" to Feed(intArrayOf(1, harmonics.size / HAR_DIM, HAR_DIM), floatBuffer(harmonics)),
          "style" to Feed(intArrayOf(1, STYLE_DIM), floatBuffer(style)),
        ),
      )
    val waveform = vocoderOut.float("Identity") // [1,600T]

    // The upstream pip package trims the trailing samples of every chunk (synthesis tail).
    val sampleCount = maxOf(waveform.size - TAIL_TRIM, MIN_SAMPLES.coerceAtMost(waveform.size))
    val audio = FloatArray(sampleCount) { waveform[it].coerceIn(-1f, 1f) }

    return Result(audio, tokenCount, frames, (System.nanoTime() - startNanos) / 1_000_000)
  }

  /** `np.repeat(x[0], durations, axis=0)` for a row-major [1,N,dim] tensor. */
  private fun repeatRows(x: FloatArray, durations: IntArray, dim: Int): FloatArray {
    var total = 0
    for (duration in durations) total += duration
    val out = FloatArray(total * dim)
    var write = 0
    for (row in durations.indices) {
      repeat(durations[row]) {
        System.arraycopy(x, row * dim, out, write, dim)
        write += dim
      }
    }
    return out
  }

  // ---- Dynamic-shape interpreter plumbing ----------------------------------------------------

  private class Feed(val shape: IntArray, val data: ByteBuffer)

  private class Outputs(private val interpreter: Interpreter) {
    private fun buffer(nameSuffix: String): ByteBuffer {
      for (i in 0 until interpreter.outputTensorCount) {
        val tensor = interpreter.getOutputTensor(i)
        if (tensor.name().endsWith(nameSuffix)) {
          // Copy out: the tensor buffer is only valid until the next allocation.
          val source = tensor.asReadOnlyBuffer().order(ByteOrder.nativeOrder())
          val copy = ByteBuffer.allocate(source.remaining()).order(ByteOrder.nativeOrder())
          copy.put(source).rewind()
          return copy
        }
      }
      throw IllegalStateException("Output not found: $nameSuffix")
    }

    fun float(nameSuffix: String): FloatArray {
      val buffer = buffer(nameSuffix).asFloatBuffer()
      return FloatArray(buffer.remaining()).also { buffer.get(it) }
    }

    fun int(nameSuffix: String): IntArray {
      val buffer = buffer(nameSuffix).asIntBuffer()
      return IntArray(buffer.remaining()).also { buffer.get(it) }
    }
  }

  /**
   * Runs one graph with dynamic shapes: resize each input to the sentence's actual shape,
   * re-allocate, invoke. Inputs are matched by canonical tensor name ("serving_default_x:0" and
   * "x" both -> "x") so the same path drives signature and signature-less graphs.
   */
  private fun run(interpreter: Interpreter, feeds: Map<String, Feed>): Outputs {
    var resized = false
    val inputs = arrayOfNulls<Any>(interpreter.inputTensorCount)
    for (i in 0 until interpreter.inputTensorCount) {
      val tensor = interpreter.getInputTensor(i)
      val name = tensor.name().removePrefix("serving_default_").substringBefore(':')
      val feed = checkNotNull(feeds[name]) { "Missing input: $name" }
      if (!tensor.shape().contentEquals(feed.shape)) {
        interpreter.resizeInput(i, feed.shape)
        resized = true
      }
      inputs[i] = feed.data.rewind()
    }
    if (resized) interpreter.allocateTensors()
    // The fused dynamic-length LSTM kernels keep their hidden state in variable tensors, which
    // persist across invocations. A genuine length change resets them via re-allocation, but a
    // same-length invoke would start from the previous sentence's final state and corrupt the
    // predicted durations (repro: speaking the same text twice). Reset unconditionally.
    interpreter.resetVariableTensors()
    interpreter.runForMultipleInputsOutputs(inputs, emptyMap<Int, Any>())
    return Outputs(interpreter)
  }

  private fun floatBuffer(data: FloatArray): ByteBuffer =
    ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.nativeOrder()).also {
      it.asFloatBuffer().put(data)
    }

  private fun intBuffer(data: IntArray): ByteBuffer =
    ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.nativeOrder()).also {
      it.asIntBuffer().put(data)
    }

  override fun close() {
    predictor.close()
    prosody.close()
    vocoder.close()
  }

  companion object {
    const val SAMPLE_RATE = 24000

    /** The 8 bundled voices, in `voices.bin` order (see conversion/export_voices.py). */
    val VOICES =
      listOf(
        "expr-voice-2-m",
        "expr-voice-2-f",
        "expr-voice-3-m",
        "expr-voice-3-f",
        "expr-voice-4-m",
        "expr-voice-4-f",
        "expr-voice-5-m",
        "expr-voice-5-f",
      )

    private const val PREDICTOR_MODEL = "kitten_predictor_fp16.tflite"
    private const val PROSODY_MODEL = "kitten_prosody_fp16.tflite"
    private const val VOCODER_MODEL = "kitten_vocoder_fp16.tflite"
    private const val VOICES_FILE = "voices.bin"

    private const val NUM_THREADS = 4
    private const val STYLE_ROWS = 400
    private const val STYLE_DIM = 256
    private const val D_DIM = 256
    private const val ASR_DIM = 128
    private const val HAR_DIM = 22

    /** The upstream pip package trims this many samples from the end of every chunk. */
    private const val TAIL_TRIM = 5000
    private const val MIN_SAMPLES = 1200
  }
}
