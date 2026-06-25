/*
 * Copyright 2024 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.aiedge.examples.texttospeech

import android.content.Context
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.cos
import kotlin.math.sin

/**
 * Kokoro-82M (StyleTTS2 + ISTFTNet) text-to-speech on the LiteRT CPU runtime.
 *
 * The .tflite graph computes everything up to the magnitude/phase spectrogram; the
 * final iSTFT overlap-add (post_n_fft = 20, hop = 5) runs here on the host — it has no
 * learned weights and is numerically exact (~2 ms). Moving it out of the graph sidesteps
 * a converter constant-dedup bug that otherwise fuses the two iSTFT conv-transpose weights.
 *
 * Model I/O (fixed-length export):
 *   in0  input_ids [1, N]  int64      in1  ref_s [1, 256]  float32
 *   out0 spec [1, 11, F]   float32    out1 phase [1, 11, F] float32   out2 pred_dur [N] int64
 *
 * NOTE: this build is FIXED-LENGTH — N and F are baked at export, so it reproduces one
 * baked utterance. Arbitrary text needs a converter-side dynamic-LSTM fix.
 */
class KokoroTtsHelper(private val context: Context) {

    data class Result(
        val audio: FloatArray,
        val inferenceMs: Long,
        val istftMs: Long,
        val audioSeconds: Float,
        val rtf: Float,
    )

    private var interpreter: Interpreter? = null
    val ready: Boolean get() = interpreter != null

    // iSTFT constants — CustomSTFT(filter_length = 20, hop = 5, freq_bins = 11).
    private val nFft = 20
    private val hop = 5
    private val freqBins = 11
    private val frameSamples = 600 // host trims audio to sum(pred_dur) * 600 samples

    private lateinit var wr: Array<FloatArray>      // [11][20] inverse-DFT cos basis
    private lateinit var wi: Array<FloatArray>      // [11][20] inverse-DFT sin basis
    private lateinit var refIds: Array<LongArray>   // [1][N]  baked demo phoneme ids
    private lateinit var refStyle: Array<FloatArray> // [1][256] voice style vector

    fun setup(threads: Int = 4) {
        val modelFile = File(context.filesDir, MODEL_FILE)
        check(modelFile.exists()) {
            "$MODEL_FILE not found in ${context.filesDir}. Stage it with scripts/install_model.sh " +
                "(adb push + run-as cp into the app's files dir)."
        }
        val buffer = RandomAccessFile(modelFile, "r").use { raf ->
            raf.channel.map(FileChannel.MapMode.READ_ONLY, 0, raf.length())
        }
        val itp = Interpreter(buffer, Interpreter.Options().apply { numThreads = threads })
        interpreter = itp
        Log.i(TAG, "Interpreter ready (${modelFile.length() / 1_000_000} MB, $threads threads)")

        // input_ids is the int64 input; its length N sizes the baked phoneme buffer.
        val idsTensor = if (itp.getInputTensor(0).dataType() == DataType.INT64) {
            itp.getInputTensor(0)
        } else {
            itp.getInputTensor(1)
        }
        val n = idsTensor.shape()[1]
        refIds = arrayOf(loadLongs("ref_input_ids_i64.bin", n))
        refStyle = arrayOf(loadFloats("ref_s_f32.bin", 256))
        val flatWr = loadFloats("istft_Wr_f32.bin", freqBins * nFft)
        val flatWi = loadFloats("istft_Wi_f32.bin", freqBins * nFft)
        wr = Array(freqBins) { k -> FloatArray(nFft) { j -> flatWr[k * nFft + j] } }
        wi = Array(freqBins) { k -> FloatArray(nFft) { j -> flatWi[k * nFft + j] } }
    }

    fun synthesize(): Result {
        val itp = interpreter ?: error("Interpreter not set up")

        // Bind inputs by dtype (export order is input_ids int64, ref_s float32).
        val inputs = arrayOfNulls<Any>(itp.inputTensorCount)
        for (i in 0 until itp.inputTensorCount) {
            inputs[i] = if (itp.getInputTensor(i).dataType() == DataType.INT64) refIds else refStyle
        }

        // Allocate outputs from the graph's declared shapes.
        val specShape = itp.getOutputTensor(0).shape() // [1, fb, F]
        val fb = specShape[1]
        val frames = specShape[2]
        val out0 = Array(1) { Array(fb) { FloatArray(frames) } }
        val out1 = Array(1) { Array(fb) { FloatArray(frames) } }
        val predDur = LongArray(itp.getOutputTensor(2).shape()[0])
        val outputs = HashMap<Int, Any>().apply {
            put(0, out0); put(1, out1); put(2, predDur)
        }

        val t0 = SystemClock.uptimeMillis()
        itp.runForMultipleInputsOutputs(inputs, outputs)
        val inferenceMs = SystemClock.uptimeMillis() - t0

        // spec = exp(.) >= 0 (never negative); phase = sin(.) carries negatives.
        val a0 = out0[0]
        val a1 = out1[0]
        val spec: Array<FloatArray>
        val phase: Array<FloatArray>
        if (hasNegative(a0) && !hasNegative(a1)) {
            spec = a1; phase = a0
        } else {
            spec = a0; phase = a1
        }

        val t1 = SystemClock.uptimeMillis()
        val audio = istft(spec, phase, predDur.sum().toInt() * frameSamples)
        val istftMs = SystemClock.uptimeMillis() - t1

        val seconds = audio.size / SAMPLE_RATE.toFloat()
        val rtf = if (seconds > 0f) inferenceMs / (seconds * 1000f) else 0f
        Log.i(TAG, "infer=${inferenceMs}ms istft=${istftMs}ms audio=${"%.2f".format(seconds)}s rtf=${"%.3f".format(rtf)}")
        return Result(audio, inferenceMs, istftMs, seconds, rtf)
    }

    /**
     * iSTFT overlap-add == CustomSTFT.inverse: real = spec*cos(phase), imag = spec*sin(phase),
     * conv_transpose1d(Wr) - conv_transpose1d(Wi) with stride = hop, then crop n_fft/2 (center
     * padding) and trim to trimSamples. Verified bit-exact vs the host reference (corr 1.000000).
     */
    private fun istft(spec: Array<FloatArray>, phase: Array<FloatArray>, trimSamples: Int): FloatArray {
        val frames = spec[0].size
        val raw = FloatArray((frames - 1) * hop + nFft)
        for (f in 0 until frames) {
            val base = f * hop
            for (k in 0 until freqBins) {
                val mg = spec[k][f]
                val ph = phase[k][f]
                val re = mg * cos(ph.toDouble()).toFloat()
                val im = mg * sin(ph.toDouble()).toFloat()
                val wrk = wr[k]
                val wik = wi[k]
                for (n in 0 until nFft) {
                    raw[base + n] += re * wrk[n] - im * wik[n]
                }
            }
        }
        val start = nFft / 2
        val len = minOf(trimSamples, raw.size - start).coerceAtLeast(0)
        return raw.copyOfRange(start, start + len)
    }

    private fun hasNegative(a: Array<FloatArray>): Boolean {
        for (row in a) for (v in row) if (v < 0f) return true
        return false
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }

    private fun loadFloats(name: String, count: Int): FloatArray {
        val bytes = context.assets.open(name).use { it.readBytes() }
        val out = FloatArray(count)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
        return out
    }

    private fun loadLongs(name: String, count: Int): LongArray {
        val bytes = context.assets.open(name).use { it.readBytes() }
        val out = LongArray(count)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer().get(out)
        return out
    }

    companion object {
        private const val TAG = "KokoroTts"
        const val MODEL_FILE = "kokoro_specout.tflite"
        const val SAMPLE_RATE = 24000
    }
}
