/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.ai.edge.examples.text_to_speech

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin

/**
 * Matcha-TTS on-device. The three heavy graphs run on the LiteRT CompiledModel GPU (ML Drift);
 * everything stochastic/sequential runs host-side here (no FFT anywhere — Matcha's HiFi-GAN
 * vocoder is a time-domain GAN, which is what lets the whole synthesis path stay on the GPU).
 *
 *   phoneme ids --(host: embed + pad)--> text_encoder(GPU) --> mu, logw
 *               --(host: durations + length-regulator)--> mu_y[1,80,T]
 *               --(host: Euler ODE loop, N steps)-->
 *                   decoder(GPU) x N  (x_t, mu_y, sin_emb(t), mask) --> v
 *               --(host: denormalize)--> mel --> vocoder(GPU) --> waveform[T*256]
 *
 * Fixed shapes (MAX_TEXT phonemes, MAX_MEL frames); a runtime float mask makes the padded
 * positions a no-op (additive attention bias), so one compiled graph handles any length.
 * Graphs are loaded from filesDir (push via scripts/install_to_device.sh).
 */
class MatchaSynthesizer(private val context: Context) : Closeable {

    companion object {
        const val MAX_TEXT = 256
        const val MAX_MEL = 512
        const val N_FEATS = 80
        const val N_CHANNELS = 192
        const val HOP = 256
        const val SAMPLE_RATE = 22050
        const val LENGTH_SCALE = 0.95f
        const val MEL_MEAN = -5.536622f
        const val MEL_STD = 2.116101f
        const val TIME_DIM = 160
        const val DEFAULT_STEPS = 10
        const val TEXTENC = "matcha_textenc_fp16.tflite"
        const val DECODER = "matcha_decoder_fp16.tflite"
        const val VOCODER = "matcha_vocoder_fp16.tflite"
        const val MAX_SAMPLES = MAX_MEL * HOP
    }

    private fun load(name: String, acc: Accelerator = Accelerator.GPU): CompiledModel {
        val f = File(context.filesDir, name)
        check(f.exists()) { "Model not found: $name. Push it first:\n  scripts/install_to_device.sh" }
        return CompiledModel.create(f.absolutePath, CompiledModel.Options(acc), null)
    }

    private val embTable: FloatArray = readFloats(context.assets.open("emb.bin").readBytes()) // [178*192]

    private val textenc = load(TEXTENC)
    // decoder on CPU: the diffusers transformer blocks are mis-fused on the Mali ML Drift
    // delegate (residual collapses to corr 0.006 in the full graph, though the SAME
    // transformer is corr 0.98 as a standalone graph -> a fusion bug, not the ops). On CPU
    // the decoder is exact; RTF stays realtime (~0.93). textenc + vocoder run on GPU.
    private val decoder = load(DECODER, Accelerator.CPU)
    private val vocoder = load(VOCODER)
    private val teIn = textenc.createInputBuffers();  private val teOut = textenc.createOutputBuffers()
    private val decIn = decoder.createInputBuffers();  private val decOut = decoder.createOutputBuffers()
    private val vocIn = vocoder.createInputBuffers();  private val vocOut = vocoder.createOutputBuffers()

    data class Result(val audio: FloatArray, val frames: Int, val steps: Int, val ms: Long)

    /** phonemeIds: matcha symbol ids (NOT yet interspersed with blanks). */
    fun synthesize(phonemeIds: IntArray, nSteps: Int = DEFAULT_STEPS, seed: Long? = null): Result {
        val t0 = System.nanoTime()

        // ---- intersperse blanks (id 0), pad to MAX_TEXT, build text mask ----
        val tx = minOf(phonemeIds.size * 2 + 1, MAX_TEXT)
        val ids = IntArray(MAX_TEXT)
        run { var i = 1; for (p in phonemeIds) { if (i >= MAX_TEXT) break; ids[i] = p; i += 2 } }
        val tmask = FloatArray(MAX_TEXT) { if (it < tx) 1f else 0f }

        // ---- host phoneme-embedding lookup -> [1, MAX_TEXT, 192] ----
        val embX = FloatArray(MAX_TEXT * N_CHANNELS)
        for (t in 0 until MAX_TEXT) {
            val src = ids[t] * N_CHANNELS
            System.arraycopy(embTable, src, embX, t * N_CHANNELS, N_CHANNELS)
        }

        // ---- text encoder (GPU) -> mu[1,80,T], logw[1,1,T] ----
        teIn[0].writeFloat(embX); teIn[1].writeFloat(tmask)
        textenc.run(teIn, teOut)
        val mu = teOut[0].readFloat()    // [80*MAX_TEXT], channel-major
        val logw = teOut[1].readFloat()  // [MAX_TEXT]

        // ---- durations -> cumulative -> length regulator (mu_y) ----
        val wceil = FloatArray(MAX_TEXT) { ceil(exp(logw[it]) * tmask[it]) * LENGTH_SCALE }
        val cum = FloatArray(MAX_TEXT)
        var acc = 0f
        for (i in 0 until MAX_TEXT) { acc += wceil[i]; cum[i] = acc }
        val yLen = minOf(maxOf(acc.toInt(), 1), MAX_MEL)

        val muY = FloatArray(N_FEATS * MAX_MEL)
        run {
            var p = 0
            for (f in 0 until yLen) {
                while (p < MAX_TEXT - 1 && cum[p] <= f) p++
                for (c in 0 until N_FEATS) muY[c * MAX_MEL + f] = mu[c * MAX_TEXT + p]
            }
        }
        val ymask = FloatArray(MAX_MEL) { if (it < yLen) 1f else 0f }

        // ---- Euler ODE: x_{k+1} = x_k + dt * decoder(x_k, mu_y, sin_emb(t), mask) ----
        val rnd = if (seed != null) java.util.Random(seed) else java.util.Random()
        val x = FloatArray(N_FEATS * MAX_MEL)
        for (c in 0 until N_FEATS) for (f in 0 until yLen) x[c * MAX_MEL + f] = rnd.nextGaussian().toFloat()
        val dt = 1f / nSteps
        var tcur = 0f
        for (step in 0 until nSteps) {
            decIn[0].writeFloat(x)
            decIn[1].writeFloat(muY)
            decIn[2].writeFloat(sinPosEmb(tcur))
            decIn[3].writeFloat(ymask)
            decoder.run(decIn, decOut)
            val v = decOut[0].readFloat()
            for (i in x.indices) x[i] += dt * v[i]
            tcur += dt
        }

        // ---- denormalize + zero pad -> mel -> vocoder (GPU) -> waveform ----
        val mel = FloatArray(N_FEATS * MAX_MEL)
        for (c in 0 until N_FEATS) for (f in 0 until yLen) {
            val i = c * MAX_MEL + f; mel[i] = x[i] * MEL_STD + MEL_MEAN
        }
        vocIn[0].writeFloat(mel)
        vocoder.run(vocIn, vocOut)
        val wavFull = vocOut[0].readFloat()    // [MAX_MEL*HOP]
        val n = yLen * HOP
        val audio = FloatArray(n) { wavFull[it].coerceIn(-1f, 1f) }

        return Result(audio, yLen, nSteps, (System.nanoTime() - t0) / 1_000_000)
    }

    /** matcha SinusoidalPosEmb (weight-free): t -> [TIME_DIM] = [sin(...) | cos(...)]. */
    private fun sinPosEmb(t: Float, scale: Float = 1000f): FloatArray {
        val half = TIME_DIM / 2
        val out = FloatArray(TIME_DIM)
        val k = -ln(10000.0) / (half - 1)
        for (i in 0 until half) {
            val e = scale * t * exp(i * k).toFloat()
            out[i] = sin(e); out[half + i] = cos(e)
        }
        return out
    }

    private fun readFloats(b: ByteArray): FloatArray {
        val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(b.size / 4) { bb.float }
    }

    override fun close() {
        teIn.forEach { it.close() }; teOut.forEach { it.close() }
        decIn.forEach { it.close() }; decOut.forEach { it.close() }
        vocIn.forEach { it.close() }; vocOut.forEach { it.close() }
        textenc.close(); decoder.close(); vocoder.close()
    }
}
