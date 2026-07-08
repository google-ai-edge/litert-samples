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

package com.google.ai.edge.examples.text_to_speech

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Random
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
 * ```
 * phoneme ids --(host: embed + pad)--> text_encoder(GPU) --> mu, logw
 *             --(host: durations + length-regulator)--> mu_y[1,80,T]
 *             --(host: Euler ODE loop, N steps)-->
 *                 decoder(GPU) x N  (x_t, mu_y, sin_emb(t), mask) --> v
 *             --(host: denormalize)--> mel --> vocoder(GPU) --> waveform[T*256]
 * ```
 *
 * Fixed shapes ([MAX_TEXT] phonemes, [MAX_MEL] frames); a runtime float mask makes the padded
 * positions a no-op (additive attention bias), so one compiled graph handles any length.
 * Graphs are loaded from filesDir (push via scripts/install_to_device.sh).
 */
class MatchaSynthesizer(private val context: Context) : Closeable {

    /** One synthesis result: float PCM plus timing metadata for the demo UI. */
    data class Result(val audio: FloatArray, val frames: Int, val steps: Int, val ms: Long)

    private val embeddingTable: FloatArray =
        readFloats(context.assets.open("emb.bin").readBytes()) // [178 * N_CHANNELS]

    private val textEncoder = loadModel(TEXT_ENCODER_MODEL)

    // Decoder on CPU: the diffusers transformer blocks are mis-fused on the Mali ML Drift
    // delegate (residual collapses to corr 0.006 in the full graph, though the SAME
    // transformer is corr 0.98 as a standalone graph -> a fusion bug, not the ops). On CPU
    // the decoder is exact; RTF stays realtime (~0.93). Text encoder + vocoder run on GPU.
    private val decoder = loadModel(DECODER_MODEL, Accelerator.CPU)
    private val vocoder = loadModel(VOCODER_MODEL)

    private val textEncoderInputs = textEncoder.createInputBuffers()
    private val textEncoderOutputs = textEncoder.createOutputBuffers()
    private val decoderInputs = decoder.createInputBuffers()
    private val decoderOutputs = decoder.createOutputBuffers()
    private val vocoderInputs = vocoder.createInputBuffers()
    private val vocoderOutputs = vocoder.createOutputBuffers()

    private fun loadModel(name: String, accelerator: Accelerator = Accelerator.GPU): CompiledModel {
        val file = File(context.filesDir, name)
        check(file.exists()) {
            "Model not found: $name. Push it first:\n  scripts/install_to_device.sh"
        }
        return CompiledModel.create(file.absolutePath, CompiledModel.Options(accelerator), null)
    }

    /**
     * Synthesizes speech from matcha symbol ids (NOT yet interspersed with blanks).
     *
     * @param phonemeIds matcha symbol ids from [MatchaG2P.phonemize].
     * @param nSteps number of Euler ODE steps (more = higher quality, slower).
     * @param seed optional fixed noise seed for reproducible output.
     */
    fun synthesize(phonemeIds: IntArray, nSteps: Int = DEFAULT_STEPS, seed: Long? = null): Result {
        val startNanos = System.nanoTime()

        // Intersperse blanks (id 0), pad to MAX_TEXT, build the text mask.
        val textLength = minOf(phonemeIds.size * 2 + 1, MAX_TEXT)
        val ids = IntArray(MAX_TEXT)
        var writeIndex = 1
        for (id in phonemeIds) {
            if (writeIndex >= MAX_TEXT) break
            ids[writeIndex] = id
            writeIndex += 2
        }
        val textMask = FloatArray(MAX_TEXT) { if (it < textLength) 1f else 0f }

        // Host phoneme-embedding lookup -> [1, MAX_TEXT, N_CHANNELS].
        val embedded = FloatArray(MAX_TEXT * N_CHANNELS)
        for (t in 0 until MAX_TEXT) {
            System.arraycopy(
                embeddingTable, ids[t] * N_CHANNELS, embedded, t * N_CHANNELS, N_CHANNELS)
        }

        // Text encoder (GPU) -> mu[1,80,T], logw[1,1,T].
        textEncoderInputs[0].writeFloat(embedded)
        textEncoderInputs[1].writeFloat(textMask)
        textEncoder.run(textEncoderInputs, textEncoderOutputs)
        val mu = textEncoderOutputs[0].readFloat() // [N_FEATS * MAX_TEXT], channel-major
        val logDurations = textEncoderOutputs[1].readFloat() // [MAX_TEXT]

        // Durations -> cumulative -> length regulator (mu_y).
        val durations = FloatArray(MAX_TEXT) {
            ceil(exp(logDurations[it]) * textMask[it]) * LENGTH_SCALE
        }
        val cumulative = FloatArray(MAX_TEXT)
        var total = 0f
        for (i in 0 until MAX_TEXT) {
            total += durations[i]
            cumulative[i] = total
        }
        val melLength = minOf(maxOf(total.toInt(), 1), MAX_MEL)

        val muY = FloatArray(N_FEATS * MAX_MEL)
        var phonemeIndex = 0
        for (frame in 0 until melLength) {
            while (phonemeIndex < MAX_TEXT - 1 && cumulative[phonemeIndex] <= frame) {
                phonemeIndex++
            }
            for (c in 0 until N_FEATS) {
                muY[c * MAX_MEL + frame] = mu[c * MAX_TEXT + phonemeIndex]
            }
        }
        val melMask = FloatArray(MAX_MEL) { if (it < melLength) 1f else 0f }

        // Euler ODE: x_{k+1} = x_k + dt * decoder(x_k, mu_y, sin_emb(t), mask).
        val random = if (seed != null) Random(seed) else Random()
        val x = FloatArray(N_FEATS * MAX_MEL)
        for (c in 0 until N_FEATS) {
            for (frame in 0 until melLength) {
                x[c * MAX_MEL + frame] = random.nextGaussian().toFloat()
            }
        }
        val dt = 1f / nSteps
        var time = 0f
        repeat(nSteps) {
            decoderInputs[0].writeFloat(x)
            decoderInputs[1].writeFloat(muY)
            decoderInputs[2].writeFloat(sinusoidalPositionEmbedding(time))
            decoderInputs[3].writeFloat(melMask)
            decoder.run(decoderInputs, decoderOutputs)
            val velocity = decoderOutputs[0].readFloat()
            for (i in x.indices) {
                x[i] += dt * velocity[i]
            }
            time += dt
        }

        // Denormalize + zero pad -> mel -> vocoder (GPU) -> waveform.
        val mel = FloatArray(N_FEATS * MAX_MEL)
        for (c in 0 until N_FEATS) {
            for (frame in 0 until melLength) {
                val i = c * MAX_MEL + frame
                mel[i] = x[i] * MEL_STD + MEL_MEAN
            }
        }
        vocoderInputs[0].writeFloat(mel)
        vocoder.run(vocoderInputs, vocoderOutputs)
        val waveform = vocoderOutputs[0].readFloat() // [MAX_MEL * HOP]
        val sampleCount = melLength * HOP
        val audio = FloatArray(sampleCount) { waveform[it].coerceIn(-1f, 1f) }

        return Result(audio, melLength, nSteps, (System.nanoTime() - startNanos) / 1_000_000)
    }

    /** Matcha SinusoidalPosEmb (weight-free): t -> [TIME_DIM] = [sin(...) | cos(...)]. */
    private fun sinusoidalPositionEmbedding(t: Float, scale: Float = 1000f): FloatArray {
        val half = TIME_DIM / 2
        val embedding = FloatArray(TIME_DIM)
        val logScale = -ln(10000.0) / (half - 1)
        for (i in 0 until half) {
            val angle = scale * t * exp(i * logScale).toFloat()
            embedding[i] = sin(angle)
            embedding[half + i] = cos(angle)
        }
        return embedding
    }

    private fun readFloats(bytes: ByteArray): FloatArray {
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(bytes.size / 4) { buffer.float }
    }

    override fun close() {
        textEncoderInputs.forEach { it.close() }
        textEncoderOutputs.forEach { it.close() }
        decoderInputs.forEach { it.close() }
        decoderOutputs.forEach { it.close() }
        vocoderInputs.forEach { it.close() }
        vocoderOutputs.forEach { it.close() }
        textEncoder.close()
        decoder.close()
        vocoder.close()
    }

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
        const val MAX_SAMPLES = MAX_MEL * HOP

        private const val TEXT_ENCODER_MODEL = "matcha_textenc_fp16.tflite"
        private const val DECODER_MODEL = "matcha_decoder_fp16.tflite"
        private const val VOCODER_MODEL = "matcha_vocoder_fp16.tflite"
    }
}
