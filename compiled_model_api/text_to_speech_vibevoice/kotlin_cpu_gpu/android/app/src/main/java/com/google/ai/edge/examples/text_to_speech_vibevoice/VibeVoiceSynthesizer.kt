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

package com.google.ai.edge.examples.text_to_speech_vibevoice

import android.content.Context
import android.util.Half
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * VibeVoice-Realtime-0.5B text-to-speech on LiteRT CompiledModel (hybrid GPU/CPU).
 *
 * Four LiteRT graphs plus host-side orchestration mirror the reference `generate()` —
 * a stochastic, autoregressive, next-token-diffusion TTS:
 *
 *   text --BPE--> ids
 *   per text token:  embed(host) -> base_lm(CPU) -> +type -> tts_lm(CPU) -> cond
 *   per speech token (6 per window):
 *     5-step DPM-Solver++:  head(GPU) x2 (cond + null) -> CFG -> v -> latent[64]
 *     connector(host) + type -> tts_lm(CPU) & neg tts_lm(CPU) -> next cond
 *     EOS classifier(host) stops generation
 *   accumulate latents -> decoder(CPU) -> 24 kHz waveform
 *
 * Placement is dictated by the Pixel 8a Mali ML Drift delegate *at the LiteRT version this
 * sample pins* (2.1.3): the two Qwen2 LMs are rejected on GPU with "unsupported
 * FULLY_CONNECTED weights shape". LiteRT 2.1.5 fixes that — there both LMs delegate every
 * node and match the CPU reference to corr >= 0.9998 — but until this sample moves to it they
 * run as fp32 graphs on CPU, which fp16 requires anyway (fp16 collapses their 20-layer stack
 * on ARM XNNPACK). The σ-VAE decoder
 * compiles on GPU but ML Drift miscomputes it (a graph-assembly buffer bug — every op is
 * bit-exact in isolation, but the assembled ConvNeXt block is wrong), so it too runs as an
 * fp32 graph on CPU. Only the tiny diffusion head runs on GPU (fp32 precision).
 *
 * The two Qwen2 LMs keep their KV cache host-side (packed `[1, L*nkv, Pmax, 64]` tensors
 * fed in and read back each step — the ML-Drift-safe "state as graph I/O" pattern). The
 * voice is a precomputed prompt KV cache (voice_*.bin). No FFT anywhere. Graphs load from
 * the external files dir (push via install_to_device.sh); the fp16 embedding table is mmapped.
 */
class VibeVoiceSynthesizer(private val context: Context) : Closeable {

    companion object {
        const val H = 896            // hidden size
        const val HD = 64            // head dim
        const val NKV = 2            // kv heads (GQA)
        const val VAE = 64           // acoustic latent dim
        const val THETA = 1_000_000.0
        const val BASE_L = 4
        const val BASE_PMAX = 128
        const val TTS_L = 20
        const val TTS_PMAX = 384
        const val TEXT_WIN = 5
        const val SPEECH_WIN = 6
        const val CFG = 1.3f
        const val DECODE_FRAMES = 128
        const val HOP = 3200
        const val SAMPLE_RATE = 24000
        // speech latent normalization (model.speech_scaling_factor / speech_bias_factor)
        const val SCALE = 0.2333984375f
        const val BIAS = -0.0703125f
        val TIMESTEPS = floatArrayOf(999f, 799f, 599f, 400f, 200f)
        val SIGMAS = doubleArrayOf(
            20291.302734375, 3.103729724884033, 1.3907514810562134,
            0.7402812242507935, 0.3374606668949127, 0.0,
        )
        const val BASE_LM = "vv_base_lm_kv_fp32.tflite"
        const val TTS_LM = "vv_tts_lm_kv_fp32.tflite"
        const val HEAD = "vv_diffhead_fp16.tflite"
        const val DECODER = "vv_decoder_fp32.tflite"
        const val EMB = "embed_tokens.f16"
        const val GLUE = "glue.f32"
        const val VOICE = "voice_en-Emma_woman.bin"
    }

    // Model files (~1.7 GB) live in the app's external files dir so `adb push` can write them
    // directly (no tmp double-copy) — see install_to_device.sh.
    private val modelDir =
        requireNotNull(context.getExternalFilesDir(null)) { "External storage unavailable" }

    // ---- graphs -----------------------------------------------------------
    private fun load(
        name: String,
        acc: Accelerator = Accelerator.GPU,
        gpuFp32: Boolean = false,
    ): CompiledModel {
        val f = File(modelDir, name)
        check(f.exists()) { "Model not found: $name. Push it first:\n  install_to_device.sh" }
        val opts = CompiledModel.Options(acc)
        // Mali ML Drift defaults to fp16; the deep graphs (safe-RMS, large conv accumulation) need
        // fp32 on-device or the audio degrades to noise (desktop upcasts to fp32 and is clean).
        if (gpuFp32) {
            opts.gpuOptions =
                CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
        }
        return CompiledModel.create(f.absolutePath, opts, null)
    }

    // Placement (see the class KDoc): the two Qwen2 LMs and the σ-VAE decoder run on CPU as
    // fp32 graphs — LiteRT 2.1.3's Mali delegate rejects the LM FULLY_CONNECTED shape (2.1.5
    // fixes it) and miscomputes the decoder (every version tried). Only the small diffusion
    // head runs on the GPU.
    private val baseLm = load(BASE_LM, Accelerator.CPU)
    private val ttsLm = load(TTS_LM, Accelerator.CPU)
    private val head = load(HEAD, Accelerator.GPU, gpuFp32 = true)
    private val decoder = load(DECODER, Accelerator.CPU)
    private val baseIn = baseLm.createInputBuffers()
    private val baseOut = baseLm.createOutputBuffers()
    private val ttsIn = ttsLm.createInputBuffers()
    private val ttsOut = ttsLm.createOutputBuffers()
    private val headIn = head.createInputBuffers()
    private val headOut = head.createOutputBuffers()
    private val decIn = decoder.createInputBuffers()
    private val decOut = decoder.createOutputBuffers()

    // ---- assets -----------------------------------------------------------
    private fun path(name: String): File {
        val f = File(modelDir, name)
        check(f.exists()) { "Asset not found: $name. Push it first:\n  install_to_device.sh" }
        return f
    }

    private val embChannel = RandomAccessFile(path(EMB), "r").channel
    private val embMap = embChannel
        .map(FileChannel.MapMode.READ_ONLY, 0, embChannel.size()).order(ByteOrder.LITTLE_ENDIAN)

    private val glue: FloatArray = readF32(path(GLUE))
    // glue.f32 offsets (float indices), see conversion/build_vibevoice.py
    private val connFc1W = 0
    private val connFc1B = 57344
    private val connNormW = 58240
    private val connFc2W = 59136
    private val connFc2B = 861952
    private val typeEmb = 862848
    private val eosFc1W = 864640
    private val eosFc1B = 1667456
    private val eosFc2W = 1668352
    private val eosFc2B = 1669248

    private val tokenizer = BpeTokenizer(context)
    private val rnd = java.util.Random()

    data class Result(val audio: FloatArray, val speechTokens: Int, val ms: Long)

    /** A host-managed packed KV cache: `[L*nkv, Pmax, HD]` for keys and values, filled left. */
    private inner class Cache(val layers: Int, val pmax: Int) {
        val g = layers * NKV
        val pk = FloatArray(g * pmax * HD)
        val pv = FloatArray(g * pmax * HD)
        var pos = 0

        fun mask(): FloatArray {
            val m = FloatArray(pmax + 1) { -30000f }
            for (p in 0 until pos) {
                m[p] = 0f
            }
            m[pmax] = 0f                       // current token, concatenated at the tail
            return m
        }

        fun append(nk: FloatArray, nv: FloatArray) {
            for (gi in 0 until g) {
                val dst = gi * pmax * HD + pos * HD
                val src = gi * HD
                System.arraycopy(nk, src, pk, dst, HD)
                System.arraycopy(nv, src, pv, dst, HD)
            }
            pos++
        }
    }

    private val baseCache = Cache(BASE_L, BASE_PMAX)
    private val ttsCache = Cache(TTS_L, TTS_PMAX)
    private val negCache = Cache(TTS_L, TTS_PMAX)
    private val ttsInitHidden = FloatArray(H)
    private val negInitHidden = FloatArray(H)

    init {
        seedVoice(path(VOICE))
    }

    /** Seed the three KV caches + init hiddens from the precomputed voice preset. */
    private fun seedVoice(f: File) {
        val bb = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        fun seed(c: Cache, seedLen: Int) {
            fun read(dst: FloatArray) {
                for (gi in 0 until c.g) for (s in 0 until seedLen) {
                    val base = gi * c.pmax * HD + s * HD
                    for (d in 0 until HD) {
                        dst[base + d] = Half.toFloat(bb.short)
                    }
                }
            }
            read(c.pk)
            read(c.pv)
            c.pos = seedLen
        }
        seed(baseCache, 74)
        seed(ttsCache, 251)
        seed(negCache, 1)
        for (i in 0 until H) {
            ttsInitHidden[i] = bb.float
        }
        for (i in 0 until H) {
            negInitHidden[i] = bb.float
        }
    }

    // ---- small host math --------------------------------------------------
    private fun embRow(id: Int): FloatArray {
        val out = FloatArray(H)
        var b = id.toLong() * H * 2L
        for (j in 0 until H) {
            out[j] = Half.toFloat(embMap.getShort(b.toInt()))
            b += 2
        }
        return out
    }

    private fun linear(x: FloatArray, w: Int, b: Int, outDim: Int, inDim: Int): FloatArray {
        val out = FloatArray(outDim)
        for (o in 0 until outDim) {
            var acc = glue[b + o]
            val row = w + o * inDim
            for (i in 0 until inDim) {
                acc += glue[row + i] * x[i]
            }
            out[o] = acc
        }
        return out
    }

    private fun connector(latent: FloatArray): FloatArray {
        val h1 = linear(latent, connFc1W, connFc1B, H, VAE)          // fc1: 64 -> 896
        var ms = 0f
        for (v in h1) {
            ms += v * v
        }
        ms /= H
        val r = (1.0 / sqrt(ms + 1e-6)).toFloat()                    // RMSNorm (eps 1e-6)
        for (o in 0 until H) {
            h1[o] = h1[o] * r * glue[connNormW + o]
        }
        return linear(h1, connFc2W, connFc2B, H, H)                  // fc2: 896 -> 896
    }

    private fun eosProb(hidden: FloatArray): Float {
        val e1 = linear(hidden, eosFc1W, eosFc1B, H, H)
        for (o in 0 until H) {
            if (e1[o] < 0f) {
                e1[o] = 0f              // relu
            }
        }
        var acc = glue[eosFc2B]
        val row = eosFc2W
        for (i in 0 until H) {
            acc += glue[row + i] * e1[i]           // fc2: 896 -> 1
        }
        return (1.0 / (1.0 + exp(-acc.toDouble()))).toFloat()
    }

    private fun addType(hidden: FloatArray, type: Int): FloatArray {
        val out = FloatArray(H)
        val base = typeEmb + type * H
        for (o in 0 until H) {
            out[o] = hidden[o] + glue[base + o]
        }
        return out
    }

    private fun rope(pos: Int): Pair<FloatArray, FloatArray> {
        val cos = FloatArray(HD)
        val sin = FloatArray(HD)
        for (j in 0 until HD / 2) {
            val ang = pos * (1.0 / Math.pow(THETA, 2.0 * j / HD))
            val c = kotlin.math.cos(ang).toFloat()
            val s = kotlin.math.sin(ang).toFloat()
            cos[j] = c
            cos[j + HD / 2] = c
            sin[j] = s
            sin[j + HD / 2] = s
        }
        return cos to sin
    }

    private fun timestepEmb(t: Float): FloatArray {
        val out = FloatArray(256)
        for (j in 0 until 128) {
            val a = t * exp(-ln(10000.0) * j / 128).toFloat()
            out[j] = kotlin.math.cos(a.toDouble()).toFloat()
            out[128 + j] = kotlin.math.sin(a.toDouble()).toFloat()
        }
        return out
    }

    private fun alphaSigma(sigma: Double): Pair<Double, Double> {
        val a = 1.0 / sqrt(sigma * sigma + 1.0)
        return a to sigma * a
    }

    // ---- one LM step over a packed KV cache -------------------------------
    private fun lmStep(
        model: CompiledModel, inB: List<TensorBuffer>, outB: List<TensorBuffer>,
        cache: Cache, x: FloatArray,
    ): FloatArray {
        val (cos, sin) = rope(cache.pos)
        inB[0].writeFloat(x)
        inB[1].writeFloat(cos)
        inB[2].writeFloat(sin)
        inB[3].writeFloat(cache.mask())
        inB[4].writeFloat(cache.pk)
        inB[5].writeFloat(cache.pv)
        model.run(inB, outB)
        val hidden = outB[0].readFloat()
        cache.append(outB[1].readFloat(), outB[2].readFloat())
        return hidden
    }

    private fun headV(noisy: FloatArray, tEmb: FloatArray, cond: FloatArray): FloatArray {
        headIn[0].writeFloat(noisy)
        headIn[1].writeFloat(tEmb)
        headIn[2].writeFloat(cond)
        head.run(headIn, headOut)
        return headOut[0].readFloat()
    }

    /** 5-step DPM-Solver++ (order 2, v-prediction) with classifier-free guidance. */
    private fun sampleLatent(cond: FloatArray, negCond: FloatArray): FloatArray {
        var speech = FloatArray(VAE) { rnd.nextGaussian().toFloat() }
        var x0Prev: DoubleArray? = null
        for (i in 0 until TIMESTEPS.size) {
            val tEmb = timestepEmb(TIMESTEPS[i])
            val vc = headV(speech, tEmb, cond)
            val vu = headV(speech, tEmb, negCond)
            val (aS0, sgS0) = alphaSigma(SIGMAS[i])
            val (aT, sgT) = alphaSigma(SIGMAS[i + 1])
            val lamT = ln(aT) - ln(sgT)
            val lamS0 = ln(aS0) - ln(sgS0)
            val h = lamT - lamS0
            val c1 = aT * (exp(-h) - 1.0)
            val x0 = DoubleArray(VAE) {
                val v = vu[it] + CFG * (vc[it] - vu[it])            // CFG guidance
                aS0 * speech[it] - sgS0 * v                          // v-prediction -> x0
            }
            val lowerOrder = i == 0 || i == TIMESTEPS.size - 1       // first & last step: 1st order
            val next = FloatArray(VAE)
            if (lowerOrder || x0Prev == null) {
                for (k in 0 until VAE) {
                    next[k] = ((sgT / sgS0) * speech[k] - c1 * x0[k]).toFloat()
                }
            } else {
                val (aS1, sgS1) = alphaSigma(SIGMAS[i - 1])
                val h0 = lamS0 - (ln(aS1) - ln(sgS1))
                val r0 = h0 / h
                for (k in 0 until VAE) {
                    val d1 = (1.0 / r0) * (x0[k] - x0Prev!![k])
                    next[k] = ((sgT / sgS0) * speech[k] - c1 * x0[k] - 0.5 * c1 * d1).toFloat()
                }
            }
            speech = next
            x0Prev = x0
        }
        return speech
    }

    /** Generate speech for `text` using the bundled voice preset. */
    fun synthesize(text: String, maxSpeechTokens: Int = TTS_PMAX - 260): Result {
        val t0 = System.nanoTime()
        // Reset the KV caches to the voice preset so each call is independent (otherwise a second
        // call continues from the first's end state and stops almost immediately).
        seedVoice(path(VOICE))
        val ids = tokenizer.encode(text)
        val latents = ArrayList<FloatArray>()
        var cond = ttsInitHidden.copyOf()
        var negCond = negInitHidden.copyOf()

        var winIdx = 0
        var finished = false
        while (!finished) {
            val start = winIdx * TEXT_WIN
            winIdx++
            for (j in start until minOf(start + TEXT_WIN, ids.size)) {   // text tokens
                val bh = lmStep(baseLm, baseIn, baseOut, baseCache, embRow(ids[j]))
                cond = lmStep(ttsLm, ttsIn, ttsOut, ttsCache, addType(bh, 1))
            }
            for (s in 0 until SPEECH_WIN) {                              // speech tokens
                val latent = sampleLatent(cond, negCond)
                val scaled = FloatArray(VAE) { latent[it] / SCALE - BIAS }
                latents.add(scaled)
                val ae = connector(latent)
                cond = lmStep(ttsLm, ttsIn, ttsOut, ttsCache, addType(ae, 0))
                negCond = lmStep(ttsLm, ttsIn, ttsOut, negCache, addType(ae, 0))
                if (eosProb(cond) > 0.5f || latents.size >= maxSpeechTokens ||
                    ttsCache.pos >= TTS_PMAX - 1
                ) { finished = true
                break }
            }
        }
        return Result(decode(latents), latents.size, (System.nanoTime() - t0) / 1_000_000)
    }

    /** Decode accumulated 64-d latents to a 24 kHz waveform (fixed-frame graph, tail discarded). */
    private fun decode(latents: List<FloatArray>): FloatArray {
        val n = minOf(latents.size, DECODE_FRAMES)
        val x = FloatArray(VAE * DECODE_FRAMES)                          // [64, DECODE_FRAMES]
        for (f in 0 until n) {
            for (c in 0 until VAE) {
                x[c * DECODE_FRAMES + f] = latents[f][c]
            }
        }
        decIn[0].writeFloat(x)
        decoder.run(decIn, decOut)
        val wav = decOut[0].readFloat()
        val len = n * HOP
        return FloatArray(len) { wav[it].coerceIn(-1f, 1f) }
    }

    override fun close() {
        listOf(baseIn, baseOut, ttsIn, ttsOut, headIn, headOut, decIn, decOut).forEach { l ->
            l.forEach { it.close() }
        }
        baseLm.close()
        ttsLm.close()
        head.close()
        decoder.close()
        embChannel.close()
    }

    private fun readF32(f: File): FloatArray {
        val b = f.readBytes()
        val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(b.size / 4) { bb.float }
    }
}
