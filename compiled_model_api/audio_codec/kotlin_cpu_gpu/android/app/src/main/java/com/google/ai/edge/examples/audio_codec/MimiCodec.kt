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

package com.google.ai.edge.examples.audio_codec

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * Mimi (Kyutai 2024 streaming neural codec, 24 kHz), on-device. HYBRID placement: the heavy SEANet
 * convolutional halves run on the LiteRT CompiledModel GPU (ML Drift); the two 8-layer Transformers
 * run on CPU (Accelerator.CPU), and the split RVQ runs on CPU in [MimiRvq].
 *
 *   audio[L] -[GPU enc_conv]-> feat[1,512,Se] -[CPU enc_tx: enc-transformer+downsample]-> emb[1,512,Tc]
 *            -[CPU RVQ.encode]-> codes[32,Tc] -[CPU RVQ.decode]-> emb[1,512,Tc]
 *            -[CPU dec_tx: upsample+dec-transformer]-> conv_in[1,512,seq] -[GPU deconly]-> audio[L]
 *
 * Why the transformers are on CPU: the decoder transformer's residual stream reaches |x|=27, where
 * the Mali GPU delegate's internal fp16 compute loses precision (device audio ~12 dB SNR on speech),
 * while the SEANet convs are fp16-exact on GPU (~48 dB). The transformer behaves identically
 * standalone and fused on device, so this is fp16 PRECISION, not a fusion collapse. The transformers
 * are tiny (8 layers x 512, seq ~50), so CPU is trivial.
 *
 * All graphs + mimi_rvq.bin are loaded from filesDir (push via install_to_device.sh).
 * Fixed length: SECS=2 s -> SAMPLES=48000, Se=50, Tc=25, seq=50.
 */
class MimiCodec(private val context: Context) : Closeable {

    companion object {
        const val SAMPLES = 48000   // 2 s @ 24 kHz
        const val HID = 512
        const val SE = 50           // encoder SEANet output frames (25 Hz)
        const val TC = 25           // code frames (12.5 Hz)
        const val SEQ = 50          // decoder transformer length (= 2 * TC)
        const val ENC_CONV = "mimi_enc_conv_fp16.tflite"   // GPU
        const val ENC_TX = "mimi_enc_tx_fp16.tflite"       // CPU
        const val DEC_TX = "mimi_dec_tx_fp16.tflite"       // CPU
        const val DECONLY = "mimi_deconly_fp16.tflite"     // GPU
        const val RVQ_BIN = "mimi_rvq.bin"
    }

    private fun load(name: String, acc: Accelerator): CompiledModel {
        val f = File(context.filesDir, name)
        check(f.exists()) { "Model not found: $name. Push first with install_to_device.sh" }
        return CompiledModel.create(f.absolutePath, CompiledModel.Options(acc), null)
    }

    private val encConv = load(ENC_CONV, Accelerator.GPU)
    private val encTx = load(ENC_TX, Accelerator.CPU)
    private val decTx = load(DEC_TX, Accelerator.CPU)
    private val deconly = load(DECONLY, Accelerator.GPU)
    private val rvq = MimiRvq(File(context.filesDir, RVQ_BIN).absolutePath)

    private val ecIn = encConv.createInputBuffers()
    private val ecOut = encConv.createOutputBuffers()
    private val etIn = encTx.createInputBuffers()
    private val etOut = encTx.createOutputBuffers()
    private val dtIn = decTx.createInputBuffers()
    private val dtOut = decTx.createOutputBuffers()
    private val doIn = deconly.createInputBuffers()
    private val doOut = deconly.createOutputBuffers()

    data class Result(val audio: FloatArray, val codes: IntArray, val encodeMs: Long, val decodeMs: Long)

    /** (1,512,Se) channel-major c*Se+t  ->  (1,Se,512) time-major t*512+c (enc_tx input layout). */
    private fun transposeCT(x: FloatArray, c: Int, t: Int): FloatArray {
        val o = FloatArray(x.size)
        for (cc in 0 until c) {
            for (tt in 0 until t) {
                o[tt * c + cc] = x[cc * t + tt]
            }
        }
        return o
    }

    /** Full encode -> quantize -> decode round-trip of a SAMPLES-length clip. */
    fun roundTrip(audio: FloatArray): Result {
        val t0 = System.nanoTime()
        ecIn[0].writeFloat(audio)
        encConv.run(ecIn, ecOut)
        val feat = ecOut[0].readFloat()                       // (1,512,Se) c-major
        etIn[0].writeFloat(transposeCT(feat, HID, SE))
        encTx.run(etIn, etOut)
        val emb = etOut[0].readFloat()                        // (1,512,Tc) c-major
        val codes = rvq.encode(emb, TC)
        val t1 = System.nanoTime()
        val embBack = rvq.decode(codes, TC)
        dtIn[0].writeFloat(embBack)
        decTx.run(dtIn, dtOut)
        val convIn = dtOut[0].readFloat()                     // (1,512,seq) c-major
        doIn[0].writeFloat(convIn)
        deconly.run(doIn, doOut)
        val out = doOut[0].readFloat()                        // (1,1,L)
        val t2 = System.nanoTime()
        return Result(out, codes, (t1 - t0) / 1_000_000, (t2 - t1) / 1_000_000)
    }

    override fun close() {
        listOf(ecIn, ecOut, etIn, etOut, dtIn, dtOut, doIn, doOut).forEach { bs -> bs.forEach { it.close() } }
        encConv.close()
        encTx.close()
        decTx.close()
        deconly.close()
    }
}
