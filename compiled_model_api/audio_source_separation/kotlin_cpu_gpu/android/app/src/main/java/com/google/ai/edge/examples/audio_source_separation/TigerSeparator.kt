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

package com.google.ai.edge.examples.audio_source_separation

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.Closeable
import java.io.File

/**
 * TIGER-DnR cinematic sound separation on the LiteRT CompiledModel GPU — fully GPU.
 *
 * Three sibling TIGER graphs (dialog / effect / music, ~1.4 M params each, fp16 ~16 MB) take a
 * 12.06 s 44.1 kHz mono chunk (reflect-padded on the host) and emit the separated complex
 * spectrograms [1, 3, 1025, 1040] (real, imag). STFT runs inside the graph (DFT-as-conv);
 * iSTFT + overlap-add run on the host. Per DnR convention each graph contributes one stem:
 * dialog=source 2, effect=source 1, music=source 0.
 */
class TigerSeparator(private val ctx: Context) : Closeable {

    companion object {
        const val SR = 44100
        const val FRAMES = 1040
        const val CHUNK = (FRAMES - 1) * Istft.HOP      // 531968 samples = 12.06 s
        const val HOP_CHUNK = SR * 10                    // 10 s chunk hop -> ~2 s crossfade
        const val PAD = Istft.WIN / 2                    // 1024 reflect pad each side
        val STEMS = listOf("dialog", "effect", "music")
        private val SRC_INDEX = mapOf("dialog" to 2, "effect" to 1, "music" to 0)
    }

    private fun modelFile(stem: String): File {
        val f = File(ctx.filesDir, "tiger_${stem}_fp16.tflite")
        check(f.exists()) { "Model not found: ${f.name}. Run scripts/install_to_device.sh first." }
        return f
    }

    init {
        STEMS.forEach { modelFile(it) }   // fail fast if models are not installed yet
    }

    /**
     * Separate a mono 44.1 kHz track into 3 stems. Models are loaded one at a time (each sweeps
     * all chunks) to keep peak GPU memory at a single graph.
     * @return stems in STEMS order, each the same length as pcm
     */
    fun separate(pcm: FloatArray, onProgress: (String, Int, Int) -> Unit): List<FloatArray> {
        val nChunks =
            if (pcm.size <= CHUNK) 1
            else 1 + ((pcm.size - CHUNK) + HOP_CHUNK - 1) / HOP_CHUNK
        val stems = ArrayList<FloatArray>(3)
        for (stem in STEMS) {
            val out = FloatArray(pcm.size)
            val weight = FloatArray(pcm.size)
            val model = CompiledModel.create(
                modelFile(stem).absolutePath, CompiledModel.Options(Accelerator.GPU), null)
            val inBuf = model.createInputBuffers()
            val outBuf = model.createOutputBuffers()
            try {
                for (c in 0 until nChunks) {
                    onProgress(stem, c + 1, nChunks)
                    val start = c * HOP_CHUNK
                    val seg = separateChunk(model, inBuf, outBuf, pcm, start, SRC_INDEX[stem]!!)
                    // equal-weight overlap-add (reference wav_chunk_inference averages overlaps)
                    val n = minOf(CHUNK, pcm.size - start)
                    for (i in 0 until n) {
                        out[start + i] += seg[i]
                        weight[start + i] += 1f
                    }
                }
            } finally {
                inBuf.forEach { it.close() }
                outBuf.forEach { it.close() }
                model.close()
            }
            for (i in out.indices) {
                if (weight[i] > 1f) {
                    out[i] /= weight[i]
                }
            }
            stems.add(out)
        }
        return stems
    }

    private fun separateChunk(
        model: CompiledModel,
        inBuf: List<TensorBuffer>,
        outBuf: List<TensorBuffer>,
        pcm: FloatArray, start: Int, srcIdx: Int,
    ): FloatArray {
        // window + reflect pad (torch.stft center=True equivalent)
        val x = FloatArray(CHUNK + 2 * PAD)
        for (i in 0 until CHUNK) {
            val p = start + i
            x[PAD + i] = if (p < pcm.size) pcm[p] else 0f
        }
        for (j in 1..PAD) {
            x[PAD - j] = x[PAD + j]                                  // left reflect
            x[PAD + CHUNK - 1 + j] = x[PAD + CHUNK - 1 - j]          // right reflect
        }
        inBuf[0].writeFloat(x)
        model.run(inBuf, outBuf)
        val real = outBuf[0].readFloat()   // [3 * 1025 * 1040]
        val imag = outBuf[1].readFloat()
        val plane = Istft.ENC * FRAMES
        val re = real.copyOfRange(srcIdx * plane, (srcIdx + 1) * plane)
        val im = imag.copyOfRange(srcIdx * plane, (srcIdx + 1) * plane)
        return Istft.run(re, im, FRAMES, CHUNK)
    }

    override fun close() {}
}
