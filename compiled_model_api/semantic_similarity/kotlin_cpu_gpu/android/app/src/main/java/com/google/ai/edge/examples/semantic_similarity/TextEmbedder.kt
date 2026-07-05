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

package com.google.ai.edge.examples.semantic_similarity

import android.content.Context
import android.util.Half
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.sqrt

/**
 * On-device text embedding with Qwen3-Embedding-0.6B, fully on LiteRT CompiledModel GPU.
 *
 *   tokenize (BpeTokenizer) -> ids
 *   host embed lookup        -> inputs_embeds [1,128,1024]      (GATHER is GPU-banned)
 *   CompiledModel GPU        -> hidden_states [1,128,1024]
 *   pool last real token + L2-normalize (+ optional Matryoshka) -> embedding
 *
 * Two large assets are staged into filesDir by scripts/install_to_device.sh:
 *   qwen3emb_gpu_fp16.tflite  — 881 MB transformer graph
 *   embeddings_fp16.bin       — [vocab,1024] fp16 token embedding table (tied weights)
 */
class TextEmbedder(private val ctx: Context) {

    companion object {
        const val L = 128            // fixed sequence length (graph input)
        const val DIM = 1024         // embedding dimension
        private const val MODEL = "qwen3emb_gpu_fp16.tflite"
        private const val EMB = "embeddings_fp16.bin"
    }

    private fun f(n: String) = File(ctx.filesDir, n).also {
        check(it.exists()) { "Missing ${it.name}. Run scripts/install_to_device.sh first." }
    }

    private val tokenizer = BpeTokenizer(ctx)

    private val model = CompiledModel.create(
        f(MODEL).absolutePath, CompiledModel.Options(Accelerator.GPU), null)
    private val inputs = model.createInputBuffers()
    private val outputs = model.createOutputBuffers()

    // memory-map the fp16 embedding table (310 MB) for host-side lookup
    private val embChannel = RandomAccessFile(f(EMB), "r").channel
    private val embMap = embChannel.map(FileChannel.MapMode.READ_ONLY, 0, embChannel.size())
        .order(ByteOrder.LITTLE_ENDIAN)

    /** Build the [L,DIM] inputs_embeds for a token sequence via host lookup. */
    private fun lookup(ids: IntArray): FloatArray {
        val out = FloatArray(L * DIM)
        for (p in 0 until L) {
            val base = ids[p].toLong() * DIM * 2L        // fp16 => 2 bytes
            var o = p * DIM
            var b = base.toInt()
            for (j in 0 until DIM) { out[o++] = Half.toFloat(embMap.getShort(b)); b += 2 }
        }
        return out
    }

    /** Embed one text -> L2-normalized vector (Matryoshka: keep first `dim` dims, renormalize). */
    fun embed(text: String, dim: Int = DIM): FloatArray {
        val enc = tokenizer.encode(text)
        val ids = IntArray(L) { BpeTokenizer.EOS_POOL }   // pad with <|endoftext|>
        val poolPos: Int
        if (enc.size >= L) {
            for (i in 0 until L - 1) ids[i] = enc[i]
            ids[L - 1] = BpeTokenizer.EOS_POOL             // keep pooled EOS at the tail
            poolPos = L - 1
        } else {
            for (i in enc.indices) ids[i] = enc[i]
            poolPos = enc.size - 1
        }

        inputs[0].writeFloat(lookup(ids))
        model.run(inputs, outputs)
        val hidden = outputs[0].readFloat()               // [L*DIM]

        val n = if (dim in 1 until DIM) dim else DIM
        val vec = FloatArray(n)
        System.arraycopy(hidden, poolPos * DIM, vec, 0, n)
        var s = 0f; for (v in vec) s += v * v
        val inv = 1f / (sqrt(s) + 1e-9f)
        for (i in vec.indices) vec[i] *= inv
        return vec
    }

    /** Instruction-aware query embedding (Qwen3-Embedding recommends an instruction prefix). */
    fun embedQuery(query: String, dim: Int = DIM): FloatArray = embed(
        "Instruct: Given a web search query, retrieve relevant passages\nQuery:$query", dim)

    fun close() {
        inputs.forEach { it.close() }; outputs.forEach { it.close() }
        model.close(); embChannel.close()
    }
}

fun cosine(a: FloatArray, b: FloatArray): Float {
    var d = 0f; val n = minOf(a.size, b.size)
    for (i in 0 until n) d += a[i] * b[i]
    return d      // inputs are already L2-normalized
}
