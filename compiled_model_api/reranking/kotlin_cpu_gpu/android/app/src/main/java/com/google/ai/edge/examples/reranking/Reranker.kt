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

package com.google.ai.edge.examples.reranking

import android.content.Context
import android.util.Half
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp

/**
 * On-device RAG reranker with Qwen3-Reranker-0.6B, fully on LiteRT CompiledModel GPU.
 *
 *   prompt = PREFIX + "<Instruct>:… <Query>:… <Document>:…" + SUFFIX   (Qwen3-Reranker template)
 *   host embed lookup -> inputs_embeds [1,256,1024]           (GATHER is GPU-banned)
 *   CompiledModel GPU -> logits [1,256,2] (baked "no"/"yes" head, tied embedding rows)
 *   softmax over [no,yes] at the last real token -> P(yes) = relevance score
 *
 * Two large assets are staged into filesDir by scripts/install_to_device.sh:
 *   qwen3rerank_gpu_fp16.tflite  — 882 MB transformer graph (+ 2-logit head)
 *   embeddings_fp16.bin          — [vocab,1024] fp16 token embedding table (tied weights)
 */
class Reranker(private val ctx: Context) {

    companion object {
        const val L = 256
        const val DIM = 1024
        const val PAD = 151643        // <|endoftext|>
        const val DEFAULT_INSTR =
            "Given a web search query, retrieve relevant passages that answer the query"
        // Qwen3-Reranker template, pre-tokenized (contains special tokens absent from vocab.json)
        private val PREFIX_IDS = intArrayOf(
            151644, 8948, 198, 60256, 3425, 279, 11789, 20027, 279, 8502, 3118, 389, 279, 11361,
            323, 279, 758, 1235, 3897, 13, 7036, 429, 279, 4226, 646, 1172, 387, 330, 9693, 1,
            476, 330, 2152, 3263, 151645, 198, 151644, 872, 198)
        private val SUFFIX_IDS = intArrayOf(151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271)
        private const val MODEL = "qwen3rerank_gpu_fp16.tflite"
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

    private val embChannel = RandomAccessFile(f(EMB), "r").channel
    private val embMap = embChannel.map(FileChannel.MapMode.READ_ONLY, 0, embChannel.size())
        .order(ByteOrder.LITTLE_ENDIAN)

    private fun lookup(ids: IntArray): FloatArray {
        val out = FloatArray(L * DIM)
        for (p in 0 until L) {
            var b = ids[p].toLong().toInt() * DIM * 2      // fp16 => 2 bytes
            var o = p * DIM
            for (j in 0 until DIM) { out[o++] = Half.toFloat(embMap.getShort(b)); b += 2 }
        }
        return out
    }

    /** Relevance score P(yes) that `doc` answers `query`, in [0,1]. */
    fun score(query: String, doc: String, instruction: String = DEFAULT_INSTR): Float {
        val cids = tokenizer.encode("<Instruct>: $instruction\n<Query>: $query\n<Document>: $doc")
        val ids = IntArray(L) { PAD }
        val room = L - PREFIX_IDS.size - SUFFIX_IDS.size       // budget for the content
        var k = 0
        for (i in PREFIX_IDS) ids[k++] = i
        for (i in 0 until minOf(cids.size, room)) ids[k++] = cids[i]   // truncate long docs
        for (i in SUFFIX_IDS) ids[k++] = i
        val poolPos = k - 1                                    // last real token (end of suffix)

        inputs[0].writeFloat(lookup(ids))
        model.run(inputs, outputs)
        val out = outputs[0].readFloat()                       // [L*2] = [no,yes] per position
        val no = out[poolPos * 2]; val yes = out[poolPos * 2 + 1]
        val m = maxOf(no, yes)
        val en = exp((no - m).toDouble()); val ey = exp((yes - m).toDouble())
        return (ey / (en + ey)).toFloat()
    }

    fun close() {
        inputs.forEach { it.close() }; outputs.forEach { it.close() }
        model.close(); embChannel.close()
    }
}
