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

package com.google.ai.edge.examples.flux2_klein

import android.util.Half
import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Turns a typed prompt into the two tensors the graphs cannot produce themselves.
 *
 * `gen_prep_klein.py` bakes exactly two prompt-dependent tensors into its `.bin` files —
 * `inputs_embeds` and `enc_mask` — so reproducing those two on device is all it takes to make the
 * prompt editable. Everything else the script stages (both rotary tables, the timestep embeddings,
 * the scheduler deltas, the initial latents, the tail permutations) depends only on positions, the
 * schedule, or the seed.
 *
 * The embedding lookup is a host-side gather over a memory-mapped fp16 table rather than a graph:
 * `GATHER` is not a GPU op on this delegate, the table is 778 MB, and the gathered row is the
 * graph's input anyway.
 */
class PromptEncoder(assetsDir: File) : Closeable {

    private val tokenizer = QwenTokenizer(assetsDir)
    private val embeddingFile = RandomAccessFile(File(assetsDir, EMBEDDING_FILE), "r")
    private val embeddings = embeddingFile.channel
        .map(FileChannel.MapMode.READ_ONLY, 0, embeddingFile.length())
        .order(ByteOrder.LITTLE_ENDIAN)
        .asShortBuffer()

    /** Token ids for [prompt], padded to [TEXT_TOKENS] and reported with the real length. */
    fun tokenize(prompt: String): Tokens {
        val ids = tokenizer.encodePrompt(prompt, TEXT_TOKENS)
        val padded = IntArray(TEXT_TOKENS) { if (it < ids.size) ids[it] else PAD_TOKEN_ID }
        return Tokens(padded, ids.size)
    }

    /** Gathers the fp16 embedding rows for [tokens] into a `[1, 512, 2560]` fp32 tensor. */
    fun embed(tokens: Tokens): FloatArray {
        val out = FloatArray(TEXT_TOKENS * HIDDEN_SIZE)
        for (position in 0 until TEXT_TOKENS) {
            val base = tokens.ids[position].toLong() * HIDDEN_SIZE
            for (channel in 0 until HIDDEN_SIZE) {
                out[position * HIDDEN_SIZE + channel] =
                    Half.toFloat(embeddings.get((base + channel).toInt()))
            }
        }
        return out
    }

    /**
     * Builds the encoder's additive attention mask, materialized across the head axis.
     *
     * `mask[q, k] = (k > q ? NEG : 0) + (k < realTokens ? 0 : NEG)` — causal, plus a padded-key
     * block. It is `[1, 32, 512, 512]` rather than `[1, 1, 512, 512]` because ML Drift silently
     * miscomputes a broadcast `ADD` whose left operand is a `BATCH_MATMUL` result, which is exactly
     * `softmax(q @ kᵀ + mask)`. Materializing the head axis removes the broadcast.
     */
    fun attentionMask(tokens: Tokens): FloatArray {
        val plane = FloatArray(TEXT_TOKENS * TEXT_TOKENS)
        for (query in 0 until TEXT_TOKENS) {
            for (key in 0 until TEXT_TOKENS) {
                var value = 0f
                if (key > query) {
                    value += NEGATIVE_INFINITY
                }
                if (key >= tokens.realCount) {
                    value += NEGATIVE_INFINITY
                }
                plane[query * TEXT_TOKENS + key] = value
            }
        }
        val out = FloatArray(ENCODER_HEADS * plane.size)
        for (head in 0 until ENCODER_HEADS) {
            System.arraycopy(plane, 0, out, head * plane.size, plane.size)
        }
        return out
    }

    override fun close() {
        embeddingFile.close()
    }

    /** The padded ids and how many of them are real. */
    data class Tokens(val ids: IntArray, val realCount: Int) {
        override fun equals(other: Any?) =
            other is Tokens && realCount == other.realCount && ids.contentEquals(other.ids)

        override fun hashCode() = 31 * ids.contentHashCode() + realCount
    }

    companion object {
        /** True when the tokenizer and the embedding table have been staged. */
        fun isStaged(assetsDir: File) =
            File(assetsDir, EMBEDDING_FILE).exists() && File(assetsDir, "qwen_vocab.txt").exists()

        private const val EMBEDDING_FILE = "qwen_embed_fp16.bin"
        private const val TEXT_TOKENS = 512
        private const val HIDDEN_SIZE = 2560
        private const val ENCODER_HEADS = 32
        private const val PAD_TOKEN_ID = 151643
        private const val NEGATIVE_INFINITY = -1e9f
    }
}
