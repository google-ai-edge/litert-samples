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

import java.io.File

/**
 * Byte-level BPE tokenizer for the Qwen3-4B text encoder, and the chat template around it.
 *
 * A faithful port of `Qwen2Tokenizer`. Three stages, in order:
 *  1. **Pre-tokenize** with the model's own regex, so `" red"` stays one piece and `"1234"` splits
 *     into digits. Getting this wrong changes the token ids without any error.
 *  2. **Byte-level encode**: every UTF-8 byte maps to one printable code point, so a space becomes
 *     `Ġ` and a newline `Ċ`. That is why no vocabulary entry contains whitespace.
 *  3. **Merge** adjacent pairs by lowest rank until none is left, then look the pieces up.
 *
 * The chat template's wrapper tokens are *added tokens* with fixed ids, so only the prompt body
 * needs BPE:
 * ```
 * <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
 * ```
 *
 * Verified against the Python tokenizer with `scripts/export_tokenizer_klein.py`'s fixture; see
 * [tokenizeFixture].
 */
class QwenTokenizer(assetsDir: File) {

    private val tokenToId = HashMap<String, Int>(VOCAB_HINT)
    private val mergeRank = HashMap<Long, Int>(MERGE_HINT)
    private val byteToUnicode: IntArray = buildByteEncoder()

    init {
        var nextId = 0
        File(assetsDir, "qwen_vocab.txt").forEachLine { line ->
            tokenToId[line] = nextId++
        }
        var rank = 0
        File(assetsDir, "qwen_merges.txt").forEachLine { line ->
            val space = line.indexOf(' ')
            val left = tokenToId[line.substring(0, space)]
            val right = tokenToId[line.substring(space + 1)]
            if (left != null && right != null) {
                mergeRank[pairKey(left, right)] = rank
            }
            rank++
        }
    }

    /**
     * Tokenizes [prompt] and wraps it in the chat template.
     *
     * @param prompt free-form user text.
     * @param maxTokens the encoder's fixed sequence length.
     * @return the token ids, at most [maxTokens] long, body-truncated if needed.
     */
    fun encodePrompt(prompt: String, maxTokens: Int): IntArray {
        val budget = maxTokens - TEMPLATE_PREFIX.size - TEMPLATE_SUFFIX.size
        require(budget > 0) { "maxTokens too small for the chat template" }
        val body = encode(prompt).take(budget)
        return (TEMPLATE_PREFIX.toList() + body + TEMPLATE_SUFFIX.toList()).toIntArray()
    }

    /** Tokenizes raw text, with no template and no special tokens. */
    fun encode(text: String): List<Int> {
        val ids = ArrayList<Int>(text.length / 2 + 1)
        for (piece in PRE_TOKENIZER.findAll(text)) {
            val symbols = toByteLevel(piece.value).map { tokenToId.getValue(it) }
            ids.addAll(merge(symbols))
        }
        return ids
    }

    /** Splits a piece into one vocabulary symbol per UTF-8 byte. */
    private fun toByteLevel(piece: String): List<String> =
        piece.toByteArray(Charsets.UTF_8).map { byte ->
            String(Character.toChars(byteToUnicode[byte.toInt() and BYTE_MASK]))
        }

    /**
     * Repeatedly merges the lowest-ranked adjacent pair.
     *
     * Quadratic in the piece length, which the pre-tokenizer keeps small; a heap would buy nothing
     * for a 512-token prompt.
     */
    private fun merge(initial: List<Int>): List<Int> {
        if (initial.size < 2) {
            return initial
        }
        val symbols = ArrayList(initial)
        while (symbols.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestIndex = -1
            for (i in 0 until symbols.size - 1) {
                val rank = mergeRank[pairKey(symbols[i], symbols[i + 1])] ?: continue
                if (rank < bestRank) {
                    bestRank = rank
                    bestIndex = i
                }
            }
            if (bestIndex < 0) {
                break
            }
            val merged = tokenToId.getValue(
                idToToken(symbols[bestIndex]) + idToToken(symbols[bestIndex + 1]))
            symbols[bestIndex] = merged
            symbols.removeAt(bestIndex + 1)
        }
        return symbols
    }

    private val tokens: Array<String> by lazy {
        val out = arrayOfNulls<String>(tokenToId.size)
        tokenToId.forEach { (token, id) -> out[id] = token }
        @Suppress("UNCHECKED_CAST")
        out as Array<String>
    }

    private fun idToToken(id: Int) = tokens[id]

    private companion object {
        /** `<|im_start|>user\n` */
        val TEMPLATE_PREFIX = intArrayOf(151644, 872, 198)

        /** `<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n` */
        val TEMPLATE_SUFFIX = intArrayOf(151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271)

        /** Qwen2's own pre-tokenizer pattern, copied from `tokenizer.json`. */
        val PRE_TOKENIZER = Regex(
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)" +
                "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+" +
                "|\\p{N}" +
                "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*" +
                "|\\s*[\\r\\n]+" +
                "|\\s+(?!\\S)" +
                "|\\s+")

        const val VOCAB_HINT = 1 shl 18
        const val MERGE_HINT = 1 shl 18
        const val BYTE_MASK = 0xff

        fun pairKey(left: Int, right: Int): Long = (left.toLong() shl 32) or right.toLong()

        /**
         * GPT-2's `bytes_to_unicode`: the 188 printable bytes map to themselves, the rest to an
         * unused code-point block starting at 256.
         */
        fun buildByteEncoder(): IntArray {
            val printable = ('!'.code..'~'.code) + ('¡'.code..'¬'.code) +
                ('®'.code..'ÿ'.code)
            val encoder = IntArray(256)
            var next = 0
            for (byte in 0 until 256) {
                encoder[byte] = if (byte in printable) byte else 256 + next++
            }
            return encoder
        }
    }
}
