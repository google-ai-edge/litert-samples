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

package com.google.ai.edge.examples.rwkv7

/**
 * Byte-level greedy longest-match tokenizer for the RWKV "World" vocabulary
 * (`rwkv_vocab_v20230424.txt`, 65529 entries; token 0 is the implicit end-of-text).
 *
 * This is a line-for-line port of the official RWKV `TRIE_TOKENIZER`: for the
 * current byte pair (s0, s1) it looks up the candidate token list bucketed by
 * those two bytes and takes the first candidate that prefixes the input, which
 * — given the vocabulary's ordering — is the longest match. Falls back to the
 * single byte s0 when no multi-byte candidate matches.
 *
 * Vocabulary lines have the form `<idx> <python-literal> <byte-length>` where
 * the literal is either a Python string (`'...'`/`"..."`, UTF-8 encoded here)
 * or a bytes literal (`b'...'`, raw bytes). Only the escape sequences that
 * actually occur in the file are supported; anything else fails loudly.
 */
class RwkvTokenizer(vocabLines: List<String>) {

    private companion object {
        /** Buckets are addressed by the first two bytes of a token, 256x256. */
        const val BYTE_RANGE = 256
    }

    private val idxToToken = HashMap<Int, ByteArray>(vocabLines.size * 2)
    private val tokenToIdx = HashMap<String, Int>(vocabLines.size * 2)

    /** Candidate tokens bucketed by their first two bytes, longest-first. */
    private val table = Array(BYTE_RANGE) { arrayOfNulls<MutableList<ByteArray>>(BYTE_RANGE) }

    /** goodPairs[s0][s1] = true when at least one multi-byte token starts with (s0, s1). */
    private val goodPairs = Array(BYTE_RANGE) { BooleanArray(BYTE_RANGE) }

    /** Longest token length starting with byte s0 — bounds the lookahead window. */
    private val maxLen = IntArray(BYTE_RANGE)

    init {
        val tokens = ArrayList<ByteArray>(vocabLines.size)
        for (line in vocabLines) {
            if (line.isEmpty()) {
                continue
            }
            val firstSpace = line.indexOf(' ')
            val lastSpace = line.lastIndexOf(' ')
            val idx = line.substring(0, firstSpace).toInt()
            val bytes = parsePythonLiteral(line.substring(firstSpace + 1, lastSpace))
            check(bytes.size == line.substring(lastSpace + 1).toInt()) {
                "vocab length mismatch at idx $idx"
            }
            tokens.add(bytes)
            idxToToken[idx] = bytes
            tokenToIdx[byteKey(bytes)] = idx
        }
        // Reverse file order so each bucket lists longer (higher-index) tokens first,
        // making "first prefix match" equal to "longest match" — as in the reference.
        for (i in tokens.indices.reversed()) {
            val token = tokens[i]
            if (token.size < 2) {
                continue
            }
            val s0 = token[0].toInt() and 0xFF
            val s1 = token[1].toInt() and 0xFF
            val bucket = table[s0][s1] ?: mutableListOf<ByteArray>().also { table[s0][s1] = it }
            bucket.add(token)
            goodPairs[s0][s1] = true
            if (token.size > maxLen[s0]) {
                maxLen[s0] = token.size
            }
        }
    }

    /** Encodes [text] to token ids (UTF-8 bytes, greedy longest match). */
    fun encode(text: String): IntArray {
        val src = text.toByteArray(Charsets.UTF_8)
        val ids = ArrayList<Int>()
        var i = 0
        while (i < src.size) {
            var match: ByteArray? = null
            if (i < src.size - 1) {
                val s0 = src[i].toInt() and 0xFF
                val s1 = src[i + 1].toInt() and 0xFF
                if (goodPairs[s0][s1]) {
                    val window = src.copyOfRange(i, minOf(i + maxLen[s0], src.size))
                    match = table[s0][s1]?.firstOrNull { isPrefix(it, window) }
                }
            }
            val token = match ?: byteArrayOf(src[i])
            ids.add(checkNotNull(tokenToIdx[byteKey(token)]) { "byte not in vocab" })
            i += token.size
        }
        return ids.toIntArray()
    }

    /** Returns the raw bytes of one token — callers accumulate and UTF-8 decode. */
    fun tokenBytes(id: Int): ByteArray = idxToToken[id] ?: ByteArray(0)

    /** Decodes token ids to text (invalid UTF-8 becomes replacement chars). */
    fun decode(ids: IntArray): String {
        var total = 0
        for (id in ids) {
            total += tokenBytes(id).size
        }
        val bytes = ByteArray(total)
        var offset = 0
        for (id in ids) {
            val tok = tokenBytes(id)
            tok.copyInto(bytes, offset)
            offset += tok.size
        }
        return String(bytes, Charsets.UTF_8)
    }

    /** True when [prefix] is a prefix of [data]. */
    private fun isPrefix(prefix: ByteArray, data: ByteArray): Boolean {
        if (prefix.size > data.size) {
            return false
        }
        for (j in prefix.indices) {
            if (prefix[j] != data[j]) {
                return false
            }
        }
        return true
    }

    /** ISO-8859-1 maps bytes 1:1 to chars, giving a hashable key for a ByteArray. */
    private fun byteKey(bytes: ByteArray): String = String(bytes, Charsets.ISO_8859_1)

    /**
     * Parses the Python literal used in the vocab file: `'...'`, `"..."` (string,
     * UTF-8 encoded) or `b'...'` (raw bytes). Supported escapes: `\\ \' \" \n \r \t
     * \xNN` and, for strings only, `\uNNNN`.
     */
    private fun parsePythonLiteral(literal: String): ByteArray {
        val isBytes = literal.startsWith("b")
        val body = literal.substring(if (isBytes) 2 else 1, literal.length - 1)
        val chars = StringBuilder()
        val rawBytes = ArrayList<Byte>()
        var i = 0
        while (i < body.length) {
            val c = body[i]
            if (c != '\\') {
                if (isBytes) {
                    rawBytes.add(c.code.toByte())
                } else {
                    chars.append(c)
                }
                i++
                continue
            }
            i++
            when (val esc = body[i]) {
                '\\', '\'', '"' -> if (isBytes) rawBytes.add(esc.code.toByte()) else chars.append(esc)
                'n' -> if (isBytes) rawBytes.add('\n'.code.toByte()) else chars.append('\n')
                'r' -> if (isBytes) rawBytes.add('\r'.code.toByte()) else chars.append('\r')
                't' -> if (isBytes) rawBytes.add('\t'.code.toByte()) else chars.append('\t')
                'x' -> {
                    val value = body.substring(i + 1, i + 3).toInt(16)
                    if (isBytes) {
                        rawBytes.add(value.toByte())
                    } else {
                        chars.append(value.toChar())
                    }
                    i += 2
                }
                'u' -> {
                    check(!isBytes) { "\\u escape in bytes literal" }
                    chars.append(body.substring(i + 1, i + 5).toInt(16).toChar())
                    i += 4
                }
                else -> error("unsupported escape \\$esc in vocab literal")
            }
            i++
        }
        return if (isBytes) rawBytes.toByteArray() else chars.toString().toByteArray(Charsets.UTF_8)
    }
}
