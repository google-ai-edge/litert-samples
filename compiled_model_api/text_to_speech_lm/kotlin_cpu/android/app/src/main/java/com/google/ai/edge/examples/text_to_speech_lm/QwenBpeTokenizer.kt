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

package com.google.ai.edge.examples.text_to_speech_lm

import java.io.File
import java.text.Normalizer
import org.json.JSONObject

/**
 * Byte-level BPE tokenizer for the Qwen2 vocabulary (vocab.json + merges.txt).
 *
 * Implements the GPT-2 style pipeline: pre-tokenize with the Qwen2 regex,
 * map UTF-8 bytes to the byte-level unicode alphabet, then greedily apply
 * ranked merges. Only plain text is encoded here — the chat-template special
 * tokens around the TTS prompt have fixed ids and are added by the caller
 * (see [Qwen3TtsEngine]).
 */
class QwenBpeTokenizer(vocabFile: File, mergesFile: File) {

    private val vocab = HashMap<String, Int>(160_000)
    private val ranks = HashMap<Long, Int>(160_000)
    private val pieceIds = HashMap<String, Int>() // merge pieces -> id cache
    private val byteToChar = CharArray(256)

    // Qwen2 pre-tokenization regex (from tokenizer.json). The reference
    // tokenizer is Rust with Unicode-aware \s; Android's ICU-backed regex is
    // Unicode-aware by default (and rejects the desktop-JVM
    // UNICODE_CHARACTER_CLASS flag). Verified against reference vectors by
    // the startup self-test, including U+3000 whitespace.
    private val pretokenize = Regex(
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|" +
            " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")

    init {
        // GPT-2 byte-to-unicode table: printable bytes map to themselves,
        // the rest to U+0100.. in order.
        val direct = (('!'.code..'~'.code) + ('¡'.code..'¬'.code) +
            ('®'.code..'ÿ'.code)).toHashSet()
        var next = 256
        for (b in 0 until 256) {
            byteToChar[b] = if (b in direct) b.toChar() else (next++).toChar()
        }

        val json = JSONObject(vocabFile.readText())
        for (key in json.keys()) {
            vocab[key] = json.getInt(key)
        }

        var rank = 0
        mergesFile.forEachLine { line ->
            if (line.isNotBlank() && !line.startsWith("#version")) {
                val sp = line.indexOf(' ')
                val left = line.substring(0, sp)
                val right = line.substring(sp + 1)
                ranks[pairKey(pieceId(left), pieceId(right))] = rank
                rank++
            }
        }
    }

    private fun pieceId(piece: String): Int =
        pieceIds.getOrPut(piece) {
            vocab[piece] ?: throw IllegalStateException("piece not in vocab: $piece")
        }

    private fun pairKey(a: Int, b: Int): Long = (a.toLong() shl 32) or b.toLong()

    /** Encodes plain text (no special tokens) to token ids. */
    fun encode(text: String): IntArray {
        // The reference tokenizer applies NFC normalization first.
        val normalized = Normalizer.normalize(text, Normalizer.Form.NFC)
        val out = ArrayList<Int>(normalized.length / 3 + 8)
        for (match in pretokenize.findAll(normalized)) {
            val bytes = match.value.toByteArray(Charsets.UTF_8)
            val mapped = StringBuilder(bytes.size)
            for (b in bytes) {
                mapped.append(byteToChar[b.toInt() and 0xFF])
            }
            bpe(mapped.toString(), out)
        }
        return out.toIntArray()
    }

    private fun bpe(token: String, out: ArrayList<Int>) {
        // Word as a list of piece strings, merged greedily by rank.
        var pieces = token.map { it.toString() }
        if (pieces.size == 1) {
            out.add(vocab[token] ?: error("byte piece missing: $token"))
            return
        }
        while (pieces.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestIdx = -1
            for (i in 0 until pieces.size - 1) {
                val a = vocab[pieces[i]] ?: continue
                val b = vocab[pieces[i + 1]] ?: continue
                val r = ranks[pairKey(a, b)] ?: continue
                if (r < bestRank) {
                    bestRank = r
                    bestIdx = i
                }
            }
            if (bestIdx < 0) break
            val merged = pieces[bestIdx] + pieces[bestIdx + 1]
            pieces = pieces.subList(0, bestIdx) + listOf(merged) +
                pieces.subList(bestIdx + 2, pieces.size)
        }
        for (p in pieces) {
            out.add(vocab[p] ?: error("piece not in vocab: $p"))
        }
    }
}
