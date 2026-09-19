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

package com.google.ai.edge.examples.text_prompted_segmentation

import android.content.Context
import org.json.JSONObject
import java.io.File

/**
 * CLIP byte-level BPE tokenizer (vocab.json + merges.txt), enough for prompt encoding:
 * lowercase, whitespace-split with basic punctuation handling, byte-to-unicode mapping,
 * BPE merges, "</w>" word endings. Matches HF CLIPTokenizer for typical prompts.
 */
class BpeTokenizer(ctx: Context) {

    companion object {
        const val BOS = 49406
        const val EOT = 49407
        const val MAX_LEN = 77
    }

    private val encoder = HashMap<String, Int>()
    private val ranks = HashMap<Pair<String, String>, Int>()
    private val byteToUnicode = HashMap<Int, Char>()

    init {
        val vocab = JSONObject(File(ctx.filesDir, "vocab.json").readText())
        for (k in vocab.keys()) {
            encoder[k] = vocab.getInt(k)
        }
        File(ctx.filesDir, "merges.txt").readLines().drop(1).forEachIndexed { i, line ->
            val p = line.trim().split(" ")
            if (p.size == 2) ranks[Pair(p[0], p[1])] = i
        }
        // GPT-2 byte-to-unicode table
        val bs = ArrayList<Int>()
        for (c in '!'.code..'~'.code) {
            bs.add(c)
        }
        for (c in 0xA1..0xAC) {
            bs.add(c)
        }
        for (c in 0xAE..0xFF) {
            bs.add(c)
        }
        val cs = ArrayList(bs)
        var n = 0
        for (b in 0..255) {
            if (b !in bs) {
                bs.add(b)
                cs.add(256 + n)
                n++
            }
        }
        for (i in bs.indices) {
            byteToUnicode[bs[i]] = cs[i].toChar()
        }
    }

    private fun bpe(tokenIn: String): List<String> {
        var word = tokenIn.dropLast(4).map { it.toString() }.toMutableList()
        if (word.isEmpty()) return listOf(tokenIn)
        word[word.size - 1] = word.last() + "</w>"
        while (word.size > 1) {
            var best: Pair<String, String>? = null
            var bestRank = Int.MAX_VALUE
            for (i in 0 until word.size - 1) {
                val r = ranks[Pair(word[i], word[i + 1])] ?: continue
                if (r < bestRank) {
                    bestRank = r
                    best = Pair(word[i], word[i + 1])
                }
            }
            val b = best ?: break
            val merged = ArrayList<String>(word.size)
            var i = 0
            while (i < word.size) {
                if (i < word.size - 1 && word[i] == b.first && word[i + 1] == b.second) {
                    merged.add(b.first + b.second)
                    i += 2
                } else { merged.add(word[i])
                i++ }
            }
            word = merged
        }
        return word
    }

    /** prompt -> 77 ids: [BOS, tokens..., EOT, EOT-pad...]; returns (ids, eotPos). */
    fun encode(text: String): Pair<IntArray, Int> {
        val clean = text.lowercase().trim().replace(Regex("\\s+"), " ")
        val pieces = Regex("[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+").findAll(clean).map { it.value }
        val ids = ArrayList<Int>()
        for (w in pieces) {
            val mapped = w.toByteArray(Charsets.UTF_8).map { byteToUnicode[it.toInt() and 0xFF]!! }
                .joinToString("") + "</w>"
            for (t in bpe(mapped)) encoder[t]?.let { ids.add(it) } ?: run {
                for (ch in t) encoder[ch.toString()]?.let { ids.add(it) }
            }
        }
        val out = IntArray(MAX_LEN) { EOT }
        out[0] = BOS
        val n = minOf(ids.size, MAX_LEN - 2)
        for (i in 0 until n) {
            out[i + 1] = ids[i]
        }
        return Pair(out, n + 1)   // eotPos = index of the first EOT
    }
}
