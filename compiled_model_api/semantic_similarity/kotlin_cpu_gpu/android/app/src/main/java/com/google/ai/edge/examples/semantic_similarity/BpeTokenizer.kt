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
import org.json.JSONObject
import java.util.regex.Pattern

/**
 * Qwen3 byte-level BPE tokenizer (GPT-2 / cl100k style), ported for on-device use.
 * Matches `AutoTokenizer("Qwen/Qwen3-Embedding-0.6B")` exactly:
 *   1. Split on the Qwen pre-tokenizer regex (Isolated).
 *   2. Byte-level encode each piece (UTF-8 byte -> GPT-2 unicode char).
 *   3. BPE-merge by rank from merges.txt.
 *   4. Map pieces to ids via vocab.json.
 *   5. Append <|endoftext|> (151643) — the token Qwen3-Embedding pools for the sentence embedding.
 *
 * vocab.json + merges.txt are bundled in assets.
 */
class BpeTokenizer(ctx: Context) {

    companion object {
        const val EOS_POOL = 151643            // <|endoftext|>, appended and pooled
        // Qwen pre-tokenizer split pattern (from tokenizer.json)
        private const val PAT =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    }

    private val pattern: Pattern = Pattern.compile(PAT)
    private val byteToUnicode: CharArray = buildByteToUnicode()
    private val vocab = HashMap<String, Int>(160000)
    private val ranks = HashMap<String, Int>(160000)

    init {
        // vocab.json: token -> id
        val vj = JSONObject(ctx.assets.open("vocab.json").bufferedReader().use { it.readText() })
        val keys = vj.keys()
        while (keys.hasNext()) { val k = keys.next(); vocab[k] = vj.getInt(k) }
        // merges.txt: "a b" -> rank (skip the "#version" header line)
        ctx.assets.open("merges.txt").bufferedReader().useLines { lines ->
            var rank = 0
            for (line in lines) {
                if (line.isEmpty() || line.startsWith("#")) continue
                ranks[line] = rank++         // key kept as the raw "a b" merge line
            }
        }
    }

    /** GPT-2 reversible byte<->unicode map so every byte becomes a printable char. */
    private fun buildByteToUnicode(): CharArray {
        val bs = ArrayList<Int>()
        for (i in '!'.code..'~'.code) bs.add(i)
        for (i in '¡'.code..'¬'.code) bs.add(i)
        for (i in '®'.code..'ÿ'.code) bs.add(i)
        val cs = ArrayList<Int>(bs)
        var n = 0
        for (b in 0..255) if (b !in bs) { bs.add(b); cs.add(256 + n); n++ }
        val map = CharArray(256)
        for (i in bs.indices) map[bs[i]] = cs[i].toChar()
        return map
    }

    private fun bpe(word: String): List<String> {
        var symbols = ArrayList<String>(word.length)
        for (c in word) symbols.add(c.toString())
        if (symbols.size < 2) return symbols
        while (true) {
            // lowest-rank adjacent bigram across the whole word
            var bestRank = Int.MAX_VALUE; var bestA = ""; var bestB = ""
            for (i in 0 until symbols.size - 1) {
                val r = ranks[symbols[i] + " " + symbols[i + 1]] ?: continue
                if (r < bestRank) { bestRank = r; bestA = symbols[i]; bestB = symbols[i + 1] }
            }
            if (bestRank == Int.MAX_VALUE) break
            // merge every occurrence of (bestA,bestB) in one pass (GPT-2 BPE)
            val merged = ArrayList<String>(symbols.size)
            var i = 0
            while (i < symbols.size) {
                if (i < symbols.size - 1 && symbols[i] == bestA && symbols[i + 1] == bestB) {
                    merged.add(bestA + bestB); i += 2
                } else { merged.add(symbols[i]); i++ }
            }
            symbols = merged
        }
        return symbols
    }

    /** Tokenize text and append the pooling EOS. Returns token ids. */
    fun encode(text: String): IntArray {
        val ids = ArrayList<Int>()
        val m = pattern.matcher(text)
        while (m.find()) {
            val piece = m.group() ?: continue
            // byte-level: UTF-8 bytes -> mapped unicode chars
            val bytes = piece.toByteArray(Charsets.UTF_8)
            val sb = StringBuilder(bytes.size)
            for (b in bytes) sb.append(byteToUnicode[b.toInt() and 0xFF])
            for (tokenStr in bpe(sb.toString())) {
                val id = vocab[tokenStr]
                if (id != null) ids.add(id)
            }
        }
        ids.add(EOS_POOL)
        return ids.toIntArray()
    }
}
