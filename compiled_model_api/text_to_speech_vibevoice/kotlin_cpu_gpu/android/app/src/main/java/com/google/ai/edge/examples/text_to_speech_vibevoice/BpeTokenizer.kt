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

package com.google.ai.edge.examples.text_to_speech_vibevoice

import android.content.Context
import org.json.JSONObject
import java.util.regex.Pattern

/**
 * Qwen2 byte-level BPE tokenizer (GPT-2 / cl100k style), ported for on-device use.
 *
 * VibeVoice tokenizes the target text exactly as `tokenizer.encode(text.strip() + "\n",
 * add_special_tokens=False)` — plain byte-level BPE, NO special tokens (the speech/system
 * prompt tokens are already baked into the voice preset's KV cache). The pipeline:
 *   1. Split on the Qwen pre-tokenizer regex.
 *   2. Byte-level encode each piece (UTF-8 byte -> GPT-2 unicode char).
 *   3. BPE-merge by rank from merges.txt.
 *   4. Map pieces to ids via vocab.json.
 *
 * vocab.json + merges.txt (Qwen2.5, 151936 vocab) are bundled in assets.
 */
class BpeTokenizer(ctx: Context) {

    companion object {
        // Qwen pre-tokenizer split pattern (from tokenizer.json)
        private const val PAT =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|" +
                "\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    }

    private val pattern: Pattern = Pattern.compile(PAT)
    private val byteToUnicode: CharArray = buildByteToUnicode()
    private val vocab = HashMap<String, Int>(160000)
    private val ranks = HashMap<String, Int>(160000)

    init {
        val vj = JSONObject(ctx.assets.open("vocab.json").bufferedReader().use { it.readText() })
        val keys = vj.keys()
        while (keys.hasNext()) {
            val k = keys.next()
            vocab[k] = vj.getInt(k)
        }
        ctx.assets.open("merges.txt").bufferedReader().useLines { lines ->
            var rank = 0
            for (line in lines) {
                if (line.isEmpty() || line.startsWith("#")) continue
                ranks[line] = rank++
            }
        }
    }

    /** GPT-2 reversible byte<->unicode map so every byte becomes a printable char. */
    private fun buildByteToUnicode(): CharArray {
        val bs = ArrayList<Int>()
        for (i in '!'.code..'~'.code) {
            bs.add(i)
        }
        for (i in '¡'.code..'¬'.code) {
            bs.add(i)
        }
        for (i in '®'.code..'ÿ'.code) {
            bs.add(i)
        }
        val cs = ArrayList<Int>(bs)
        var n = 0
        for (b in 0..255) if (b !in bs) {
            bs.add(b)
            cs.add(256 + n)
            n++
        }
        val map = CharArray(256)
        for (i in bs.indices) {
            map[bs[i]] = cs[i].toChar()
        }
        return map
    }

    private fun bpe(word: String): List<String> {
        var symbols = ArrayList<String>(word.length)
        for (c in word) {
            symbols.add(c.toString())
        }
        if (symbols.size < 2) return symbols
        while (true) {
            var bestRank = Int.MAX_VALUE
            var bestA = ""
            var bestB = ""
            for (i in 0 until symbols.size - 1) {
                val r = ranks[symbols[i] + " " + symbols[i + 1]] ?: continue
                if (r < bestRank) {
                    bestRank = r
                    bestA = symbols[i]
                    bestB = symbols[i + 1]
                }
            }
            if (bestRank == Int.MAX_VALUE) break
            val merged = ArrayList<String>(symbols.size)
            var i = 0
            while (i < symbols.size) {
                if (i < symbols.size - 1 && symbols[i] == bestA && symbols[i + 1] == bestB) {
                    merged.add(bestA + bestB)
                    i += 2
                } else { merged.add(symbols[i])
                i++ }
            }
            symbols = merged
        }
        return symbols
    }

    /** Tokenizes `text.strip() + "\n"` with byte-level BPE, returning ids (no special tokens). */
    fun encode(text: String): IntArray {
        val ids = ArrayList<Int>()
        val m = pattern.matcher(text.trim() + "\n")
        while (m.find()) {
            val piece = m.group() ?: continue
            val bytes = piece.toByteArray(Charsets.UTF_8)
            val sb = StringBuilder(bytes.size)
            for (b in bytes) {
                sb.append(byteToUnicode[b.toInt() and 0xFF])
            }
            for (tokenStr in bpe(sb.toString())) {
                val id = vocab[tokenStr]
                if (id != null) {
                    ids.add(id)
                }
            }
        }
        return ids.toIntArray()
    }
}
