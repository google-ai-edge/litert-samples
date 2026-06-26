/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.ai.edge.examples.text_to_speech

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import org.json.JSONObject
import java.io.BufferedReader
import java.io.Closeable
import java.io.File
import java.io.InputStreamReader

/**
 * Clean English G2P (no espeak at runtime). Hybrid, same shape as kokoro's:
 *   1. a 275k-entry espeak-IPA dictionary (primary, covers ~all common words correctly), and
 *   2. a DeepPhonemizer ForwardTransformer (OpenPhonemizer's espeak-IPA checkpoint, Clear BSD /
 *      MIT) for out-of-dictionary words, converted to a fixed-shape LiteRT graph run on the
 *      CompiledModel CPU delegate (the graph uses EQUAL / SELECT_V2 / >4D attention tensors the
 *      GPU delegate rejects but the CPU accelerator runs fine).
 *
 *   word -> dict[word] OR (char ids, each repeated char_repeats, <en_us>…<end>, 0-padded to
 *           MAXT) -> g2p.tflite (CPU) -> argmax -> collapse repeats -> espeak-style IPA
 *   sentence IPA -> matcha symbol ids (keithito 178-symbol IPA set)
 *
 * Produces the same phoneme inventory Matcha-LJSpeech was trained on (espeak en-us IPA),
 * so it maps 1:1 onto the model's symbol table.
 */
class MatchaG2P(context: Context) : Closeable {

    companion object {
        const val MODEL = "dp_g2p_matcha_fp16.tflite"
        // AAPT auto-decompresses .gz assets and strips the extension, so the packaged
        // name is g2p_dict.txt (AssetManager handles the APK compression transparently).
        const val DICT = "g2p_dict.txt"
        // case-aware tokens: ACRONYM (2+ caps) | NUMBER | word | punctuation
        private val TOKEN = Regex("[A-Z]{2,}|\\d[\\d,]*(?:\\.\\d+)?|[A-Za-z']+|[.,!?;:—…\"]")
        private val ACRO = Regex("[A-Z]{2,}")
        private val WORD = Regex("[A-Za-z']+")
        // espeak letter-name IPA — acronyms are spelled out (e.g. "GPU" -> dʒˈiːpˈiːjˈuː)
        private val LETTER = mapOf(
            'a' to "ˈeɪ", 'b' to "bˈiː", 'c' to "sˈiː", 'd' to "dˈiː", 'e' to "ˈiː", 'f' to "ˈɛf",
            'g' to "dʒˈiː", 'h' to "ˈeɪtʃ", 'i' to "ˈaɪ", 'j' to "dʒˈeɪ", 'k' to "kˈeɪ", 'l' to "ˈɛl",
            'm' to "ˈɛm", 'n' to "ˈɛn", 'o' to "ˈoʊ", 'p' to "pˈiː", 'q' to "kjˈuː", 'r' to "ˈɑːɹ",
            's' to "ˈɛs", 't' to "tˈiː", 'u' to "jˈuː", 'v' to "vˈiː", 'w' to "dˈʌbəljˌuː",
            'x' to "ˈɛks", 'y' to "wˈaɪ", 'z' to "zˈiː")
        private val ONES = arrayOf("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
            "eighteen", "nineteen")
        private val TENS = arrayOf("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
        private val SCALES = arrayOf("", "thousand", "million", "billion", "trillion")
    }

    private val dict = HashMap<String, String>(300_000)

    private val g2p: CompiledModel
    private val gIn: List<TensorBuffer>
    private val gOut: List<TensorBuffer>

    private val char2idx = HashMap<Char, Int>()
    private val idx2ph = HashMap<Int, String>()
    private val special: Set<String>
    private val charRepeats: Int
    private val start: Int
    private val end: Int
    private val maxt: Int
    private val nPh: Int
    private val symbolToId = HashMap<Char, Int>()

    init {
        val f = File(context.filesDir, MODEL)
        check(f.exists()) { "G2P model not found: $MODEL. Push it first:\n  scripts/install_to_device.sh" }
        g2p = CompiledModel.create(f.absolutePath, CompiledModel.Options(Accelerator.CPU), null)
        gIn = g2p.createInputBuffers(); gOut = g2p.createOutputBuffers()

        val meta = JSONObject(context.assets.open("g2p_meta.json").readBytes().decodeToString())
        val c2i = meta.getJSONObject("char2idx")
        for (k in c2i.keys()) if (k.length == 1) char2idx[k[0]] = c2i.getInt(k)
        val i2p = meta.getJSONObject("idx2ph")
        for (k in i2p.keys()) idx2ph[k.toInt()] = i2p.getString(k)
        charRepeats = meta.getInt("char_repeats")
        start = meta.getInt("start"); end = meta.getInt("end")
        maxt = meta.getInt("MAXT"); nPh = meta.getInt("n_phonemes")
        val sp = meta.getJSONArray("special"); special = (0 until sp.length()).map { sp.getString(it) }.toSet()

        // matcha symbol -> id (keithito symbols list from config.json)
        val cfg = JSONObject(context.assets.open("config.json").readBytes().decodeToString())
        val syms = cfg.getJSONArray("symbols")
        for (i in 0 until syms.length()) {
            val s = syms.getString(i)
            if (s.length == 1) symbolToId[s[0]] = i
        }

        // espeak-IPA dictionary (primary G2P):  word<TAB>ipa per line
        BufferedReader(InputStreamReader(context.assets.open(DICT), Charsets.UTF_8)).use { r ->
            r.forEachLine { ln ->
                val t = ln.indexOf('\t')
                if (t > 0) dict[ln.substring(0, t)] = ln.substring(t + 1)
            }
        }
    }

    /** Full text -> matcha symbol ids (blanks are interspersed later by the synthesizer).
     *  Host-side text normalization: ALL-CAPS acronyms are spelled letter-by-letter (GPU ->
     *  "gee pee you") and numbers are read as words (4090 -> "four thousand ninety"). */
    fun phonemize(text: String): IntArray {
        val ipa = StringBuilder()
        var first = true
        fun add(p: String) { if (p.isNotEmpty()) { if (!first) ipa.append(' '); ipa.append(p); first = false } }
        for (m in TOKEN.findAll(text)) {
            val tok = m.value
            when {
                ACRO.matches(tok) -> add(tok.lowercase().mapNotNull { LETTER[it] }.joinToString(""))
                tok[0].isDigit() -> for (w in numToWords(tok)) add(dict[w] ?: phonemizeWord(w))
                WORD.matches(tok) -> add(dict[tok.lowercase()] ?: phonemizeWord(tok.lowercase()))  // dict primary, neural OOV
                else -> ipa.append(tok)    // punctuation: attach to the preceding word
            }
        }
        val out = ArrayList<Int>(ipa.length)
        for (ch in ipa) symbolToId[ch]?.let { out.add(it) }
        return out.toIntArray()
    }

    /** "1,234.5" -> ["one","thousand","two","hundred","thirty","four","point","five"]. */
    private fun numToWords(raw: String): List<String> {
        val tok = raw.replace(",", "")
        if (tok.contains('.')) {
            val parts = tok.split('.', limit = 2)
            val words = (if (parts[0].isNotEmpty()) intToWords(parts[0].toLongOrNull() ?: 0L) else listOf("zero")).toMutableList()
            words.add("point")
            for (d in parts[1]) if (d.isDigit()) words.add(ONES[d - '0'])
            return words
        }
        return intToWords(tok.toLongOrNull() ?: return emptyList())
    }

    private fun under1000(n0: Int): List<String> {
        var n = n0; val w = ArrayList<String>(4)
        if (n >= 100) { w.add(ONES[n / 100]); w.add("hundred"); n %= 100 }
        if (n >= 20) { w.add(TENS[n / 10]); n %= 10 }
        if (n > 0) w.add(ONES[n])
        return w
    }

    private fun intToWords(n0: Long): List<String> {
        if (n0 == 0L) return listOf("zero")
        if (n0 < 0) return listOf("minus") + intToWords(-n0)
        val groups = ArrayList<Int>(); var n = n0
        while (n > 0) { groups.add((n % 1000).toInt()); n /= 1000 }
        if (groups.size > SCALES.size) return n0.toString().map { ONES[it - '0'] }   // too big -> digit by digit
        val out = ArrayList<String>()
        for (i in groups.indices.reversed()) {
            if (groups[i] == 0) continue
            out.addAll(under1000(groups[i]))
            if (SCALES[i].isNotEmpty()) out.add(SCALES[i])
        }
        return out
    }

    /** One word -> espeak-style IPA string. */
    fun phonemizeWord(word: String): String {
        val ids = ArrayList<Int>(maxt)
        ids.add(start)
        for (c in word) char2idx[c]?.let { id -> repeat(charRepeats) { ids.add(id) } }
        ids.add(end)
        val len = minOf(ids.size, maxt)
        val inBuf = FloatArray(maxt) { if (it < len) ids[it].toFloat() else 0f }

        gIn[0].writeFloat(inBuf)
        g2p.run(gIn, gOut)
        val logits = gOut[0].readFloat()    // [maxt * nPh]

        val sb = StringBuilder()
        var prev = -1
        for (t in 0 until len) {
            var best = 0; var bestV = logits[t * nPh]
            for (k in 1 until nPh) { val v = logits[t * nPh + k]; if (v > bestV) { bestV = v; best = k } }
            if (best == prev) continue
            prev = best
            val ph = idx2ph[best] ?: continue
            if (ph in special || best == 0) continue
            for (ch in ph) if (ch != '-') sb.append(ch)   // strip acronym hyphens
        }
        return sb.toString()
    }

    override fun close() {
        gIn.forEach { it.close() }; gOut.forEach { it.close() }
        g2p.close()
    }
}
