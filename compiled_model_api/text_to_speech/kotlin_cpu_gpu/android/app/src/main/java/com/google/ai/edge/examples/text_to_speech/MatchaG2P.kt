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

package com.google.ai.edge.examples.text_to_speech

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.BufferedReader
import java.io.Closeable
import java.io.File
import java.io.InputStreamReader
import org.json.JSONObject

/**
 * Clean English G2P (no espeak at runtime). Hybrid, same shape as kokoro's:
 * 1. a 275k-entry espeak-IPA dictionary (primary, covers ~all common words correctly), and
 * 2. a DeepPhonemizer ForwardTransformer (OpenPhonemizer's espeak-IPA checkpoint, Clear BSD /
 *    MIT) for out-of-dictionary words, converted to a fixed-shape LiteRT graph run on the
 *    CompiledModel CPU delegate (the graph uses EQUAL / SELECT_V2 / >4D attention tensors the
 *    GPU delegate rejects but the CPU accelerator runs fine).
 *
 * ```
 * word -> dict[word] OR (char ids, each repeated charRepeats, <en_us>…<end>, 0-padded to
 *         maxTokens) -> g2p.tflite (CPU) -> argmax -> collapse repeats -> espeak-style IPA
 * sentence IPA -> matcha symbol ids (keithito 178-symbol IPA set)
 * ```
 *
 * Produces the same phoneme inventory Matcha-LJSpeech was trained on (espeak en-us IPA),
 * so it maps 1:1 onto the model's symbol table.
 */
class MatchaG2P(context: Context) : Closeable {

    private val dictionary = HashMap<String, String>(300_000)

    private val g2pModel: CompiledModel
    private val g2pInputs: List<TensorBuffer>
    private val g2pOutputs: List<TensorBuffer>

    private val charToIndex = HashMap<Char, Int>()
    private val indexToPhoneme = HashMap<Int, String>()
    private val specialTokens: Set<String>
    private val charRepeats: Int
    private val startId: Int
    private val endId: Int
    private val maxTokens: Int
    private val numPhonemes: Int
    private val symbolToId = HashMap<Char, Int>()

    init {
        val modelFile = File(context.filesDir, MODEL_FILE)
        check(modelFile.exists()) {
            "G2P model not found: $MODEL_FILE. Push it first:\n  scripts/install_to_device.sh"
        }
        g2pModel =
            CompiledModel.create(modelFile.absolutePath, CompiledModel.Options(Accelerator.CPU), null)
        g2pInputs = g2pModel.createInputBuffers()
        g2pOutputs = g2pModel.createOutputBuffers()

        val meta = JSONObject(context.assets.open("g2p_meta.json").readBytes().decodeToString())
        val char2idx = meta.getJSONObject("char2idx")
        for (key in char2idx.keys()) {
            if (key.length == 1) {
                charToIndex[key[0]] = char2idx.getInt(key)
            }
        }
        val idx2ph = meta.getJSONObject("idx2ph")
        for (key in idx2ph.keys()) {
            indexToPhoneme[key.toInt()] = idx2ph.getString(key)
        }
        charRepeats = meta.getInt("char_repeats")
        startId = meta.getInt("start")
        endId = meta.getInt("end")
        maxTokens = meta.getInt("MAXT")
        numPhonemes = meta.getInt("n_phonemes")
        val special = meta.getJSONArray("special")
        specialTokens = (0 until special.length()).map { special.getString(it) }.toSet()

        // Matcha symbol -> id (keithito symbols list from config.json).
        val config = JSONObject(context.assets.open("config.json").readBytes().decodeToString())
        val symbols = config.getJSONArray("symbols")
        for (i in 0 until symbols.length()) {
            val symbol = symbols.getString(i)
            if (symbol.length == 1) {
                symbolToId[symbol[0]] = i
            }
        }

        // espeak-IPA dictionary (primary G2P): word<TAB>ipa per line.
        BufferedReader(InputStreamReader(context.assets.open(DICT_ASSET), Charsets.UTF_8)).use { reader ->
            reader.forEachLine { line ->
                val tab = line.indexOf('\t')
                if (tab > 0) {
                    dictionary[line.substring(0, tab)] = line.substring(tab + 1)
                }
            }
        }
    }

    /**
     * Converts full text to matcha symbol ids (blanks are interspersed later by the synthesizer).
     *
     * Host-side text normalization: ALL-CAPS acronyms are spelled letter-by-letter (GPU ->
     * "gee pee you") and numbers are read as words (4090 -> "four thousand ninety").
     */
    fun phonemize(text: String): IntArray {
        val ipa = StringBuilder()
        var first = true
        fun append(phonemes: String) {
            if (phonemes.isEmpty()) return
            if (!first) {
                ipa.append(' ')
            }
            ipa.append(phonemes)
            first = false
        }
        for (match in TOKEN.findAll(text)) {
            val token = match.value
            when {
                ACRONYM.matches(token) -> {
                    append(token.lowercase().mapNotNull { LETTER_IPA[it] }.joinToString(""))
                }
                token[0].isDigit() -> {
                    for (word in numberToWords(token)) {
                        append(dictionary[word] ?: phonemizeWord(word))
                    }
                }
                WORD.matches(token) -> {
                    // Dictionary is primary; the neural G2P handles out-of-dictionary words.
                    append(dictionary[token.lowercase()] ?: phonemizeWord(token.lowercase()))
                }
                else -> {
                    // Punctuation: attach to the preceding word.
                    ipa.append(token)
                }
            }
        }
        val ids = ArrayList<Int>(ipa.length)
        for (ch in ipa) {
            symbolToId[ch]?.let { ids.add(it) }
        }
        return ids.toIntArray()
    }

    /** "1,234.5" -> ["one","thousand","two","hundred","thirty","four","point","five"]. */
    private fun numberToWords(raw: String): List<String> {
        val token = raw.replace(",", "")
        if (token.contains('.')) {
            val parts = token.split('.', limit = 2)
            val words =
                (if (parts[0].isNotEmpty()) integerToWords(parts[0].toLongOrNull() ?: 0L)
                 else listOf("zero"))
                    .toMutableList()
            words.add("point")
            for (digit in parts[1]) {
                if (digit.isDigit()) {
                    words.add(ONES[digit - '0'])
                }
            }
            return words
        }
        return integerToWords(token.toLongOrNull() ?: return emptyList())
    }

    private fun wordsUnderThousand(value: Int): List<String> {
        var n = value
        val words = ArrayList<String>(4)
        if (n >= 100) {
            words.add(ONES[n / 100])
            words.add("hundred")
            n %= 100
        }
        if (n >= 20) {
            words.add(TENS[n / 10])
            n %= 10
        }
        if (n > 0) {
            words.add(ONES[n])
        }
        return words
    }

    private fun integerToWords(value: Long): List<String> {
        if (value == 0L) return listOf("zero")
        if (value < 0) return listOf("minus") + integerToWords(-value)
        val groups = ArrayList<Int>()
        var n = value
        while (n > 0) {
            groups.add((n % 1000).toInt())
            n /= 1000
        }
        if (groups.size > SCALES.size) {
            // Too big to name; read digit by digit.
            return value.toString().map { ONES[it - '0'] }
        }
        val words = ArrayList<String>()
        for (i in groups.indices.reversed()) {
            if (groups[i] == 0) continue
            words.addAll(wordsUnderThousand(groups[i]))
            if (SCALES[i].isNotEmpty()) {
                words.add(SCALES[i])
            }
        }
        return words
    }

    /** Converts one word to an espeak-style IPA string with the neural G2P. */
    fun phonemizeWord(word: String): String {
        val ids = ArrayList<Int>(maxTokens)
        ids.add(startId)
        for (ch in word) {
            charToIndex[ch]?.let { id -> repeat(charRepeats) { ids.add(id) } }
        }
        ids.add(endId)
        val length = minOf(ids.size, maxTokens)
        val input = FloatArray(maxTokens) { if (it < length) ids[it].toFloat() else 0f }

        g2pInputs[0].writeFloat(input)
        g2pModel.run(g2pInputs, g2pOutputs)
        val logits = g2pOutputs[0].readFloat() // [maxTokens * numPhonemes]

        val result = StringBuilder()
        var previous = -1
        for (t in 0 until length) {
            var best = 0
            var bestScore = logits[t * numPhonemes]
            for (k in 1 until numPhonemes) {
                val score = logits[t * numPhonemes + k]
                if (score > bestScore) {
                    bestScore = score
                    best = k
                }
            }
            if (best == previous) continue
            previous = best
            val phoneme = indexToPhoneme[best] ?: continue
            if (phoneme in specialTokens || best == 0) continue
            for (ch in phoneme) {
                if (ch != '-') { // Strip acronym hyphens.
                    result.append(ch)
                }
            }
        }
        return result.toString()
    }

    override fun close() {
        g2pInputs.forEach { it.close() }
        g2pOutputs.forEach { it.close() }
        g2pModel.close()
    }

    companion object {
        private const val MODEL_FILE = "dp_g2p_matcha_fp16.tflite"

        // AAPT auto-decompresses .gz assets and strips the extension, so the packaged
        // name is g2p_dict.txt (AssetManager handles the APK compression transparently).
        private const val DICT_ASSET = "g2p_dict.txt"

        // Case-aware tokens: ACRONYM (2+ caps) | NUMBER | word | punctuation.
        private val TOKEN = Regex("[A-Z]{2,}|\\d[\\d,]*(?:\\.\\d+)?|[A-Za-z']+|[.,!?;:—…\"]")
        private val ACRONYM = Regex("[A-Z]{2,}")
        private val WORD = Regex("[A-Za-z']+")

        // espeak letter-name IPA — acronyms are spelled out (e.g. "GPU" -> dʒˈiːpˈiːjˈuː).
        private val LETTER_IPA = mapOf(
            'a' to "ˈeɪ", 'b' to "bˈiː", 'c' to "sˈiː", 'd' to "dˈiː", 'e' to "ˈiː", 'f' to "ˈɛf",
            'g' to "dʒˈiː", 'h' to "ˈeɪtʃ", 'i' to "ˈaɪ", 'j' to "dʒˈeɪ", 'k' to "kˈeɪ", 'l' to "ˈɛl",
            'm' to "ˈɛm", 'n' to "ˈɛn", 'o' to "ˈoʊ", 'p' to "pˈiː", 'q' to "kjˈuː", 'r' to "ˈɑːɹ",
            's' to "ˈɛs", 't' to "tˈiː", 'u' to "jˈuː", 'v' to "vˈiː", 'w' to "dˈʌbəljˌuː",
            'x' to "ˈɛks", 'y' to "wˈaɪ", 'z' to "zˈiː",
        )
        private val ONES = arrayOf(
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
            "eighteen", "nineteen",
        )
        private val TENS = arrayOf(
            "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
        )
        private val SCALES = arrayOf("", "thousand", "million", "billion", "trillion")
    }
}
