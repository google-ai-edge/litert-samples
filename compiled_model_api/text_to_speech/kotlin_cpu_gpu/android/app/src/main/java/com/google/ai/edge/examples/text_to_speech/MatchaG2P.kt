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
        private val WORD = Regex("[a-z']+")
        private val TOKEN = Regex("[a-z']+|[.,!?;:—…\"]")
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

    /** Full text -> matcha symbol ids (blanks are interspersed later by the synthesizer). */
    fun phonemize(text: String): IntArray {
        val ipa = StringBuilder()
        var first = true
        for (m in TOKEN.findAll(text.lowercase())) {
            val tok = m.value
            if (WORD.matches(tok)) {
                val p = dict[tok] ?: phonemizeWord(tok)   // dict primary, neural OOV fallback
                if (p.isNotEmpty()) { if (!first) ipa.append(' '); ipa.append(p); first = false }
            } else {                       // punctuation: attach to the preceding word
                ipa.append(tok)
            }
        }
        val out = ArrayList<Int>(ipa.length)
        for (ch in ipa) symbolToId[ch]?.let { out.add(it) }
        return out.toIntArray()
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
