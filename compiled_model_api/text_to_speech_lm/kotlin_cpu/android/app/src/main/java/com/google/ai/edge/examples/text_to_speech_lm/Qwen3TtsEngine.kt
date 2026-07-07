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

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.File
import java.nio.ShortBuffer
import java.util.Random
import kotlin.math.exp
import kotlin.math.min

/**
 * Qwen3-TTS on LiteRT: the host-orchestrated Compiled Model decode loop.
 *
 * Runs three graphs — talker LM (prefill_32/decode signatures, KV 1024), MTP
 * code-predictor decode step (17-slot KV, invoked 17x per audio frame), and
 * the codec decoder (64-frame chunks -> 24 kHz PCM) — plus host-side BPE
 * tokenization, embedding-table lookups, prompt assembly, and sampling. It is
 * a Kotlin port of the Python reference pipeline in the sibling `python/`
 * directory, which reproduces the PyTorch implementation token-for-token
 * under greedy decoding.
 *
 * All model files live in [dir] (pushed by `install_to_device.sh`).
 */
class Qwen3TtsEngine(private val dir: File) {

    companion object {
        private const val TAG = "Qwen3Tts"
        private const val HIDDEN = 1024
        private const val CODEC_VOCAB = 3072
        private const val CACHE = 1024
        private const val NEG = -1e9f
        private const val EOS = 2150
        private const val PAD_ID = 2148
        private const val BOS_ID = 2149
        private const val THINK = 2154
        private const val THINK_BOS = 2156
        private const val THINK_EOS = 2157
        private const val NOTHINK = 2155
        private const val TTS_BOS = 151672
        private const val TTS_EOS = 151673
        private const val TTS_PAD = 151671
        private const val MTP_LAYERS = 5
        private const val MTP_CACHE = 17
        private const val MTP_KV_FLOATS = MTP_LAYERS * 8 * MTP_CACHE * 128
        private const val MTP_VOCAB = 2048
        private const val CODEC_CHUNK = 64
        private const val CODEC_CTX = 25
        private const val UPSAMPLE = 1920
        const val SAMPLE_RATE = 24000
        const val MAX_FRAMES = 512

        // <|im_start|>assistant\n   and   <|im_end|>\n<|im_start|>assistant\n
        private val PROMPT_PREFIX = intArrayOf(151644, 77091, 198)
        private val PROMPT_SUFFIX = intArrayOf(151645, 198, 151644, 77091, 198)

        val LANGUAGE_IDS = mapOf(
            "chinese" to 2055, "english" to 2050, "german" to 2053,
            "italian" to 2070, "portuguese" to 2071, "spanish" to 2054,
            "japanese" to 2058, "korean" to 2064, "french" to 2061,
            "russian" to 2069,
        )
    }

    private val tokenizer =
        QwenBpeTokenizer(File(dir, "vocab.json"), File(dir, "merges.txt"))

    /** Exposes plain-text tokenization for the startup self-test. */
    fun encodeText(text: String): IntArray = tokenizer.encode(text)

    // Host tables. The two big ones stay memory-mapped fp16.
    private val codecEmb = Npy.loadFloats(File(dir, "codec_embedding_fp32.npy"))
    private val mtpEmb: ShortBuffer = Npy.mmapHalf(File(dir, "mtp_embeddings_fp16.npy"))
    private val textEmb: ShortBuffer = Npy.mmapHalf(File(dir, "text_embedding_fp16.npy"))
    private val proj = Npy.loadNpz(
        File(dir, "text_projection_fp32.npz"), listOf("w1", "b1", "w2", "b2"))
    val speaker: FloatArray = Npy.loadFloats(File(dir, "demo_speaker.npy"))

    private fun load(name: String, threads: Int): CompiledModel {
        val f = File(dir, name)
        check(f.exists()) { "Model not found: $name — run install_to_device.sh" }
        val options = CompiledModel.Options(Accelerator.CPU)
        options.cpuOptions = CompiledModel.CpuOptions(threads, null, null)
        return CompiledModel.create(f.absolutePath, options, null)
    }

    private val talker = load("talker_int4.tflite", 4)
    private val mtp = load("mtp_fp32.tflite", 2)
    private val codec = load("codec_decoder_fp32.tflite", 4)

    private val kvNames = (0 until 28).flatMap {
        listOf("kv_cache_k_$it", "kv_cache_v_$it")
    }

    data class Result(
        val audio: FloatArray, val frames: Int,
        val prefillMs: Long, val talkerMs: Long, val mtpMs: Long,
        val codecMs: Long,
    )

    interface Progress { fun onFrame(frame: Int) }

    /**
     * Synthesizes [text] in the voice of [spk] (1024-d x-vector).
     *
     * With [greedy] the loop matches the Python reference (and hence the
     * PyTorch implementation) token-for-token; otherwise top-k/temperature
     * sampling with the model's default parameters.
     */
    fun synthesize(
        text: String, language: String = "english",
        spk: FloatArray = speaker, greedy: Boolean = false,
        seed: Long? = null, progress: Progress? = null,
    ): Result {
        val rnd = if (seed != null) Random(seed) else Random()

        // ---- prompt assembly (host-side, plain lookups + the tiny MLP) ----
        val textIds = tokenizer.encode(text)
        val ttsBos = embedText(intArrayOf(TTS_BOS))[0]
        val ttsEos = embedText(intArrayOf(TTS_EOS))[0]
        val ttsPad = embedText(intArrayOf(TTS_PAD))[0]

        val control = if (language == "auto") {
            intArrayOf(NOTHINK, THINK_BOS, THINK_EOS)
        } else {
            val lang = LANGUAGE_IDS[language.lowercase()]
                ?: throw IllegalArgumentException("language: $language")
            intArrayOf(THINK, THINK_BOS, lang, THINK_EOS)
        }
        // control embeds | speaker | pad,bos  -> [C+3, 1024]
        val codecPre = ArrayList<FloatArray>()
        for (id in control) codecPre.add(codecRow(id))
        codecPre.add(spk.copyOf())
        codecPre.add(codecRow(PAD_ID))
        codecPre.add(codecRow(BOS_ID))

        val role = embedText(PROMPT_PREFIX)                      // [3,1024]
        val body = Array(codecPre.size - 1) { i ->              // pads+bos + codecPre[:-1]
            val cond = if (i < codecPre.size - 2) ttsPad else ttsBos
            add(cond, codecPre[i])
        }
        val firstText = add(embedText(intArrayOf(textIds[0]))[0], codecPre.last())
        val prefill = role + body + arrayOf(firstText)           // [P,1024]
        val trailing = ArrayList<FloatArray>()                   // streamed text cond
        for (i in 1 until textIds.size) {
            trailing.add(embedText(intArrayOf(textIds[i]))[0])
        }
        trailing.add(ttsEos)

        // ---- talker prefill + first decode ----
        var t0 = System.nanoTime()
        val decodeKv = TalkerState()
        decodeKv.prefill(prefill)
        var pos = prefill.size - 1
        var step = decodeKv.decode(prefill.last(), pos)
        val prefillMs = (System.nanoTime() - t0) / 1_000_000

        // ---- frame loop ----
        val suppress = FloatArray(CODEC_VOCAB)
        for (i in 2048 until CODEC_VOCAB) suppress[i] = NEG
        suppress[EOS] = 0f

        val frames = ArrayList<IntArray>()
        val history = HashSet<Int>()
        var talkerNs = 0L
        var mtpNs = 0L
        while (frames.size < MAX_FRAMES) {
            val scores = FloatArray(CODEC_VOCAB) { step.logits[it] + suppress[it] }
            if (frames.size < 2) scores[EOS] = NEG // min_new_tokens = 2
            for (t in history) {
                scores[t] = if (scores[t] > 0) scores[t] / 1.05f else scores[t] * 1.05f
            }
            val cb0 = pick(scores, greedy, rnd)
            history.add(cb0)
            if (cb0 == EOS) break

            t0 = System.nanoTime()
            val residual = mtpFrame(step.hidden, cb0, greedy, rnd)
            mtpNs += System.nanoTime() - t0

            val frame = IntArray(16); frame[0] = cb0
            for (i in 0 until 15) frame[i + 1] = residual[i]
            frames.add(frame)
            progress?.onFrame(frames.size)

            // next input embed = sum of 16 codebook embeds + text conditioning
            val embed = codecRow(cb0)
            for (i in 0 until 15) addMtpRow(embed, i, residual[i])
            val stepIdx = frames.size - 1
            val cond = if (stepIdx < trailing.size) trailing[stepIdx] else ttsPad
            for (i in 0 until HIDDEN) embed[i] += cond[i]

            pos += 1
            t0 = System.nanoTime()
            step = decodeKv.decode(embed, pos)
            talkerNs += System.nanoTime() - t0
        }

        // ---- codec decode (chunks with left context) ----
        t0 = System.nanoTime()
        val audio = decodeCodes(frames)
        val codecMs = (System.nanoTime() - t0) / 1_000_000

        return Result(audio, frames.size, prefillMs,
            talkerNs / 1_000_000, mtpNs / 1_000_000, codecMs)
    }

    // ------------------------------------------------------------------
    // Talker: prefill_32 + decode signatures with ping-pong KV buffers.
    // ------------------------------------------------------------------
    private inner class TalkerState {
        // Two full KV sets; run() alternates them as inputs/outputs to avoid
        // copying ~235 MB of cache per step.
        val setA = kvNames.associateWith { talker.createOutputBuffer(it, "decode") }
        val setB = kvNames.associateWith { talker.createOutputBuffer(it, "decode") }
        var current = setA // holds the cache AFTER the latest step

        val embIn = talker.createInputBuffer("embeddings", "decode")
        val posIn = talker.createInputBuffer("input_pos", "decode")
        val maskIn = talker.createInputBuffer("mask", "decode")
        val logitsOut = talker.createOutputBuffer("logits", "decode")
        val mask = FloatArray(CACHE) { NEG }

        fun prefill(embeds: Array<FloatArray>) {
            val p = embeds.size
            check(p <= 32) { "prompt too long for prefill_32: $p" }
            val flat = FloatArray(32 * HIDDEN)
            for (t in embeds.indices) {
                System.arraycopy(embeds[t], 0, flat, t * HIDDEN, HIDDEN)
            }
            val maskFlat = FloatArray(32 * CACHE) { NEG }
            for (row in 0 until 32) {
                val allowed = min(row, p - 1) + 1
                for (c in 0 until allowed) maskFlat[row * CACHE + c] = 0f
            }
            val inputs = HashMap<String, com.google.ai.edge.litert.TensorBuffer>()
            inputs["embeddings"] = talker.createInputBuffer("embeddings", "prefill_32")
                .also { it.writeFloat(flat) }
            inputs["input_pos"] = talker.createInputBuffer("input_pos", "prefill_32")
                .also { it.writeInt(IntArray(32) { i -> i }) }
            inputs["mask"] = talker.createInputBuffer("mask", "prefill_32")
                .also { it.writeFloat(maskFlat) }
            val zero = FloatArray(8 * CACHE * 128)
            for (name in kvNames) {
                inputs[name] = talker.createInputBuffer(name, "prefill_32")
                    .also { it.writeFloat(zero) }
            }
            talker.run(inputs, setA.mapValues { it.value }, "prefill_32")
            current = setA
            for (buffer in inputs.values) buffer.close()
        }

        fun decode(embed: FloatArray, pos: Int): Step {
            embIn.writeFloat(embed)
            posIn.writeInt(intArrayOf(pos))
            java.util.Arrays.fill(mask, NEG)
            for (c in 0..pos) mask[c] = 0f
            maskIn.writeFloat(mask)
            val next = if (current === setA) setB else setA
            val inputs = HashMap<String, com.google.ai.edge.litert.TensorBuffer>(64)
            inputs["embeddings"] = embIn
            inputs["input_pos"] = posIn
            inputs["mask"] = maskIn
            for (name in kvNames) inputs[name] = current.getValue(name)
            val outputs = HashMap<String, com.google.ai.edge.litert.TensorBuffer>(64)
            outputs["logits"] = logitsOut
            for (name in kvNames) outputs[name] = next.getValue(name)
            talker.run(inputs, outputs, "decode")
            current = next
            val logits = logitsOut.readFloat() // [4096] = codec logits | hidden
            return Step(
                logits.copyOfRange(0, CODEC_VOCAB),
                logits.copyOfRange(CODEC_VOCAB, CODEC_VOCAB + HIDDEN))
        }
    }

    class Step(val logits: FloatArray, val hidden: FloatArray)

    // ------------------------------------------------------------------
    // MTP inner loop: one decode-step graph invoked 17x per frame.
    // Inputs (positional): embed, pos, mask, k_all, v_all.
    // Outputs (positional): logits_all [15,2048], k_all, v_all.
    // ------------------------------------------------------------------
    private val mtpIn = mtp.createInputBuffers()   // embed, pos, mask, k, v
    private val mtpOutPing = mtp.createOutputBuffers() // logits, k, v
    private val mtpOutPong = mtp.createOutputBuffers()

    private fun mtpFrame(
        hidden: FloatArray, cb0: Int, greedy: Boolean, rnd: Random,
    ): IntArray {
        val zero = FloatArray(MTP_KV_FLOATS)
        mtpIn[3].writeFloat(zero)
        mtpIn[4].writeFloat(zero)
        // KV ping-pong: read from kIn, write into ping/pong alternately;
        // the freshly written pair becomes the next step's input. The
        // input-created pair (mtpIn[3/4]) only carries the initial zeros.
        var kIn = Pair(mtpIn[3], mtpIn[4])
        val ping = Pair(mtpOutPing[1], mtpOutPing[2])
        val pong = Pair(mtpOutPong[1], mtpOutPong[2])
        var usePing = true
        val logitsOut = mtpOutPing[0]
        val codes = IntArray(15)
        val mask = FloatArray(MTP_CACHE) { NEG }
        for (t in 0 until 16) {
            val embed = when {
                t == 0 -> hidden
                t == 1 -> codecRow(cb0)
                else -> mtpRow(t - 2, codes[t - 2])
            }
            mtpIn[0].writeFloat(embed)
            mtpIn[1].writeInt(intArrayOf(t))
            mask[t] = 0f
            mtpIn[2].writeFloat(mask)
            val write = if (usePing) ping else pong
            mtp.run(
                listOf(mtpIn[0], mtpIn[1], mtpIn[2], kIn.first, kIn.second),
                listOf(logitsOut, write.first, write.second))
            if (t >= 1) {
                val all = logitsOut.readFloat() // [15 * 2048]
                val head = t - 1
                val logits = FloatArray(MTP_VOCAB)
                System.arraycopy(all, head * MTP_VOCAB, logits, 0, MTP_VOCAB)
                codes[head] = pick(logits, greedy, rnd)
            }
            kIn = write
            usePing = !usePing
        }
        return codes
    }

    // ------------------------------------------------------------------
    // Codec decode: fixed 64-frame chunks, 25 frames of left context.
    // ------------------------------------------------------------------
    private fun decodeCodes(frames: List<IntArray>): FloatArray {
        if (frames.isEmpty()) return FloatArray(0)
        val codecIn = codec.createInputBuffers()
        val codecOut = codec.createOutputBuffers()
        val pieces = ArrayList<FloatArray>()
        var i = 0
        while (i < frames.size) {
            val j = min(i + CODEC_CHUNK, frames.size)
            val ctx = min(CODEC_CTX, i)
            val n = j - (i - ctx)
            val buf = IntArray(16 * CODEC_CHUNK)
            for (t in 0 until n) {
                val frame = frames[i - ctx + t]
                for (q in 0 until 16) buf[q * CODEC_CHUNK + t] = frame[q]
            }
            codecIn[0].writeInt(buf)
            codec.run(codecIn, codecOut)
            val wav = codecOut[0].readFloat()
            pieces.add(wav.copyOfRange(ctx * UPSAMPLE, n * UPSAMPLE))
            i = j
        }
        var total = 0
        for (p in pieces) total += p.size
        val out = FloatArray(total)
        var off = 0
        for (p in pieces) { System.arraycopy(p, 0, out, off, p.size); off += p.size }
        return out
    }

    // ------------------------------------------------------------------
    // Host math helpers.
    // ------------------------------------------------------------------
    private fun codecRow(id: Int): FloatArray {
        val out = FloatArray(HIDDEN)
        System.arraycopy(codecEmb, id * HIDDEN, out, 0, HIDDEN)
        return out
    }

    private fun mtpRow(table: Int, id: Int): FloatArray {
        val out = FloatArray(HIDDEN)
        val base = (table * MTP_VOCAB + id) * HIDDEN
        for (i in 0 until HIDDEN) out[i] = Npy.halfToFloat(mtpEmb.get(base + i))
        return out
    }

    private fun addMtpRow(acc: FloatArray, table: Int, id: Int) {
        val base = (table * MTP_VOCAB + id) * HIDDEN
        for (i in 0 until HIDDEN) acc[i] += Npy.halfToFloat(mtpEmb.get(base + i))
    }

    private fun add(a: FloatArray, b: FloatArray): FloatArray =
        FloatArray(HIDDEN) { a[it] + b[it] }

    /** text_embedding lookup + the 2048->1024 SiLU projection MLP. */
    private fun embedText(ids: IntArray): Array<FloatArray> {
        val w1 = proj.getValue("w1"); val b1 = proj.getValue("b1")
        val w2 = proj.getValue("w2"); val b2 = proj.getValue("b2")
        return Array(ids.size) { n ->
            val x = FloatArray(2048)
            val base = ids[n] * 2048
            for (i in 0 until 2048) x[i] = Npy.halfToFloat(textEmb.get(base + i))
            val h = FloatArray(2048)
            for (r in 0 until 2048) {
                var acc = b1[r]
                val wBase = r * 2048
                for (c in 0 until 2048) acc += w1[wBase + c] * x[c]
                h[r] = acc / (1f + exp(-acc)) // SiLU
            }
            val y = FloatArray(HIDDEN)
            for (r in 0 until HIDDEN) {
                var acc = b2[r]
                val wBase = r * 2048
                for (c in 0 until 2048) acc += w2[wBase + c] * h[c]
                y[r] = acc
            }
            y
        }
    }

    /** Greedy argmax or top-50/temperature-0.9 sampling. */
    private fun pick(logits: FloatArray, greedy: Boolean, rnd: Random): Int {
        if (greedy) {
            var best = 0
            for (i in logits.indices) if (logits[i] > logits[best]) best = i
            return best
        }
        val k = 50
        val idx = logits.indices.sortedByDescending { logits[it] }.take(k)
        val probs = DoubleArray(k)
        val maxLogit = logits[idx[0]] / 0.9
        var sum = 0.0
        for (i in 0 until k) {
            probs[i] = exp(logits[idx[i]] / 0.9 - maxLogit)
            sum += probs[i]
        }
        var r = rnd.nextDouble() * sum
        for (i in 0 until k) {
            r -= probs[i]
            if (r <= 0) return idx[i]
        }
        return idx[k - 1]
    }

    fun close() {
        talker.close(); mtp.close(); codec.close()
    }
}
