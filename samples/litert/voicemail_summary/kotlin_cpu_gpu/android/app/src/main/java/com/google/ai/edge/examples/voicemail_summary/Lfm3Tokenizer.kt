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

package com.google.ai.edge.examples.voicemail_summary

import android.util.JsonReader
import android.util.JsonToken
import java.io.ByteArrayOutputStream
import java.io.Reader
import java.util.regex.Pattern

/**
 * The LFM3 tokenizer (the bundle's `tokenizer.json`) without JNI. The file describes a byte-level BPE:
 * 1. added tokens (`<|im_start|>`, `<|audio_start|>`, ...) are matched first, leftmost-longest, on the raw text (the
 *    file has no normalizer);
 * 2. the rest is split by one regex (pre-tokenizer `Split`, behavior `Isolated`: every match and every gap is a piece);
 * 3. each piece's UTF-8 bytes are mapped to GPT-2's byte characters (`ByteLevel`, no prefix space);
 * 4. BPE with `ignore_merges`: a piece that is itself a vocabulary entry is one token, otherwise the merges apply by
 *    rank, leftmost first among equal ranks (as `tokenizers`' `Word::merge_all`).
 * [encode] adds no special tokens: [ChatPrompt] writes them into the text.
 */
class Lfm3Tokenizer private constructor(
  private val vocab: HashMap<String, Int>,
  /** (left id shl 32 | right id) -> (rank shl 32 | merged id). */
  private val merges: HashMap<Long, Long>,
  private val addedByFirst: Map<Char, List<Added>>,
  private val addedIds: Map<String, Int>,
  private val idToToken: Array<String?>,
  private val special: BooleanArray,
  private val byteIds: IntArray,
) {
  class Added(val content: String, val id: Int)

  /** Number of token ids (the vocabulary and the added tokens). */
  val size: Int get() = idToToken.size

  /** The id of an added token or a vocabulary entry. */
  fun id(token: String): Int = addedIds[token] ?: vocab[token] ?: throw IllegalArgumentException("no token $token")

  fun encode(text: String): IntArray {
    val out = IntList()
    var segment = 0
    var i = 0
    while (i < text.length) {
      val match = matchAdded(text, i)
      if (match == null) {
        i++
        continue
      }
      if (i > segment) encodePlain(text.substring(segment, i), out)
      out.add(match.id)
      i += match.content.length
      segment = i
    }
    if (segment < text.length) encodePlain(text.substring(segment), out)
    return out.toArray()
  }

  /** Text of [ids]; special tokens are left out when [skipSpecial] (as `decode(..., skip_special_tokens=True)`). */
  fun decode(ids: List<Int>, skipSpecial: Boolean = true): String {
    val bytes = ByteArrayOutputStream()
    for (id in ids) {
      if (id < 0 || id >= idToToken.size) continue
      if (skipSpecial && special[id]) continue
      val token = idToToken[id] ?: continue
      // ByteLevel decoder: every character back to its byte; a token with a character outside the map keeps its UTF-8.
      val mapped = ByteArray(token.length)
      var ok = true
      for ((k, ch) in token.withIndex()) {
        val b = if (ch.code < CHAR_TO_BYTE.size) CHAR_TO_BYTE[ch.code] else -1
        if (b < 0) {
          ok = false
          break
        }
        mapped[k] = b.toByte()
      }
      if (ok) bytes.write(mapped) else bytes.write(token.toByteArray(Charsets.UTF_8))
    }
    return String(bytes.toByteArray(), Charsets.UTF_8)
  }

  private fun matchAdded(text: String, at: Int): Added? {
    val candidates = addedByFirst[text[at]] ?: return null
    for (a in candidates) if (text.startsWith(a.content, at)) return a
    return null
  }

  private fun encodePlain(segment: String, out: IntList) {
    val m = SPLIT.matcher(segment)
    var last = 0
    while (m.find()) {
      if (m.start() > last) bpe(segment.substring(last, m.start()), out)
      if (m.end() > m.start()) bpe(m.group(), out)
      last = m.end()
    }
    if (last < segment.length) bpe(segment.substring(last), out)
  }

  private fun bpe(piece: String, out: IntList) {
    val bytes = piece.toByteArray(Charsets.UTF_8)
    val word = StringBuilder(bytes.size)
    for (b in bytes) word.append(BYTE_TO_CHAR[b.toInt() and 0xff])
    vocab[word.toString()]?.let {
      out.add(it)
      return
    }
    val ids = IntArray(bytes.size) { byteIds[bytes[it].toInt() and 0xff] }
    var n = ids.size
    while (n >= 2) {
      var best = -1
      var bestRank = Int.MAX_VALUE
      var bestMerged = 0
      for (k in 0 until n - 1) {
        val m = merges[(ids[k].toLong() shl 32) or (ids[k + 1].toLong() and 0xffffffffL)] ?: continue
        val rank = (m ushr 32).toInt()
        if (rank < bestRank) {
          bestRank = rank
          best = k
          bestMerged = (m and 0xffffffffL).toInt()
        }
      }
      if (best < 0) break
      ids[best] = bestMerged
      System.arraycopy(ids, best + 2, ids, best + 1, n - best - 2)
      n--
    }
    for (k in 0 until n) out.add(ids[k])
  }

  class IntList {
    private var data = IntArray(64)
    var size = 0
      private set

    fun add(v: Int) {
      if (size == data.size) data = data.copyOf(size * 2)
      data[size++] = v
    }

    fun toArray(): IntArray = data.copyOf(size)
  }

  companion object {
    /** onig's Unicode `\s` (White_Space), spelled out: Android's regex engine (ICU) and the JVM differ on `\s`. */
    private const val WS =
      "\\x{09}-\\x{0d}\\x{20}\\x{85}\\x{a0}\\x{1680}\\x{2000}-\\x{200a}\\x{2028}\\x{2029}\\x{202f}\\x{205f}\\x{3000}"

    /** The file's Split regex with `\s` / `\S` spelled out. */
    private val SPLIT: Pattern = Pattern.compile(
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^$WS\\p{L}\\p{N}]+[\\r\\n]*" +
        "|[$WS]*[\\r\\n]+|[$WS]+(?![^$WS])|[$WS]+"
    )

    /** GPT-2's bytes_to_unicode: every byte maps to one printable character. */
    private val BYTE_TO_CHAR: CharArray = CharArray(256).also { t ->
      val direct = (0x21..0x7e) + (0xa1..0xac) + (0xae..0xff)
      var extra = 0
      for (b in 0 until 256) t[b] = if (b in direct) b.toChar() else (256 + extra++).toChar()
    }
    private val CHAR_TO_BYTE: IntArray = IntArray(512) { -1 }.also { t ->
      for (b in 0 until 256) t[BYTE_TO_CHAR[b].code] = b
    }

    /** Reads `tokenizer.json` (streamed: the file is 17.9 MB) and checks it is the pipeline this class implements. */
    fun load(reader: Reader): Lfm3Tokenizer {
      val vocab = HashMap<String, Int>(1 shl 18)
      val mergeLeft = ArrayList<String>(300_000)
      val mergeRight = ArrayList<String>(300_000)
      val added = ArrayList<Triple<String, Int, Boolean>>()
      var modelType: String? = null
      var ignoreMerges = false
      var byteFallback = false
      var splitPattern: String? = null
      var byteLevelPrefix: Boolean? = null
      JsonReader(reader).use { js ->
        js.beginObject()
        while (js.hasNext()) {
          when (js.nextName()) {
            "added_tokens" -> {
              js.beginArray()
              while (js.hasNext()) {
                var content = ""
                var id = -1
                var isSpecial = false
                js.beginObject()
                while (js.hasNext()) {
                  when (js.nextName()) {
                    "content" -> content = js.nextString()
                    "id" -> id = js.nextInt()
                    "special" -> isSpecial = js.nextBoolean()
                    else -> js.skipValue()
                  }
                }
                js.endObject()
                added.add(Triple(content, id, isSpecial))
              }
              js.endArray()
            }
            "normalizer" -> {
              check(js.peek() == JsonToken.NULL) { "tokenizer.json: a normalizer is not supported" }
              js.nextNull()
            }
            "pre_tokenizer" -> {
              // {"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": ...}, ...},
              //  {"type": "ByteLevel", "add_prefix_space": false, ...}]}
              js.beginObject()
              while (js.hasNext()) {
                when (js.nextName()) {
                  "pretokenizers" -> {
                    js.beginArray()
                    while (js.hasNext()) {
                      var type = ""
                      js.beginObject()
                      while (js.hasNext()) {
                        when (js.nextName()) {
                          "type" -> type = js.nextString()
                          "pattern" -> {
                            js.beginObject()
                            while (js.hasNext()) {
                              if (js.nextName() == "Regex") splitPattern = js.nextString() else js.skipValue()
                            }
                            js.endObject()
                          }
                          "add_prefix_space" -> byteLevelPrefix = js.nextBoolean()
                          else -> js.skipValue()
                        }
                      }
                      js.endObject()
                      check(type == "Split" || type == "ByteLevel") { "tokenizer.json: pre-tokenizer $type" }
                    }
                    js.endArray()
                  }
                  else -> js.skipValue()
                }
              }
              js.endObject()
            }
            "model" -> {
              js.beginObject()
              while (js.hasNext()) {
                when (js.nextName()) {
                  "type" -> modelType = js.nextString()
                  "ignore_merges" -> ignoreMerges = js.nextBoolean()
                  "byte_fallback" -> byteFallback = js.nextBoolean()
                  "vocab" -> {
                    js.beginObject()
                    while (js.hasNext()) {
                      val token = js.nextName()
                      vocab[token] = js.nextInt()
                    }
                    js.endObject()
                  }
                  "merges" -> {
                    js.beginArray()
                    while (js.hasNext()) {
                      if (js.peek() == JsonToken.BEGIN_ARRAY) {
                        js.beginArray()
                        mergeLeft.add(js.nextString())
                        mergeRight.add(js.nextString())
                        js.endArray()
                      } else {
                        val s = js.nextString()
                        val space = s.indexOf(' ', 1)
                        mergeLeft.add(s.substring(0, space))
                        mergeRight.add(s.substring(space + 1))
                      }
                    }
                    js.endArray()
                  }
                  else -> js.skipValue()
                }
              }
              js.endObject()
            }
            else -> js.skipValue()
          }
        }
        js.endObject()
      }
      check(modelType == "BPE" && ignoreMerges && !byteFallback) {
        "tokenizer.json: model $modelType ignore_merges=$ignoreMerges byte_fallback=$byteFallback is not this pipeline"
      }
      check(byteLevelPrefix == false) { "tokenizer.json: ByteLevel add_prefix_space=$byteLevelPrefix" }
      check(splitPattern == EXPECTED_SPLIT) { "tokenizer.json: Split regex differs from the one spelled out here" }

      val merges = HashMap<Long, Long>(mergeLeft.size * 4 / 3 + 1)
      for (rank in mergeLeft.indices) {
        val l = vocab[mergeLeft[rank]] ?: error("merge $rank: unknown ${mergeLeft[rank]}")
        val r = vocab[mergeRight[rank]] ?: error("merge $rank: unknown ${mergeRight[rank]}")
        val merged = vocab[mergeLeft[rank] + mergeRight[rank]] ?: error("merge $rank: no merged token")
        merges[(l.toLong() shl 32) or (r.toLong() and 0xffffffffL)] = (rank.toLong() shl 32) or merged.toLong()
      }
      var size = (vocab.values.maxOrNull() ?: -1) + 1
      for ((_, id, _) in added) size = maxOf(size, id + 1)
      val idToToken = arrayOfNulls<String>(size)
      for ((token, id) in vocab) idToToken[id] = token
      val special = BooleanArray(size)
      val addedIds = HashMap<String, Int>()
      for ((content, id, isSpecial) in added) {
        idToToken[id] = content
        special[id] = isSpecial
        addedIds[content] = id
      }
      val byFirst = added.filter { it.first.isNotEmpty() }.map { Added(it.first, it.second) }
        .groupBy { it.content[0] }.mapValues { (_, v) -> v.sortedByDescending { it.content.length } }
      val byteIds = IntArray(256) { vocab[BYTE_TO_CHAR[it].toString()] ?: error("no byte token for $it") }
      return Lfm3Tokenizer(vocab, merges, byFirst, addedIds, idToToken, special, byteIds)
    }

    /** The regex of LiquidAI/LFM3-Audio-2B's tokenizer.json, as the file writes it. */
    private const val EXPECTED_SPLIT =
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*" +
        "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
  }
}

/** The chat prompts of the repo's chat_template.jinja for the four tasks of this app (one user turn, no history). */
object ChatPrompt {
  const val TTS_SYSTEM = "Perform TTS in American English using a female voice."
  const val INTERLEAVED_SYSTEM = "Respond with interleaved text and audio."
  const val ASR_SYSTEM = "Perform ASR in English."
  const val AUDIO = "<|audio|>"

  /** The assistant turn opened with [prefill] (continue_final_message): the model goes on from that token. */
  fun continued(system: String, user: String, prefill: String): String =
    "<|startoftext|><|im_start|>system\n$system<|im_end|>\n<|im_start|>user\n$user<|im_end|>\n" +
      "<|im_start|>assistant\n$prefill"

  /** The assistant turn opened empty (add_generation_prompt). */
  fun generation(system: String, user: String): String = continued(system, user, "")
}
