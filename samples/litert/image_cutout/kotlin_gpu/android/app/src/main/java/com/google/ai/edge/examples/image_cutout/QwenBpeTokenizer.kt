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

// Follows the tokenizer.json contract of Hugging Face tokenizers (Apache-2.0; see NOTICE).
package com.google.ai.edge.examples.image_cutout

import com.google.gson.stream.JsonReader
import com.google.gson.stream.JsonToken
import java.io.File
import java.io.Reader
import java.text.Normalizer

/**
 * Byte-level BPE tokenizer for the Qwen2 vocabulary of WeVisOne-VL-2B, read from the checkpoint's
 * `tokenizer.json`.
 *
 * The pipeline of the reference `tokenizers` library with add_special_tokens=False: split the raw
 * text on added-token strings (each becomes its own id; all 1,036 are `normalized: false` without
 * strip flags), NFC-normalize the rest, pre-tokenize with the Qwen2 Split regex, map UTF-8 bytes to
 * the byte-level alphabet, apply ranked merges. [decode] keeps the added tokens (`<ref>`, `<box>`,
 * `<0>`..`<1000>`, `<OBJ>`, ...) as their text, as `skip_special_tokens=False` does. The JSON is
 * read with Gson's streaming reader so the same class runs in the JVM tests.
 */
class QwenBpeTokenizer
private constructor(
  private val vocab: HashMap<String, Int>,
  private val ranks: HashMap<Long, Int>,
  private val added: HashMap<String, Int>,
) {
  private val addedLengths: IntArray =
    added.keys.map { it.length }.distinct().sortedDescending().toIntArray()
  private val addedAllStartWithLt = added.keys.all { it.startsWith("<") }
  private val idToToken: Array<String?>
  private val addedIds = HashSet<Int>(added.values)
  private val byteToChar = CharArray(256)
  private val charToByte = HashMap<Char, Int>(512)

  val vocabSize: Int
    get() = vocab.size

  val mergeCount: Int
    get() = ranks.size

  val addedCount: Int
    get() = added.size

  init {
    // GPT-2 byte-to-unicode table: printable bytes map to themselves, the rest to U+0100.. in
    // order.
    val direct = (('!'.code..'~'.code) + ('¡'.code..'¬'.code) + ('®'.code..'ÿ'.code)).toHashSet()
    var next = 256
    for (b in 0 until 256) {
      byteToChar[b] = if (b in direct) b.toChar() else (next++).toChar()
      charToByte[byteToChar[b]] = b
    }
    val maxId = maxOf(vocab.values.maxOrNull() ?: 0, added.values.maxOrNull() ?: 0)
    idToToken = arrayOfNulls(maxId + 1)
    for ((piece, id) in vocab) {
      idToToken[id] = piece
    }
    for ((content, id) in added) {
      idToToken[id] = content
    }
  }

  /**
   * Encodes text to ids (add_special_tokens=False: no BOS/EOS, but added-token strings map to their
   * ids).
   */
  fun encode(text: String): IntArray {
    val out = ArrayList<Int>(text.length / 2 + 8)
    var start = 0
    var i = 0
    while (i < text.length) {
      val hit = if (!addedAllStartWithLt || text[i] == '<') matchAdded(text, i) else null
      if (hit != null) {
        if (i > start) {
          encodePlain(text.substring(start, i), out)
        }
        out.add(hit.second)
        i += hit.first
        start = i
      } else {
        i++
      }
    }
    if (start < text.length) {
      encodePlain(text.substring(start), out)
    }
    return out.toIntArray()
  }

  /** The text of [ids] with every added token kept as its string (`skip_special_tokens=False`). */
  fun decode(ids: IntArray): String {
    val bytes = java.io.ByteArrayOutputStream(ids.size * 4)
    for (id in ids) {
      val piece = idToToken.getOrNull(id) ?: continue
      if (id in addedIds) {
        bytes.write(piece.toByteArray(Charsets.UTF_8))
        continue
      }
      // the ByteLevel decoder: every char of a vocabulary piece is one byte of the byte-level
      // alphabet
      val mapped = piece.map { charToByte[it] }
      if (mapped.all { it != null }) {
        for (b in mapped) {
          bytes.write(b!!)
        }
      } else {
        bytes.write(piece.toByteArray(Charsets.UTF_8))
      }
    }
    return String(bytes.toByteArray(), Charsets.UTF_8)
  }

  /** The id of an added token's string, or null. */
  fun addedId(content: String): Int? = added[content]

  private fun pairKey(a: Int, b: Int): Long = (a.toLong() shl 32) or (b.toLong() and 0xffffffffL)

  /**
   * Longest added token starting at [at] (leftmost-longest, as the reference's added-vocabulary
   * matcher).
   */
  private fun matchAdded(text: String, at: Int): Pair<Int, Int>? {
    for (len in addedLengths) {
      if (at + len > text.length) continue
      val id = added[text.substring(at, at + len)] ?: continue
      return len to id
    }
    return null
  }

  private fun encodePlain(text: String, out: ArrayList<Int>) {
    val normalized = Normalizer.normalize(text, Normalizer.Form.NFC)
    for (match in PRETOKENIZE.findAll(normalized)) {
      val bytes = match.value.toByteArray(Charsets.UTF_8)
      val mapped = StringBuilder(bytes.size)
      for (b in bytes) {
        mapped.append(byteToChar[b.toInt() and 0xFF])
      }
      bpe(mapped.toString(), out)
    }
  }

  private fun bpe(token: String, out: ArrayList<Int>) {
    // The word as piece ids, merged greedily by rank (leftmost occurrence of the best pair first,
    // as the
    // reference's priority queue does).
    val ids = ArrayList<Int>(token.length)
    val pieces = ArrayList<String>(token.length)
    for (c in token) {
      val s = c.toString()
      pieces.add(s)
      ids.add(vocab[s] ?: error("byte piece missing: $s"))
    }
    while (ids.size > 1) {
      var bestRank = Int.MAX_VALUE
      var bestIdx = -1
      for (k in 0 until ids.size - 1) {
        val r = ranks[pairKey(ids[k], ids[k + 1])] ?: continue
        if (r < bestRank) {
          bestRank = r
          bestIdx = k
        }
      }
      if (bestIdx < 0) break
      val merged = pieces[bestIdx] + pieces[bestIdx + 1]
      val mergedId = vocab[merged] ?: error("merged piece not in vocab: $merged")
      pieces[bestIdx] = merged
      ids[bestIdx] = mergedId
      pieces.removeAt(bestIdx + 1)
      ids.removeAt(bestIdx + 1)
    }
    out.addAll(ids)
  }

  companion object {
    // Unicode White_Space, the `\s` of the reference's regex engine (Oniguruma), spelled out so the
    // pattern means
    // the same on the JVM (ASCII `\s`) and on Android (ICU, Unicode `\s`).
    private const val WS =
      "\\t\\n\\u000B\\f\\r \\u0085\\u00A0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000"

    // The Qwen2 pre-tokenization regex of tokenizer.json (pre_tokenizer.Split, behavior Isolated)
    // with `\s` and
    // `\S` written as [WS] and [^WS].
    private val PRETOKENIZE =
      Regex(
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|" +
          " ?[^$WS\\p{L}\\p{N}]+[\\r\\n]*|[$WS]*[\\r\\n]+|[$WS]+(?![^$WS])|[$WS]+"
      )

    fun fromFile(file: File): QwenBpeTokenizer =
      file.bufferedReader(Charsets.UTF_8, 1 shl 16).use { read(it) }

    /**
     * Reads `added_tokens`, `model.vocab` and `model.merges` (a list of [left, right] pairs or
     * "left right").
     */
    fun read(reader: Reader): QwenBpeTokenizer {
      val vocab = HashMap<String, Int>(160_000)
      val added = HashMap<String, Int>()
      val pendingMerges = ArrayList<Pair<String, String>>(160_000)
      JsonReader(reader).use { r ->
        r.beginObject()
        while (r.hasNext()) {
          when (r.nextName()) {
            "added_tokens" -> {
              r.beginArray()
              while (r.hasNext()) {
                var id = -1
                var content: String? = null
                r.beginObject()
                while (r.hasNext()) {
                  when (r.nextName()) {
                    "id" -> id = r.nextInt()
                    "content" -> content = r.nextString()
                    else -> r.skipValue()
                  }
                }
                r.endObject()
                if (content != null && id >= 0) {
                  added[content] = id
                }
              }
              r.endArray()
            }
            "model" -> {
              r.beginObject()
              while (r.hasNext()) {
                when (r.nextName()) {
                  "vocab" -> {
                    r.beginObject()
                    while (r.hasNext()) {
                      val k = r.nextName()
                      vocab[k] = r.nextInt()
                    }
                    r.endObject()
                  }
                  "merges" -> {
                    r.beginArray()
                    while (r.hasNext()) {
                      if (r.peek() == JsonToken.BEGIN_ARRAY) {
                        r.beginArray()
                        val a = r.nextString()
                        val b = r.nextString()
                        r.endArray()
                        pendingMerges.add(a to b)
                      } else {
                        val line = r.nextString()
                        val sp = line.indexOf(' ')
                        pendingMerges.add(line.substring(0, sp) to line.substring(sp + 1))
                      }
                    }
                    r.endArray()
                  }
                  else -> r.skipValue()
                }
              }
              r.endObject()
            }
            else -> r.skipValue()
          }
        }
        r.endObject()
      }
      val ranks = HashMap<Long, Int>(pendingMerges.size * 2)
      for ((rank, m) in pendingMerges.withIndex()) {
        val a = vocab[m.first] ?: error("merge piece not in vocab: ${m.first}")
        val b = vocab[m.second] ?: error("merge piece not in vocab: ${m.second}")
        ranks.putIfAbsent((a.toLong() shl 32) or (b.toLong() and 0xffffffffL), rank)
      }
      return QwenBpeTokenizer(vocab, ranks, added)
    }
  }
}
