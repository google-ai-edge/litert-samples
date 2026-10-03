// Vendored from https://huggingface.co/litert-community/Julia-1-LiteRT/blob/8f36857c56e891c023060586759c6cdc8baf6b3e/android/app/src/main/java/com/julia1/JuliaTokenizer.kt (Apache-2.0)
// formatted for this repository's 100-column and brace rules; no logic change
// SPDX-License-Identifier: Apache-2.0
package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.julia

import java.io.File
import java.util.PriorityQueue

/**
 * Encoder for the Julia-1 tokenizer (the Gemma-style BPE tokenizer.json inherited from
 * mmBERT-small; the same file as the multilingual Laya sample); no native or Android APIs.
 */
class JuliaTokenizer(file: File) {
  private data class AddedToken(
    val content: String,
    val id: Int,
    val leftStrip: Boolean,
    val rightStrip: Boolean,
  )

  private class TrieNode {
    val children = HashMap<Char, TrieNode>()
    var token: AddedToken? = null
  }

  private data class Merge(val rank: Int, val id: Int)

  private data class Candidate(
    val rank: Int,
    val left: Int,
    val right: Int,
    val leftId: Int,
    val rightId: Int,
    val id: Int,
  )

  private val vocabulary: Map<String, Int>
  private val merges: Map<Long, Merge>
  private val addedRoot = TrieNode()
  private val byteIds = IntArray(256)
  /** JSON parsing and index construction time, excluding encoding. */
  val loadTimeMs: Double
  /** Number of vocabulary entries checked against the pinned tokenizer. */
  val vocabularySize: Int
    get() = vocabulary.size

  /** Number of ranked BPE merge pairs loaded from tokenizer.json. */
  val mergeCount: Int
    get() = merges.size

  init {
    val start = System.nanoTime()
    @Suppress("UNCHECKED_CAST") val root = JuliaJson.parse(file) as Map<String, Any?>
    @Suppress("UNCHECKED_CAST") val model = root.getValue("model") as Map<String, Any?>
    require(model["type"] == "BPE" && model["byte_fallback"] == true)
    require(
      model["dropout"] == null &&
        model["continuing_subword_prefix"] == null &&
        model["end_of_word_suffix"] == null
    )
    require(model["ignore_merges"] == false && model["unk_token"] == "<unk>")
    @Suppress("UNCHECKED_CAST") val normalizer = root.getValue("normalizer") as Map<String, Any?>
    require(normalizer["type"] == "Replace" && normalizer["content"] == "▁")
    require((normalizer["pattern"] as Map<*, *>)["String"] == " ")
    @Suppress("UNCHECKED_CAST")
    val preTokenizer = root.getValue("pre_tokenizer") as Map<String, Any?>
    require(preTokenizer["type"] == "Metaspace" && preTokenizer["replacement"] == "▁")
    require(preTokenizer["prepend_scheme"] == "always" && preTokenizer["split"] == true)
    val rawVocabulary = model.getValue("vocab") as Map<*, *>
    vocabulary =
      HashMap<String, Int>(rawVocabulary.size * 4 / 3 + 1).also { values ->
        for ((token, id) in rawVocabulary) {
          values[token as String] = (id as Number).toInt()
        }
      }
    require(vocabulary.size == 256000)
    require(
      vocabulary["<pad>"] == 0 &&
        vocabulary["<eos>"] == 1 &&
        vocabulary["<bos>"] == 2 &&
        vocabulary["<unk>"] == 3 &&
        vocabulary["<mask>"] == 4
    )
    // This checkpoint has 255 byte tokens: TAB has its own scalar token (226), no <0x09>.
    for (i in byteIds.indices) {
      byteIds[i] =
        vocabulary["<0x" + i.toString(16).uppercase().padStart(2, '0') + ">"] ?: -1
    }
    val rawMerges = model.getValue("merges") as List<*>
    merges =
      HashMap<Long, Merge>(rawMerges.size * 4 / 3 + 1).also { ranks ->
        for ((rank, entry) in rawMerges.withIndex()) {
          val pair = entry as List<*>
          val left = pair[0] as String
          val right = pair[1] as String
          ranks[key(vocabulary.getValue(left), vocabulary.getValue(right))] =
            Merge(rank, vocabulary.getValue(left + right))
        }
      }
    for (value in root.getValue("added_tokens") as List<*>) {
      val entry = value as Map<*, *>
      require(entry["normalized"] == false && entry["single_word"] == false)
      val token =
        AddedToken(
          entry["content"] as String,
          (entry["id"] as Number).toInt(),
          entry["lstrip"] == true,
          entry["rstrip"] == true,
        )
      var node = addedRoot
      for (character in token.content) {
        node = node.children.getOrPut(character) { TrieNode() }
      }
      node.token = token
    }
    loadTimeMs = (System.nanoTime() - start) / 1e6
  }

  /** Added tokens match before normalization, then each remaining segment uses Metaspace/BPE. */
  fun encode(text: String, addSpecialTokens: Boolean = false): IntArray {
    val result = ArrayList<Int>()
    if (addSpecialTokens) {
      result.add(2)
    }
    var unprocessed = 0
    var position = 0
    while (position < text.length) {
      var node = addedRoot
      var end = position
      var matched: AddedToken? = null
      var matchedEnd = position
      while (end < text.length) {
        node = node.children[text[end]] ?: break
        end++
        node.token?.let {
          matched = it
          matchedEnd = end
        }
      }
      val token = matched
      if (token == null) {
        position++
        continue
      }
      var begin = position
      if (token.leftStrip) {
        while (begin > unprocessed && isRustWhitespace(text.codePointBefore(begin))) {
          begin -= Character.charCount(text.codePointBefore(begin))
        }
      }
      encodeOrdinary(text.substring(unprocessed, begin), result)
      result.add(token.id)
      if (token.rightStrip) {
        while (matchedEnd < text.length && isRustWhitespace(text.codePointAt(matchedEnd))) {
          matchedEnd += Character.charCount(text.codePointAt(matchedEnd))
        }
      }
      position = matchedEnd
      unprocessed = matchedEnd
    }
    encodeOrdinary(text.substring(unprocessed), result)
    if (addSpecialTokens) {
      result.add(1)
    }
    return result.toIntArray()
  }

  private fun encodeOrdinary(text: String, output: MutableList<Int>) {
    if (text.isEmpty()) return
    var normalized = text.replace(' ', '▁')
    if (!normalized.startsWith('▁')) {
      normalized = "▁$normalized"
    }
    var begin = 0
    for (index in 1 until normalized.length) {
      if (normalized[index] == '▁') {
        bpe(normalized.substring(begin, index), output)
        begin = index
      }
    }
    bpe(normalized.substring(begin), output)
  }

  private fun bpe(piece: String, output: MutableList<Int>) {
    val symbols = ArrayList<Int>()
    var offset = 0
    while (offset < piece.length) {
      val codePoint = piece.codePointAt(offset)
      val character = String(Character.toChars(codePoint))
      val id = vocabulary[character]
      if (id != null) {
        symbols.add(id)
      } else {
        for (byte in character.toByteArray(Charsets.UTF_8)) {
          val fallback = byteIds[byte.toInt() and 255]
          check(fallback >= 0) {
            "Missing byte fallback for unknown scalar U+${codePoint.toString(16)}"
          }
          symbols.add(fallback)
        }
      }
      offset += Character.charCount(codePoint)
    }
    if (symbols.size < 2) {
      output.addAll(symbols)
      return
    }
    val ids = symbols.toIntArray()
    val previous = IntArray(ids.size) { it - 1 }
    val next = IntArray(ids.size) { if (it + 1 < ids.size) it + 1 else -1 }
    val alive = BooleanArray(ids.size) { true }
    val candidates = PriorityQueue<Candidate>(compareBy<Candidate> { it.rank }.thenBy { it.left })
    fun offer(left: Int) {
      if (left < 0 || !alive[left]) return
      val right = next[left]
      if (right < 0) return
      val merge = merges[key(ids[left], ids[right])] ?: return
      candidates.add(Candidate(merge.rank, left, right, ids[left], ids[right], merge.id))
    }
    for (index in 0 until ids.size - 1) {
      offer(index)
    }
    while (candidates.isNotEmpty()) {
      val candidate = candidates.remove()
      val left = candidate.left
      val right = candidate.right
      if (
        !alive[left] ||
          !alive[right] ||
          next[left] != right ||
          ids[left] != candidate.leftId ||
          ids[right] != candidate.rightId
      ) {
        continue
      }
      ids[left] = candidate.id
      alive[right] = false
      next[left] = next[right]
      if (next[right] >= 0) {
        previous[next[right]] = left
      }
      offer(previous[left])
      offer(left)
    }
    for (index in ids.indices) {
      if (alive[index]) {
        output.add(ids[index])
      }
    }
  }

  private fun key(left: Int, right: Int): Long =
    (left.toLong() shl 32) or (right.toLong() and 0xffffffffL)

  // Rust char::is_whitespace follows Unicode White_Space, unlike Java isWhitespace alone.
  private fun isRustWhitespace(cp: Int): Boolean =
    cp in 0x09..0x0d ||
      cp == 0x20 ||
      cp == 0x85 ||
      cp == 0xa0 ||
      cp == 0x1680 ||
      cp in 0x2000..0x200a ||
      cp == 0x2028 ||
      cp == 0x2029 ||
      cp == 0x202f ||
      cp == 0x205f ||
      cp == 0x3000
}
