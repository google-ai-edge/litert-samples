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

// Vendored from https://huggingface.co/litert-community/GLiClass-Edge-v3.0-LiteRT/blob/88c90950587eb951974c094eef91afa0fe3552c0/android/sample/app/src/main/java/com/gliclass/GliclassTokenizer.kt (Apache-2.0)
package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.gliclass

import java.io.File
import java.text.Normalizer
import java.util.PriorityQueue
import java.util.regex.Pattern
import org.json.JSONArray
import org.json.JSONObject

/**
 * Kotlin port of the Hugging Face `tokenizers` pipeline that GLiClass-Edge v3.0's `tokenizer.json`
 * defines (ModernBERT's byte-level BPE), without JNI. `encode` returns `[CLS] … [SEP]` like
 * `Tokenizer.encode(text)` with special tokens, in the library's order:
 * 1. split the raw text on the added tokens marked `normalized: false` (`<<LABEL>>`, `<<SEP>>`,
 *    `[CLS]`, …; `[MASK]` also takes the whitespace on its left);
 * 2. NFC-normalize each remaining segment, then split it on the added tokens marked `normalized:
 *    true` (the 2–24 space runs, `[unused0]`–`[unused82]`, `|||…|||`);
 * 3. per remaining split: prepend one space unless it starts with U+0020 (`add_prefix_space`
 *    applies to every split, so the prompt after `<<SEP>>` starts with `Ġ` while text glued to the
 *    prompt does not), split with the GPT-2 regex, map UTF-8 bytes to GPT-2's byte characters and
 *    apply the BPE merges by rank;
 * 4. add `[CLS]` and `[SEP]`.
 *
 * Added tokens are matched leftmost-longest. The regex spells out the whitespace class because
 * Java's `\s` is ASCII-only and Android rejects `UNICODE_CHARACTER_CLASS`; the class is onig's
 * Unicode `\s` (Unicode White_Space).
 */
class GliclassTokenizer(tokenizerJson: File) {
  private class AddedToken(val id: Int, val leftStrip: Boolean)

  private class TrieNode {
    val children = HashMap<Char, TrieNode>()
    var token: AddedToken? = null
  }

  private class Merge(val rank: Int, val id: Int)

  private class Candidate(
    val rank: Int,
    val left: Int,
    val right: Int,
    val leftId: Int,
    val rightId: Int,
    val id: Int,
  )

  private val vocabulary: HashMap<String, Int>
  private val merges: HashMap<Long, Merge>
  private val addedIds = HashMap<String, Int>()
  // `normalized: false`: matched in the raw text.
  private val rawTokens = TrieNode()
  // `normalized: true`: matched in each NFC-normalized segment.
  private val normalizedTokens = TrieNode()
  // Vocabulary ID of each byte's single GPT-2 character; -1 if the vocabulary has none.
  private val byteIds = IntArray(BYTE_VALUES)

  /** `[CLS]`, added first. */
  val clsId: Int

  /** `[SEP]`, added last. */
  val sepId: Int

  /** `[PAD]`, used for right padding (its embedding row is looked up too). */
  val padId: Int

  /** Number of token IDs: the BPE vocabulary plus the added tokens. */
  val vocabularySize: Int

  init {
    val json = JSONObject(tokenizerJson.readText())
    validatePipeline(json)
    val model = json.getJSONObject("model")
    val vocab = model.getJSONObject("vocab")
    vocabulary = HashMap(hashMapCapacity(vocab.length()))
    for (token in vocab.keys()) {
      vocabulary[token] = vocab.getInt(token)
    }
    val bytes = byteCharacters()
    for (byte in 0 until BYTE_VALUES) {
      byteIds[byte] = vocabulary[bytes[byte].toString()] ?: -1
    }
    val rawMerges = model.getJSONArray("merges")
    merges = HashMap(hashMapCapacity(rawMerges.length()))
    for (rank in 0 until rawMerges.length()) {
      val (left, right) = mergePair(rawMerges.get(rank))
      val leftId = requireNotNull(vocabulary[left]) { "Merge $rank: unknown token $left" }
      val rightId = requireNotNull(vocabulary[right]) { "Merge $rank: unknown token $right" }
      val merged = requireNotNull(vocabulary[left + right]) { "Merge $rank: no token $left$right" }
      // As tokenizers' merge map: a repeated pair would keep its last rank.
      merges[key(leftId, rightId)] = Merge(rank, merged)
    }
    var size = vocabulary.values.maxOrNull()?.plus(1) ?: 0
    val added = json.getJSONArray("added_tokens")
    for (index in 0 until added.length()) {
      val entry = added.getJSONObject(index)
      require(!entry.getBoolean("single_word") && !entry.getBoolean("rstrip")) {
        "Unsupported added-token matching policy: ${entry.getString("content")}"
      }
      val content = entry.getString("content")
      val id = entry.getInt("id")
      val token = AddedToken(id, entry.getBoolean("lstrip"))
      if (entry.getBoolean("normalized")) {
        insert(normalizedTokens, normalize(content), token)
      } else {
        insert(rawTokens, content, token)
      }
      addedIds[content] = id
      size = maxOf(size, id + 1)
    }
    vocabularySize = size
    clsId = requireNotNull(addedIds["[CLS]"])
    sepId = requireNotNull(addedIds["[SEP]"])
    padId = requireNotNull(addedIds["[PAD]"])
    validateTemplate(json.getJSONObject("post_processor"))
  }

  /** ID of an added token or vocabulary entry, or null. */
  fun tokenId(token: String): Int? = addedIds[token] ?: vocabulary[token]

  /** `[CLS]` + token IDs of [text] + `[SEP]`, as `Tokenizer.encode(text).ids`. */
  fun encode(text: String): IntArray {
    val ids = ArrayList<Int>(text.length / CHARACTERS_PER_TOKEN_ESTIMATE + SPECIAL_TOKEN_COUNT)
    ids.add(clsId)
    split(
      text,
      rawTokens,
      { segment ->
        split(
          normalize(segment),
          normalizedTokens,
          { piece -> byteLevel(piece, ids) },
          { ids.add(it) },
        )
      },
    ) {
      ids.add(it)
    }
    ids.add(sepId)
    return ids.toIntArray()
  }

  /**
   * Splits [text] on the tokens of [root] (leftmost-longest, non-overlapping) and reports the
   * non-empty text between them and each token's ID in order. A token with `lstrip` also takes the
   * whitespace to its left, but never text before the previous token.
   */
  private fun split(
    text: String,
    root: TrieNode,
    onText: (String) -> Unit,
    onToken: (Int) -> Unit,
  ) {
    var unprocessed = 0
    var position = 0
    while (position < text.length) {
      var node = root
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
        while (begin > unprocessed && isWhitespace(text.codePointBefore(begin))) {
          begin -= Character.charCount(text.codePointBefore(begin))
        }
      }
      if (begin > unprocessed) {
        onText(text.substring(unprocessed, begin))
      }
      onToken(token.id)
      position = matchedEnd
      unprocessed = matchedEnd
    }
    if (unprocessed < text.length) {
      onText(text.substring(unprocessed))
    }
  }

  /** ByteLevel pre-tokenizer (prefix space, GPT-2 regex) followed by BPE on every piece. */
  private fun byteLevel(split: String, output: MutableList<Int>) {
    val text = if (split.startsWith(' ')) split else " $split"
    val matcher = SPLIT_PATTERN.matcher(text)
    var last = 0
    while (matcher.find()) {
      if (matcher.start() > last) {
        bpe(text.substring(last, matcher.start()), output)
      }
      bpe(matcher.group(), output)
      last = matcher.end()
    }
    if (last < text.length) {
      bpe(text.substring(last), output)
    }
  }

  /**
   * BPE on the UTF-8 bytes of one piece: start from each byte's character, then repeatedly apply
   * the lowest-rank merge, leftmost first, as `tokenizers`' `Word::merge_all`. A byte without a
   * vocabulary character is dropped (the model has no unknown token); valid UTF-8 never has one.
   */
  private fun bpe(piece: String, output: MutableList<Int>) {
    val bytes = piece.toByteArray(Charsets.UTF_8)
    val symbols = ArrayList<Int>(bytes.size)
    for (byte in bytes) {
      val id = byteIds[byte.toInt() and 0xff]
      if (id >= 0) {
        symbols.add(id)
      }
    }
    if (symbols.size < 2) {
      output.addAll(symbols)
      return
    }
    val ids = symbols.toIntArray()
    val previous = IntArray(ids.size) { it - 1 }
    val next = IntArray(ids.size) { if (it + 1 < ids.size) it + 1 else -1 }
    val alive = BooleanArray(ids.size) { true }
    val queue = PriorityQueue<Candidate>(compareBy<Candidate> { it.rank }.thenBy { it.left })
    fun offer(left: Int) {
      if (left < 0 || !alive[left]) {
        return
      }
      val right = next[left]
      if (right < 0) {
        return
      }
      val merge = merges[key(ids[left], ids[right])] ?: return
      queue.add(Candidate(merge.rank, left, right, ids[left], ids[right], merge.id))
    }
    for (index in 0 until ids.size - 1) {
      offer(index)
    }
    while (queue.isNotEmpty()) {
      val candidate = queue.remove()
      val left = candidate.left
      val right = candidate.right
      // Skip entries made stale by an earlier merge on either side.
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

  private fun validatePipeline(json: JSONObject) {
    val model = json.getJSONObject("model")
    require(
      model.getString("type") == "BPE" &&
        model.isNull("dropout") &&
        model.isNull("unk_token") &&
        model.isNull("continuing_subword_prefix") &&
        model.isNull("end_of_word_suffix") &&
        !model.optBoolean("byte_fallback", false) &&
        !model.optBoolean("ignore_merges", false)
    ) {
      "tokenizer.json model differs from the GLiClass-Edge byte-level BPE contract"
    }
    require(json.getJSONObject("normalizer").getString("type") == "NFC") {
      "tokenizer.json normalizer is not NFC"
    }
    val pre = json.getJSONObject("pre_tokenizer")
    require(
      pre.getString("type") == "ByteLevel" &&
        pre.getBoolean("add_prefix_space") &&
        pre.getBoolean("use_regex")
    ) {
      "tokenizer.json pre-tokenizer differs from ByteLevel(add_prefix_space, use_regex)"
    }
  }

  private fun validateTemplate(post: JSONObject) {
    require(post.getString("type") == "TemplateProcessing") { "Unexpected post-processor" }
    val single = post.getJSONArray("single")
    val pieces =
      (0 until single.length()).map { index ->
        val item = single.getJSONObject(index)
        if (item.has("SpecialToken")) {
          item.getJSONObject("SpecialToken").getString("id")
        } else {
          "$" + item.getJSONObject("Sequence").getString("id")
        }
      }
    require(pieces == listOf("[CLS]", "\$A", "[SEP]")) { "Template is not [CLS] A [SEP]: $pieces" }
    val special = post.getJSONObject("special_tokens")
    require(
      special.getJSONObject("[CLS]").getJSONArray("ids").getInt(0) == clsId &&
        special.getJSONObject("[SEP]").getJSONArray("ids").getInt(0) == sepId
    ) {
      "Template special-token IDs differ from the added tokens"
    }
  }

  companion object {
    /** Distinct byte values; GPT-2 also maps unprintable bytes to code points from here up. */
    private const val BYTE_VALUES = 256

    /** Initial size of the ID list: about one token per three characters of English text. */
    private const val CHARACTERS_PER_TOKEN_ESTIMATE = 3

    /** `[CLS]` and `[SEP]`. */
    private const val SPECIAL_TOKEN_COUNT = 2

    // onig's Unicode \s = Unicode White_Space; also Rust's char::is_whitespace for lstrip.
    private const val WHITESPACE =
      "\\x{09}-\\x{0d}\\x{20}\\x{85}\\x{a0}\\x{1680}\\x{2000}-\\x{200a}" +
        "\\x{2028}\\x{2029}\\x{202f}\\x{205f}\\x{3000}"

    /** GPT-2's split regex with `\s` / `\S` spelled out for java.util.regex and Android ICU. */
    private val SPLIT_PATTERN: Pattern =
      Pattern.compile(
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^$WHITESPACE\\p{L}\\p{N}]+" +
          "|[$WHITESPACE]+(?![^$WHITESPACE])|[$WHITESPACE]+"
      )

    private fun normalize(text: String): String = Normalizer.normalize(text, Normalizer.Form.NFC)

    private fun insert(root: TrieNode, content: String, token: AddedToken) {
      require(content.isNotEmpty()) { "Empty added token" }
      var node = root
      for (character in content) {
        node = node.children.getOrPut(character) { TrieNode() }
      }
      node.token = token
    }

    /** A merge written as "left right" or as ["left", "right"]. */
    private fun mergePair(entry: Any): Pair<String, String> =
      when (entry) {
        is JSONArray -> {
          require(entry.length() == 2) { "Merge is not a pair: $entry" }
          entry.getString(0) to entry.getString(1)
        }
        is String -> {
          val space = entry.indexOf(' ', 1)
          require(space > 0) { "Merge is not a pair: $entry" }
          entry.substring(0, space) to entry.substring(space + 1)
        }
        else -> throw IllegalArgumentException("Unsupported merge entry: $entry")
      }

    /** HashMap capacity that holds [entries] without a rehash at the default load factor 0.75. */
    private fun hashMapCapacity(entries: Int): Int = entries * 4 / 3 + 1

    private fun key(left: Int, right: Int): Long =
      (left.toLong() shl 32) or (right.toLong() and 0xffffffffL)

    /** GPT-2 `bytes_to_unicode`: every byte maps to one printable character. */
    private fun byteCharacters(): CharArray {
      val printable = (0x21..0x7e) + (0xa1..0xac) + (0xae..0xff)
      val map = CharArray(BYTE_VALUES)
      var extra = 0
      for (byte in 0 until BYTE_VALUES) {
        map[byte] =
          if (byte in printable) {
            byte.toChar()
          } else {
            (BYTE_VALUES + extra++).toChar()
          }
      }
      return map
    }

    private fun isWhitespace(codePoint: Int): Boolean =
      codePoint in 0x09..0x0d ||
        codePoint == 0x20 ||
        codePoint == 0x85 ||
        codePoint == 0xa0 ||
        codePoint == 0x1680 ||
        codePoint in 0x2000..0x200a ||
        codePoint == 0x2028 ||
        codePoint == 0x2029 ||
        codePoint == 0x202f ||
        codePoint == 0x205f ||
        codePoint == 0x3000
  }
}
