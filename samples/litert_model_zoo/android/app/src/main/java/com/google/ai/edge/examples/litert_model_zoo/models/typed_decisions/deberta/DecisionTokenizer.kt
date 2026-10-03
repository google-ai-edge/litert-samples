// Vendored from https://huggingface.co/litert-community/Open-Decision-DeBERTa-v3-Large-LiteRT/blob/7a276235b795e8ad3ae7ac6a9f237daa2098863a/android/sample/app/src/main/java/com/opendecision/DecisionTokenizer.kt (Apache-2.0)
// formatted for this repository's 100-column and brace rules; no logic change
package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.deberta

import java.io.File
import java.text.BreakIterator
import java.util.Base64
import org.json.JSONObject

/**
 * The source checkpoint's `tokenizer.json` (DeBERTa-v3 SentencePiece Unigram, 128,000 pieces + 8
 * added tokens) without JNI, for `tokenizer(text, add_special_tokens=False)` as the author's
 * `Collator` calls it.
 *
 * Pipeline, in Hugging Face tokenizers' order:
 * 1. Added tokens with `normalized: false` (`[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`, `[STATE]`, `[Q]`,
 *    `[OPT]`) are matched in the raw text, leftmost-longest.
 * 2. Each remaining segment is normalized: `Strip` (both sides, Unicode White_Space), `Precompiled`
 *    (the SentencePiece charsmap: a double-array trie over UTF-8 bytes, applied per extended
 *    grapheme cluster shorter than 6 bytes, else per code point; the first common-prefix match
 *    wins), then `Replace` of two or more spaces by one.
 * 3. Added tokens with `normalized: true` (`[UNK]`) are matched in the normalized segment; the
 *    pieces around them are not normalized again.
 * 4. `Metaspace` (replacement U+2581, prepend always, split) and the Unigram Viterbi lattice per
 *    pre-token, with `[UNK]` only for a code point that has no single-piece entry; consecutive
 *    unknowns fuse into one.
 *
 * Lattice scores stay doubles to keep the library's path selection. There is no byte fallback.
 * The Unigram and added-token code follows the GLiNER2.5-Decide sample's port; the normalizer is
 * new.
 */
class DecisionTokenizer(tokenizerJson: File) {
  private data class Piece(val id: Int, val score: Double)

  private val vocabulary = HashMap<String, Piece>(170_000)
  private val addedTokens = HashMap<String, Int>()
  private val normalizedAddedTokens = HashMap<String, Int>()
  private val maxPieceLength: Int
  private val unknownScore: Double
  private val unknownId: Int
  private val charsmap: Precompiled

  /** `[PAD]` id, used for right padding (its table row is looked up too). */
  val padId: Int
  val clsId: Int
  val sepId: Int
  val stateId: Int
  val questionId: Int
  val optionId: Int

  init {
    val json = JSONObject(tokenizerJson.readText())
    val model = json.getJSONObject("model")
    require(model.getString("type") == "Unigram") { "Expected a Unigram tokenizer" }
    require(!model.optBoolean("byte_fallback", false)) {
      "Byte-fallback tokenizers are unsupported"
    }
    charsmap = Precompiled(validateNormalization(json))
    unknownId = model.getInt("unk_id")
    val vocab = model.getJSONArray("vocab")
    var minScore = Double.POSITIVE_INFINITY
    var maxLength = 0
    for (id in 0 until vocab.length()) {
      val entry = vocab.getJSONArray(id)
      val text = entry.getString(0)
      val score = entry.getDouble(1)
      vocabulary[text] = Piece(id, score)
      minScore = minOf(minScore, score)
      maxLength = maxOf(maxLength, text.length)
    }
    val extra = json.getJSONArray("added_tokens")
    for (i in 0 until extra.length()) {
      val token = extra.getJSONObject(i)
      require(
        !token.getBoolean("lstrip") &&
          !token.getBoolean("rstrip") &&
          !token.getBoolean("single_word")
      ) {
        "Unsupported added-token matching policy"
      }
      val content = token.getString("content")
      if (token.getBoolean("normalized")) {
        val normalized = normalize(content)
        require(normalized.isNotEmpty()) { "Added token normalizes to nothing: $content" }
        normalizedAddedTokens[normalized] = token.getInt("id")
      } else {
        addedTokens[content] = token.getInt("id")
      }
    }
    padId = requireNotNull(addedTokens["[PAD]"])
    clsId = requireNotNull(addedTokens["[CLS]"])
    sepId = requireNotNull(addedTokens["[SEP]"])
    stateId = requireNotNull(addedTokens["[STATE]"])
    questionId = requireNotNull(addedTokens["[Q]"])
    optionId = requireNotNull(addedTokens["[OPT]"])
    maxPieceLength = maxLength
    unknownScore = minScore - 10.0
  }

  /**
   * `tokenizer(text, add_special_tokens=False)["input_ids"]`: no CLS or SEP, markers only where
   * typed.
   */
  fun encode(text: String): IntArray {
    if (text.isEmpty()) {
      return IntArray(0)
    }
    addedTokens[text]?.let {
      return intArrayOf(it)
    }
    val result = ArrayList<Int>()
    var from = 0
    while (from < text.length) {
      val (next, match) = leftmostLongest(text, from, addedTokens.keys)
      if (next > from) {
        encodeNormalized(text.substring(from, next), result)
      }
      val special = match ?: break
      result.add(addedTokens.getValue(special))
      from = next + special.length
    }
    return result.toIntArray()
  }

  private fun encodeNormalized(text: String, result: MutableList<Int>) {
    val normalized = normalize(text)
    if (normalized.isEmpty()) {
      return
    }
    var from = 0
    while (from < normalized.length) {
      val (next, match) = leftmostLongest(normalized, from, normalizedAddedTokens.keys)
      if (next > from) {
        encodePreTokenized(normalized.substring(from, next), result)
      }
      val special = match ?: break
      result.add(normalizedAddedTokens.getValue(special))
      from = next + special.length
    }
  }

  private fun encodePreTokenized(normalized: String, result: MutableList<Int>) {
    val escaped = normalized.replace(' ', '▁')
    val prefixed = if (escaped.startsWith('▁')) escaped else "▁$escaped"
    var start = 0
    for (i in 1 until prefixed.length) {
      if (prefixed[i] == '▁') {
        result.addAll(viterbi(prefixed.substring(start, i)))
        start = i
      }
    }
    result.addAll(viterbi(prefixed.substring(start)))
  }

  private fun viterbi(text: String): List<Int> {
    val best = DoubleArray(text.length + 1) { Double.NEGATIVE_INFINITY }
    val backPosition = IntArray(text.length + 1)
    val backId = IntArray(text.length + 1)
    best[0] = 0.0
    var position = 0
    while (position < text.length) {
      val characterLength = Character.charCount(text.codePointAt(position))
      var hasSingleCharacter = false
      if (best[position] != Double.NEGATIVE_INFINITY) {
        val limit = minOf(text.length, position + maxPieceLength)
        for (end in position + 1..limit) {
          val piece = vocabulary[text.substring(position, end)] ?: continue
          if (end - position == characterLength) {
            hasSingleCharacter = true
          }
          val score = best[position] + piece.score
          if (score > best[end]) {
            best[end] = score
            backPosition[end] = position
            backId[end] = piece.id
          }
        }
        if (!hasSingleCharacter) {
          val end = position + characterLength
          val score = best[position] + unknownScore
          if (score > best[end]) {
            best[end] = score
            backPosition[end] = position
            backId[end] = unknownId
          }
        }
      }
      position += characterLength
    }
    val reversed = ArrayList<Int>()
    position = text.length
    while (position > 0) {
      val id = backId[position]
      if (id != unknownId || reversed.lastOrNull() != unknownId) {
        reversed.add(id)
      }
      val previous = backPosition[position]
      check(previous < position) { "Unigram lattice has no path" }
      position = previous
    }
    reversed.reverse()
    return reversed
  }

  /**
   * Strip (both sides, Unicode White_Space), the precompiled charsmap, then runs of spaces
   * collapsed to one.
   */
  fun normalize(text: String): String {
    val stripped = text.trim { isUnicodeWhitespace(it.code) }
    return REPEATED_SPACES.matcher(charsmap.normalize(stripped)).replaceAll(" ")
  }

  private fun validateNormalization(json: JSONObject): ByteArray {
    val normalizers = json.getJSONObject("normalizer").getJSONArray("normalizers")
    require(
      normalizers.length() == 3 &&
        normalizers.getJSONObject(0).getString("type") == "Strip" &&
        normalizers.getJSONObject(0).getBoolean("strip_left") &&
        normalizers.getJSONObject(0).getBoolean("strip_right") &&
        normalizers.getJSONObject(1).getString("type") == "Precompiled" &&
        normalizers.getJSONObject(2).getString("type") == "Replace" &&
        normalizers.getJSONObject(2).getJSONObject("pattern").getString("Regex") == " {2,}" &&
        normalizers.getJSONObject(2).getString("content") == " "
    ) {
      "Tokenizer normalizer differs from the published contract"
    }
    val pre = json.getJSONObject("pre_tokenizer").getJSONArray("pretokenizers")
    require(
      pre.length() == 1 &&
        pre.getJSONObject(0).getString("type") == "Metaspace" &&
        pre.getJSONObject(0).getString("replacement") == "▁" &&
        pre.getJSONObject(0).getString("prepend_scheme") == "always" &&
        pre.getJSONObject(0).getBoolean("split")
    ) {
      "Tokenizer pre-tokenizer differs from the published contract"
    }
    return Base64.getDecoder()
      .decode(normalizers.getJSONObject(1).getString("precompiled_charsmap"))
  }

  /**
   * SentencePiece's precompiled charsmap as Hugging Face tokenizers (crate spm_precompiled 0.1.4)
   * reads it: a little-endian u32 trie size, the darts-clone double array, then the NUL-separated
   * normalized strings.
   */
  class Precompiled(blob: ByteArray) {
    private val array: IntArray
    private val normalized: ByteArray

    init {
      val trieSize = readU32(blob, 0)
      require(trieSize % 4 == 0 && 4 + trieSize <= blob.size) {
        "Cannot parse precompiled_charsmap"
      }
      array = IntArray(trieSize / 4) { readU32(blob, 4 + it * 4) }
      normalized = blob.copyOfRange(4 + trieSize, blob.size)
    }

    fun normalize(text: String): String {
      val out = StringBuilder(text.length)
      val graphemes = BreakIterator.getCharacterInstance()
      graphemes.setText(text)
      var start = graphemes.first()
      var end = graphemes.next()
      while (end != BreakIterator.DONE) {
        val grapheme = text.substring(start, end)
        val bytes = grapheme.toByteArray(Charsets.UTF_8)
        val whole = if (bytes.size < 6) transform(bytes) else null
        if (whole != null) {
          out.append(whole)
        } else {
          var i = 0
          while (i < grapheme.length) {
            val length = Character.charCount(grapheme.codePointAt(i))
            val part = grapheme.substring(i, i + length)
            out.append(transform(part.toByteArray(Charsets.UTF_8)) ?: part)
            i += length
          }
        }
        start = end
        end = graphemes.next()
      }
      return out.toString()
    }

    /** The normalized string of the FIRST common-prefix match of [key], or null. */
    private fun transform(key: ByteArray): String? {
      var nodePos = 0
      var unit = array[nodePos]
      nodePos = nodePos xor offset(unit)
      for (b in key) {
        val c = b.toInt() and 0xff
        if (c == 0) {
          break
        }
        nodePos = nodePos xor c
        unit = array[nodePos]
        if (label(unit) != c) {
          return null
        }
        nodePos = nodePos xor offset(unit)
        if (hasLeaf(unit)) {
          val index = value(array[nodePos])
          var stop = index
          while (stop < normalized.size && normalized[stop] != 0.toByte()) {
            stop++
          }
          return String(normalized, index, stop - index, Charsets.UTF_8)
        }
      }
      return null
    }

    private fun hasLeaf(unit: Int) = (unit ushr 8) and 1 == 1

    private fun value(unit: Int) = unit and 0x7fffffff

    private fun label(unit: Int) = unit and (0x80000000.toInt() or 0xff)

    private fun offset(unit: Int) = (unit ushr 10) shl ((unit and (1 shl 9)) ushr 6)

    private fun readU32(bytes: ByteArray, at: Int): Int =
      (bytes[at].toInt() and 0xff) or
        ((bytes[at + 1].toInt() and 0xff) shl 8) or
        ((bytes[at + 2].toInt() and 0xff) shl 16) or
        ((bytes[at + 3].toInt() and 0xff) shl 24)
  }

  companion object {
    private val REPEATED_SPACES = java.util.regex.Pattern.compile(" {2,}")

    /**
     * Aho-Corasick LeftmostLongest as the library matches added tokens: earliest start, then
     * longest.
     */
    private fun leftmostLongest(text: String, from: Int, tokens: Set<String>): Pair<Int, String?> {
      var next = text.length
      var match: String? = null
      for (token in tokens) {
        val position = text.indexOf(token, from)
        if (
          position >= 0 &&
            (position < next || (position == next && token.length > (match?.length ?: 0)))
        ) {
          next = position
          match = token
        }
      }
      return next to match
    }

    // Unicode White_Space, as Rust's char::is_whitespace used by the Strip normalizer.
    fun isUnicodeWhitespace(codePoint: Int): Boolean =
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
