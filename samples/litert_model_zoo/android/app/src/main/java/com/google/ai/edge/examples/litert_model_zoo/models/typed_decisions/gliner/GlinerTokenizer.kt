// Vendored from https://huggingface.co/litert-community/GLiNER2.5-Decide-LiteRT/blob/db80197282d11373df084c0ceed67a54544cfa84/android/app/src/main/java/com/gliner25decide/GlinerTokenizer.kt (Apache-2.0)
package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.gliner

import java.io.File
import java.text.Normalizer
import java.util.regex.Pattern
import org.json.JSONObject

/**
 * Supplies the per-word tokenizer used by gliner2 2.0.0
 * `processor.py:SchemaTransformer._format_input_with_mapping`, without JNI.
 *
 * The published `tokenizer.json` defines DeBERTa-v3's SentencePiece Unigram vocabulary,
 * normalization and added tokens. Each schema or text word is encoded independently, without
 * CLS/SEP postprocessing, because routing refers to its first subword. Lattice scores remain
 * doubles to preserve Hugging Face tokenizers' path selection; the 128011 IDs address the published
 * embedding table. This tokenizer has no byte fallback.
 *
 * Copied from the GLiNER2.5 Small sample. One change: GLiNER2.5-Decide's `tokenizer.json` marks
 * `[UNK]` as `normalized: true`, which Small's copy rejected at construction. Such tokens are now
 * matched the way Hugging Face tokenizers' `AddedVocabulary::extract_and_normalize` does it: after
 * the non-normalized tokens are split out, each remaining segment is normalized once, then split on
 * the normalized tokens, and the pieces go to Metaspace without being normalized again (so "a [UNK]
 * b" keeps the space before `[UNK]` as its own U+2581 piece, as in Python).
 */
class GlinerTokenizer(tokenizerJson: File) {
  private data class Piece(val id: Int, val score: Double)

  private val vocabulary = HashMap<String, Piece>(170_000)
  // Matched in the raw text, before normalization (`normalized: false`).
  private val addedTokens = HashMap<String, Int>()
  // Matched in each normalized segment (`normalized: true`), keyed by the normalized content.
  private val normalizedAddedTokens = HashMap<String, Int>()
  private val maxPieceLength: Int
  private val unknownScore: Double
  private val unknownId: Int
  /** ID of `[PAD]`, used for right padding (its embedding row is looked up too). */
  val padId: Int

  /** Number of token IDs: the Unigram vocabulary plus the added tokens. */
  val vocabularySize: Int

  init {
    val json = JSONObject(tokenizerJson.readText())
    val model = json.getJSONObject("model")
    require(model.getString("type") == "Unigram") { "Expected published Unigram tokenizer" }
    require(!model.optBoolean("byte_fallback", false)) {
      "Byte-fallback tokenizers are unsupported"
    }
    validateNormalization(json)
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
    var size = vocab.length()
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
      size = maxOf(size, token.getInt("id") + 1)
    }
    vocabularySize = size
    padId = requireNotNull(addedTokens["[PAD]"])
    maxPieceLength = maxLength
    unknownScore = minScore - 10.0
  }

  /**
   * Encodes one word for `SchemaTransformer._format_input_with_mapping` in gliner2 2.0.0.
   *
   * Returns an unpadded ID vector with no CLS or SEP. Added tokens are matched before the published
   * normalizer so structural markers keep their exact IDs and routing positions.
   */
  fun encodeWord(text: String): IntArray {
    if (text.isEmpty()) {
      return IntArray(0)
    }
    addedTokens[text]?.let {
      return intArrayOf(it)
    }
    // Added tokens are identified before normalization, even inside a word.
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
    // Normalized added tokens split the normalized segment; its pieces are not normalized again.
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
    val escaped = normalized.replace(' ', '\u2581')
    val prefixed =
      if (escaped.startsWith('\u2581')) {
        escaped
      } else {
        "\u2581$escaped"
      }
    // Metaspace split=true starts a new pre-token at every replacement.
    var start = 0
    for (i in 1 until prefixed.length) {
      if (prefixed[i] == '\u2581') {
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
        // Hugging Face Unigram adds UNK only when there is no
        // vocabulary piece for this single Unicode scalar.
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
      // Unigram fuses consecutive unknown characters into one UNK.
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

  private fun validateNormalization(json: JSONObject) {
    val normalizers = json.getJSONObject("normalizer").getJSONArray("normalizers")
    require(
      normalizers.length() == 3 &&
        normalizers.getJSONObject(0).getString("type") == "Replace" &&
        normalizers.getJSONObject(0).getJSONObject("pattern").getString("Regex") ==
          "\\s{2,}|[\\n\\r\\t]" &&
        normalizers.getJSONObject(0).getString("content") == " " &&
        normalizers.getJSONObject(1).getString("type") == "NFC" &&
        normalizers.getJSONObject(2).getString("type") == "Strip" &&
        !normalizers.getJSONObject(2).getBoolean("strip_left") &&
        normalizers.getJSONObject(2).getBoolean("strip_right")
    ) {
      "Tokenizer normalizer differs from the published contract"
    }
    val pre = json.getJSONObject("pre_tokenizer").getJSONArray("pretokenizers")
    require(
      pre.length() == 1 &&
        pre.getJSONObject(0).getString("type") == "Metaspace" &&
        pre.getJSONObject(0).getString("replacement") == "\u2581" &&
        pre.getJSONObject(0).getString("prepend_scheme") == "always" &&
        pre.getJSONObject(0).getBoolean("split")
    ) {
      "Tokenizer pre-tokenizer differs from the published contract"
    }
  }

  companion object {
    // Android rejects UNICODE_CHARACTER_CLASS; spelling out Unicode White_Space keeps the
    // published normalizer identical on Android's ICU regex engine and the desktop JVM.
    private val REPEATED_WHITESPACE =
      Pattern.compile(
        "[\\x{09}-\\x{0d}\\x{20}\\x{85}\\x{a0}\\x{1680}\\x{2000}-\\x{200a}" +
          "\\x{2028}\\x{2029}\\x{202f}\\x{205f}\\x{3000}]{2,}|[\\n\\r\\t]"
      )

    /** The published normalizer: Replace(`\s{2,}|[\n\r\t]` → " "), NFC, Strip(right). */
    private fun normalize(text: String): String {
      val replaced = REPEATED_WHITESPACE.matcher(text).replaceAll(" ")
      return Normalizer.normalize(replaced, Normalizer.Form.NFC).trimEnd {
        isUnicodeWhitespace(it.code)
      }
    }

    /**
     * Aho-Corasick LeftmostLongest, as Hugging Face tokenizers matches added tokens: the earliest
     * start wins, then the longest token at that start. Returns (text.length, null) when none.
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

    // Unicode White_Space, used by Rust's char::is_whitespace for Strip.
    private fun isUnicodeWhitespace(codePoint: Int): Boolean =
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
