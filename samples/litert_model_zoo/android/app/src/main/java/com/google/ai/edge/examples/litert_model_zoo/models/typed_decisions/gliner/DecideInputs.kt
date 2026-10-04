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

// Vendored from https://huggingface.co/litert-community/GLiNER2.5-Decide-LiteRT/blob/db80197282d11373df084c0ceed67a54544cfa84/android/app/src/main/java/com/gliner25decide/DecideInputs.kt (Apache-2.0)
// One change from the source: EmbeddingTable.lookup reads each row with one bulk read into
// reused arrays.
package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.gliner

import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.ShortBuffer
import java.nio.channels.FileChannel
import java.util.Locale
import java.util.regex.Pattern

/**
 * Ports the classification input path of gliner2 2.0.0 (`processor.py:_collate_batch`,
 * `_transform_record`, `_format_input_with_mapping`) and `conversion/decide_graph.py:fixed_inputs`
 * in the model repository.
 *
 * Encoded sequence = per task `( [P] prompt_str ( [L] label … ) )`, tasks joined by `[SEP_STRUCT]`,
 * then `[SEP_TEXT]` and the lower-cased text words; every token is tokenized on its own, with no
 * CLS/SEP. The graph inputs are float32 embeddings `[1,N,1024]`, attention `[1,N]` and label
 * routing `[1,32,N]` (row j one-hot at the j-th `[L]` marker in task order, label order), flattened
 * row-major. N is the smallest of 128/256/512 that holds the sequence; longer inputs and more than
 * 32 labels are rejected, never truncated.
 */
class DecideInputs(private val tokenizer: GlinerTokenizer) {
  /**
   * A gliner2 2.0.0 `processing/word_splitter.py:WhitespaceTokenSplitter` token. Lowercasing
   * changes only [text]; half-open [start]/[end] offsets count Unicode code points in the original
   * string, so Java UTF-16 indices must not be substituted.
   */
  data class Word(val text: String, val start: Int, val end: Int)

  /**
   * Window-independent encoding: unpadded IDs, each task's `[P]`-then-`[L]` positions
   * (`batch.schema_special_indices[0]`) and the `[L]` positions in request order.
   */
  data class Encoded(
    val text: String,
    val words: List<Word>,
    val schemaTokens: List<List<String>>,
    val inputIds: IntArray,
    val schemaSpecialPositions: List<IntArray>,
    val labelPositions: IntArray,
  ) {
    val encodedLength: Int
      get() = inputIds.size

    val labelCount: Int
      get() = labelPositions.size
  }

  /** Graph-ready inputs for one window. [inputIds] and [attentionMask] have N entries. */
  data class Prepared(
    val encoded: Encoded,
    val window: Int,
    val inputIds: IntArray,
    val attentionMask: FloatArray,
    val labelRouting: FloatArray,
  )

  /**
   * gliner2 2.0.0 `_collate_batch` + `_transform_record` + `_format_input_with_mapping` for a
   * classification-only schema (no classification prefix, no truncation).
   */
  fun encode(text: String, tasks: List<Task>): Encoded {
    DecideSchema.validate(tasks, LABEL_SLOTS)
    val collated = collateText(text)
    val words = splitWords(collated)
    val schemaTokens = tasks.map { DecideSchema.schemaTokens(it) }
    val combined = ArrayList<String>()
    for (struct in schemaTokens) {
      combined.addAll(struct)
      combined.add(DecideSchema.SEP_STRUCT)
    }
    if (combined.isNotEmpty()) {
      combined.removeAt(combined.size - 1)
    }
    combined.add(DecideSchema.SEP_TEXT)
    words.forEach { combined.add(it.text) }

    // Only structural slots are routed: [P] at offset+1 and the [L] slots 4, 6, … before ") )".
    // Prompt or label text that itself tokenizes to a marker ID is never counted as a marker.
    val markerIndices = HashSet<Int>()
    var offset = 0
    for (struct in schemaTokens) {
      if (struct.size > 1) {
        markerIndices.add(offset + 1)
      }
      for (index in 4 until struct.size - 2 step 2) {
        markerIndices.add(offset + index)
      }
      offset += struct.size + 1
    }

    val ids = ArrayList<Int>()
    val special = List(tasks.size) { ArrayList<Int>() }
    var currentSchema = 0
    var foundSeparator = false
    for ((originalIndex, token) in combined.withIndex()) {
      var schemaIndex = -1
      if (token == DecideSchema.SEP_TEXT) {
        foundSeparator = true
      } else if (!foundSeparator) {
        schemaIndex = currentSchema
        if (token == DecideSchema.SEP_STRUCT) {
          currentSchema++
        }
      }
      val position = ids.size
      tokenizer.encodeWord(token).forEach { ids.add(it) }
      if (schemaIndex >= 0 && originalIndex in markerIndices) {
        require(schemaIndex < tasks.size) { MARKER_TEXT_MESSAGE }
        special[schemaIndex].add(position)
      }
    }
    // A label or prompt equal to "[SEP_STRUCT]"/"[SEP_TEXT]" shifts gliner2's segment bookkeeping;
    // reject instead of routing labels to the wrong head.
    tasks.forEachIndexed { index, task ->
      require(special[index].size == task.labels.size + 1) { MARKER_TEXT_MESSAGE }
    }
    return Encoded(
      collated,
      words,
      schemaTokens,
      ids.toIntArray(),
      special.map { it.toIntArray() },
      special.flatMap { it.drop(1) }.toIntArray(),
    )
  }

  /**
   * [encode] plus `fixed_inputs` padding for [window], or for the smallest fitting window when
   * [window] is null. Padding uses the tokenizer's PAD ID, zero attention and zero routing rows.
   */
  fun prepare(text: String, tasks: List<Task>, window: Int? = null): Prepared =
    pad(encode(text, tasks), window)

  /** Pads an [Encoded] request to [window] (or the smallest fitting window). */
  fun pad(encoded: Encoded, window: Int? = null): Prepared {
    val n =
      window
        ?: WINDOWS.firstOrNull { encoded.encodedLength <= it }
        ?: throw IllegalArgumentException(
          "Input is ${encoded.encodedLength} encoded tokens; the largest graph holds " +
            "${WINDOWS.last()}. Shorten the text or the task list."
        )
    require(n in WINDOWS) { "Unsupported graph window $n" }
    require(encoded.encodedLength <= n) {
      "Input does not fit s$n: ${encoded.encodedLength} encoded tokens"
    }
    require(encoded.labelCount <= LABEL_SLOTS) { "More than $LABEL_SLOTS labels" }
    val paddedIds = IntArray(n) { tokenizer.padId }
    encoded.inputIds.copyInto(paddedIds)
    val attention = FloatArray(n)
    attention.fill(1f, 0, encoded.encodedLength)
    val routing = FloatArray(LABEL_SLOTS * n)
    encoded.labelPositions.forEachIndexed { row, position -> routing[row * n + position] = 1f }
    return Prepared(encoded, n, paddedIds, attention, routing)
  }

  /**
   * The published float16 `[128011,1024]` word-embedding table (little-endian, headerless), upcast
   * to float32 on lookup exactly as numpy's `float16.astype(float32)`: every float16 value is
   * representable in float32, so the upcast is exact. The 262,166,528-byte file is memory-mapped
   * instead of copied to the Java heap. Padding IDs are looked up too.
   */
  class EmbeddingTable(file: File) : Closeable {
    private val channel = RandomAccessFile(file, "r").channel
    private val values: ShortBuffer

    init {
      try {
        require(channel.size() == TABLE_BYTES) {
          "word_embeddings_fp16.bin is not the published [128011,1024] float16 table"
        }
        values =
          channel
            .map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
            .order(ByteOrder.LITTLE_ENDIAN)
            .asShortBuffer()
      } catch (failure: Throwable) {
        channel.close()
        throw failure
      }
    }

    /** One table row of float16 bits, filled by one bulk read per looked-up ID. */
    private val row = ShortArray(HIDDEN_SIZE)

    /** The embeddings [lookup] returns, reused while N stays the same. */
    private var embeds = FloatArray(0)

    /** Raw float16 bits of one table value, for tests. */
    fun halfBits(id: Int, column: Int): Int =
      values.get(id * HIDDEN_SIZE + column).toInt() and 0xffff

    /**
     * Returns row-major `[1,N,1024]` float32 embeddings for N padded IDs, matching `fixed_inputs`
     * fed from the float16 table with host upcast. Each row is copied with one bulk read. The
     * returned array is reused by the next call: write it to the graph before looking up again.
     */
    fun lookup(ids: IntArray): FloatArray {
      if (embeds.size != ids.size * HIDDEN_SIZE) {
        embeds = FloatArray(ids.size * HIDDEN_SIZE)
      }
      for ((position, id) in ids.withIndex()) {
        require(id in 0 until VOCABULARY_SIZE) { "Tokenizer ID outside embedding table: $id" }
        values.position(id * HIDDEN_SIZE)
        values.get(row, 0, HIDDEN_SIZE)
        val target = position * HIDDEN_SIZE
        for (column in 0 until HIDDEN_SIZE) {
          embeds[target + column] = HALF_TO_FLOAT[row[column].toInt() and 0xffff]
        }
      }
      return embeds
    }

    /** Releases the file channel when the host runtime no longer needs embedding lookups. */
    override fun close() = channel.close()
  }

  companion object {
    /** Encoded-token windows N of the three shipped graphs, smallest first. */
    val WINDOWS: List<Int> = listOf(128, 256, 512)

    /** Label slots of every graph: rows of `label_routing` and entries of `logits`. */
    const val LABEL_SLOTS = 32

    /** DeBERTa-v3-large hidden size: one embedding row. */
    const val HIDDEN_SIZE = 1024

    /** Rows of the embedding table (tokenizer vocabulary including GLiNER2's markers). */
    const val VOCABULARY_SIZE = 128011

    /** 128011 × 1024 × 2 bytes = 262,166,528. */
    val TABLE_BYTES: Long = VOCABULARY_SIZE.toLong() * HIDDEN_SIZE * 2
    private const val MARKER_TEXT_MESSAGE =
      "A task name, label or prompt equals a gliner2 structural token ([SEP_STRUCT]/[SEP_TEXT])."

    /** gliner2 `_collate_batch`: append "." unless the text ends with ".", "!" or "?". */
    fun collateText(text: String): String =
      when {
        text.isEmpty() -> "."
        text.endsWith(".") || text.endsWith("!") || text.endsWith("?") -> text
        else -> "$text."
      }

    /**
     * IEEE-754 binary16 → binary32 bits, as numpy's `npy_halfbits_to_floatbits` (sign, exponent
     * rebias, subnormal normalization, infinity/NaN with the payload kept). Hand-written so the
     * same code runs on API 26 and in JVM unit tests: `java.lang.Float.float16ToFloat` is Java 20+,
     * and `android.util.Half` is only a stub on the desktop JVM.
     */
    fun halfToFloat(bits: Int): Float {
      val sign = (bits and 0x8000) shl 16
      val exponent = (bits ushr 10) and 0x1f
      val mantissa = bits and 0x03ff
      val floatBits =
        when {
          exponent == 0x1f -> sign or 0x7f800000 or (mantissa shl 13)
          exponent != 0 -> sign or ((exponent + 112) shl 23) or (mantissa shl 13)
          mantissa == 0 -> sign
          else -> {
            var normalized = mantissa
            var floatExponent = 113
            while (normalized and 0x400 == 0) {
              normalized = normalized shl 1
              floatExponent--
            }
            sign or (floatExponent shl 23) or ((normalized and 0x3ff) shl 13)
          }
        }
      return Float.fromBits(floatBits)
    }

    private val HALF_TO_FLOAT = FloatArray(1 shl 16) { halfToFloat(it) }

    /**
     * Ports gliner2 2.0.0 `processing/word_splitter.py:WhitespaceTokenSplitter.__call__`, the
     * default splitter. Python's Unicode word/space classes are explicit because Java's `\w`
     * includes combining marks and its default `\s` is ASCII-only. Original code-point offsets
     * survive lowercasing, including characters whose lowercase mapping changes length.
     */
    fun splitWords(text: String): List<Word> {
      val result = ArrayList<Word>()
      val matcher = WORD_PATTERN.matcher(text)
      while (matcher.find()) {
        // Lower only the token value. Lowercasing the source first can
        // change its length (for example U+0130) and corrupt offsets.
        result.add(
          Word(
            matcher.group().lowercase(Locale.ROOT),
            text.codePointCount(0, matcher.start()),
            text.codePointCount(0, matcher.end()),
          )
        )
      }
      return result
    }

    private const val SPACE =
      "\\x{09}-\\x{0d}\\x{1c}-\\x{20}\\x{85}\\x{a0}\\x{1680}" +
        "\\x{2000}-\\x{200a}\\x{2028}\\x{2029}\\x{202f}\\x{205f}\\x{3000}"
    private const val WORD = "\\p{L}\\p{N}_"
    private val WORD_PATTERN =
      Pattern.compile(
        "(?:https?://[^$SPACE]+|www\\.[^$SPACE]+)" +
          "|[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,}" +
          "|@[a-z0-9_]+" +
          "|[$WORD]+(?:[-_][$WORD]+)*" +
          "|[^$SPACE]",
        Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE,
      )
  }
}
