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

// Vendored from https://huggingface.co/litert-community/GLiClass-Edge-v3.0-LiteRT/blob/88c90950587eb951974c094eef91afa0fe3552c0/android/sample/app/src/main/java/com/gliclass/GliclassInputs.kt (Apache-2.0)
package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.gliclass

import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.ShortBuffer
import java.nio.channels.FileChannel

/**
 * Ports the input side of the `gliclass` 0.1.20 uni-encoder pipeline (`prepare_input` with
 * `prompt_first`, then the tokenizer call) and the graph contract of GLiClass-Edge v3.0 on LiteRT.
 *
 * Linearized request = `<<LABEL>>label1<<LABEL>>label2…<<SEP>>` + prompt + text (no separator
 * between prompt and text), tokenized with `[CLS]`/`[SEP]`. The graph inputs are float32
 * `inputs_embeds [1,N,384]` (rows of the float16 table, `[PAD]` rows past the request), attention
 * `[1,N]` and `label_routing [1,25,N]` (row k one-hot at the k-th `<<LABEL>>` token), flattened
 * row-major. N is the smallest of 128/256 that holds the sequence; longer inputs and more than 25
 * labels are rejected, never truncated.
 */
class GliclassInputs(private val tokenizer: GliclassTokenizer, private val table: EmbeddingTable) {
  /** Window-independent encoding of one request. */
  data class Encoded(
    val linearized: String,
    val inputIds: IntArray,
    val labelPositions: IntArray,
  ) {
    /** Tokens of the request, `[CLS]` and `[SEP]` included; the window must hold them all. */
    val encodedLength: Int
      get() = inputIds.size

    /** Labels of the request: one `<<LABEL>>` token and one `label_routing` row each. */
    val labelCount: Int
      get() = labelPositions.size
  }

  /** Graph-ready inputs for one window: [inputIds] and [attentionMask] have N entries. */
  data class Prepared(
    val encoded: Encoded,
    val window: Int,
    val inputIds: IntArray,
    val embeds: FloatArray,
    val attentionMask: FloatArray,
    val labelRouting: FloatArray,
  )

  private val labelId = requireNotNull(tokenizer.tokenId(LABEL_TOKEN))

  init {
    require(tokenizer.vocabularySize == VOCABULARY_SIZE) {
      "tokenizer.json has ${tokenizer.vocabularySize} IDs; the embedding table has $VOCABULARY_SIZE"
    }
    require(labelId == LABEL_ID && tokenizer.tokenId(TEXT_SEPARATOR_TOKEN) == TEXT_SEPARATOR_ID) {
      "tokenizer.json does not hold the GLiClass markers at $LABEL_ID / $TEXT_SEPARATOR_ID"
    }
  }

  /** Token IDs and `<<LABEL>>` positions of one request; rejects what the graph cannot hold. */
  fun encode(text: String, labels: List<String>, prompt: String? = null): Encoded {
    require(labels.isNotEmpty()) { "Add at least one label." }
    require(labels.size <= LABEL_SLOTS) {
      "${labels.size} labels; the graph holds $LABEL_SLOTS. Remove some labels."
    }
    val linearized = linearize(text, labels, prompt)
    val ids = tokenizer.encode(linearized)
    val positions = labelPositions(ids, labelId)
    require(positions.size == labels.size) {
      "The text, prompt or a label contains $LABEL_TOKEN; remove it."
    }
    return Encoded(linearized, ids, positions)
  }

  /** [encode] plus padding for [window], or for the smallest fitting window when null. */
  fun prepare(
    text: String,
    labels: List<String>,
    prompt: String? = null,
    window: Int? = null,
  ): Prepared = pad(encode(text, labels, prompt), window)

  /** Pads an [Encoded] request to [window] (or the smallest fitting window) and looks up rows. */
  fun pad(encoded: Encoded, window: Int? = null): Prepared {
    val n =
      window
        ?: WINDOWS.firstOrNull { encoded.encodedLength <= it }
        ?: throw IllegalArgumentException(
          "Input is ${encoded.encodedLength} tokens; the largest graph holds ${WINDOWS.last()}. " +
            "Shorten the text, the prompt or the labels."
        )
    require(n in WINDOWS) { "Unsupported graph window $n" }
    require(encoded.encodedLength <= n) {
      "Input does not fit s$n: ${encoded.encodedLength} tokens"
    }
    val paddedIds = IntArray(n) { tokenizer.padId }
    encoded.inputIds.copyInto(paddedIds)
    val attention = FloatArray(n)
    attention.fill(1f, 0, encoded.encodedLength)
    val routing = FloatArray(LABEL_SLOTS * n)
    encoded.labelPositions.forEachIndexed { row, position -> routing[row * n + position] = 1f }
    return Prepared(encoded, n, paddedIds, table.lookup(paddedIds), attention, routing)
  }

  /**
   * The float16 `[50370,384]` token-embedding table (little-endian, headerless), upcast to float32
   * on lookup exactly as numpy's `float16.astype(float32)`. The 38,684,160-byte file is
   * memory-mapped instead of copied to the Java heap.
   */
  class EmbeddingTable(file: File) : Closeable {
    private val channel = RandomAccessFile(file, "r").channel
    private val values: ShortBuffer

    init {
      try {
        require(channel.size() == TABLE_BYTES) {
          "${file.name} is not the float16 [$VOCABULARY_SIZE,$HIDDEN_SIZE] embedding table"
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

    /** Row-major `[1,N,384]` float32 embeddings of N IDs. Absolute reads keep the position. */
    fun lookup(ids: IntArray): FloatArray {
      val out = FloatArray(ids.size * HIDDEN_SIZE)
      for ((row, id) in ids.withIndex()) {
        require(id in 0 until VOCABULARY_SIZE) { "Token ID outside the embedding table: $id" }
        val source = id * HIDDEN_SIZE
        val target = row * HIDDEN_SIZE
        for (column in 0 until HIDDEN_SIZE) {
          out[target + column] = HALF_TO_FLOAT[values.get(source + column).toInt() and 0xffff]
        }
      }
      return out
    }

    override fun close() = channel.close()
  }

  companion object {
    /** Encoded-token windows N of the shipped graphs, smallest first. */
    val WINDOWS: List<Int> = listOf(128, 256)

    /** Label slots of every graph: rows of `label_routing` and entries of `logits`. */
    const val LABEL_SLOTS = 25

    /** ModernBERT (ettin-encoder-32m) hidden size: one embedding row. */
    const val HIDDEN_SIZE = 384

    /** Rows of the embedding table: 50,280 BPE tokens plus the added tokens up to `<<SEP>>`. */
    const val VOCABULARY_SIZE = 50370

    /** 50370 × 384 × 2 bytes = 38,684,160. */
    val TABLE_BYTES: Long = VOCABULARY_SIZE.toLong() * HIDDEN_SIZE * 2

    /** Marker before every label in the linearized request (`class_token_index`). */
    const val LABEL_TOKEN = "<<LABEL>>"

    /** Marker between the labels and the prompt + text (`text_token_index`). */
    const val TEXT_SEPARATOR_TOKEN = "<<SEP>>"

    /** Token ID of [LABEL_TOKEN] in `tokenizer.json`. */
    const val LABEL_ID = 50368

    /** Token ID of [TEXT_SEPARATOR_TOKEN] in `tokenizer.json`. */
    const val TEXT_SEPARATOR_ID = 50369

    /** The pipeline's input string: labels, `<<SEP>>`, the prompt (if any), then the text. */
    fun linearize(text: String, labels: List<String>, prompt: String?): String = buildString {
      labels.forEach { append(LABEL_TOKEN).append(it) }
      append(TEXT_SEPARATOR_TOKEN)
      prompt?.let { append(it) }
      append(text)
    }

    /** Positions of [labelId] in [ids], in order. */
    fun labelPositions(ids: IntArray, labelId: Int = LABEL_ID): IntArray =
      ids.indices.filter { ids[it] == labelId }.toIntArray()

    /**
     * IEEE-754 binary16 → binary32 bits, as numpy's `npy_halfbits_to_floatbits`. Hand-written so
     * the same code runs on API 26 and in JVM unit tests (`android.util.Half` is a stub there).
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
  }
}
