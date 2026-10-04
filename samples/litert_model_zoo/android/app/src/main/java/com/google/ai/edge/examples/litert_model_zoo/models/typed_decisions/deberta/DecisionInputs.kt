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

// Vendored from https://huggingface.co/litert-community/Open-Decision-DeBERTa-v3-Large-LiteRT/blob/7a276235b795e8ad3ae7ac6a9f237daa2098863a/android/sample/app/src/main/java/com/opendecision/DecisionInputs.kt (Apache-2.0)
// Formatted for this repository. One change from the source: EmbeddingTable.lookup reads each
// row with one bulk read into reused arrays.
package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.deberta

import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.ShortBuffer
import java.nio.channels.FileChannel

/**
 * The author's `Collator.encode_one` and the graph's input contract (`decision_litert.py` in the
 * model repository): `[CLS] [STATE] state[:256] ([Q] instructions ([OPT] option)+)+ [SEP]`, the
 * text span of every question and option (markers excluded), then for one window N the float32
 * inputs `inputs_embeds [1,N,1024]`, `attention_mask [1,N]`, `q_routing [1,128,N]` and `o_routing
 * [1,128,N]`, flattened row-major. N is the smallest of 256 / 512 that holds the sequence; longer
 * requests and more than 128 options are rejected, never truncated.
 */
class DecisionInputs(private val tokenizer: DecisionTokenizer) {
  /**
   * Half-open token range of one text, markers excluded. An empty range (no tokens) routes to zero.
   */
  data class Span(val start: Int, val end: Int)

  /** Window-independent encoding of one request. */
  data class Encoded(
    val inputIds: IntArray,
    val questionSpans: List<Span>,
    val optionSpans: List<List<Span>>,
  ) {
    val encodedLength: Int
      get() = inputIds.size

    val optionCount: Int
      get() = optionSpans.sumOf { it.size }
  }

  /**
   * Graph-ready inputs for one window; [attentionMask], [qRouting] and [oRouting] are flattened
   * row-major.
   */
  data class Prepared(
    val encoded: Encoded,
    val window: Int,
    val inputIds: IntArray,
    val attentionMask: FloatArray,
    val qRouting: FloatArray,
    val oRouting: FloatArray,
  )

  fun encode(state: String, questions: List<Question>): Encoded {
    Question.validate(questions)
    val ids = ArrayList<Int>()
    ids.add(tokenizer.clsId)
    ids.add(tokenizer.stateId)
    val stateIds = tokenizer.encode(state)
    for (i in 0 until minOf(stateIds.size, MAX_STATE_TOKENS)) {
      ids.add(stateIds[i])
    }
    val questionSpans = ArrayList<Span>()
    val optionSpans = ArrayList<List<Span>>()
    for (question in questions) {
      val instructions = tokenizer.encode(question.instructions)
      questionSpans.add(Span(ids.size + 1, ids.size + 1 + instructions.size))
      ids.add(tokenizer.questionId)
      instructions.forEach { ids.add(it) }
      val spans = ArrayList<Span>()
      for (option in question.options) {
        val tokens = tokenizer.encode(option)
        spans.add(Span(ids.size + 1, ids.size + 1 + tokens.size))
        ids.add(tokenizer.optionId)
        tokens.forEach { ids.add(it) }
      }
      optionSpans.add(spans)
    }
    ids.add(tokenizer.sepId)
    require(ids.size <= MAX_LENGTH) {
      "The request has ${ids.size} tokens; the model's limit is $MAX_LENGTH " +
        "(state + ${questions.size} questions)."
    }
    val optionCount = optionSpans.sumOf { it.size }
    require(optionCount <= OPTION_SLOTS) {
      "$optionCount options exceed the graph's $OPTION_SLOTS option slots."
    }
    return Encoded(ids.toIntArray(), questionSpans, optionSpans)
  }

  /** Pads to [window] (null = the smallest window that fits) and builds the four inputs. */
  fun prepare(encoded: Encoded, window: Int? = null): Prepared {
    val n = encoded.encodedLength
    val chosen = window ?: WINDOWS.firstOrNull { it >= n }
      ?: throw IllegalArgumentException(
        "The request has $n tokens; the largest window is ${WINDOWS.last()}."
      )
    require(n <= chosen) { "The request has $n tokens; the window is $chosen." }
    val ids = IntArray(chosen) { tokenizer.padId }
    encoded.inputIds.copyInto(ids)
    val attention = FloatArray(chosen) { if (it < n) 1f else 0f }
    val qRouting = FloatArray(OPTION_SLOTS * chosen)
    val oRouting = FloatArray(OPTION_SLOTS * chosen)
    var slot = 0
    for ((q, spans) in encoded.questionSpans.zip(encoded.optionSpans)) {
      for (o in spans) {
        if (q.end > q.start) {
          val weight = 1f / (q.end - q.start)
          for (t in q.start until q.end) {
            qRouting[slot * chosen + t] = weight
          }
        }
        if (o.end > o.start) {
          val weight = 1f / (o.end - o.start)
          for (t in o.start until o.end) {
            oRouting[slot * chosen + t] = weight
          }
        }
        slot++
      }
    }
    return Prepared(encoded, chosen, ids, attention, qRouting, oRouting)
  }

  fun prepare(state: String, questions: List<Question>, window: Int? = null): Prepared =
    prepare(encode(state, questions), window)

  /**
   * The float16 word table `[128100,1024]` (little-endian, no header), memory-mapped; rows are
   * widened to float32 exactly (every float16 value is representable), as NumPy's
   * `astype(np.float32)` does on the desktop.
   */
  class EmbeddingTable(file: File) : Closeable {
    private val channel = RandomAccessFile(file, "r").channel
    private val shorts: ShortBuffer

    init {
      require(file.length() == TABLE_BYTES) { "Unexpected table size ${file.length()}" }
      shorts =
        channel
          .map(FileChannel.MapMode.READ_ONLY, 0, file.length())
          .order(ByteOrder.LITTLE_ENDIAN)
          .asShortBuffer()
    }

    /** One table row of float16 bits, filled by one bulk read per looked-up ID. */
    private val row = ShortArray(HIDDEN_SIZE)

    /** The embeddings [lookup] returns, reused while N stays the same. */
    private var embeds = FloatArray(0)

    /**
     * `inputs_embeds` for the padded [inputIds]: `[N,1024]` row-major, pad rows included. Each row
     * is copied with one bulk read and widened through [HALF_TO_FLOAT]. The returned array is
     * reused by the next call.
     */
    fun lookup(inputIds: IntArray): FloatArray {
      if (embeds.size != inputIds.size * HIDDEN_SIZE) {
        embeds = FloatArray(inputIds.size * HIDDEN_SIZE)
      }
      for ((position, id) in inputIds.withIndex()) {
        require(id in 0 until VOCABULARY_SIZE) { "Token id $id is outside the table" }
        shorts.position(id * HIDDEN_SIZE)
        shorts.get(row, 0, HIDDEN_SIZE)
        val offset = position * HIDDEN_SIZE
        for (c in 0 until HIDDEN_SIZE) {
          embeds[offset + c] = HALF_TO_FLOAT[row[c].toInt() and 0xffff]
        }
      }
      return embeds
    }

    override fun close() = channel.close()
  }

  companion object {
    const val HIDDEN_SIZE = 1024
    const val VOCABULARY_SIZE = 128_100
    const val OPTION_SLOTS = 128
    const val MAX_STATE_TOKENS = 256
    const val MAX_LENGTH = 512
    const val TABLE_BYTES = VOCABULARY_SIZE.toLong() * HIDDEN_SIZE * 2
    val WINDOWS = listOf(256, 512)

    /**
     * IEEE 754 binary16 → binary32, exact for every bit pattern (normals, subnormals, zeros,
     * infinities, NaN).
     */
    fun halfToFloat(half: Short): Float {
      val h = half.toInt() and 0xffff
      val sign = (h and 0x8000) shl 16
      val exponent = (h shr 10) and 0x1f
      val mantissa = h and 0x3ff
      val bits =
        when {
          exponent == 0 -> {
            if (mantissa == 0) {
              sign
            } else {
              // Subnormal: normalize the mantissa.
              var m = mantissa
              var e = 127 - 15 + 1
              while (m and 0x400 == 0) {
                m = m shl 1
                e--
              }
              sign or (e shl 23) or ((m and 0x3ff) shl 13)
            }
          }
          exponent == 0x1f -> sign or 0x7f800000 or (mantissa shl 13)
          else -> sign or ((exponent + 127 - 15) shl 23) or (mantissa shl 13)
        }
      return java.lang.Float.intBitsToFloat(bits)
    }

    /** [halfToFloat] of every float16 bit pattern, indexed by the unsigned bits. */
    private val HALF_TO_FLOAT = FloatArray(1 shl 16) { halfToFloat(it.toShort()) }
  }
}
