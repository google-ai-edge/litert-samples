// Vendored from https://huggingface.co/litert-community/Julia-1-LiteRT/blob/8f36857c56e891c023060586759c6cdc8baf6b3e/android/app/src/main/java/com/julia1/JuliaEmbeddings.kt (Apache-2.0)
// SPDX-License-Identifier: Apache-2.0
package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.julia

import java.io.Closeable
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/** Read-only float16 token table; the graph receives gathered float32 rows, PAD row included. */
class JuliaEmbeddings(tableFile: File) : Closeable {
  private var table: ByteBuffer?

  init {
    require(tableFile.isFile && tableFile.length() == SIZE_BYTES) {
      "Invalid token table size: ${tableFile.length()} B; expected $SIZE_BYTES B"
    }
    // Closing the channel leaves the read-only mapping valid. No copy of the 197 MB table is made.
    table =
      FileInputStream(tableFile).channel.use {
        it.map(FileChannel.MapMode.READ_ONLY, 0, SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
      }
  }

  /** Fill row-major [1, window, 384]; positions after ids use token id 0, not a zero vector. */
  fun gather(
    ids: IntArray,
    window: Int,
    destination: FloatArray = FloatArray(window * WIDTH),
  ): FloatArray {
    val mapped = checkNotNull(table) { "JuliaEmbeddings is closed" }
    require(window >= ids.size) { "Sequence exceeds embedding window $window" }
    require(window <= Int.MAX_VALUE / WIDTH && destination.size == window * WIDTH) {
      "Embedding destination must contain window * 384 float32 values"
    }
    ids.forEach { require(it in 0 until VOCABULARY_SIZE) { "Token id out of range: $it" } }
    var output = 0
    for (position in 0 until window) {
      val token = if (position < ids.size) ids[position] else PAD_ID
      var offset = token * WIDTH * 2
      repeat(WIDTH) {
        destination[output++] = halfToFloat(mapped.getShort(offset).toInt())
        offset += 2
      }
    }
    return destination
  }

  /** The JVM owns unmapping; releasing this reference avoids retaining it after engine close. */
  override fun close() {
    table = null
  }

  companion object {
    /** Vocabulary rows of the mmBERT-small tokenizer. */
    const val VOCABULARY_SIZE = 256000
    /** Float16 values per token row (the encoder hidden size). */
    const val WIDTH = 384
    /** Padding gathers the real PAD embedding row rather than a zero vector. */
    const val PAD_ID = 0
    /** Complete row-major binary16 table size: 256000 * 384 * 2 bytes. */
    const val SIZE_BYTES = 196_608_000L

    /** Exact binary16 to binary32 expansion: signed zero, subnormals and NaNs included. */
    fun halfToFloat(bits: Int): Float {
      val sign = (bits and 0x8000) shl 16
      val exponent = (bits ushr 10) and 0x1f
      val fraction = bits and 0x3ff
      val expanded =
        when (exponent) {
          0 ->
            if (fraction == 0) {
              sign
            } else {
              val shift = Integer.numberOfLeadingZeros(fraction) - 21
              sign or ((113 - shift) shl 23) or (((fraction shl shift) and 0x3ff) shl 13)
            }
          // IEEE conversion quiets signaling NaNs while retaining the sign and payload, as NumPy
          // does.
          31 -> sign or 0x7f800000 or (fraction shl 13) or (if (fraction == 0) 0 else 0x400000)
          else -> sign or ((exponent + 112) shl 23) or (fraction shl 13)
        }
      return Float.fromBits(expanded)
    }
  }
}
