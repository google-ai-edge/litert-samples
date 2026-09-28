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

package com.google.ai.edge.examples.zero_shot_classification

import java.io.Closeable
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/** Read-only token table; the graph receives gathered float32 embeddings, including the PAD row. */
class LayaEmbeddings(tableFile: File, metadataFile: File) : Closeable {
  /** Table layout and checkpoint provenance, loaded without copying the table. */
  val metadata: Map<String, Any?> = LayaJson.asObject(LayaJson.parse(metadataFile))
  /** Expected table hash from metadata; the installer verifies the file content. */
  val sha256: String
  private var table: ByteBuffer?

  init {
    val shape = LayaJson.asArray(metadata["shape"]).map { (it as Number).toInt() }
    require(shape == listOf(VOCABULARY_SIZE, WIDTH)) { "Expected token table shape [256000, 768]" }
    require(metadata["dtype"] == "float16") { "Expected float16 token table" }
    require(metadata["byte_order"] == "little") { "Expected little-endian token table" }
    require(tableFile.isFile && tableFile.length() == SIZE_BYTES) {
      "Invalid token table size: ${tableFile.length()} B; expected $SIZE_BYTES B"
    }
    (metadata["size_bytes"] as? Number)?.let {
      require(it.toLong() == SIZE_BYTES) { "Token table metadata size mismatch" }
    }
    sha256 = metadata["sha256"] as? String ?: error("Missing token table SHA256")
    require(sha256.length == 64 && sha256.all { it in '0'..'9' || it in 'a'..'f' }) {
      "Invalid token table SHA256"
    }
    // Closing the channel leaves the read-only mapping valid. No copy of the 393 MB table is made.
    table =
      FileInputStream(tableFile).channel.use {
        it.map(FileChannel.MapMode.READ_ONLY, 0, SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
      }
  }

  /** Fill row-major [1, window, 768]; positions after ids use token id 0, not a zero vector. */
  fun gather(
    ids: IntArray,
    window: Int = ids.size,
    destination: FloatArray = FloatArray(window * WIDTH),
  ): FloatArray {
    val mapped = checkNotNull(table) { "LayaEmbeddings is closed" }
    require(window >= ids.size) { "Sequence exceeds embedding window $window" }
    require(window <= Int.MAX_VALUE / WIDTH && destination.size == window * WIDTH) {
      "Embedding destination must contain window * 768 float32 values"
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
    /** Vocabulary rows in the multilingual checkpoint. */
    const val VOCABULARY_SIZE = 256000
    /** Float16 values per token row. */
    const val WIDTH = 768
    /** Padding gathers a real embedding row rather than a zero vector. */
    const val PAD_ID = 0
    /** Complete row-major binary16 table size. */
    const val SIZE_BYTES = 393_216_000L

    /** Exact IEEE-754 binary16 → binary32 expansion, including signed zero, subnormals and NaNs. */
    fun halfToFloat(bits: Int): Float {
      val sign = (bits and 0x8000) shl 16
      val exponent = (bits ushr 10) and 0x1f
      val fraction = bits and 0x3ff
      val expanded =
        when (exponent) {
          0 ->
            if (fraction == 0) sign
            else {
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
