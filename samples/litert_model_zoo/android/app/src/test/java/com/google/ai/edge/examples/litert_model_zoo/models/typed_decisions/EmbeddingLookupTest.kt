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

package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions

import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.deberta.DecisionInputs
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.gliclass.GliclassInputs
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.gliner.DecideInputs
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.julia.JuliaEmbeddings
import com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.laya.LayaEmbeddings
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.ShortBuffer
import java.nio.channels.FileChannel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.fail
import org.junit.Test

/**
 * The five text hosts read each embedding row with one bulk read, where their sources read value by
 * value. Same IDs and same table give the same bits: `lookup` into reused arrays (GLiNER2.5-Decide,
 * GLiClass-Edge, Open-Decision) and `gather` into the caller's array, PAD rows included (Laya,
 * Julia-1).
 */
class EmbeddingLookupTest {
  @Test
  fun bulkLookupGivesTheSourceBits() {
    check(
      "GLiNER2.5-Decide",
      DecideInputs.VOCABULARY_SIZE,
      DecideInputs.HIDDEN_SIZE,
      windows = listOf(128, 512),
      open = { DecideInputs.EmbeddingTable(it) },
      lookup = DecideInputs.EmbeddingTable::lookup,
      widen = { DecideInputs.halfToFloat(it) },
    )
    check(
      "GLiClass-Edge",
      GliclassInputs.VOCABULARY_SIZE,
      GliclassInputs.HIDDEN_SIZE,
      windows = listOf(256, 128),
      open = { GliclassInputs.EmbeddingTable(it) },
      lookup = GliclassInputs.EmbeddingTable::lookup,
      widen = { GliclassInputs.halfToFloat(it) },
    )
    check(
      "Open-Decision",
      DecisionInputs.VOCABULARY_SIZE,
      DecisionInputs.HIDDEN_SIZE,
      windows = listOf(256, 512),
      open = { DecisionInputs.EmbeddingTable(it) },
      lookup = DecisionInputs.EmbeddingTable::lookup,
      widen = { DecisionInputs.halfToFloat(it.toShort()) },
    )
  }

  @Test
  fun bulkGatherGivesTheSourceBits() {
    val metadata = File.createTempFile("token_embeddings", ".json")
    try {
      metadata.writeText(
        "{\"shape\": [${LayaEmbeddings.VOCABULARY_SIZE}, ${LayaEmbeddings.WIDTH}], " +
          "\"dtype\": \"float16\", \"byte_order\": \"little\", " +
          "\"size_bytes\": ${LayaEmbeddings.SIZE_BYTES}, \"sha256\": \"${"0".repeat(64)}\"}"
      )
      checkGather(
        "Laya",
        LayaEmbeddings.VOCABULARY_SIZE,
        LayaEmbeddings.WIDTH,
        LayaEmbeddings.PAD_ID,
        window = 256,
        open = { LayaEmbeddings(it, metadata) },
        gatherNew = { table, ids, window -> table.gather(ids, window) },
        gatherInto = LayaEmbeddings::gather,
        widen = { LayaEmbeddings.halfToFloat(it) },
      )
    } finally {
      metadata.delete()
    }
    checkGather(
      "Julia-1",
      JuliaEmbeddings.VOCABULARY_SIZE,
      JuliaEmbeddings.WIDTH,
      JuliaEmbeddings.PAD_ID,
      window = 512,
      open = { JuliaEmbeddings(it) },
      gatherNew = { table, ids, window -> table.gather(ids, window) },
      gatherInto = JuliaEmbeddings::gather,
      widen = { JuliaEmbeddings.halfToFloat(it) },
    )
  }

  /**
   * A table file of the published size whose last rows hold all 65,536 float16 bit patterns (zeros
   * elsewhere), looked up twice at the first window and once at the second.
   */
  private fun <T : AutoCloseable> check(
    family: String,
    vocabulary: Int,
    hidden: Int,
    windows: List<Int>,
    open: (File) -> T,
    lookup: (T, IntArray) -> FloatArray,
    widen: (Int) -> Float,
  ) {
    val patternRows = (PATTERNS + hidden - 1) / hidden
    val file = File.createTempFile("embeddings", ".bin")
    try {
      writeTable(file, vocabulary, hidden, patternRows, alsoFirstRows = false)
      val values = map(file)
      val first = ids(vocabulary, patternRows, windows[0], salt = 1)
      val second = ids(vocabulary, patternRows, windows[0], salt = 2)
      val resized = ids(vocabulary, patternRows, windows[1], salt = 3)
      open(file).use { table ->
        val a = lookup(table, first)
        assertSameBits("$family first", sourceLookup(values, hidden, first, widen), a)
        val b = lookup(table, second)
        assertSame("$family reuses its array", a, b)
        assertSameBits("$family second", sourceLookup(values, hidden, second, widen), b)
        val c = lookup(table, resized)
        assertSameBits("$family resized", sourceLookup(values, hidden, resized, widen), c)
      }
    } finally {
      file.delete()
    }
  }

  /**
   * The same table, with the patterns also in the first rows so that the PAD row (row 0) is not a
   * zero vector, gathered into a new array and twice into one caller's array (stale values from the
   * previous call must not survive), each time with PAD positions after the IDs.
   */
  private fun <T : AutoCloseable> checkGather(
    family: String,
    vocabulary: Int,
    hidden: Int,
    pad: Int,
    window: Int,
    open: (File) -> T,
    gatherNew: (T, IntArray, Int) -> FloatArray,
    gatherInto: (T, IntArray, Int, FloatArray) -> FloatArray,
    widen: (Int) -> Float,
  ) {
    val patternRows = (PATTERNS + hidden - 1) / hidden
    val file = File.createTempFile("embeddings", ".bin")
    try {
      writeTable(file, vocabulary, hidden, patternRows, alsoFirstRows = true)
      val values = map(file)
      val first = ids(vocabulary, patternRows, window - PAD_POSITIONS, salt = 4)
      val second = ids(vocabulary, patternRows, window - PAD_POSITIONS, salt = 5)
      open(file).use { table ->
        val a = gatherNew(table, first, window)
        val expectedFirst = sourceGather(values, hidden, pad, first, window, widen)
        assertSameBits("$family new array", expectedFirst, a)
        val destination = FloatArray(window * hidden) { Float.NaN }
        val b = gatherInto(table, second, window, destination)
        assertSame("$family fills the caller's array", destination, b)
        val expectedSecond = sourceGather(values, hidden, pad, second, window, widen)
        assertSameBits("$family caller's array", expectedSecond, b)
        val c = gatherInto(table, first, window, destination)
        assertSameBits("$family caller's array again", expectedFirst, c)
      }
    } finally {
      file.delete()
    }
  }

  /**
   * A table file of the published size, zeros except all 65,536 float16 bit patterns in its last
   * [patternRows] rows and, with [alsoFirstRows], in its first rows too.
   */
  private fun writeTable(
    file: File,
    vocabulary: Int,
    hidden: Int,
    patternRows: Int,
    alsoFirstRows: Boolean,
  ) {
    RandomAccessFile(file, "rw").use { raf ->
      raf.setLength(vocabulary.toLong() * hidden * 2)
      val bytes = ByteBuffer.allocate(patternRows * hidden * 2).order(ByteOrder.LITTLE_ENDIAN)
      for (bits in 0 until PATTERNS) {
        bytes.putShort(bits.toShort())
      }
      raf.seek((vocabulary - patternRows).toLong() * hidden * 2)
      raf.write(bytes.array())
      if (alsoFirstRows) {
        raf.seek(0)
        raf.write(bytes.array())
      }
    }
  }

  /** Every pattern row, then the first, a middle and the last row, then spread-out rows. */
  private fun ids(vocabulary: Int, patternRows: Int, window: Int, salt: Int): IntArray {
    val chosen = ArrayList<Int>()
    for (row in vocabulary - patternRows until vocabulary) {
      chosen += row
    }
    chosen += listOf(0, vocabulary / 2, vocabulary - 1)
    var next = salt
    while (chosen.size < window) {
      next = (next * 7919 + 104729) % vocabulary
      chosen += next
    }
    return chosen.take(window).shuffled(java.util.Random(salt.toLong())).toIntArray()
  }

  private fun map(file: File): ShortBuffer =
    RandomAccessFile(file, "r").channel.use { channel ->
      channel
        .map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
        .order(ByteOrder.LITTLE_ENDIAN)
        .asShortBuffer()
    }

  /** The sources' lookup: a new array and one absolute read per value. */
  private fun sourceLookup(
    values: ShortBuffer,
    hidden: Int,
    ids: IntArray,
    widen: (Int) -> Float,
  ): FloatArray {
    val out = FloatArray(ids.size * hidden)
    for ((row, id) in ids.withIndex()) {
      for (column in 0 until hidden) {
        out[row * hidden + column] = widen(values.get(id * hidden + column).toInt() and 0xffff)
      }
    }
    return out
  }

  /** The sources' gather: one absolute read per value, the PAD row after the IDs. */
  private fun sourceGather(
    values: ShortBuffer,
    hidden: Int,
    pad: Int,
    ids: IntArray,
    window: Int,
    widen: (Int) -> Float,
  ): FloatArray {
    val out = FloatArray(window * hidden)
    for (position in 0 until window) {
      val token = if (position < ids.size) ids[position] else pad
      for (column in 0 until hidden) {
        out[position * hidden + column] = widen(values.get(token * hidden + column).toInt())
      }
    }
    return out
  }

  private fun assertSameBits(what: String, expected: FloatArray, actual: FloatArray) {
    assertEquals("$what: size", expected.size, actual.size)
    for (i in expected.indices) {
      val bits = actual[i].toRawBits()
      val sourceBits = expected[i].toRawBits()
      if (bits != sourceBits) {
        fail("$what: value $i has bits $bits, the source $sourceBits")
      }
    }
  }

  private companion object {
    /** Every float16 bit pattern once. */
    const val PATTERNS = 1 shl 16

    /** Positions after the IDs that the gather hosts fill with the PAD row. */
    const val PAD_POSITIONS = 16
  }
}
