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

package com.google.ai.edge.examples.ocr_verification

import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.ShortBuffer
import java.nio.channels.FileChannel
import org.json.JSONObject

/**
 * The three bfloat16 tables of `tables/tables.safetensors` (the source checkpoint's bytes): the
 * token embedding (248,320 x 2048, memory-mapped, rows read on demand), the picture position
 * embedding (2304 x 1024) and the two rows of `score.weight`. A bfloat16 value is the top 16 bits
 * of a float32, so the rows widen exactly.
 */
class Bf16Tables(file: File) : Closeable {
  private val raf = RandomAccessFile(file, "r")
  private val channel: FileChannel = raf.channel
  private val embed: ShortBuffer
  val embedRows: Int
  val posTable: FloatArray
  val scoreWeight: FloatArray

  init {
    try {
      val lenBuf = java.nio.ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
      channel.read(lenBuf, 0)
      lenBuf.flip()
      val headerLen = lenBuf.long
      require(headerLen in 2..10_000_000) { "not a safetensors file: header of $headerLen bytes" }
      val header = java.nio.ByteBuffer.allocate(headerLen.toInt())
      channel.read(header, 8)
      val json = JSONObject(String(header.array(), Charsets.UTF_8))
      val dataStart = 8 + headerLen

      fun entry(key: String, rows: Int, cols: Int): Pair<Long, Long> {
        val e = json.getJSONObject(key)
        require(e.getString("dtype") == "BF16") { "$key is ${e.getString("dtype")}, not BF16" }
        val shape = e.getJSONArray("shape")
        require(shape.length() == 2 && shape.getInt(0) == rows && shape.getInt(1) == cols) {
          "$key has shape $shape, expected [$rows, $cols]"
        }
        val offsets = e.getJSONArray("data_offsets")
        val start = dataStart + offsets.getLong(0)
        val end = dataStart + offsets.getLong(1)
        require(end - start == rows.toLong() * cols * 2) { "$key: byte count does not match" }
        return start to end
      }

      embedRows = 248_320
      val (eStart, eEnd) = entry(EMBED_KEY, embedRows, HostMath.HIDDEN)
      embed =
        channel
          .map(FileChannel.MapMode.READ_ONLY, eStart, eEnd - eStart)
          .order(ByteOrder.LITTLE_ENDIAN)
          .asShortBuffer()
      posTable = readAll(entry(POS_KEY, HostMath.SIDE * HostMath.SIDE, HostMath.POS_DIM))
      scoreWeight = readAll(entry(SCORE_KEY, 2, HostMath.HIDDEN))
    } catch (e: Throwable) {
      runCatching { raf.close() }
      throw e
    }
  }

  private fun readAll(range: Pair<Long, Long>): FloatArray {
    val n = ((range.second - range.first) / 2).toInt()
    val buf = java.nio.ByteBuffer.allocate(n * 2).order(ByteOrder.LITTLE_ENDIAN)
    var at = range.first
    while (buf.hasRemaining()) {
      val r = channel.read(buf, at)
      require(r > 0) { "short read of the tables file" }
      at += r
    }
    buf.flip()
    val shorts = buf.asShortBuffer()
    return FloatArray(n) { widen(shorts.get(it)) }
  }

  /** Writes the embedding row of [id] (2048 values) into [out] at [at]. */
  fun embedRow(id: Int, out: FloatArray, at: Int) {
    require(id in 0 until embedRows) { "token id $id outside the table" }
    val base = id * HostMath.HIDDEN
    for (d in 0 until HostMath.HIDDEN) {
      out[at + d] = widen(embed.get(base + d))
    }
  }

  override fun close() {
    runCatching { channel.close() }
    runCatching { raf.close() }
  }

  companion object {
    const val EMBED_KEY = "model.language_model.embed_tokens.weight"
    const val POS_KEY = "model.visual.pos_embed.weight"
    const val SCORE_KEY = "score.weight"

    fun widen(bits: Short): Float = Float.fromBits((bits.toInt() and 0xFFFF) shl 16)
  }
}
