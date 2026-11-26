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

package com.google.ai.edge.examples.voicemail_summary

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * What a .litertlm file says about itself. The file starts with "LITERTLM", a version and the
 * offset where its header ends; the header, a flatbuffer from byte 32 on, lists the file's
 * sections. A model's weights are either inside its graphs or in TFLiteWeights sections of their
 * own, and the list shows which.
 */
class LitertlmHeader private constructor(val sections: List<Section>) {
  /** [type] is the schema's AnySectionDataType; [modelType] the section's "model_type" item. */
  class Section(val type: Int, val begin: Long, val end: Long, val modelType: String?)

  /** The length of a whole copy of the file: where its last section ends. */
  val fileBytes: Long
    get() = sections.maxOf { it.end }

  /** Whether the file's weights are in TFLiteWeights sections, not inside its graphs. */
  val weightsInSections: Boolean
    get() = sections.any { it.type == TFLITE_WEIGHTS }

  /** A flatbuffer table at [pos]. A field is named by its place in the vtable: 4, 6, 8... */
  private class Table(val bytes: ByteBuffer, val pos: Int) {
    private fun offset(field: Int): Int {
      val vtable = pos - bytes.getInt(pos)
      val size = bytes.getShort(vtable).toInt() and 0xffff
      return if (field < size) bytes.getShort(vtable + field).toInt() and 0xffff else 0
    }

    private fun target(at: Int) = at + bytes.getInt(at)

    fun ubyte(field: Int): Int =
      offset(field).let { if (it == 0) 0 else bytes.get(pos + it).toInt() and 0xff }

    fun long(field: Int): Long = offset(field).let { if (it == 0) 0 else bytes.getLong(pos + it) }

    fun table(field: Int): Table? =
      offset(field).takeIf { it != 0 }?.let { Table(bytes, target(pos + it)) }

    fun tables(field: Int): List<Table> {
      val at = offset(field)
      if (at == 0) return emptyList()
      val vector = target(pos + at)
      val count = bytes.getInt(vector)
      check(count in 0..(bytes.limit() - vector) / 4)
      return List(count) { Table(bytes, target(vector + 4 + 4 * it)) }
    }

    fun string(field: Int): String? =
      offset(field).takeIf { it != 0 }?.let {
        val at = target(pos + it)
        val length = bytes.getInt(at)
        check(length in 0..bytes.limit() - at - 4)
        String(bytes.array(), at + 4, length, Charsets.UTF_8)
      }
  }

  companion object {
    const val TFLITE_MODEL = 3
    const val HF_TOKENIZER_ZLIB = 6
    const val TFLITE_WEIGHTS = 7
    private const val STRING_VALUE = 9
    private const val START_BYTES = 32
    private const val MAX_HEADER_BYTES = 1L shl 24

    /** The header of [file]; null when it has none: not a .litertlm file, or cut inside it. */
    fun read(file: File): LitertlmHeader? =
      runCatching {
          RandomAccessFile(file, "r").use { input ->
            val start = ByteArray(START_BYTES).also { input.readFully(it) }
            check(String(start, 0, 8, Charsets.US_ASCII) == "LITERTLM")
            val end = ByteBuffer.wrap(start).order(ByteOrder.LITTLE_ENDIAN).getLong(24)
            check(end in START_BYTES..MAX_HEADER_BYTES)
            val header = ByteArray((end - START_BYTES).toInt()).also { input.readFully(it) }
            val bytes = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
            val metadata = Table(bytes, bytes.getInt(0))
            val sections =
              requireNotNull(metadata.table(6)).tables(4).map { section ->
                val modelType =
                  section
                    .tables(4)
                    .firstOrNull { it.string(4) == "model_type" && it.ubyte(6) == STRING_VALUE }
                    ?.table(8)
                    ?.string(4)
                Section(section.ubyte(10), section.long(6), section.long(8), modelType)
              }
            check(sections.isNotEmpty())
            LitertlmHeader(sections)
          }
        }
        .getOrNull()
  }
}
