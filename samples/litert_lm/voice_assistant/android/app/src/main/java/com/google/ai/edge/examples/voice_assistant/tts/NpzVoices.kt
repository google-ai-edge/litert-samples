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

// Adapted from john-rocky/hfmodels-android (commit 3086d647):
// litert/src/main/kotlin/io/github/johnrocky/hfmodels/litert/NpzVoices.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.tts

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.ZipFile

/**
 * The style tables of a kitten-family `voices.npz`: a zip of `<voice>.npy`, each a C-order
 * little-endian float32 array of shape (rows, dim). Read with java.util.zip and the npy header
 * (format 1.0 to 3.0); stored and deflated entries both work.
 */
internal class NpzVoices private constructor(private val tables: Map<String, Table>) {
  class Table(val rows: Int, val dim: Int, val data: FloatArray) {
    /** Row [row] as a new array of [dim] floats. */
    fun row(row: Int): FloatArray = data.copyOfRange(row * dim, (row + 1) * dim)
  }

  /** The voice names in the zip's entry order. */
  val names: List<String>
    get() = tables.keys.toList()

  operator fun get(name: String): Table? = tables[name]

  companion object {
    fun read(file: File): NpzVoices =
      ZipFile(file).use { zip ->
        val out = LinkedHashMap<String, Table>()
        for (e in zip.entries()) {
          if (e.isDirectory || !e.name.endsWith(".npy")) {
            continue
          }
          val bytes = zip.getInputStream(e).use { it.readBytes() }
          out[e.name.removeSuffix(".npy")] = parseNpy(bytes, e.name)
        }
        NpzVoices(out)
      }

    /** One npy file holding a 2-D little-endian float32 array in C order. */
    fun parseNpy(b: ByteArray, name: String): Table {
      require(
        b.size >= 10 && b[0] == 0x93.toByte() && String(b, 1, 5, Charsets.US_ASCII) == "NUMPY"
      ) {
        "$name: not an npy file"
      }
      val le = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
      val (headerLen, start) =
        when (val major = b[6].toInt()) {
          1 -> (le.getShort(8).toInt() and 0xffff) to 10
          2,
          3 -> le.getInt(8) to 12
          else -> throw IllegalArgumentException("$name: npy format $major.${b[7]} is not known")
        }
      require(start + headerLen <= b.size) { "$name: truncated npy header" }
      val header = String(b, start, headerLen, Charsets.ISO_8859_1)
      val descr = Regex("'descr'\\s*:\\s*'([^']*)'").find(header)?.groupValues?.get(1)
      val fortran =
        Regex("'fortran_order'\\s*:\\s*(True|False)").find(header)?.groupValues?.get(1)
      val shape =
        Regex("'shape'\\s*:\\s*\\(([^)]*)\\)")
          .find(header)
          ?.groupValues
          ?.get(1)
          ?.split(',')
          ?.map { it.trim() }
          ?.filter { it.isNotEmpty() }
          ?.map { it.toInt() }
      require(descr == "<f4") { "$name: dtype '$descr', expected '<f4' (little-endian float32)" }
      require(fortran == "False") { "$name: fortran_order=$fortran, expected False" }
      require(shape != null && shape.size == 2) { "$name: shape $shape, expected (rows, dim)" }
      val n = shape[0] * shape[1]
      val dataStart = start + headerLen
      require(b.size - dataStart == n * 4) {
        "$name: ${b.size - dataStart} data bytes, expected ${n * 4} for shape $shape"
      }
      val data = FloatArray(n)
      ByteBuffer.wrap(b, dataStart, n * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(data)
      return Table(shape[0], shape[1], data)
    }
  }
}
