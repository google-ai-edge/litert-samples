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

package com.google.ai.edge.examples.text_to_speech_lm

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.ShortBuffer
import java.nio.channels.FileChannel
import java.util.zip.ZipFile

/**
 * Minimal NumPy .npy / .npz readers.
 *
 * Supports the two dtypes the Qwen3-TTS host tables use: little-endian float32
 * ('<f4') and float16 ('<f2'). Large fp16 tables are memory-mapped and
 * up-converted per lookup (see [halfToFloat]).
 */
object Npy {

    /** Parsed .npy header: element dtype size, element count, data offset. */
    data class Header(val isHalf: Boolean, val count: Int, val dataOffset: Int)

    private fun parseHeader(bytes: ByteArray): Header {
        require(
            bytes.size > 10 && bytes[0] == 0x93.toByte() &&
                String(bytes, 1, 5) == "NUMPY"
        ) { "not an .npy file" }
        val headerLen: Int
        val headerStart: Int
        if (bytes[6].toInt() == 1) { // version 1.0: u16 header length
            headerLen = (bytes[8].toInt() and 0xFF) or
                ((bytes[9].toInt() and 0xFF) shl 8)
            headerStart = 10
        } else { // version 2.0+: u32 header length
            headerLen = (bytes[8].toInt() and 0xFF) or
                ((bytes[9].toInt() and 0xFF) shl 8) or
                ((bytes[10].toInt() and 0xFF) shl 16) or
                ((bytes[11].toInt() and 0xFF) shl 24)
            headerStart = 12
        }
        val header = String(bytes, headerStart, headerLen)
        val isHalf = when {
            header.contains("'<f2'") -> true
            header.contains("'<f4'") -> false
            else -> throw IllegalArgumentException("unsupported dtype: $header")
        }
        require(!header.contains("'fortran_order': True")) { "fortran order" }
        val shape = Regex("'shape':\\s*\\(([^)]*)\\)")
            .find(header)!!.groupValues[1]
            .split(',').mapNotNull { it.trim().toIntOrNull() }
        val count = shape.fold(1) { a, b -> a * b }
        return Header(isHalf, count, headerStart + headerLen)
    }

    /** Loads a small .npy (fp32 or fp16) fully into a FloatArray. */
    fun loadFloats(file: File): FloatArray {
        val bytes = file.readBytes()
        return floatsFromNpyBytes(bytes)
    }

    /** Parses .npy content already in memory (e.g. an .npz zip entry). */
    fun floatsFromNpyBytes(bytes: ByteArray): FloatArray {
        val h = parseHeader(bytes)
        val buf = ByteBuffer.wrap(bytes, h.dataOffset, bytes.size - h.dataOffset)
            .order(ByteOrder.LITTLE_ENDIAN)
        val out = FloatArray(h.count)
        if (h.isHalf) {
            val sb = buf.asShortBuffer()
            for (i in 0 until h.count) {
                out[i] = halfToFloat(sb.get(i))
            }
        } else {
            buf.asFloatBuffer().get(out)
        }
        return out
    }

    /** Memory-maps a large fp16 .npy; returns a ShortBuffer over the data. */
    fun mmapHalf(file: File): ShortBuffer {
        val head = ByteArray(128)
        RandomAccessFile(file, "r").use { it.readFully(head) }
        val h = parseHeader(head)
        require(h.isHalf) { "expected fp16 table: $file" }
        val channel = RandomAccessFile(file, "r").channel
        val map: MappedByteBuffer = channel.map(
            FileChannel.MapMode.READ_ONLY, h.dataOffset.toLong(),
            file.length() - h.dataOffset)
        return map.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
    }

    /** Loads named fp32 arrays out of an .npz archive. */
    fun loadNpz(file: File, names: List<String>): Map<String, FloatArray> {
        val out = HashMap<String, FloatArray>()
        ZipFile(file).use { zip ->
            for (name in names) {
                val entry = zip.getEntry("$name.npy")
                    ?: throw IllegalArgumentException("$name.npy not in $file")
                zip.getInputStream(entry).use { stream ->
                    out[name] = floatsFromNpyBytes(stream.readBytes())
                }
            }
        }
        return out
    }

    /** IEEE 754 half-precision to float (scalar; hot path is 2048/lookup). */
    fun halfToFloat(h: Short): Float {
        val bits = h.toInt() and 0xFFFF
        val sign = (bits and 0x8000) shl 16
        val exp = (bits ushr 10) and 0x1F
        val mant = bits and 0x3FF
        return when {
            exp == 0 -> {
                if (mant == 0) {
                    Float.fromBits(sign)
                } else { // subnormal
                    var m = mant
                    var e = -1
                    while (m and 0x400 == 0) {
                        m = m shl 1
                        e++
                    }
                    m = m and 0x3FF
                    Float.fromBits(sign or ((127 - 15 - e) shl 23) or (m shl 13))
                }
            }
            exp == 0x1F -> Float.fromBits(
                sign or 0x7F800000 or (mant shl 13)) // inf / nan
            else -> Float.fromBits(sign or ((exp - 15 + 127) shl 23) or (mant shl 13))
        }
    }
}
