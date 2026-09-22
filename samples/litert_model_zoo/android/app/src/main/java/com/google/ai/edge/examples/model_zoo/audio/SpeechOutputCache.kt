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

package com.google.ai.edge.examples.model_zoo.audio

import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.StandardCopyOption

/** Private, replaceable WAV output. Samples are the same float32 values used by AudioTrack. */
class SpeechOutputCache(private val directory: File) {
  private val output = File(directory, "last-speech.wav")

  @Synchronized
  fun save(samples: FloatArray, sampleRate: Int) {
    check(directory.isDirectory || directory.mkdirs()) { "The speech cache is unavailable." }
    val temporary = File.createTempFile("last-speech-", ".tmp", directory)
    try {
      FileOutputStream(temporary).use { file ->
        val buffered = file.buffered()
        FloatWaveWriter.write(samples, sampleRate, buffered)
        buffered.flush()
        file.fd.sync()
      }
      Files.move(
        temporary.toPath(),
        output.toPath(),
        StandardCopyOption.ATOMIC_MOVE,
        StandardCopyOption.REPLACE_EXISTING,
      )
    } finally {
      temporary.delete()
    }
  }

  @Synchronized
  fun copyTo(destination: OutputStream) {
    check(output.isFile) { "Generate speech again before saving its WAV file." }
    output.inputStream().use { it.copyTo(destination) }
  }
}

/** IEEE float32 mono WAV, including the fact chunk required for non-PCM formats. */
internal object FloatWaveWriter {
  fun write(samples: FloatArray, sampleRate: Int, destination: OutputStream) {
    require(sampleRate in 8000..192000) { "The speech sample rate is unsupported." }
    require(samples.isNotEmpty()) { "There is no speech to save." }
    require(samples.size <= (Int.MAX_VALUE - 58) / 4) { "The speech output is too large." }
    require(samples.all { it.isFinite() }) { "The speech output contains invalid samples." }
    val dataBytes = samples.size * 4
    val header = ByteBuffer.allocate(58).order(ByteOrder.LITTLE_ENDIAN)
    header.put("RIFF".toByteArray(Charsets.US_ASCII)).putInt(50 + dataBytes)
    header.put("WAVEfmt ".toByteArray(Charsets.US_ASCII)).putInt(18)
    header.putShort(3).putShort(1).putInt(sampleRate).putInt(sampleRate * 4)
    header.putShort(4).putShort(32).putShort(0)
    header.put("fact".toByteArray(Charsets.US_ASCII)).putInt(4).putInt(samples.size)
    header.put("data".toByteArray(Charsets.US_ASCII)).putInt(dataBytes)
    destination.write(header.array())
    val buffer = ByteBuffer.allocate(4096).order(ByteOrder.LITTLE_ENDIAN)
    for (sample in samples) {
      if (buffer.remaining() < 4) {
        destination.write(buffer.array(), 0, buffer.position())
        buffer.clear()
      }
      buffer.putInt(sample.toRawBits())
    }
    if (buffer.position() > 0) destination.write(buffer.array(), 0, buffer.position())
  }
}
