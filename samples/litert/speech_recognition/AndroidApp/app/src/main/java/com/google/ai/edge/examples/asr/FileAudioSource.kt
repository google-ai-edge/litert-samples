/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.examples.asr

import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds
import kotlin.time.DurationUnit

class FileAudioSource(
  private val inputStream: InputStream,
  override val samplingRate: Int,
  override val interval: Duration,
  override val overlap: Duration,
) : AudioSource {

  init {
    if (interval <= overlap) {
      throw IllegalArgumentException("interval must be strictly longer than overlap")
    }
  }

  private val samplesPerInterval = getNumSamples(interval)
  private val overlapSamples = getNumSamples(overlap)
  private val step = samplesPerInterval - overlapSamples

  override fun close() {}

  override fun getAudioData(): Sequence<FloatArray> = sequence {
    val dataInputStream = validateHeaderAndGetDataStream(inputStream)

    val maxChunkSizeBytes = samplesPerInterval * BYTES_PER_SAMPLE
    val stepSizeBytes = step * BYTES_PER_SAMPLE

    val bufferForSamples = FloatArray(samplesPerInterval)
    val rawBytes = ByteArray(maxChunkSizeBytes)
    var isFirstChunk = true

    while (true) {
      val bytesToRead = if (isFirstChunk) maxChunkSizeBytes else stepSizeBytes
      val readBytes = dataInputStream.read(rawBytes, 0, bytesToRead)
      if (readBytes <= 0) {
        // Stop even when there are some samples in the buffer as those samples are overlapped with
        // the previous chunk and processed already. No need to process them again as the results
        // will likely be worse than the previous chunk due to shorter audio duration.
        break
      }

      val newSamples = readBytes / BYTES_PER_SAMPLE
      val byteBuffer = ByteBuffer.wrap(rawBytes, 0, readBytes)
      byteBuffer.order(ByteOrder.LITTLE_ENDIAN)

      val sampleIndexStart = if (isFirstChunk) 0 else overlapSamples
      for (i in sampleIndexStart until sampleIndexStart + newSamples) {
        bufferForSamples[i] = byteBuffer.short / MAX_PCM_VALUE
      }
      // Fill the rest of the buffer with zeros as silence.
      for (i in sampleIndexStart + newSamples until bufferForSamples.size) {
        bufferForSamples[i] = 0.0f
      }
      isFirstChunk = false

      yield(bufferForSamples)

      // Shift the buffer to the left by 'step' samples to make room for the next samples.
      System.arraycopy(bufferForSamples, step, bufferForSamples, 0, overlapSamples)
    }
  }

  private fun validateHeaderAndGetDataStream(inputStream: InputStream): InputStream {
    val riffHeader = ByteArray(12)
    if (inputStream.read(riffHeader) != 12) {
      throw IllegalArgumentException("Invalid WAV file: Too short")
    }
    require(String(riffHeader, 0, 4) == "RIFF" && String(riffHeader, 8, 4) == "WAVE") {
      "Not a valid RIFF/WAVE file"
    }

    while (true) {
      val chunkHeader = ByteArray(8)
      require(inputStream.read(chunkHeader) == 8) { "Invalid WAV file: Too short chunk header" }

      val headerBuffer = ByteBuffer.wrap(chunkHeader).order(ByteOrder.LITTLE_ENDIAN)
      val chunkId = String(chunkHeader, 0, 4)
      headerBuffer.position(4)
      val chunkSize = headerBuffer.int

      when (chunkId) {
        FMT_CHUNK_ID -> {
          val fmtData = ByteArray(chunkSize)
          readFully(inputStream, fmtData)

          val fmtBuffer = ByteBuffer.wrap(fmtData).order(ByteOrder.LITTLE_ENDIAN)
          val audioFormat = fmtBuffer.short.toInt() // 2 bytes
          require(audioFormat == 1) {
            "Unsupported audio format: expected PCM (1), but got $audioFormat"
          }

          fmtBuffer.position(4) // Skip channels (2) and block align (2)
          val fileSampleRate = fmtBuffer.int // 4 bytes
          require(fileSampleRate == samplingRate) {
            "Unsupported sample rate: expected $samplingRate, but got $fileSampleRate"
          }
        }
        DATA_CHUNK_ID -> return inputStream // InputStream is now at the start of the data chunk.
        else -> skipFully(inputStream, chunkSize.toLong()) // Skip other chunks
      }
    }
  }

  private fun readFully(inputStream: InputStream, buffer: ByteArray) {
    var totalRead = 0
    while (totalRead < buffer.size) {
      val readBytes = inputStream.read(buffer, totalRead, buffer.size - totalRead)
      if (readBytes < 0) throw IllegalArgumentException("Invalid WAV file: Chunk truncated")
      totalRead += readBytes
    }
  }

  private fun skipFully(inputStream: InputStream, byteCount: Long) {
    var skipped = 0L
    while (skipped < byteCount) {
      val skippedBytes = inputStream.skip(byteCount - skipped)
      if (skippedBytes <= 0) throw IllegalArgumentException("Invalid WAV file: Chunk truncated")
      skipped += skippedBytes
    }
  }

  private fun getNumSamples(duration: Duration): Int {
    return (samplingRate * duration.toDouble(DurationUnit.SECONDS)).toInt()
  }

  private companion object {
    const val FMT_CHUNK_ID = "fmt "
    const val DATA_CHUNK_ID = "data"

    // The number of bytes in a standard WAV file header.
    const val WAV_HEADER_SIZE = 44
    // The number of bytes per audio sample in 16-bit PCM format.
    const val BYTES_PER_SAMPLE = 2
    // The normalization factor for 16-bit PCM audio to convert to float in range [-1.0, 1.0].
    const val MAX_PCM_VALUE = 32768.0f
  }
}
