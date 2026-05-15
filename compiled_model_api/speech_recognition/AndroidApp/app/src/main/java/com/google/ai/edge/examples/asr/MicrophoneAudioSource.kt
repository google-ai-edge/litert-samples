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

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlin.math.sqrt
import kotlin.time.Duration
import kotlin.time.DurationUnit

/** Interface for reading audio data from a source which allows unittest to override the source. */
interface AudioReader : AutoCloseable {
  fun read(audioData: ShortArray, offsetInShorts: Int, sizeInShorts: Int): Int
}

/** Default implementation of AudioReader using Android's AudioRecord API. */
class AudioRecordReader(samplingRate: Int, minBufferSizeInBytes: Int) : AudioReader {
  @SuppressLint("MissingPermission")
  private val audioRecord =
    AudioRecord(
      MediaRecorder.AudioSource.VOICE_RECOGNITION,
      samplingRate,
      AudioFormat.CHANNEL_IN_MONO,
      AudioFormat.ENCODING_PCM_16BIT,
      maxOf(
        minBufferSizeInBytes,
        AudioRecord.getMinBufferSize(
          samplingRate,
          AudioFormat.CHANNEL_IN_MONO,
          AudioFormat.ENCODING_PCM_16BIT,
        ),
      ),
    )

  init {
    if (audioRecord.state != AudioRecord.STATE_INITIALIZED) {
      throw IllegalStateException("AudioRecord initialization failed")
    }

    audioRecord.startRecording()
  }

  override fun close() {
    audioRecord.stop()
    audioRecord.release()
  }

  override fun read(audioData: ShortArray, offsetInShorts: Int, sizeInShorts: Int) =
    audioRecord.read(audioData, offsetInShorts, sizeInShorts)
}

class MicrophoneAudioSource(
  override val samplingRate: Int,
  override val interval: Duration,
  override val overlap: Duration,
  private val getAudioReader: (Int) -> AudioReader = { minBufferSize ->
    AudioRecordReader(samplingRate, minBufferSize)
  },
) : AudioSource {
  @Volatile private var isClosing = false

  init {
    if (interval <= overlap) {
      throw IllegalArgumentException("interval must be strictly longer than overlap")
    }
  }

  @SuppressLint("MissingPermission")
  override fun getAudioData(): Sequence<FloatArray> = sequence {
    val numOfSamplesPerInterval = getNumSamples(interval, samplingRate)
    val numOfSamplesOverlapped = getNumSamples(overlap, samplingRate)
    val step = numOfSamplesPerInterval - numOfSamplesOverlapped

    getAudioReader(step * BYTES_PER_SAMPLE).use { audioReader ->
      // Allocate a buffer large enough to hold one interval plus step samples to avoid overflow.
      val bufferForSamples = FloatArray(numOfSamplesPerInterval + step)
      val rawBuffer = ShortArray(step)
      var numOfSamplesRead = 0

      while (true) {
        // Read audio data into rawBuffer until we have 'step' samples.
        val readBytes = audioReader.read(rawBuffer, numOfSamplesRead, step - numOfSamplesRead)
        when {
          readBytes == 0 ||
            readBytes == AudioRecord.ERROR_DEAD_OBJECT ||
            readBytes == AudioRecord.ERROR_INVALID_OPERATION -> {
            // Permanent errors or end of stream.
            return@sequence
          }
          readBytes < 0 -> {
            // Transient errors like ERROR (-1) or ERROR_BAD_VALUE (-2) can happen if the
            // audio buffer is temporarily locked or starved. We skip the bad frame and
            // retry rather than permanently killing the user's recording session.
            continue
          }
          else -> { // readBytes > 0
            numOfSamplesRead += readBytes
            if (numOfSamplesRead < step) continue
            numOfSamplesRead = 0
          }
        }

        // Now we have 'step' new samples in rawBuffer. Normalize and copy them to bufferForSamples.
        var sumSq = 0.0f
        for (i in 0 until step) {
          val sample = rawBuffer[i] / MAX_16BIT_PCM_VALUE
          sumSq += sample * sample
        }

        // Normalize audio samples and fill them at then end of bufferForSamples.
        val rms = sqrt((sumSq / step).toDouble()).toFloat()
        val isSilent = rms < SILENCE_THRESHOLD
        val sampleStartIndex = bufferForSamples.size - step
        for (i in 0 until step) {
          val normalizedSample = if (isSilent) 0.0f else rawBuffer[i] / MAX_16BIT_PCM_VALUE
          bufferForSamples[sampleStartIndex + i] = normalizedSample
        }

        yield(bufferForSamples)

        // Shift the buffer to the left by 'step' samples to make room for the next samples.
        System.arraycopy(bufferForSamples, step, bufferForSamples, 0, bufferForSamples.size - step)
      }
    }
  }

  override fun close() {}

  private companion object {
    const val MAX_16BIT_PCM_VALUE = 32768.0f
    const val SILENCE_THRESHOLD = 0.005f
    const val BYTES_PER_SAMPLE = 4

    fun getNumSamples(duration: Duration, samplingRate: Int): Int =
      (samplingRate * duration.toDouble(DurationUnit.SECONDS)).toInt()
  }
}
