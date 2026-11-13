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

package com.google.ai.edge.examples.voice_to_text

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/** Why a recording ended. */
enum class StopReason {
  /** Stop was tapped. */
  USER,

  /** The recording reached [Prompts.MAX_SECONDS]. */
  CAP,

  /** The app went to the background (the microphone is only the app's while it is in front). */
  BACKGROUND,
}

/** The microphone could not be opened or stopped delivering audio; the message says why. */
class MicrophoneException(message: String) : IllegalStateException(message)

/** One recording: 16 kHz mono samples in [-1, 1]. */
class Recording(val samples: FloatArray, val reason: StopReason, val peak: Float) {
  val seconds: Double
    get() = samples.size / Wav.SAMPLE_RATE.toDouble()
}

/**
 * The microphone as 16 kHz mono 16-bit PCM, from `VOICE_RECOGNITION` (the input tuned for
 * recognizers), for one recording at a time. The capture loop follows this repository's
 * voice_assistant sample (loop/MicSource.kt): reads of 20 ms, the recorder released on the thread
 * that reads it.
 */
class Recorder(private val sampleRate: Int = Wav.SAMPLE_RATE) {
  private val stop = AtomicReference<StopReason?>(null)

  /**
   * Records until [stop] is called or [maxSamples] are in, on the calling thread (a blocking call:
   * run it off the main thread). [onProgress] gets the samples so far and the last 20 ms's peak.
   * `RECORD_AUDIO` must be granted before.
   */
  @SuppressLint("MissingPermission")
  fun capture(maxSamples: Int, onProgress: (samples: Int, chunkPeak: Float) -> Unit): Recording {
    stop.set(null)
    val chunk = sampleRate / 50
    val minBytes =
      AudioRecord.getMinBufferSize(
        sampleRate,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
      )
    if (minBytes <= 0) {
      throw MicrophoneException("This phone cannot record 16 kHz mono audio ($minBytes).")
    }
    val record =
      AudioRecord(
        MediaRecorder.AudioSource.VOICE_RECOGNITION,
        sampleRate,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
        maxOf(minBytes, chunk * 2 * 8),
      )
    if (record.state != AudioRecord.STATE_INITIALIZED) {
      record.release()
      throw MicrophoneException("The microphone did not open.")
    }
    val pcm = ShortArray(maxSamples)
    var n = 0
    var peak = 0
    try {
      record.startRecording()
      if (record.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
        throw MicrophoneException("The microphone is in use by another app.")
      }
      var failures = 0
      while (stop.get() == null) {
        val want = minOf(chunk, maxSamples - n)
        if (want == 0) {
          stop.compareAndSet(null, StopReason.CAP)
          break
        }
        val got = record.read(pcm, n, want)
        when {
          got > 0 -> {
            failures = 0
            var chunkPeak = 0
            for (i in n until n + got) chunkPeak = maxOf(chunkPeak, abs(pcm[i].toInt()))
            peak = maxOf(peak, chunkPeak)
            n += got
            onProgress(n, chunkPeak / 32768f)
          }
          got == AudioRecord.ERROR_DEAD_OBJECT || got == AudioRecord.ERROR_INVALID_OPERATION ->
            throw MicrophoneException("The microphone stopped (AudioRecord.read returned $got).")
          else -> {
            // 0 or a transient error: read again after a pause, for at most two seconds.
            failures++
            if (failures >= MAX_TRANSIENT_FAILURES) {
              throw MicrophoneException("The microphone gave no audio for two seconds.")
            }
            Thread.sleep(TRANSIENT_RETRY_MS)
          }
        }
      }
    } finally {
      runCatching { record.stop() }
      record.release()
    }
    return Recording(Wav.toFloat(pcm, n), stop.get() ?: StopReason.USER, peak / 32768f)
  }

  /** Ends the recording in progress within 20 ms; the first reason given wins. */
  fun stop(reason: StopReason) {
    stop.compareAndSet(null, reason)
  }

  private companion object {
    const val TRANSIENT_RETRY_MS = 5L
    const val MAX_TRANSIENT_FAILURES = 400
  }
}

/** Checks on a recording before it goes to the model. */
object RecordingChecks {
  /** Shorter than this, a recording holds no word. */
  const val MIN_SECONDS = 0.5

  /**
   * At or below this peak (8 of 32,768) the microphone gave digital silence: a phone with
   * microphone access turned off records zeros. Any room noise is above it.
   */
  const val SILENT_PEAK = 8f / 32768f

  /** Null when the recording can go to the model, else what to tell the user. */
  fun problem(seconds: Double, peak: Float): String? =
    when {
      seconds < MIN_SECONDS ->
        "Nothing was recorded: tap Record, speak, then tap Stop."
      peak <= SILENT_PEAK ->
        "The recording is silent. If microphone access is turned off in the phone's quick " +
          "settings, turn it on and record again."
      else -> null
    }

  /** What to tell the user when [SpeechTrim] finds no speech. */
  const val NO_SPEECH =
    "No speech was heard in this recording. Hold the phone closer, speak, and record again."
}

/**
 * Where the speech starts in a recording. The model writes a sentence for room noise alone (5 s of a
 * quiet room came back as a sentence in Spanish on a Galaxy S26), and 1.4 s of room noise before
 * quiet speech turned the English half of a transcript into French, so the app sends the recording
 * from a little before the first loud 20 ms frame to its end, and nothing when fewer than 5 frames
 * are loud. A frame is loud at 4 times the recording's own noise floor (the 20th percentile of the
 * frame levels), and never below [MIN_RMS]. The end is kept: quiet last words fell under the floor.
 */
object SpeechTrim {
  /** 20 ms at 16 kHz. */
  const val FRAME = 320

  /** About -56 dBFS: speech through the air reached a Pixel 11 Pro at 0.002 to 0.007. */
  const val MIN_RMS = 0.0015f

  const val FLOOR_FACTOR = 4f

  /** 100 ms of loud frames, at least. */
  const val MIN_LOUD_FRAMES = 5

  /** Kept before the first loud frame: 200 ms. */
  const val FRAMES_BEFORE = 10

  /** The first sample to send, or null when no speech was found. */
  fun start(samples: FloatArray): Int? {
    val frames = samples.size / FRAME
    if (frames == 0) return null
    val rms =
      FloatArray(frames) { f ->
        var sum = 0.0
        for (i in f * FRAME until (f + 1) * FRAME) sum += samples[i].toDouble() * samples[i]
        sqrt(sum / FRAME).toFloat()
      }
    val floor = rms.sortedArray()[frames / 5]
    val threshold = max(MIN_RMS, FLOOR_FACTOR * floor)
    var first = -1
    var loud = 0
    for (f in 0 until frames) {
      if (rms[f] >= threshold) {
        if (first < 0) first = f
        loud++
      }
    }
    if (loud < MIN_LOUD_FRAMES) return null
    return max(0, first - FRAMES_BEFORE) * FRAME
  }
}
